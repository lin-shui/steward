"""Ego Runtime (ER): perception loop, Ego-State store, ESim receipts, skill execution and the
operator view of the Ego-State (rendered on its own thread, see :mod:`steward.viewloop`).

The runtime talks to the world only through the robot port (sensors, actuator commands,
telemetry). It never holds World State identifiers or ground truth.
"""

from __future__ import annotations

import base64
import collections
import io
import itertools
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from steward.ego.classify import classify
from steward.ego.perception import box_of, decode_rgbd, segments
from steward.ego.esim import Esim, EsimConfig
from steward.ego.estimator import SELF_ID, EgoEstimator
from steward.ego.roles import RoleConfig, assign_roles
from steward.ego.skills import MotionConfig, SkillExecutor, SkillFailed
from steward.ego.view import EgoViewRenderer
from steward.math3d import matrix_from_quat, yaw_from_quat
from steward.net import HttpError, call
from steward.scene.schema import SceneState
from steward.viewloop import ViewLoop


@dataclass
class RuntimeConfig:
    perception_hz: float = 10.0
    telemetry_hz: float = 2.0
    max_risk: float = 0.35
    # A grasp verifies itself and a miss costs one retreat, so a pick may be tried at a risk that
    # would be reckless for a move through the room or a release over a surface.
    max_pick_risk: float = 0.6
    # A fresh look taken for an imminent pick may restate the target's position when the new
    # reading agrees with the belief this closely; farther readings are different objects.
    relocate_max_jump_m: float = 0.35
    relocate_min_points: int = 30
    relocate_fallback_m: float = 0.6  # beyond the agree-gate: accept the nearest confident reading
    relocate_fallback_score: float = 0.55
    relocate_max_dz_m: float = 0.12  # an instance this far off in height is a different thing
    fresh_look_every_s: float = 10.0  # a retried pick of the same target reuses the last look
    redock_max_m: float = 0.12  # a correction this small is folded into the plan; larger re-plans
    esim: EsimConfig = field(default_factory=EsimConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    roles: RoleConfig = field(default_factory=RoleConfig)


def body_ok(category: str) -> bool:
    """A text prompt needs a real name; unknown things have nothing to ask for."""
    return bool(category) and category != "unknown"


# A category is one word of the robot's vocabulary; what a person would call the thing may be
# another. The class name alone misses a mug asked for as a cup, so a second name is tried.
PICK_PROMPTS: dict[str, tuple[str, ...]] = {
    "cup": ("cup", "mug"),
    "mug": ("mug", "cup"),
    "wine_glass": ("wine glass", "glass"),
    "bowl": ("bowl", "dish"),
}


def _fresh_look_prompts(category: str, color: str | None) -> tuple[str, ...]:
    """Text prompts to try for one track, best first.

    SAM3's text head needs more than a category to find tableware from above: at 0.95 m "red
    bowl" scores 0.89 and "bowl" nothing at all on the same frame, and the same holds for "black
    mug" against "mug". The colour is the robot's own reading of the object, so putting it in
    front of the name is not being told the answer — it is using the description it already has.
    """
    name = category.replace("_", " ")
    out = []
    if color and color not in ("clear", "unknown"):
        out.append(f"{color} {name}")
    out.append(name)
    out.extend(p for p in PICK_PROMPTS.get(category, ()) if p != name)
    return tuple(dict.fromkeys(out))[:3]  # three prompts at most: each one is a segmenter round trip


class RobotLink:
    """HTTP client of one robot port: the only path between the Ego System and the world."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def sensors(self) -> dict:
        return call("GET", f"{self.url}/sensors")

    def rgbd(self) -> bytes:
        """One RGB-D frame (``npz``) when the world provides raw imaging instead of detections."""
        return call("GET", f"{self.url}/sensors/rgbd")

    def command(self, cmd: dict) -> None:
        call("POST", f"{self.url}/command", cmd)

    def telemetry(self, scene: dict) -> None:
        call("POST", f"{self.url}/telemetry", scene)

    def camera_png(self, width: int = 320, height: int = 240) -> bytes:
        return call("GET", f"{self.url}/camera/rgb?width={width}&height={height}")


ALERT_TEXT_MAX = 1500
QUERY_MIN_PROBABILITY = 0.2
ACTIVITY_MAX = 300
PLAN_STEP_STATES = frozenset({"pending", "active", "done", "failed", "skipped"})

class EgoRuntime:
    def __init__(self, link: RobotLink, robot_model: dict, config: RuntimeConfig | None = None,
                 out_dir: Path | None = None):
        self.link = link
        self.config = config or RuntimeConfig()
        self.estimator = EgoEstimator(robot_model=robot_model)
        self.esim = Esim(robot_model, self.config.esim)
        self.out_dir = out_dir
        self._lock = threading.RLock()
        self._running = False
        self._thread: threading.Thread | None = None
        self.receipts: dict[str, dict] = {}
        self.last_error: str | None = None
        self.control_owner: str | None = None
        self.perception_mode: str | None = None
        self.alerts: list[dict] = []
        self.plan: dict | None = None
        self._looked_at: dict[str, float] = {}  # target id -> body stamp of its last fresh look
        self._roles: dict[str, str] = {}  # the latest role assignment, kept unless clearly outweighed
        self.activity: collections.deque[dict] = collections.deque(maxlen=ACTIVITY_MAX)
        self._activity_ids = itertools.count(1)
        self._alert_ids = itertools.count(1)
        self.executor = SkillExecutor(self.link.command, self._motion_state, self._mark_held, self.config.motion,
                                      shift=self._shift_track, on_error=lambda msg: self.alert(msg, "skill"),
                                      on_finish=self._skill_finished, on_release=self._released,
                                      on_begin=self._skill_begin)
        self._view_scene: tuple[dict, int, float] | None = None  # (ES dict, version, stamp) for the view
        renderer = EgoViewRenderer()
        self.view = ViewLoop("ego", lambda settings: self._render_view(renderer, settings), renderer.close)

    # --------------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Start perception in a worker thread. The first frame is taken there too: RGB-D frames
        are rendered on the world's main thread, which may be the caller."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="ego-perception", daemon=True)
        self._thread.start()
        self.view.start()

    def stop(self) -> None:
        self._running = False
        self.view.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _poll_once(self) -> None:
        frame = self.link.sensors()
        mode = "rgbd" if frame.get("imaging") == "rgbd" else "detections"
        if mode != self.perception_mode:
            with self._lock:
                if self.perception_mode is not None:
                    # The sensors changed kind (a phase switch): objects are perceived afresh with
                    # the new pipeline; what the gripper holds and the running skill are kept.
                    self.estimator.forget_objects()
                self.perception_mode = mode
        if mode == "rgbd":
            rgbd = decode_rgbd(self.link.rgbd())
            with self._lock:
                if rgbd["header"]["stamp"] < self.estimator.stamp - 1e-6:
                    self._restart()
                self.estimator.update_rgbd(rgbd)
                self._publish_view_scene()
            return
        with self._lock:
            if frame["stamp"] < self.estimator.stamp - 1e-6:
                self._restart()
            self.estimator.update(frame)
            self._publish_view_scene()

    def _telemetry_scene(self) -> dict:
        """The Ego-State sent as telemetry, each object carrying its scene-level role as the state
        variable ``role`` (``src: inferred``) next to its perceived category."""
        roles = {o["id"]: (o["role"], o["roleConfidence"]) for o in self.summary()["objects"]}
        scene = self.ego_state().to_dict()
        for ent in scene["entities"]:
            if ent["id"] in roles:
                role, p = roles[ent["id"]]
                ent.setdefault("stateVars", {})["role"] = {"value": role, "unc": {"kind": "categorical", "p": {role: p}},
                                                           "src": "inferred"}
        return scene

    def _publish_view_scene(self) -> None:
        """Hand the view a copy of the Ego-State taken right after a perception update (the caller
        holds the lock), so rendering never waits for the next update to finish."""
        self._view_scene = (self.estimator.to_scene().to_dict(), self.estimator.version, self.estimator.stamp)

    def _render_view(self, renderer: EgoViewRenderer, settings) -> tuple[bytes, str, dict]:
        scene = self._view_scene
        if scene is None:
            with self._lock:
                self._publish_view_scene()
            scene = self._view_scene
        ego, version, stamp = scene
        return renderer.render(ego, settings), "image/jpeg", {"esVersion": version, "t": round(stamp, 3)}

    def _restart(self) -> None:
        """The world clock went backwards: a new episode started, which the robot experiences as a
        reboot. Forget the Ego-State, receipts and any running skill."""
        self.executor.cancel()
        self.estimator = EgoEstimator(robot_model=self.estimator.robot_model, config=self.estimator.config)
        self.receipts.clear()

    def _loop(self) -> None:
        period = 1.0 / self.config.perception_hz
        telemetry_every = max(1, int(self.config.perception_hz / self.config.telemetry_hz))
        k = 0
        while self._running:
            t0 = time.monotonic()
            try:
                self._poll_once()
                k += 1
                if k % telemetry_every == 0:
                    self.link.telemetry(self._telemetry_scene())
                self.last_error = None
            except (HttpError, OSError) as exc:
                self.last_error = f"robot port unavailable: {exc}"
            except Exception as exc:  # noqa: BLE001 - perception must keep running; the operator is alerted
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.alert(f"感知循环出现异常：{type(exc).__name__}: {exc}", "runtime")
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    # --------------------------------------------------------------- state
    def ego_state(self) -> SceneState:
        with self._lock:
            return self.estimator.to_scene()

    def _motion_state(self) -> dict:
        """What the body is doing now, for the motion loop.

        Odometry, joints and the gripper come from the robot's live sensor header, not from the
        belief: with a real segmenter the next RGB-D frame can be seconds or tens of seconds
        away, and a motion loop that grades its progress on the perception clock either stalls
        or reads a frame rendered before it started and calls the turn failed. Objects stay the
        estimate they are — the body only needs to know where it is."""
        with self._lock:
            est = self.estimator
            entities = {t.id: {"p": t.pos.tolist()} for t in est.tracks.values()}
            state = {"stamp": est.stamp, "odometry": dict(est.odometry), "joints": dict(est.joints),
                     "gripper": dict(est.gripper), "entities": entities, "toolOffset": self.esim.tool_offset}
        try:
            live = self.link.sensors()
        except Exception:  # noqa: BLE001 - the belief is still a usable answer when the body cannot be reached
            return state
        for key in ("stamp", "odometry", "joints", "gripper"):
            if live.get(key) is not None:
                state[key] = live[key]
        if live.get("stamp") is not None and live["stamp"] < state["stamp"] - 1e-6:
            state["stamp"] = est.stamp  # an episode reset race: prefer the belief's own clock
        return state

    def _mark_held(self, track_id: str) -> None:
        at = self.estimator.stamp
        try:
            at = float(self.link.sensors().get("stamp", at))  # the body's clock, now
        except Exception:  # noqa: BLE001 - the belief's own clock is a usable second best
            pass
        with self._lock:
            self.estimator.mark_held(track_id, at)

    def _released(self, track_id: str, spot: list[float]) -> None:
        """A place knows where it let go; the belief is told, not left to infer it later."""
        with self._lock:
            self.estimator.release_at(track_id, spot)

    def _shift_track(self, track_id: str, delta: list[float]) -> None:
        with self._lock:
            self.estimator.shift(track_id, delta)

    # ------------------------------------------------------------- Ego API
    def summary(self) -> dict:
        es = self.ego_state()
        me = es.entity(SELF_ID)
        p = me.pose.value
        held = next((r.a["entity"] for r in es.relations if r.type == "attached"), None)
        supports = {r.a["entity"]: r.b["entity"] for r in es.relations if r.type == "supported-by"}
        objects = []
        for ent in es.entities:
            if ent.id == SELF_ID:
                continue
            cov = ent.pose.unc["cov"]
            size = ent.rigid.links[0].geometry["collision"][0]["size"]
            objects.append({
                "id": ent.id,
                "category": ent.category.value,
                "categoryConfidence": round(ent.category.unc["p"].get(ent.category.value, 1.0), 3),
                "categories": ent.category.unc["p"],
                "position": [round(v, 3) for v in ent.pose.value["p"]],
                "yawDeg": round(math.degrees(yaw_from_quat(ent.pose.value["q"])), 1),
                "uncertaintyM": round(math.sqrt((cov[0] + cov[7]) / 2), 4),
                "sizeM": [round(v, 3) for v in size],
                "on": supports.get(ent.id),
                "existence": ent.existence,
                "lastSeenAgoS": round(es.stamp - (ent.pose.t or es.stamp), 2),
                "affordances": [a.value for a in ent.rigid.links[0].affordances],
                "color": ent.state_vars["color"].value if "color" in ent.state_vars else None,
                "extentM": self._extent(ent.id),
                **({"openingM": round(self._opening(ent.id), 3)} if ent.category.value == "drawer" else {}),
                **self._front(ent),
            })
        with self._lock:
            roles = assign_roles(objects, self.config.roles, self._roles)
            self._roles = {oid: role for oid, (role, _) in roles.items()}
        for obj in objects:
            obj["role"], confidence = roles[obj["id"]]
            obj["roleConfidence"] = round(confidence, 3)
        job = self.executor.current.to_dict() if self.executor.current else None
        return {
            "version": es.version,
            "t": round(es.stamp, 3),
            "self": {"x": round(p["p"][0], 3), "y": round(p["p"][1], 3),
                     "yawDeg": round(math.degrees(yaw_from_quat(p["q"])), 1),
                     "holding": held, "joints": {j.id: round(j.q.value[0], 3) for j in me.rigid.joints if j.q.value}},
            "objects": objects,
            "currentSkill": job,
            "perception": self.perception_mode,
            "runtimeError": self.last_error,
        }

    def _extent(self, track_id: str) -> list[float] | None:
        """Estimated physical size of a track, without the uncertainty margin added to ``sizeM``."""
        with self._lock:
            track = self.estimator.tracks.get(track_id)
            return [round(float(v), 3) for v in track.size] if track is not None else None

    def _front(self, ent) -> dict:
        """Heading of the face looking into the room, for furniture and drawers."""
        affordances = [a.value for a in ent.rigid.links[0].affordances]
        if "support-surface" not in affordances and "pullable" not in affordances:
            return {}
        with self._lock:
            track = self.estimator.tracks.get(ent.id)
            if track is None:
                return {}
            box = self.estimator.front(track)
        return {"frontDeg": round(math.degrees(box.yaw), 1), "frontHalfM": round(box.hx, 3)}

    def _opening(self, track_id: str) -> float:
        with self._lock:
            track = self.estimator.tracks.get(track_id)
            return self.estimator.drawer_opening(track) if track is not None else 0.0

    def query(self, category: str | None = None, near: str | None = None, radius: float | None = None) -> list[dict]:
        """Objects matching a category by role or, failing that, by probability (best match first),
        optionally within ``radius`` of another object."""
        objs = self.summary()["objects"]
        if category:
            score = lambda o: 2.0 if o["role"] == category else o["categories"].get(category, 0.0)  # noqa: E731
            objs = sorted((o for o in objs if score(o) >= QUERY_MIN_PROBABILITY), key=score, reverse=True)
        if near:
            anchor = next((o for o in self.summary()["objects"] if o["id"] == near), None)
            if anchor is None:
                raise ValueError(f"unknown entity {near!r}")
            r = radius if radius is not None else 1.0
            objs = [o for o in objs if o["id"] != near and math.dist(o["position"][:2], anchor["position"][:2]) <= r]
        return objs

    def observe(self) -> dict:
        frame = self.link.sensors()
        if frame.get("imaging") == "rgbd":
            rgbd = decode_rgbd(self.link.rgbd())
            with self._lock:
                cam = self.estimator.camera_pose()
            visible = []
            for seg in segments(rgbd, cam):
                centre, size, _ = box_of(seg.points)
                probs = classify(size, centre[2] - size[2] / 2, seg.color, seg.invalid_frac)
                best = max(probs, key=probs.get)
                visible.append({"category": best, "confidence": round(probs[best], 2), "rangeM": round(seg.range, 2)})
        else:
            visible = [{"category": d["category"], "rangeM": round(d["range"], 2)} for d in frame["detections"]]
        out = {"t": frame["stamp"], "visible": visible}
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            path = self.out_dir / f"observe-{frame['stamp']:.2f}.png"
            path.write_bytes(self.link.camera_png())
            out["imagePath"] = str(path)
        return out

    def predict(self, skill: str, params: dict) -> dict:
        es = self.ego_state()
        receipt = self.esim.predict(es, skill, params)
        limit = self.config.max_pick_risk if receipt.get("skill") == "pick" else self.config.max_risk
        receipt["withinRiskLimit"] = receipt["feasible"] and receipt["risk"] <= limit
        receipt["maxRisk"] = limit
        with self._lock:
            self.receipts[receipt["id"]] = receipt
        return receipt

    def receipt(self, receipt_id: str) -> dict:
        if receipt_id not in self.receipts:
            raise KeyError(f"unknown receipt {receipt_id!r}")
        return self.receipts[receipt_id]

    # ------------------------------------------------------------------ alerts
    def alert(self, message: str, source: str, level: str = "error", goal: str = "", step: str = "",
              diagnosis: str = "") -> dict:
        """Record a situation the robot cannot resolve by itself (an unexpected exception, or a
        problem its agent or mission reports) with the goal and step it was pursuing and the
        reporter's own diagnosis. A goal or step left empty is taken from the published plan. A
        repeat of an unacknowledged message only bumps its count."""
        message = message[:ALERT_TEXT_MAX]
        with self._lock:
            plan = self.plan or {}
            for existing in reversed(self.alerts):
                if not existing["acknowledged"] and existing["message"] == message:
                    existing["count"] += 1
                    existing["t"] = round(self.estimator.stamp, 2)
                    return existing
            record = {"id": next(self._alert_ids), "t": round(self.estimator.stamp, 2), "level": level,
                      "source": source, "message": message, "count": 1, "acknowledged": False,
                      "goal": (goal or plan.get("goal", ""))[:ALERT_TEXT_MAX],
                      "step": (step or plan.get("current", ""))[:ALERT_TEXT_MAX],
                      "diagnosis": diagnosis[:ALERT_TEXT_MAX]}
            self.alerts.append(record)
            del self.alerts[:-50]
        self._log("alert", message)
        return record

    def alerts_view(self) -> dict:
        with self._lock:
            return {"alerts": [dict(a) for a in self.alerts],
                    "active": [dict(a) for a in self.alerts if not a["acknowledged"]]}

    def acknowledge(self, alert_id: int | None = None) -> dict:
        with self._lock:
            for record in self.alerts:
                if alert_id is None or record["id"] == alert_id:
                    record["acknowledged"] = True
        return self.alerts_view()

    # ---------------------------------------------------------------- activity
    def _log(self, kind: str, text: str) -> None:
        """Append to the activity record the console shows below the Ego-State view: ``step``,
        ``difficulty``, ``state``, ``skill`` and ``alert`` entries, newest last."""
        with self._lock:
            self.activity.append({"id": next(self._activity_ids), "t": round(self.estimator.stamp, 1),
                                  "kind": kind, "text": text[:ALERT_TEXT_MAX]})

    def _skill_begin(self, skill: str, receipt: dict) -> None:
        """Run inside the job thread, right before the skill moves.

        A pick takes one fresh look at its target from where the robot stands; that is worth a
        few seconds, but not on the request path of ``run``. The look moves the belief, and the
        receipt's dock pose was derived from the old position: a small correction is folded into
        the plan here, a large one invalidates it — the mission re-plans from the corrected
        belief instead of the robot driving to a pose that belongs to an object that is no
        longer there."""
        plan = (receipt.get("plan") or {}) if skill == "pick" else {}
        target, before = plan.get("target"), plan.get("targetP")
        if not target or before is None:
            return
        if not self._fresh_look_once(target):
            return  # no new observation, so the plan is still as good as the belief
        track = self.estimator.tracks.get(target)
        if track is None:
            return
        moved = float(np.linalg.norm(track.pos[:2] - np.asarray(before[:2])))
        if moved <= 0.02:
            return
        if moved > self.config.redock_max_m:
            raise SkillFailed(f"target {target} moved {moved * 100:.0f} cm since planning; re-plan")
        heading = (math.cos(plan["yaw"]), math.sin(plan["yaw"]))
        backoff = math.hypot(plan["dock"][0] - plan["predock"][0], plan["dock"][1] - plan["predock"][1])
        reach = float(plan.get("reach", 0.35)) + self.esim.tool_offset
        dock = [float(track.pos[0]) - heading[0] * reach, float(track.pos[1]) - heading[1] * reach]
        predock = [dock[0] - heading[0] * backoff, dock[1] - heading[1] * backoff]
        plan.update({"targetP": [float(v) for v in track.pos], "dock": dock, "predock": predock})
        if plan.get("path"):
            plan["path"][-1] = predock
        print(f"ego: pick plan for {target} re-docked after a {moved * 100:.0f} cm correction",
              file=sys.stderr, flush=True)

    def _skill_finished(self, job) -> None:
        result = {"succeeded": "完成", "failed": "失败", "stopped": "被停止"}.get(job.state, job.state)
        self._log("skill", f"{job.skill} {result}" + (f"：{job.message}" if job.state != "succeeded" else ""))

    def activity_since(self, since: int = 0) -> list[dict]:
        """Activity entries with an id above ``since``, oldest first.

        :param since: the id of the newest entry the caller already has.
        :returns: the newer entries.
        """
        with self._lock:
            return [dict(e) for e in self.activity if e["id"] > since]

    # -------------------------------------------------------------------- plan
    def set_plan(self, plan: dict) -> dict:
        """Publish what the robot is trying to achieve: a goal, its steps with their status, the
        step in progress and the latest difficulty. A client (a mission script, the robot agent)
        owns its content; the runtime only stores the latest one and stamps it."""
        goal = plan.get("goal")
        steps = plan.get("steps", [])
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("a plan needs a goal")
        if not isinstance(steps, list) or not all(isinstance(st, dict) and isinstance(st.get("name"), str)
                                                   and st.get("status") in PLAN_STEP_STATES for st in steps):
            raise ValueError(f"plan steps need a name and a status in {sorted(PLAN_STEP_STATES)}")
        record = {"goal": goal[:300], "source": str(plan.get("source", "agent"))[:60],
                  "steps": [{"name": st["name"][:200], "status": st["status"]} for st in steps[:30]],
                  "round": plan.get("round"), "current": str(plan.get("current") or "")[:400],
                  "difficulty": str(plan.get("difficulty") or "")[:600], "state": str(plan.get("state") or "running")[:40],
                  "t": round(self.estimator.stamp, 2)}
        with self._lock:
            previous, self.plan = self.plan or {}, record
        if record["current"] and record["current"] != previous.get("current"):
            self._log("step", record["current"])
        if record["difficulty"] and record["difficulty"] != previous.get("difficulty"):
            self._log("difficulty", record["difficulty"])
        if record["state"] != previous.get("state", "running"):
            self._log("state", record["state"])
        return record

    def plan_view(self, since: int = 0) -> dict:
        """The latest published plan (``None`` before any), the skill running now and the activity
        entries newer than ``since``.

        :param since: the id of the newest activity entry the caller already has.
        :returns: ``{"plan", "skill", "activity", "activityLast", "t"}`` (``activityLast`` is the newest
            id, lower than ``since`` after the runtime restarted).
        """
        running = next((j for j in reversed(list(self.executor.jobs.values())) if j.state == "running"), None)
        with self._lock:
            plan = dict(self.plan) if self.plan else None
        with self._lock:
            newest = self.activity[-1]["id"] if self.activity else 0
        return {"plan": plan, "skill": running.to_dict() if running else None, "activity": self.activity_since(since),
                "activityLast": newest, "t": round(self.estimator.stamp, 2)}

    # ----------------------------------------------------------------- control
    def control(self) -> dict:
        return {"owner": self.control_owner}

    def acquire_control(self, owner: str) -> dict:
        """Give ``owner`` exclusive use of skills until it releases control; the running skill stops.
        Clients that are not the owner cannot start skills meanwhile, but anyone may still stop."""
        if not owner:
            raise ValueError("control needs an owner name")
        with self._lock:
            if self.control_owner not in (None, owner):
                raise PermissionError(f"the robot is already controlled by {self.control_owner}")
            self.control_owner = owner
        self.executor.stop()
        return self.control()

    def release_control(self, owner: str) -> dict:
        with self._lock:
            if self.control_owner == owner:
                self.control_owner = None
        return self.control()

    def _check_control(self, owner: str | None) -> None:
        if self.control_owner is not None and owner != self.control_owner:
            raise PermissionError(f"the robot is controlled by {self.control_owner}; wait until it releases control")

    def _fresh_look_once(self, target_id: str) -> bool:
        """At most one fresh look per target per few seconds of body time.

        :returns: whether a look was taken (and so whether the belief may have moved).
        """
        now = self.estimator.stamp
        if now - self._looked_at.get(target_id, -1e9) < self.config.fresh_look_every_s:
            return False
        self._looked_at[target_id] = now
        return self._fresh_look(target_id)

    def _fresh_look(self, target_id: str) -> bool:
        """Re-place the pick's target from a class-prompted segmentation taken right now.

        An accumulated position can be biased by the partial views it was fused from — a rim arc
        read as confidently as the bowl itself. The action deserves the best single look there is,
        and this is it: the robot stands beside the object and the camera points at it. The class
        name is the robot's own prior (another name for the same thing is tried too, because a mug
        is a cup to half the world); the believed position only decides which returned instance is
        the one meant. The track keeps its identity; only where it stands is restated."""
        url = os.environ.get("STEWARD_SEGMENTER_URL")
        track = self.estimator.tracks.get(target_id)
        if url is None or track is None or not body_ok(track.category):
            print(f"ego: fresh look for {target_id} skipped (url={'set' if url else 'unset'}, "
                  f"track={'known' if track else 'unknown'})", file=sys.stderr, flush=True)
            return False
        try:
            frame = decode_rgbd(self.link.rgbd())
            cam = self.estimator.camera_pose()
            buf = io.BytesIO()
            Image.fromarray(frame["rgb"]).save(buf, format="PNG")
            image = base64.b64encode(buf.getvalue()).decode("ascii")
            h, w = frame["depth"].shape
            f = (h / 2.0) / math.tan(math.radians(frame["fovy"]) / 2.0)
            cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
            rot = matrix_from_quat(cam["q"])
            origin = np.asarray(cam["p"], float)

            def found_in(body: dict) -> list[tuple[np.ndarray, float]]:
                """World centre and score of every instance the depth can place."""
                out = []
                if not body.get("mask_png_base64"):
                    return out
                labels = np.array(Image.open(io.BytesIO(base64.b64decode(body["mask_png_base64"]))))
                for info in body.get("instances", []):
                    ys, xs = np.nonzero(labels == int(info["id"]))
                    dd = frame["depth"][ys, xs].astype(np.float32)  # decode_rgbd gives metres
                    ok = (dd > 0.05) & (dd < 6.0)
                    if os.environ.get("STEWARD_FRESH_LOOK_DEBUG"):
                        print(f"ego dbg: instance {info['id']} score {info['score']:.2f} "
                              f"px {len(ys)} depth-valid {int(ok.sum())} "
                              f"depth range {float(dd.min()) if len(dd) else -1:.2f}..{float(dd.max()) if len(dd) else -1:.2f}",
                              file=sys.stderr, flush=True)
                    if int(ok.sum()) < self.config.relocate_min_points:
                        continue
                    cam_pts = np.stack([dd[ok], -(xs[ok] - cx) * dd[ok] / f,
                                        -(ys[ok] - cy) * dd[ok] / f], axis=1)
                    out.append((np.median(cam_pts @ rot.T + origin, axis=0), float(info["score"])))
                return out

            prompts = _fresh_look_prompts(track.category, track.color)
            found: list[tuple[np.ndarray, float]] = []
            asked = []
            for prompt in prompts:
                body = call("POST", f"{url.rstrip('/')}/segment_text",
                            {"image_base64": image, "text_prompt": prompt, "min_score": 0.4},
                            timeout=30.0)
                asked.append(f"{prompt}:{len(body.get('instances', []))}")
                found += found_in(body)
            best, best_d = None, self.config.relocate_max_jump_m
            distances = []
            for centre, score in found:
                dist = float(np.linalg.norm(centre - track.pos[:3]))
                distances.append(f"{dist * 100:.0f}cm z{centre[2]:.2f}(score {score:.2f})")
                if abs(float(centre[2]) - float(track.pos[2])) > self.config.relocate_max_dz_m:
                    continue  # an instance at another height is another thing (a table edge, the floor)
                if dist < best_d:
                    best, best_d = centre, dist
            fallback = self.config.relocate_fallback_m
            if best is None and found:
                # The belief itself can be the wrong thing here (a rim fragment decimetres from
                # the bowl it belongs to). A confidently segmented nearest instance farther away
                # is still the best reading of where this object stands.
                best, best_d = None, fallback
                for centre, score in found:
                    dist = float(np.linalg.norm(centre - track.pos[:3]))
                    if abs(float(centre[2]) - float(track.pos[2])) > self.config.relocate_max_dz_m:
                        continue
                    if dist <= fallback and score >= self.config.relocate_fallback_score and dist < best_d:
                        best, best_d = centre, dist
            if best is None:
                # Text prompts die at dock range, where the robot often stands after a first
                # failed attempt. There the point sweep is at its best instead: the object fills
                # the frame, so prompt the pixels around where the belief says it projects.
                rel = track.pos[:3] - origin
                cam_rel = rel @ rot
                if cam_rel[0] > 0.15:
                    u = int(round(cx - cam_rel[1] * f / cam_rel[0]))
                    v = int(round(cy - cam_rel[2] * f / cam_rel[0]))
                    if 0 <= u < w and 0 <= v < h:
                        points = [[u + du, v + dv] for du in (-10, 0, 10) for dv in (-10, 0, 10)]
                        near = call("POST", f"{url.rstrip('/')}/segment_instances",
                                    {"image_base64": image, "points": points, "min_score": 0.4},
                                    timeout=40.0)
                        best, best_d = None, fallback
                        for centre, _score in found_in(near):
                            dist = float(np.linalg.norm(centre - track.pos[:3]))
                            if dist < best_d:
                                best, best_d = centre, dist
            if best is None:
                print(f"ego: fresh look for {target_id} ({track.category}/{track.color}): "
                      f"{len(found)} text instances from [{', '.join(asked)}], "
                      f"none within {fallback:.2f} m of the belief "
                      f"({', '.join(distances) or 'none returned'})", file=sys.stderr, flush=True)
                return False

            with self._lock:
                moved = self.estimator.relocate(target_id, best)
            if moved:
                print(f"ego: fresh look placed {target_id} at ({best[0]:.3f}, {best[1]:.3f}, {best[2]:.3f})",
                      file=sys.stderr, flush=True)
            return bool(moved)
        except Exception as exc:
            print(f"ego: fresh look for {target_id} failed: {exc}", file=sys.stderr, flush=True)
            return False

    def run(self, receipt_id: str, owner: str | None = None) -> dict:
        """Execute a predicted plan. The runtime re-checks the receipt here, where the motion happens."""
        self._check_control(owner)
        receipt = self.receipt(receipt_id)
        now = self.estimator.stamp
        if receipt.get("used"):
            raise PermissionError(f"receipt {receipt_id} was already executed; predict again")
        if not receipt["feasible"]:
            raise PermissionError(f"receipt {receipt_id} is not feasible: {'; '.join(receipt['issues'])}")
        if now > receipt["expiresAt"]:
            raise PermissionError(f"receipt {receipt_id} expired at t={receipt['expiresAt']:.1f}s; predict again")
        limit = receipt.get("maxRisk", self.config.max_risk)
        if receipt["risk"] > limit:
            raise PermissionError(f"receipt risk {receipt['risk']:.2f} exceeds the limit {limit:.2f}")
        receipt["used"] = True
        job = self.executor.start(receipt["skill"], receipt).to_dict()
        params = receipt.get("params") or {}
        target = params.get("target") or params.get("on") or (f"({params['x']:.2f}, {params['y']:.2f})" if "x" in params else "")
        self._log("skill", f"{receipt['skill']} 开始" + (f"：{target}" if target else ""))
        return job

    def explore(self, owner: str | None = None) -> dict:
        """Look around in place. Rotation without translation needs no receipt."""
        self._check_control(owner)
        return self.executor.start("look_around", {"id": "look-around", "plan": {}}).to_dict()

    def job(self, job_id: str) -> dict:
        if job_id not in self.executor.jobs:
            raise KeyError(f"unknown job {job_id!r}")
        return self.executor.jobs[job_id].to_dict()

    def wait(self, job_id: str, timeout: float = 120.0) -> dict:
        return self.executor.wait(job_id, timeout).to_dict()

    def stop_motion(self) -> dict:
        return self.executor.stop()
