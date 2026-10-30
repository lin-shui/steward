"""Skills for the Cobot Magic twin: IK-driven arm motion, grasping and placement.

The verbs here are the cobot's version of :mod:`steward.ego.skills` (which drives the mobile
kitchen robot's base, lift and reach slide). Both expose the same ``start`` / ``wait`` / ``stop``
job interface, so the Ego Runtime can drive either one.

Grasp geometry comes from the model, not from a constant: the pinch point is
``assets.cobot.TCP_OFFSET`` along link6's +z, so a grasp targets the pinch point and the IK
target is that pose pulled back by the offset. That is why a 1 cm perception error costs 1 cm
here instead of the mobile robot's 6 cm capture radius.

Grasping still uses the simulator's ideal attach (the finger joint closes and the world rigidly
attaches what is between the fingers); :meth:`CobotSkills.pick` is what makes the approach and
retreat real.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from steward.assets.cobot import TCP_OFFSET
from steward.ego.ik import CobotIK, IkError
from steward.math3d import pose_compose, quat_from_axis_angle

ARM_OF_GRIPPER = {"L_link6": "L", "R_link6": "R"}


class SkillFailed(RuntimeError):
    """A skill could not be carried out; the message is what the operator sees."""


class SkillStopped(RuntimeError):
    """The skill was cancelled."""


@dataclass
class MotionConfig:
    tick_s: float = 0.02
    joint_tol: float = 0.03        # rad, all six joints
    settle_timeout_s: float = 12.0
    stall_s: float = 1.5
    approach_height: float = 0.12  # straight up from the grasp point before descending
    retreat_height: float = 0.15   # lift after closing the fingers
    line_step: float = 0.04        # straight-line descents and lifts are walked in these increments
    max_tool_tilt: float = 0.7     # rad the tool may tilt off the requested orientation (wrist limits)
    grasp_grip_wait_s: float = 0.4


@dataclass
class Job:
    id: str
    skill: str
    receipt: str
    state: str = "running"
    message: str = ""
    started: float = 0.0
    finished: float = 0.0
    log: list[str] = field(default_factory=list)


class CobotSkills:
    """Runs at most one cobot skill at a time in a worker thread."""

    def __init__(self, command: Callable[[dict], None], state: Callable[[], dict],
                 ik: CobotIK | None = None, config: MotionConfig | None = None,
                 on_error: Callable[[str], None] | None = None,
                 on_finish: Callable[[Job], None] | None = None):
        self._command = command
        self._state = state
        self.ik = ik if ik is not None else CobotIK()
        self.config = config or MotionConfig()
        self._on_error = on_error
        self._on_finish = on_finish
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.jobs: dict[str, Job] = {}
        self.current: Job | None = None

    # ------------------------------------------------------------------ job api
    def start(self, skill: str, receipt: dict) -> Job:
        with self._lock:
            if self.current is not None and self.current.state == "running":
                raise RuntimeError(f"skill {self.current.skill} ({self.current.id}) is still running")
            job = Job(id=f"job-{next(self._ids)}", skill=skill, receipt=receipt["id"],
                      started=self._state()["stamp"])
            self.jobs[job.id] = job
            self.current = job
            self._stop.clear()
        threading.Thread(target=self._run, args=(job, skill, receipt), name=f"skill-{job.id}",
                         daemon=True).start()
        return job

    def cancel(self) -> None:
        self._stop.set()

    def stop(self) -> dict:
        self._stop.set()
        self._open_all_grippers()
        return {"stopped": self.current.id if self.current and self.current.state == "running" else None}

    def wait(self, job_id: str, timeout: float) -> Job:
        deadline = time.monotonic() + timeout
        job = self.jobs[job_id]
        while job.state == "running" and time.monotonic() < deadline:
            time.sleep(0.02)
        return job

    def _run(self, job: Job, skill: str, receipt: dict) -> None:
        try:
            plan = receipt.get("plan", {})
            if skill == "move_pose":
                self.move_pose(job, plan)
            elif skill == "move_joints":
                self.move_joints(job, plan)
            elif skill == "pick":
                self.pick(job, plan)
            elif skill == "place":
                self.place(job, plan)
            elif skill == "push":
                self.push(job, plan)
            elif skill in ("open", "close"):
                self.gripper(job, plan["gripper"], close=(skill == "close"))
            else:
                raise SkillFailed(f"unknown skill {skill}")
            job.state, job.message = "succeeded", job.message or f"{skill} completed"
        except SkillStopped:
            job.state, job.message = "stopped", "stopped on request"
        except SkillFailed as exc:
            job.state, job.message = "failed", str(exc)
            self._retreat_on_failure()
        except Exception as exc:  # noqa: BLE001 - reported through the job record and the alert channel
            job.state, job.message = "failed", f"{type(exc).__name__}: {exc}"
            if self._on_error is not None:
                self._on_error(f"技能 {skill}（{job.id}）出现未预料的异常：{type(exc).__name__}: {exc}")
            self._retreat_on_failure()
        finally:
            job.finished = self._state()["stamp"]
            if self._on_finish is not None:
                self._on_finish(job)

    # ------------------------------------------------------------------- verbs
    def pick(self, job: Job, plan: dict) -> None:
        """Reach the pinch point over ``plan['at']``, descend, close, and lift clear."""
        arm = self._arm(plan)
        at = _xyz(plan["at"], "plan.at")
        grip = plan.get("gripper", f"{arm}_link6")
        tool = _tool_down()
        if (self._state().get("grippers") or {}).get(grip, {}).get("state") != "open":
            self.gripper(job, grip, close=False)
        above = [at[0], at[1], at[2] + self.config.approach_height]
        job.log.append(f"{arm} approach above {_fmt(above)}")
        self.move_to(job, arm, above, tool)
        self.move_to(job, arm, at, tool, line=True)
        self.gripper(job, grip, close=True)
        lifted = [at[0], at[1], at[2] + self.config.retreat_height]
        job.log.append(f"{arm} lift to {_fmt(lifted)}")
        self.move_to(job, arm, lifted, tool, line=True)
        st = self._state()
        holding = (st.get("grippers") or {}).get(grip, {}).get("holding", False)
        if not holding:
            raise SkillFailed(f"closed the gripper at {_fmt(at)} but grabbed nothing")

    def place(self, job: Job, plan: dict) -> None:
        """Carry the load to ``plan['at']``, lower it, open, and retreat."""
        arm = self._arm(plan)
        at = _xyz(plan["at"], "plan.at")
        grip = plan.get("gripper", f"{arm}_link6")
        tool = _tool_down()
        above = [at[0], at[1], at[2] + self.config.approach_height]
        job.log.append(f"{arm} carry to {_fmt(above)}")
        self.move_to(job, arm, above, tool)
        self.move_to(job, arm, at, tool, line=True)
        self.gripper(job, grip, close=False)
        self.move_to(job, arm, [at[0], at[1], at[2] + self.config.retreat_height], tool, line=True)

    def push(self, job: Job, plan: dict) -> None:
        """Push an object side-on: approach behind it, then travel ``plan['by']`` along the surface."""
        arm = self._arm(plan)
        start = _xyz(plan["at"], "plan.at")
        travel = _xyz(plan["by"], "plan.by")
        tool = _tool_sideways(plan.get("direction", [1.0, 0.0, 0.0]))
        self.move_to(job, arm, start, tool)
        self.move_to(job, arm, [start[i] + travel[i] for i in range(3)], tool, line=True)

    def gripper(self, job: Job, grip: str, close: bool) -> None:
        if grip not in ARM_OF_GRIPPER:
            raise SkillFailed(f"unknown gripper {grip!r}; expected one of {sorted(ARM_OF_GRIPPER)}")
        self._command({"gripper": {grip: "close" if close else "open"}})
        self._settle_fingers(close)
        job.log.append(f"{grip} {'closed' if close else 'opened'}")

    # -------------------------------------------------------------- primitives
    def move_to(self, job: Job, arm: str, tcp: list[float], orientation: list[float] | None = None,
                line: bool = False) -> list[float]:
        """Move the arm so the pinch point reaches ``tcp``; returns the joint targets.

        ``line=True`` walks the pinch point straight there in ``line_step`` increments instead of
        one joint interpolation, which is what a descent or a lift has to be."""
        if line:
            here = self.tcp(arm)
            distance = sum((a - b) ** 2 for a, b in zip(tcp, here, strict=True)) ** 0.5
            steps = max(1, int(distance / self.config.line_step) + 1)
            for i in range(1, steps + 1):
                frac = i / steps
                waypoint = [here[j] + (tcp[j] - here[j]) * frac for j in range(3)]
                self._move_to(job, arm, waypoint, orientation)
        else:
            self._move_to(job, arm, tcp, orientation)
        return self.ik.current(arm)

    def _move_to(self, job: Job, arm: str, tcp: list[float], orientation: list[float] | None) -> None:
        self._sync(arm)
        seed = self.ik.current(arm)
        target = self._link6_target(tcp, orientation)
        try:
            if orientation is None:
                joints = self.ik.solve(arm, target["p"], target["q"], q_init=seed)
            else:
                # The wrist cannot hold a strictly vertical tool everywhere, so a small tilt is
                # accepted rather than declaring a reachable place unreachable.
                joints, tilt = self.ik.solve_tool(arm, target["p"], target["q"],
                                                  max_tilt=self.config.max_tool_tilt, q_init=seed)
                if tilt > 0.02:
                    job.log.append(f"{arm} tool tilted {tilt:.2f} rad to reach {_fmt(tcp)}")
        except IkError as exc:
            raise SkillFailed(f"{arm} arm cannot reach {_fmt(tcp)}: {exc}") from exc
        self._command({"joints": dict(zip(self.ik.joint_names(arm), joints, strict=True))})
        self._wait_joints(arm, joints)
        job.log.append(f"{arm} at {_fmt(self.tcp(arm))}")

    def move_pose(self, job: Job, plan: dict) -> None:
        """Put the arm's pinch point at ``plan['at']`` (with an optional wxyz ``orientation``)."""
        arm = self._arm(plan)
        at = _xyz(plan["at"], "plan.at")
        orientation = plan.get("orientation")
        if orientation is not None and len(orientation) != 4:
            raise SkillFailed(f"plan.orientation must be a wxyz quaternion, got {orientation!r}")
        self.move_to(job, arm, at, orientation, line=bool(plan.get("line", False)))

    def move_joints(self, job: Job, plan: dict) -> None:
        """Drive one arm's six joints to the requested posture."""
        arm = self._arm(plan)
        joints = plan.get("joints") or {}
        names = self.ik.joint_names(arm)
        missing = [n for n in names if n not in joints]
        if missing:
            raise SkillFailed(f"move_joints needs all six {arm} joints; missing {missing}")
        target = [float(joints[n]) for n in names]
        for value, (lo, hi) in zip(target, self.ik.joint_limits(arm), strict=True):
            if not lo <= value <= hi:
                raise SkillFailed(f"{arm} joint target {value:.3f} is outside [{lo:.3f}, {hi:.3f}]")
        self._command({"joints": dict(zip(names, target, strict=True))})
        self._wait_joints(arm, target)

    def _sync(self, arm: str) -> None:
        """Seed the solver from the robot's own encoders, not from the last solution."""
        names = self.ik.joint_names(arm)
        st = self._state()
        if all(n in st.get("joints", {}) for n in names):
            self.ik.set_joints(arm, [st["joints"][n] for n in names])

    def _wait_joints(self, arm: str, target: list[float]) -> None:
        names = self.ik.joint_names(arm)
        deadline = time.monotonic() + self.config.settle_timeout_s
        last, since = None, time.monotonic()
        while time.monotonic() < deadline:
            self._tick()
            now = [self._state()["joints"][n] for n in names]
            if max(abs(a - b) for a, b in zip(now, target, strict=True)) <= self.config.joint_tol:
                return
            # A joint that stops moving short of the target means the arm is blocked or the
            # actuator is saturated: report it instead of burning the whole timeout.
            if last is not None and max(abs(a - b) for a, b in zip(now, last, strict=True)) < 1e-3:
                if time.monotonic() - since > self.config.stall_s:
                    raise SkillFailed(f"{arm} arm stalled {_fmt(list(now))} toward {_fmt(target)}")
            else:
                since = time.monotonic()
            last = now
        raise SkillFailed(f"{arm} arm did not reach {_fmt(target)} within {self.config.settle_timeout_s:.0f} s")

    def _settle_fingers(self, close: bool) -> None:
        time.sleep(self.config.grasp_grip_wait_s)

    def tcp(self, arm: str) -> list[float]:
        """Where the pinch point is now, from the robot's own joint encoders."""
        self._sync(arm)
        return self._tcp_of(self.ik.fk(arm))

    def _tcp_of(self, link6_pose: dict) -> list[float]:
        return pose_compose(link6_pose, {"p": TCP_OFFSET, "q": [1.0, 0.0, 0.0, 0.0]})["p"]

    def _link6_target(self, tcp: list[float], orientation: list[float] | None) -> dict:
        """The link6 pose that puts the pinch point at ``tcp`` with the given tool orientation."""
        tool = orientation if orientation is not None else _tool_down()
        offset = {"p": TCP_OFFSET, "q": [1.0, 0.0, 0.0, 0.0]}
        from steward.math3d import pose_inverse

        return pose_compose({"p": list(tcp), "q": tool}, pose_inverse(offset))

    def _tick(self) -> dict:
        if self._stop.is_set():
            raise SkillStopped()
        time.sleep(self.config.tick_s)
        return self._state()

    def _arm(self, plan: dict) -> str:
        grip = plan.get("gripper")
        if grip in ARM_OF_GRIPPER:
            return ARM_OF_GRIPPER[grip]
        arm = plan.get("arm")
        if arm not in ("L", "R"):
            raise SkillFailed(f"plan needs arm L/R or a gripper link, got {plan!r}")
        return arm

    def _retreat_on_failure(self) -> None:
        for arm in ("L", "R"):
            try:
                here = self.tcp(arm)
                self._move_to(Job("retreat", "retreat", ""), arm,
                              [here[0], here[1], here[2] + self.config.retreat_height], None)
            except Exception:  # noqa: BLE001 - best effort after a failure that is already reported
                pass

    def _open_all_grippers(self) -> None:
        try:
            self._command({"gripper": {g: "open" for g in ARM_OF_GRIPPER}})
        except Exception:  # noqa: BLE001 - best effort while stopping
            pass


def _tool_down() -> list[float]:
    """Tool pointing straight down: link6 +z along world -z (180 degrees about x)."""
    return [0.0, 1.0, 0.0, 0.0]


def _tool_sideways(direction: list[float]) -> list[float]:
    """Tool pointing along a horizontal direction (used to push): link6 +z onto world +/-x."""
    sign = 1.0 if direction[0] >= 0 else -1.0
    return [0.7071068, 0.0, sign * 0.7071068, 0.0]


def _fmt(p: list[float]) -> str:
    return "(" + ", ".join(f"{v:.3f}" for v in p) + ")"


def _xyz(value, what: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise SkillFailed(f"{what} must be [x, y, z], got {value!r}")
    return [float(v) for v in value]
