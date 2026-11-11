"""The episode mirror: a running episode's world, driven into this host's WSim.

A Steward console renders the world its own host owns, so an episode running elsewhere -- a
uni-robot-agent twin service serving one robot -- is invisible to it.  This module is the
receiving end of the operator plane: the episode host posts every recorded step's joint
positions here and the world is put into them, paused and kinematic, so the console shows the
run while it happens.

The payload is the same shape an exported episode trace holds (``trace/steps.jsonl``), so
replaying a finished episode is this module's other half; :class:`Mirror` keeps the last state
and the counters both paths report.

The twin side is opt-in (``TWIN_MIRROR`` there, ``PIPER_MIRROR_URL`` here): it hands out world
truth -- object poses -- which is not observation and must not reach a policy.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from ..net import HttpError

#: Ignore a state older than this in the sequence (a restarted episode numbers from 0 again).
RESET_SEQ = 0


class Mirror:
    """One host's view of the episode being mirrored into it."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.episode: str | None = None
        self.scenario: str | None = None
        self.task: str | None = None
        self.seq = -1
        self.t = 0.0
        self.frame: str | None = None
        self.updated = 0.0
        self.states = 0
        self.written = 0
        self.missing: list[str] = []
        self.error: str | None = None

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "live": self.episode is not None,
                "episode": self.episode,
                "scenario": self.scenario,
                "task": self.task,
                "seq": self.seq,
                "t": round(self.t, 3),
                "frame": self.frame,
                "states": self.states,
                "written": self.written,
                "missing": self.missing[:8],
                "updated_ago_s": round(time.time() - self.updated, 2) if self.updated else None,
                "error": self.error,
            }

    def apply(self, host, payload: dict[str, Any]) -> dict[str, Any]:
        """Put the host's world into ``payload``'s state, loading its scenario when it differs."""

        joints = payload.get("joints") or {}
        if not isinstance(joints, dict) or not joints:
            raise HttpError(400, "mirror state needs 'joints' (joint name -> positions)")
        scenario = str(payload.get("scenario") or "")
        episode = str(payload.get("episode") or "")
        seq = int(payload.get("seq") or 0)
        with self.lock:
            restarted = bool(episode) and (episode != self.episode or seq <= RESET_SEQ)
            stale = (not restarted) and seq < self.seq and episode == self.episode
            if stale:
                return {"applied": False, "reason": "stale", "seq": self.seq}
        if scenario and (host.scenario or {}).get("name") != scenario:
            # Building a scene allocates GL renderers: main thread, like every episode load.
            host.on_main(lambda: host.load_episode(scenario))
            host.sim.paused = True
        elif host.sim is not None and not host.sim.paused:
            # The episode's own clock is the truth here; this world must not integrate anything.
            host.sim.paused = True
        if host.sim is None:
            raise HttpError(409, f"the host has no episode loaded and the state names {scenario!r}")
        result = host.sim.apply_joint_state(
            {name: list(values) for name, values in joints.items()},
            time=payload.get("t"),
        )
        with self.lock:
            self.episode, self.scenario, self.task = episode or self.episode, scenario or self.scenario, str(
                payload.get("task") or self.task or ""
            )
            self.seq = seq
            self.t = float(payload.get("t") or 0.0)
            self.frame = str(payload.get("frame") or "") or self.frame
            self.updated = time.time()
            self.states += 1
            self.written = int(result["written"])
            self.missing = list(result["missing"]) + list(result["mismatched"])
            self.error = None if result["written"] else "no joint of the state exists in this scene"
        return {"applied": True, "seq": seq, "written": result["written"],
                "missing": len(result["missing"]), "mismatched": len(result["mismatched"])}


def mirror_of(host) -> Mirror:
    """The host's mirror, created on first use (no state before any episode is mirrored)."""

    mirror = getattr(host, "_episode_mirror", None)
    if mirror is None:
        mirror = Mirror()
        host._episode_mirror = mirror
    return mirror


def mirror_routes(host) -> dict[tuple[str, str], Any]:
    """WCA routes of the operator plane: ``POST /mirror/state``, ``GET /mirror/status``."""

    def state(query: dict, body: dict) -> dict:
        return mirror_of(host).apply(host, body or {})

    def status(query: dict, body: dict) -> dict:
        return mirror_of(host).status()

    return {("POST", "/mirror/state"): state, ("GET", "/mirror/status"): status}
