"""The episode mirror: a running episode's world put into this host's WSim.

The twin side is covered where it lives (uni-robot-agent's adapter tests); here the receiving
end is exercised -- a state row arrives over WCA, the world takes it, the console's view loop
renders it, and a stale or foreign row is refused or reported instead of silently accepted.
"""

from __future__ import annotations

import threading
import time

from steward.authoring.service import AuthoringService
from steward.net import call
from steward.world.server import StewardHost
from steward.world.sim import WorldSim


def _pump_while(host: StewardHost, work) -> dict:
    result: dict = {}

    def run() -> None:
        try:
            result["value"] = work()
        except Exception as exc:  # noqa: BLE001 - captured for the assertion
            result["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    while thread.is_alive():
        host.pump(0.05)
    return result


def _state_of(kitchen_root, joint: str, delta: float) -> dict:
    """One world state of the same scene, with one joint nudged: what a twin would post."""

    scene = AuthoringService(kitchen_root).load_scenario("phase0-kitchen")["sceneState"]
    sim = WorldSim(scene)
    try:
        state = sim.joint_state()
        state[joint] = [state[joint][0] + delta, *state[joint][1:]]
        return {"joints": state, "time": 1.5}
    finally:
        sim.stop()


def test_joint_state_round_trips_through_apply_joint_state(kitchen_scene):
    sim = WorldSim(kitchen_scene)
    try:
        before = sim.joint_state()
        assert before and all(isinstance(values, list) for values in before.values())
        moved = dict(before)
        free = next(name for name, values in before.items() if len(values) == 7)
        moved[free] = [before[free][0] + 0.25, *before[free][1:]]
        report = sim.apply_joint_state(moved, time=4.0)
        assert report["written"] == len(before) and not report["missing"]
        assert abs(sim.joint_state()[free][0] - (before[free][0] + 0.25)) < 1e-9
        assert sim.time == 4.0
        sim.apply_joint_state(before)
        assert sim.joint_state() == before
    finally:
        sim.stop()


def test_a_mirrored_state_moves_the_world_and_the_console_shows_it(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-kitchen", run_dir=tmp_path).start()
    try:
        joint = "w:robot-1::left_joint2" if any(
            name.endswith("left_joint2") for name in host.sim.joint_state()
        ) else next(iter(host.sim.joint_state()))
        state = _state_of(kitchen_root, joint, 0.3)
        wca = host.wca.url
        reply = call("POST", f"{wca}/mirror/state", {
            "episode": "demo-run", "scenario": "phase0-kitchen", "task": "chores",
            "seq": 0, "t": 1.5, "frame": "0000", **state,
        })
        assert reply["applied"] and reply["written"] == len(state["joints"])
        assert host.sim.paused, "a mirrored world must not integrate its own physics"
        status = call("GET", f"{wca}/mirror/status")
        assert status["live"] and status["episode"] == "demo-run" and status["seq"] == 0
        # The view loop sees the mirrored state: a frame is rendered once someone asks for it.
        _pump_while(host, lambda: call("POST", f"{wca}/console/view", {
            "camera": {"lookat": [0.0, 0.0, 0.5], "distance": 4.0, "azimuth": 90.0, "elevation": -40.0},
            "fps": 10, "width": 160, "height": 120,
        }))
        frames = _pump_while(host, lambda: [
            (time.sleep(0.15), call("GET", f"{wca}/console/frame"))[1] for _ in range(4)
        ])
        assert any(isinstance(frame, bytes) and frame.startswith(b"\xff\xd8")
                   for frame in frames["value"]), "the mirrored state never reached the view"
        stale = call("POST", f"{wca}/mirror/state", {
            "episode": "demo-run", "scenario": "phase0-kitchen", "seq": 0, "joints": state["joints"],
        })
        assert stale["applied"] is False or stale["seq"] == 0
        mismatch = call("POST", f"{wca}/mirror/state", {
            "episode": "other", "scenario": "phase0-kitchen", "seq": 0,
            "joints": {joint: [1.0, 2.0]},
        })
        assert mismatch["applied"] and mismatch["written"] == 0
        assert call("GET", f"{wca}/mirror/status")["missing"]
    finally:
        host.stop()


def test_a_state_without_joints_is_refused(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-kitchen", run_dir=tmp_path).start()
    try:
        refused = _pump_while(host, lambda: call("POST", f"{host.wca.url}/mirror/state",
                                                 {"episode": "x", "seq": 0}))
        assert refused["error"].status == 400
    finally:
        host.stop()
