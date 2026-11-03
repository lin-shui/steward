"""The simulation speed changes how long a run takes in wall time, not what happens in it."""

from __future__ import annotations

import math

from steward.assets.generators import steward_bot
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.server import ego_api, replay
from steward.mission import MissionFailed, cup_to_table
from steward.world.server import StewardHost


def _cup_to_table(kitchen_root: str, run_dir, speed: float, inputs: list[dict] | None = None) -> tuple[list[dict], dict]:
    """Run the scripted cup mission, or replay its recorded Ego API inputs, from world time 0; return
    the inputs as applied and the World State one second after the last of them."""
    host = StewardHost(kitchen_root, scenario="phase0-kitchen", speed=speed, seed=11, run_dir=run_dir)
    host.sim.paused = True  # the robot starts at world time 0, as in the recording
    host.start()
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    runtime.start()
    api = ego_api(runtime).start()
    if inputs is not None:
        runtime.link.arm("input", inputs[0]["t"], rank=1)
    host.sim.paused = False
    try:
        if inputs is None:
            try:
                cup_to_table(api.url)
            except MissionFailed:
                pass  # its end check reads the robot's belief; what counts here is that a replay repeats it
        else:
            replay(runtime, inputs)
        host.sim.wait_until(runtime.inputs[-1]["t"] + 1.0, hold="test")
        return list(runtime.inputs), host.sim.world_state()
    finally:
        host.sim.release("test")
        api.stop()
        runtime.stop()
        host.stop()


def test_a_recorded_run_repeats_exactly_at_any_speed(kitchen_root, tmp_path):
    inputs, recorded = _cup_to_table(kitchen_root, tmp_path / "recorded", math.inf)
    assert any(r["type"] == "supported-by" and r["a"]["entity"] == "w:cup-1" for r in recorded["relations"])
    for speed in (1.0, math.inf):
        replayed_inputs, replayed = _cup_to_table(kitchen_root, tmp_path / f"replay-{speed}", speed, inputs)
        # Exact equality, no tolerance: on one machine with the same binaries, floating point repeats bit
        # for bit when the same operations run on the same values in the same order. The holds make the
        # order fixed: MuJoCo steps single-threaded at a fixed timestep with every command landing at the
        # same step, world time accumulates from 0 the same way, the seeded world RNG is drawn at the same
        # steps, the robot's loops and inputs run one at a time, and JSON round-trips floats exactly. A
        # failure here is a real source of nondeterminism. Replaying a recording made on another machine
        # or with other MuJoCo/numpy builds may differ in rounding and would need a tolerance.
        assert replayed_inputs == inputs, f"at speed {speed} every input is applied at its recorded world time"
        assert replayed == recorded, f"at speed {speed} the world ends in exactly the recorded state"
