"""The twin perceives from RGB-D frames alone: no labelled boxes, no categories handed over.

Phase 1 for the cobot: the world gives RGB, depth and anonymous instance masks; the estimator has
to segment, measure and classify by itself before the skills can act.
"""

import numpy as np
import pytest

from steward.assets.cobot import cobot_magic
from steward.ego.cobot_mission import bowl_onto_plate, perceive
from steward.ego.cobot_skills import CobotSkills, MotionConfig
from steward.ego.estimator import EgoEstimator
from steward.ego.perception import decode_rgbd
from steward.world.sim import PerceptionConfig, WorldSim

from test_cobot_skills import _scene


@pytest.fixture()
def rig():
    sim = WorldSim(_scene(), perception=PerceptionConfig(mode="rgbd"))
    sim.start()
    skills = CobotSkills(command=lambda cmd: sim.command("w:cobot", cmd),
                         state=lambda: sim.sensors("w:cobot"), config=MotionConfig(tick_s=0.01))
    est = EgoEstimator(robot_model=cobot_magic().to_dict())
    frame = lambda: decode_rgbd(sim.rgbd("w:cobot", camera="top_camera"))  # noqa: E731
    try:
        yield sim, est, skills, frame
    finally:
        sim.stop()


def _truth(sim: WorldSim, entity_id: str) -> list[float]:
    state = sim.world_state()
    return list(next(e for e in state["entities"] if e["id"] == entity_id)["pose"]["value"]["p"])


def test_rgbd_frames_carry_no_labels(rig) -> None:
    sim, _, _, frame = rig
    payload = frame()
    assert set(payload) == {"rgb", "depth", "mask", "fovy", "header"}
    assert "detections" not in payload["header"]
    assert payload["mask"].dtype == np.uint8
    assert payload["depth"].dtype == np.float32
    assert payload["mask"].max() > 0, "the segmenter must have found something to measure"


def test_the_twin_measures_tableware_from_rgbd(rig) -> None:
    sim, est, _, frame = rig
    seen = perceive(est, frame, frames=10, rgbd=True, pause_s=0)
    assert "bowl" in seen and "plate" in seen, f"categories seen: {sorted(seen)}"
    for name, entity in (("bowl", "w:bowl-1"), ("plate", "w:plate-1")):
        truth = np.asarray(_truth(sim, entity))
        error = float(np.linalg.norm(seen[name].pos[:2] - truth[:2]))
        assert error < 0.02, f"{name} believed {seen[name].pos} vs truth {truth} ({error*100:.1f} cm)"
    # The table is seen but must not be mistaken for tableware.
    assert "table" not in seen or seen["table"].existence < 0.9


def test_mission_runs_on_rgbd_perception(rig) -> None:
    sim, est, skills, frame = rig
    result = bowl_onto_plate(est, skills, frame, rgbd=True)
    assert result["ok"], "\n".join(result["log"])
    sim.step(60)
    bowl = _truth(sim, "w:bowl-1")
    plate = _truth(sim, "w:plate-1")
    assert np.linalg.norm(np.subtract(bowl[:2], plate[:2])) < 0.06, f"bowl {bowl} is not on {plate}"
    assert bowl[2] > plate[2] + 0.005
