"""Cobot skills end to end in the simulation: pick a bowl off a table and put it down elsewhere."""

import numpy as np
import pytest

from steward.assets.cobot import TCP_OFFSET, cobot_magic
from steward.assets.generators import bowl, building, dining_table, plate
from steward.ego.cobot_skills import CobotSkills, MotionConfig
from steward.world.sim import WorldSim

# The left arm's comfortable tabletop band, measured by scanning tool-down IK against the URDF:
# grasp points are reachable with a strictly vertical tool at x <= ~0.30, y >= ~0.15.
TABLE_TOP = 0.75
BOWL_AT = [0.28, 0.25, TABLE_TOP]
PLATE_AT = [0.30, 0.02, TABLE_TOP]


def _scene() -> dict:
    room = building(width=4.0, depth=4.0, height=2.6).to_dict()
    room["id"] = "w:room"

    robot = cobot_magic().to_dict()
    robot["id"] = "w:cobot"

    table = dining_table(width=0.7, depth=0.8, height=TABLE_TOP).to_dict()
    table["id"] = "w:table"
    table["pose"]["value"]["p"] = [0.45, 0.0, 0.0]

    dish = bowl().to_dict()
    dish["id"] = "w:bowl-1"
    dish["pose"]["value"]["p"] = list(BOWL_AT)

    saucer = plate().to_dict()
    saucer["id"] = "w:plate-1"
    saucer["pose"]["value"]["p"] = list(PLATE_AT)
    return {"mode": "world", "entities": [room, robot, table, dish, saucer]}


@pytest.fixture()
def rig():
    sim = WorldSim(_scene())
    sim.start()
    skills = CobotSkills(command=lambda cmd: sim.command("w:cobot", cmd),
                         state=lambda: sim.sensors("w:cobot"),
                         config=MotionConfig(tick_s=0.01))
    try:
        yield sim, skills
    finally:
        sim.stop()


def _bowl(sim: WorldSim) -> list[float]:
    state = sim.world_state()
    return next(e for e in state["entities"] if e["id"] == "w:bowl-1")["pose"]["value"]["p"]


def test_tcp_sits_between_the_finger_bodies(rig) -> None:
    sim, skills = rig
    tcp = np.asarray(skills.tcp("L"))
    left = np.asarray(sim.data.xpos[sim.model.body("w:cobot/L_link7").id])
    right = np.asarray(sim.data.xpos[sim.model.body("w:cobot/L_link8").id])
    assert np.linalg.norm(tcp - (left + right) / 2) < 5e-3, "TCP must be the finger midpoint"
    assert np.linalg.norm(np.asarray(TCP_OFFSET)) == pytest.approx(0.135, abs=1e-6)


def test_pick_and_place_moves_the_object(rig) -> None:
    """The full verb pair: descend, close, lift, carry, lower, open, retreat."""
    sim, skills = rig
    bowl_before = _bowl(sim)

    job = skills.start("pick", {"id": "r1", "plan": {"arm": "L", "at": BOWL_AT}})
    done = skills.wait(job.id, timeout=90)
    assert done.state == "succeeded", f"pick failed: {done.message}\n" + "\n".join(done.log)
    assert any(e["type"] == "grasp/attach" for e in sim.events), "the gripper closed on nothing"

    carried = skills.tcp("L")
    target = [carried[0] - 0.08, carried[1] + 0.10, carried[2] - skills.config.retreat_height]
    job = skills.start("place", {"id": "r2", "plan": {"arm": "L", "at": target}})
    done = skills.wait(job.id, timeout=90)
    assert done.state == "succeeded", f"place failed: {done.message}\n" + "\n".join(done.log)

    sim.step(60)
    bowl_after = _bowl(sim)
    moved = np.linalg.norm(np.subtract(bowl_after[:2], bowl_before[:2]))
    assert moved > 0.05, f"the bowl did not move on the table (displacement {moved:.3f} m)"
    assert any(e["type"] == "grasp/release" for e in sim.events), "the gripper never opened"


def test_pick_where_there_is_nothing_reports_a_failed_grasp(rig) -> None:
    sim, skills = rig
    # Table top, midway between the bowl (y 0.25) and the plate (y 0.02): reachable with a vertical
    # tool, and both are farther than the 6 cm capture radius.
    job = skills.start("pick", {"id": "r3", "plan": {"arm": "L", "at": [0.28, 0.15, TABLE_TOP]}})
    done = skills.wait(job.id, timeout=90)
    assert done.state == "failed"
    assert "grabbed nothing" in done.message


def test_move_joints_verb_reaches_the_requested_posture(rig) -> None:
    sim, skills = rig
    target = dict(zip(skills.ik.joint_names("R"), skills.ik.neutral("R"), strict=True))
    job = skills.start("move_joints", {"id": "r4", "plan": {"arm": "R", "joints": target}})
    done = skills.wait(job.id, timeout=90)
    assert done.state == "succeeded", f"{done.message}\n" + "\n".join(done.log)
    for name, value in target.items():
        assert sim.sensors("w:cobot")["joints"][name] == pytest.approx(value, abs=0.06)


def test_joint_target_outside_the_urdf_range_is_refused(rig) -> None:
    sim, skills = rig
    target = dict(zip(skills.ik.joint_names("L"), skills.ik.neutral("L"), strict=True))
    target["L_joint2"] = 9.0
    job = skills.start("move_joints", {"id": "r5", "plan": {"arm": "L", "joints": target}})
    done = skills.wait(job.id, timeout=30)
    assert done.state == "failed"
    assert "outside" in done.message
