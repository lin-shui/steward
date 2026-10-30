"""Cobot Magic embodiment twin: URDF fidelity and MJCF compilability."""

import mujoco
import numpy as np
import pytest

from steward.assets.cobot import cobot_magic
from steward.scene.mjcf import build_mjcf
from steward.scene.schema import SceneState
from steward.scene.validate import validate_world


@pytest.fixture()
def robot() -> dict:
    return cobot_magic().to_dict()


def test_valid_world_entity(robot: dict) -> None:
    assert validate_world({"mode": "world", "entities": [robot]}) == []


def test_urdf_joint_fidelity(robot: dict) -> None:
    """Joint1..joint6 of both arms carry the Piper URDF origins, axes and limits."""
    joints = {j["id"]: j for j in robot["body"]["joints"]}
    spec = {
        "L_joint1": ([0.0, 0.0, 0.123], -2.618, 2.618),
        "L_joint3": ([0.2836, 0.0287, 0.0], -2.697, 0.0),
        "L_joint6": ([0.0, 0.091, 0.0014], -3.14, 3.14),
        "R_joint4": ([-0.2422, 0.0685, 0.0], -1.832, 1.832),
    }
    for jid, (xyz, lo, hi) in spec.items():
        j = joints[jid]
        assert j["type"]["value"] == "revolute"
        assert np.allclose(j["origin"]["value"]["p"], xyz, atol=1e-4)
        assert j["limits"]["value"] == {"lo": lo, "hi": hi}
    fingers = [joints[f"{a}_joint{i}"] for a in ("L", "R") for i in (7, 8)]
    for j in fingers:
        assert j["type"]["value"] == "prismatic"
        assert abs(j["limits"]["value"]["hi"] - j["limits"]["value"]["lo"]) == pytest.approx(0.05)


def test_compiles_and_steps() -> None:
    build = build_mjcf(SceneState(mode="world", entities=[cobot_magic()]), timestep=0.005)
    model = mujoco.MjModel.from_xml_string(build.xml)
    data = mujoco.MjData(model)
    # 2 arms x (6 revolute + 2 prismatic fingers) joints and actuators; fixed root adds none.
    assert model.njnt == 16
    assert model.nu == 16
    # Top camera plus one wrist camera per arm.
    cam_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)}
    assert any(n.endswith("top_camera") for n in cam_names)
    assert any(n.endswith("L_wrist_camera") for n in cam_names)
    assert any(n.endswith("R_wrist_camera") for n in cam_names)
    # Holds the zero pose (targets default to qpos) without blowing up.
    for _ in range(200):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert data.qvel.shape == (16,)
