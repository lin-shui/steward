"""The estimator's own kinematics must agree with the simulation, for every camera and pose.

This is what replaces the mobile robot's fixed head-mount assumption: a wrist camera has to move
with the arm, and the pinch point has to come out of the joint encoders.
"""

import numpy as np
import pytest

from steward.assets.cobot import cobot_magic
from steward.ego.estimator import EgoEstimator
from steward.world.sim import WorldSim

from test_cobot_skills import _scene


@pytest.fixture()
def rig():
    sim = WorldSim(_scene())
    est = EgoEstimator(robot_model=cobot_magic().to_dict())
    yield sim, est
    sim.stop()


def _sync(est: EgoEstimator, sim: WorldSim) -> None:
    """Give the estimator a clean frame: encoders from the sim, odometry at the fixed origin."""
    frame = sim.sensors("w:cobot")
    frame["odometry"] = {"x": 0.0, "y": 0.0, "yaw": 0.0, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.update(frame)


def test_every_camera_matches_the_simulation(rig) -> None:
    sim, est = rig
    _sync(est, sim)
    poses = est.camera_poses()
    assert set(poses) == {"top_camera", "L_wrist_camera", "R_wrist_camera"}
    for sensor_id, pose in poses.items():
        truth, _ = sim.camera_pose("w:cobot", sensor_id)
        # 1 mm is far tighter than any perception error and absorbs the float path difference
        # between MuJoCo's body quaternions and this joint-chain composition.
        assert np.linalg.norm(np.subtract(pose["p"], truth["p"])) < 1e-3, sensor_id
        assert abs(float(np.dot(pose["q"], truth["q"]))) > 0.99999, sensor_id


def test_wrist_camera_follows_the_arm(rig) -> None:
    sim, est = rig
    _sync(est, sim)
    before = est.camera_pose("L_wrist_camera")["p"]
    sim.command("w:cobot", {"joints": {"L_joint1": 1.0, "L_joint2": 0.4, "L_joint5": 0.5}})
    sim.step(400)
    _sync(est, sim)
    after = est.camera_pose("L_wrist_camera")["p"]
    moved = np.linalg.norm(np.subtract(after, before))
    assert moved > 0.15, f"the wrist camera barely moved ({moved:.3f} m)"
    truth, _ = sim.camera_pose("w:cobot", "L_wrist_camera")
    assert np.linalg.norm(np.subtract(after, truth["p"])) < 1e-3


def test_pinch_point_comes_out_of_the_encoders(rig) -> None:
    sim, est = rig
    for joints in ({"L_joint1": 0.0, "L_joint2": 1.57, "L_joint3": -1.35},
                   {"L_joint1": -0.6, "L_joint2": 0.9, "L_joint3": -2.0}):
        sim.command("w:cobot", {"joints": joints})
        sim.step(400)
        _sync(est, sim)
        truth = sim.grasp_point("w:cobot", "L_link6")["p"]
        got = est.gripper_position("L_link6")
        assert np.linalg.norm(got - np.asarray(truth)) < 1e-3, f"joints {joints}"


def test_single_camera_robot_still_uses_the_first_sensor(rig) -> None:
    from steward.assets.generators import steward_bot

    sim, _ = rig
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    frame = {"stamp": 0.0, "odometry": {"x": 1.0, "y": 2.0, "yaw": 0.3, "sigma": {"pos": 0.0, "yaw": 0.0}},
             "joints": {"lift": 0.6, "reach": 0.1}, "gripper": {"state": "open", "holding": False},
             "detections": [], "camera": {"id": "head_camera", "intrinsics": {}}}
    est.update(frame)
    # base (1, 2) yawed 0.3, head mount (0.16, 0, 1.50) and camera mount position (0.05, 0, 0),
    # both offsets turned by the base yaw; the mount's own 30 deg pitch turns the image only.
    pose = est.camera_pose()
    reach = 0.16 + 0.05
    assert pose["p"][0] == pytest.approx(1.0 + reach * np.cos(0.3), abs=1e-6)
    assert pose["p"][1] == pytest.approx(2.0 + reach * np.sin(0.3), abs=1e-6)
    assert pose["p"][2] == pytest.approx(1.50, abs=1e-6)
    # The optical axis (+x of the mount frame) points 30 degrees down.
    from steward.math3d import quat_rotate

    forward = quat_rotate(pose["q"], [1.0, 0.0, 0.0])
    assert forward[2] == pytest.approx(-np.sin(np.radians(30)), abs=1e-6)
