"""Cobot IK on the robot's own model: FK/IK round trips, limits, and per-arm isolation."""

import numpy as np
import pytest

from steward.ego.ik import CobotIK, IkError


@pytest.fixture(scope="module")
def ik() -> CobotIK:
    return CobotIK()


def test_arms_are_mirrored_across_the_mount_separation(ik: CobotIK) -> None:
    left, right = ik.fk("L"), ik.fk("R")
    # Neutral posture is identical per arm, so the arms differ only by the mount offset.
    assert left["p"][0] == pytest.approx(right["p"][0], abs=1e-6)
    assert left["p"][1] == pytest.approx(-right["p"][1], abs=0.01)
    assert left["p"][2] == pytest.approx(right["p"][2], abs=1e-6)


def test_fk_ik_round_trip_is_exact_at_a_reached_posture(ik: CobotIK) -> None:
    seed = ik.neutral("R")
    target = ik.fk("R", seed)
    solution = ik.solve("R", target["p"], target["q"], q_init=seed)
    got = ik.fk("R", solution)
    assert np.linalg.norm(np.subtract(got["p"], target["p"])) < 1e-3
    assert abs(float(np.dot(got["q"], target["q"]))) > 0.999


def test_ik_reaches_a_tabletop_point_and_stays_inside_the_urdf_limits(ik: CobotIK) -> None:
    goal = [0.30, 0.20, 0.55]
    solution = ik.solve("L", goal, q_init=ik.neutral("L"))
    assert len(solution) == 6
    assert np.linalg.norm(np.subtract(ik.fk("L", solution)["p"], goal)) < 5e-3
    for value, (lo, hi) in zip(solution, ik.joint_limits("L"), strict=True):
        assert lo <= value <= hi


def test_solving_one_arm_leaves_the_other_arm_untouched(ik: CobotIK) -> None:
    # Seed the left arm somewhere distinctive first, then solve the right arm.
    ik.solve("L", [0.30, 0.20, 0.55], q_init=ik.neutral("L"))
    left_before = ik.current("L")
    ik.solve("R", [0.30, -0.20, 0.55], q_init=ik.neutral("R"))
    assert ik.current("L") == pytest.approx(left_before)


def test_unreachable_goal_raises_with_the_distance_missed(ik: CobotIK) -> None:
    with pytest.raises(IkError, match="missed"):
        ik.solve("L", [2.0, 0.0, 1.0], q_init=ik.neutral("L"))
    assert not ik.reachable("L", [2.0, 0.0, 1.0], q_init=ik.neutral("L"))


def test_orientation_constrained_solve_points_the_tool_down(ik: CobotIK) -> None:
    # Tool pointing straight down: -z of the world expressed in the link's own frame.
    down = [0.0, 0.7071068, 0.7071068, 0.0]  # wxyz: 180 deg about the (1, 1, 0) axis
    goal = [0.32, 0.16, 0.60]
    solution = ik.solve("L", goal, down, q_init=ik.neutral("L"))
    pose = ik.fk("L", solution)
    assert np.linalg.norm(np.subtract(pose["p"], goal)) < 5e-3
    assert abs(float(np.dot(pose["q"], down))) > 0.99


def test_bad_arguments_are_rejected(ik: CobotIK) -> None:
    with pytest.raises(KeyError):
        ik.fk("X")
    with pytest.raises(ValueError):
        ik.solve("L", [0.3, 0.2, 0.6], q_init=[0.0, 0.0])
