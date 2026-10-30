"""Cobot Magic inverse kinematics on the robot's own kinematic model.

The twin's model is the vendored Piper URDF (see :mod:`steward.assets.cobot`), compiled to
MJCF and solved with mink: 6 revolute joints per arm, no analytic IK, and 6-DoF is exactly the
case PyRoki refuses, so the solver is mink exactly as on the YAM arm.

The solver freezes every DOF outside the arm being solved — the other arm's joints and both
fingers — so a left-arm goal never drags the right arm along. Joint limits come from the model
(:class:`mink.ConfigurationLimit`), so a solution is always inside the URDF ranges.

Frames: the end-effector frame of an arm is its last revolute link (``L_link6`` / ``R_link6``).
Its +z axis points out through the pinch point, 0.135 m away (``assets.cobot.TCP_OFFSET``), and
the fingers separate along its y axis; the result is measured, not assumed — see
``tests/test_cobot.py::test_tcp_is_where_the_fingers_pinch``.
"""

from __future__ import annotations

import numpy as np

from steward.assets.cobot import HOME, cobot_magic
from steward.scene.mjcf import build_mjcf
from steward.scene.schema import SceneState

import mink

ARMS = ("L", "R")
_ENT = "w:robot"  # entity id used to compile the robot-only model; MJCF names are prefixed with it
_DOF_PER_ARM = 6
_FINGER_JOINTS = ("joint7", "joint8")


class IkError(RuntimeError):
    """No joint solution reached the requested pose within the iteration budget."""


class CobotIK:
    """FK/IK for one Cobot Magic twin, on the robot's own (URDF-derived) model."""

    def __init__(self, robot: dict | None = None, timestep: float = 0.002) -> None:
        self.robot = robot if robot is not None else cobot_magic().to_dict()
        # Only the robot is compiled: IK must not depend on scene furniture.
        ent = dict(self.robot)
        ent["id"] = _ENT
        build = build_mjcf(SceneState(mode="world", entities=[_as_entity(ent)]), timestep=timestep)
        self.model = build.compile()
        self.data = mink.Configuration(self.model)
        # mink starts from a zero posture; the twin starts from the URDF's home, and FK must agree
        # with the simulation before the first solve.
        self.data.update(np.asarray(self.model.qpos0, dtype=float))
        self.dt = timestep
        self._arms = {}
        for arm in ARMS:
            plain = [f"{arm}_joint{i}" for i in range(1, _DOF_PER_ARM + 1)]
            names = [f"{_ENT}::{p}" for p in plain]
            qadrs = []
            dofadrs = []
            for name in names:
                jid = _joint_id(self.model, name)
                qadrs.append(int(self.model.jnt_qposadr[jid]))
                dofadrs.append(int(self.model.jnt_dofadr[jid]))
            self._arms[arm] = {
                "joints": plain,
                "qadr": qadrs,
                "dofadr": dofadrs,
                "ee": f"{_ENT}/{arm}_link6",
                "limits": [(float(self.model.jnt_range[_joint_id(self.model, n)][0]),
                            float(self.model.jnt_range[_joint_id(self.model, n)][1])) for n in names],
            }
        # Every DOF that is not part of a solved arm is frozen (other arm, both fingers).
        self._freeze = {}
        for arm in ARMS:
            frozen = [d for other in ARMS if other != arm for d in self._arms[other]["dofadr"]]
            frozen += [int(self.model.jnt_dofadr[_joint_id(self.model, f"{_ENT}::{a}_{j}")])
                       for a in ARMS for j in _FINGER_JOINTS]
            self._freeze[arm] = mink.DofFreezingTask(self.model, dof_indices=frozen)
        self._posture = mink.PostureTask(self.model, cost=1e-2)
        self._limit = mink.ConfigurationLimit(self.model)

    # ------------------------------------------------------------------ queries
    def joint_names(self, arm: str) -> list[str]:
        return list(self._arms[self._check(arm)]["joints"])

    def joint_limits(self, arm: str) -> list[tuple[float, float]]:
        return list(self._arms[self._check(arm)]["limits"])

    def neutral(self, arm: str) -> list[float]:
        """A mid-range starting posture: the middle of each joint's URDF range."""
        return [0.5 * (lo + hi) for lo, hi in self.joint_limits(arm)]

    def current(self, arm: str) -> list[float]:
        spec = self._arms[self._check(arm)]
        q = self.data.q
        return [float(q[a]) for a in spec["qadr"]]

    def set_joints(self, arm: str, joints: list[float]) -> None:
        """Seed the solver from measured joint angles (the robot's encoders, not a past solution)."""
        spec = self._arms[self._check(arm)]
        _check_len(joints, _DOF_PER_ARM, f"{arm} arm joints")
        q = np.array(self.data.q, dtype=float)
        for qadr, value in zip(spec["qadr"], joints, strict=True):
            q[qadr] = value
        self.data.update(q)

    def fk(self, arm: str, joints: list[float] | None = None) -> dict:
        """End-effector pose ``{p: [x, y, z], q: wxyz}`` in the world frame."""
        spec = self._arms[self._check(arm)]
        q = np.array(self.data.q, dtype=float)
        if joints is not None:
            _check_len(joints, _DOF_PER_ARM, f"{arm} arm joints")
            for qadr, value in zip(spec["qadr"], joints, strict=True):
                q[qadr] = value
        self.data.update(q)
        pose = self.data.get_transform_frame_to_world(spec["ee"], "body")
        p = np.asarray(pose.translation())
        rot = np.asarray(pose.rotation().as_matrix())
        return {"p": [float(v) for v in p], "q": _quat_from_matrix(rot)}

    # --------------------------------------------------------------------- ik
    def solve(self, arm: str, position: list[float], orientation: list[float] | None = None,
              q_init: list[float] | None = None, iterations: int = 120,
              tolerance: float = 5e-3, orientation_tolerance: float = 0.05) -> list[float]:
        """Joint angles (6) that put the arm's end-effector at ``position`` (metres, world).

        ``orientation`` is a wxyz quaternion; when omitted only position is constrained.
        Raises :class:`IkError` when the goal is unreachable within ``tolerance``.
        """
        arm = self._check(arm)
        spec = self._arms[arm]
        start = self.current(arm) if q_init is None else list(q_init)
        _check_len(start, _DOF_PER_ARM, f"{arm} arm seed")

        q = np.array(self.data.q, dtype=float)
        for qadr, value in zip(spec["qadr"], start, strict=True):
            q[qadr] = value
        self.data.update(q)

        if orientation is not None:
            _check_len(orientation, 4, "orientation quaternion")
        # A position-only goal still needs a target pose; the zero orientation cost ignores it.
        task = mink.FrameTask(spec["ee"], "body", position_cost=1.0,
                              orientation_cost=0.0 if orientation is None else 1.0,
                              gain=0.6, lm_damping=1e-1)
        task.set_target(_se3(position, orientation if orientation is not None else [1.0, 0.0, 0.0, 0.0]))
        self._posture.set_target(self.data.q)
        tasks = [task, self._posture]

        best, best_err, oerr = None, float("inf"), 0.0
        for _ in range(iterations):
            try:
                velocity = mink.solve_ik(self.data, tasks, self.dt, solver="daqp",
                                         limits=[self._limit], constraints=[self._freeze[arm]])
            except mink.NoSolutionFound as exc:  # solver gave up: report the best pose so far
                raise IkError(f"{arm} arm IK found no solution toward {list(position)}") from exc
            self.data.integrate_inplace(velocity, self.dt)
            err, oerr = self._errors(arm, position, orientation)
            if err <= tolerance and oerr <= orientation_tolerance:
                return self.current(arm)
            if err < best_err:
                best, best_err = self.current(arm), err
        raise IkError(
            f"{arm} arm IK missed {list(position)}: {best_err * 1000:.1f} mm after {iterations} steps"
            + ("" if best is None else f" (best q={[round(v, 3) for v in best]}, rotation {oerr:.3f} rad)")
        )

    def reachable(self, arm: str, position: list[float], orientation: list[float] | None = None,
                  q_init: list[float] | None = None) -> bool:
        try:
            self.solve(arm, position, orientation, q_init)
        except IkError:
            return False
        return True

    def solve_tool(self, arm: str, position: list[float], orientation: list[float],
                   max_tilt: float = 0.6, q_init: list[float] | None = None) -> tuple[list[float], float]:
        """Reach ``position`` preferring ``orientation``, but accepting a tilt up to ``max_tilt``.

        The Piper wrist has a +-70 deg roll joint (joint5), so a strictly vertical tool is only
        reachable in a narrow band in front of the base; a real pick tilts the tool a little
        instead of refusing. Returns ``(joints, tilt_radians)``; the tilt is 0 when the strict
        orientation was reached.

        Seeds are tried in order — the caller's posture, then the home and mid-range postures —
        because a 6-DoF solve started from a folded posture can miss a reachable goal entirely.
        The strict orientation wins outright when any seed reaches it; otherwise the least-tilted
        solution within ``max_tilt`` is returned.
        """
        seeds = [self.current(arm) if q_init is None else list(q_init), list(HOME), self.neutral(arm)]
        strict_errors: list[str] = []
        for seed in seeds:
            try:
                return self.solve(arm, position, orientation, q_init=seed), 0.0
            except IkError as exc:
                strict_errors.append(str(exc))
        best: tuple[list[float], float] | None = None
        position_errors: list[str] = []
        for seed in seeds:
            try:
                joints = self.solve(arm, position, None, q_init=seed)
            except IkError as exc:
                position_errors.append(str(exc))
                continue
            tilt = _angle_between(self.fk(arm, joints)["q"], orientation)
            if tilt <= max_tilt and (best is None or tilt < best[1]):
                best = (joints, tilt)
        if best is not None:
            return best
        detail = position_errors[-1] if position_errors else strict_errors[-1]
        raise IkError(f"{arm} arm cannot reach {[round(v, 3) for v in position]} in a usable tool "
                      f"orientation: {detail}")

    # ------------------------------------------------------------------ private
    def _errors(self, arm: str, position: list[float], orientation: list[float] | None) -> tuple[float, float]:
        pose = self.data.get_transform_frame_to_world(self._arms[arm]["ee"], "body")
        err = float(np.linalg.norm(np.asarray(pose.translation()) - np.asarray(position)))
        if orientation is None:
            return err, 0.0
        target = _quat_from_matrix(np.asarray(_se3(position, orientation).rotation().as_matrix()))
        got = _quat_from_matrix(np.asarray(pose.rotation().as_matrix()))
        dot = abs(float(np.dot(target, got)))
        return err, 2.0 * float(np.arccos(min(1.0, dot)))

    @staticmethod
    def _check(arm: str) -> str:
        if arm not in ARMS:
            raise KeyError(f"unknown arm {arm!r}; expected one of {ARMS}")
        return arm


def _joint_id(model, name: str) -> int:
    import mujoco

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(f"model has no joint {name!r}")
    return jid


def _as_entity(d: dict):
    from steward.scene.schema import Entity

    return Entity.from_dict(d)


def _check_len(values, n: int, what: str) -> None:
    if len(values) != n:
        raise ValueError(f"{what} must have {n} values, got {len(values)}")


def _se3(position: list[float], orientation: list[float]) -> "mink.SE3":
    from steward.math3d import matrix_from_quat

    rot = np.asarray(matrix_from_quat(list(orientation)), dtype=float).reshape(3, 3)
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = np.asarray(position, dtype=float)
    return mink.SE3.from_matrix(m)


def _quat_from_matrix(m: np.ndarray) -> list[float]:
    """Rotation matrix to a wxyz quaternion, normalised."""
    from steward.math3d import quat_from_matrix as _qf

    return [float(v) for v in _qf(np.asarray(m, dtype=float))]


def _angle_between(a: list[float], b: list[float]) -> float:
    """Angle between two orientations, in radians (0 when identical)."""
    dot = abs(float(np.dot(np.asarray(a, dtype=float), np.asarray(b, dtype=float))))
    return 2.0 * float(np.arccos(min(1.0, dot)))
