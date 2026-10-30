"""A Cobot Magic mission driven by what the robot sees, not by ground truth.

The twin's skills take positions; this module is what supplies them. It feeds sensor frames to
the Ego estimator, reads the resulting tracks, and hands the *estimated* object positions to the
skills. That is the whole point of the exercise: a pick succeeds or fails on the robot's own
perception of where the bowl is.

Phase 0 detection frames carry labelled boxes from the world; the RGB-D phase replaces them with
:meth:`WorldSim.rgbd` frames and the same estimator, which is why nothing here names a simulator
concept beyond ``frame()``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from steward.ego.cobot_skills import CobotSkills, SkillFailed, SkillStopped
from steward.ego.estimator import EgoEstimator


@dataclass
class Sighting:
    track_id: str
    category: str
    pos: np.ndarray
    size: np.ndarray
    existence: float
    range_m: float


def perceive(est: EgoEstimator, frame: Callable[[], dict], frames: int = 8,
             pause_s: float = 0.02, min_existence: float = 0.4,
             rgbd: bool = False) -> dict[str, Sighting]:
    """Update the estimator ``frames`` times and report the best track per category.

    ``rgbd=True`` means ``frame()`` returns a decoded RGB-D frame (rgb, depth, anonymous masks,
    header) instead of labelled detections: the robot then has to find and measure the objects
    itself, which is the whole point of the exercise.
    """
    for _ in range(frames):
        payload = frame()
        if rgbd:
            est.update_rgbd(payload)
        else:
            est.update(payload)
        if pause_s:
            time.sleep(pause_s)
    best: dict[str, Sighting] = {}
    for track in est.tracks.values():
        if not track.category_counts or track.existence < min_existence:
            continue
        category = max(track.category_counts, key=track.category_counts.get)
        cam = est.camera_pose()
        sighting = Sighting(track_id=track.id, category=category, pos=np.array(track.pos, dtype=float),
                            size=np.array(track.size, dtype=float), existence=float(track.existence),
                            range_m=float(np.linalg.norm(sighting_distance(track.pos, cam))))
        if category not in best or sighting.existence > best[category].existence:
            best[category] = sighting
    return best


def sighting_distance(pos, cam_pose: dict) -> np.ndarray:
    from steward.math3d import pose_compose, pose_inverse

    return np.asarray(pose_compose(pose_inverse(cam_pose), {"p": list(pos), "q": [1.0, 0.0, 0.0, 0.0]})["p"])


def bowl_onto_plate(est: EgoEstimator, skills: CobotSkills, frame: Callable[[], dict],
                    grip: str = "L_link6", retries: int = 2, rgbd: bool = False) -> dict:
    """Look at the table, pick up the bowl where the robot believes it is, and set it on the plate.

    A missed grasp is fed back: the robot looks again before the next attempt, so a stale or wrong
    belief costs one retry instead of the mission.
    """
    log: list[str] = []
    for attempt in range(1, retries + 1):
        seen = perceive(est, frame, rgbd=rgbd, frames=10 if rgbd else 8)
        missing = [c for c in ("bowl", "plate") if c not in seen]
        if missing:
            log.append(f"attempt {attempt}: not seen: {', '.join(missing)}")
            continue
        bowl, plate = seen["bowl"], seen["plate"]
        log.append(f"attempt {attempt}: bowl {_fmt(bowl.pos)} (e:{bowl.track_id}, {bowl.range_m:.2f} m), "
                   f"plate {_fmt(plate.pos)}")
        if not _arm_can_reach(skills, bowl.pos):
            log.append(f"attempt {attempt}: the bowl at {_fmt(bowl.pos)} is outside the left arm's workspace")
            continue
        # Set the bowl down on the plate's top surface, from the sizes the robot measured: placing
        # it at the plate's *centre* puts the bowl's base through the plate, and every round then
        # shoves the plate a couple of centimetres until the bowl hides it from the camera.
        plate_top = float(plate.pos[2]) + 0.5 * _extent(plate.size, 2, 0.01)
        target = [float(plate.pos[0]), float(plate.pos[1]),
                  plate_top + 0.5 * _extent(bowl.size, 2, 0.05) + 0.005]
        try:
            _run(skills, "pick", {"grip": grip, "arm": "L", "at": [float(v) for v in bowl.pos]}, log)
            _run(skills, "place", {"grip": grip, "arm": "L", "at": target}, log)
        except SkillStopped:
            raise
        except SkillFailed as exc:
            log.append(f"attempt {attempt} failed: {exc}")
            self_holding = (est.gripper or {}).get("holding") or self._holding(skills, grip)
            if self_holding:
                # Put the load down where it is: a retry that reaches for an object still in the
                # gripper can only fail, and leaving it held hides the failure from the operator.
                _run(skills, "open", {"grip": grip}, log)
                log.append(f"attempt {attempt}: put the load down where it was held")
            continue
        return {"ok": True, "attempts": attempt, "bowl": bowl.pos.tolist(), "plate": plate.pos.tolist(), "log": log}
    return {"ok": False, "log": log}


def _run(skills: CobotSkills, name: str, plan: dict, log: list[str]) -> None:
    job = skills.start(name, {"id": f"mission-{name}", "plan": plan})
    while job.state == "running":
        time.sleep(0.02)
    log.extend(f"{name}: {line}" for line in job.log)
    if job.state != "succeeded":
        raise SkillFailed(job.message or job.state)


def _extent(size, axis: int, fallback: float) -> float:
    """One dimension of a measured box, with a fallback when the estimate did not carry sizes."""
    try:
        value = float(size[axis])
    except (IndexError, TypeError, ValueError):
        return fallback
    return value if value > 1e-4 else fallback


def _arm_can_reach(skills: CobotSkills, pos: np.ndarray) -> bool:
    """A crude gate before commanding: is this anywhere near the arm's measured workspace?"""
    return 0.10 <= float(pos[0]) <= 0.60 and 0.02 <= float(pos[1]) <= 0.55 and 0.55 <= float(pos[2]) <= 1.05


def _holding(skills: CobotSkills, grip: str) -> bool:
    st = skills._state()
    return bool((st.get("grippers") or {}).get(grip, {}).get("holding"))


def _fmt(p) -> str:
    return "(" + ", ".join(f"{float(v):.3f}" for v in p) + ")"
