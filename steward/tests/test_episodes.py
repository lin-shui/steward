"""The episode page's data plane: a campaign directory read over WCA, and playback of it.

The manifest is the whole contract between a learning campaign and this console, so the fixture
here is a hand-written campaign of the documented shape -- no PyRUA code involved: an episode
with a trace, a finished one without, plus the worker round that produced its revision.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from steward.authoring.service import AuthoringService
from steward.net import call
from steward.world.server import StewardHost
from steward.world.sim import WorldSim

TASK = "bowl_on_plate"
REVISION = "0123456789abcdef"


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


def _write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) if not isinstance(payload, str) else payload)
    return path


def _rows(scene: dict, count: int = 4) -> list[dict]:
    sim = WorldSim(scene)
    try:
        base = sim.joint_state()
        free = next(name for name, values in base.items() if len(values) == 7)
        rows = []
        for index in range(count):
            joints = dict(base)
            joints[free] = [base[free][0] + 0.01 * index, *base[free][1:]]
            rows.append({"seq": index, "t": 0.5 * index, "frame": f"{index:04d}", "joints": joints})
        return rows
    finally:
        sim.stop()


def _campaign(root: Path, scene: dict) -> Path:
    campaign = root / "campaign"
    _write(campaign / "config.json", {"benchmark": "piper", "label": "fixture"})
    (campaign / "episodes.jsonl").write_text("")
    _write(campaign / "index.json", {
        "schema": "steward-episode-trace-v1", "campaign": "campaign",
        "config": {"benchmark": "piper"},
        "tasks": {TASK: {"rounds": [{"round": 1, "parent": "e3b0c44298fc1c14", "revision": REVISION,
                                     "dev": {"episodes": 1, "successes": 1}}], "test": None}},
    })
    run = campaign / "runs" / "dev" / TASK / f"{REVISION}-l0-r1-a0"
    episode = run / "episode"
    rows = _rows(scene)
    (episode / "trace").mkdir(parents=True)
    (episode / "trace" / "steps.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (episode / "workspace").mkdir()
    (episode / "workspace" / "cell_001.py").write_text("print('hello from the model')\n")
    (episode / "calls.jsonl").write_text('{"tool": "move_pose"}\n')
    (episode / "notes.bin").write_bytes(b"\x00\x01")
    (episode / "steps" / "0000").mkdir(parents=True)
    (episode / "steps" / "0000" / "cam_head.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    _write(episode / "result.json", {"environment_success": True, "end": "success", "total_wall_s": 12.5})
    _write(episode / "trace" / "manifest.json", {
        "schema": "steward-episode-trace-v1", "scenario": "phase0-kitchen", "task": TASK,
        "instruction": "put it there", "model": "test-model", "runtime": "openai",
        "identity": {"campaign": "campaign", "run": run.name, "task": TASK, "revision": REVISION,
                     "round": 1, "split": "dev", "layout": 0, "claim": "agreed"},
        "outcome": {"environment_success": True, "end": "success", "seconds": 12.5, "cells": 1,
                    "calls": 1},
        "steps": {"file": "trace/steps.jsonl", "count": len(rows), "last_t": rows[-1]["t"]},
        "frames": {"dir": "steps", "count": 1, "names": ["0000"], "cameras": ["cam_head"]},
        "records": {"cells": [{"index": 1, "file": "workspace/cell_001.py",
                               "code": "print('hello from the model')\n", "stdout": "hello"}],
                    "calls": "calls.jsonl"},
        "worker": {"round": 1, "parent": "e3b0c44298fc1c14", "revision": REVISION,
                   "findings": "the handle was white", "diff": [], "replays": [], "accepted": {}},
        "worker_trace": None,
    })
    # An episode that has started but not finished: no manifest yet, only its request.
    started = campaign / "runs" / "test" / TASK / "running-l1-r0-a0"
    _write(started / "request.json", {"task": TASK, "split": "test", "layout": 1,
                                      "revision": REVISION, "config": {"execution": {"model": "test-model"},
                                                                       "scenarios": {"test": ["phase0-kitchen"]}}})
    (started / "episode").mkdir(parents=True)
    # ... and an older finished one that predates the manifest: recorded, not running.
    old = campaign / "runs" / "dev" / TASK / "old-l0-a0"
    _write(old / "request.json", {"task": TASK, "split": "dev", "layout": 0, "revision": "e3b0c44298fc1c14",
                                  "config": {}})
    _write(old / "episode" / "result.json", {"environment_success": False, "end": "budget"})
    return campaign


def test_the_index_lists_finished_and_running_episodes(kitchen_root, kitchen_scene, tmp_path):
    campaign = _campaign(tmp_path, kitchen_scene)
    host = StewardHost(kitchen_root, scenario="phase0-kitchen", run_dir=tmp_path / "run",
                       campaigns=[campaign]).start()
    try:
        index = call("GET", f"{host.wca.url}/episodes/index")
        listed = {episode["episode"]: episode for episode in index["campaigns"][0]["episodes"]}
        finished = f"runs/dev/{TASK}/{REVISION}-l0-r1-a0"
        running = f"runs/test/{TASK}/running-l1-r0-a0"
        old = f"runs/dev/{TASK}/old-l0-a0"
        assert set(listed) == {finished, old, running}
        assert listed[finished]["success"] is True and listed[finished]["steps"] == 4
        assert listed[finished]["has_worker"] is True
        assert listed[finished]["live"] is False
        assert listed[running]["live"] is True and listed[running]["success"] is None
        assert listed[old]["live"] is False and listed[old]["success"] is False
        assert index["campaigns"][0]["tasks"][TASK]["rounds"][0]["revision"] == REVISION
    finally:
        host.stop()


def test_manifest_steps_and_artifacts_are_served_from_the_campaign(kitchen_root, kitchen_scene, tmp_path):
    campaign = _campaign(tmp_path, kitchen_scene)
    host = StewardHost(kitchen_root, scenario="phase0-kitchen", run_dir=tmp_path / "run",
                       campaigns=[campaign]).start()
    episode = f"runs/dev/{TASK}/{REVISION}-l0-r1-a0"
    try:
        base = f"{host.wca.url}/episodes"
        manifest = call("GET", f"{base}/manifest?campaign=campaign&episode={episode}")
        assert manifest["scenario"] == "phase0-kitchen" and manifest["steps"]["count"] == 4
        assert manifest["records"]["cells"][0]["stdout"] == "hello"
        steps = call("GET", f"{base}/steps?campaign=campaign&episode={episode}&start=2&count=1")
        assert steps["total"] == 4 and steps["count"] == 1 and steps["rows"][0]["seq"] == 2
        png = call("GET", f"{base}/artifact?campaign=campaign&path={episode}/episode/steps/0000/cam_head.png")
        assert png.startswith(b"\x89PNG")
        live = call("GET", f"{base}/live?campaign=campaign&episode={episode}")
        assert live["calls"] == 1 and live["steps"] == 4 and live["finished"] is True
        for path, expected in ((f"{episode}/episode/notes.bin", 415),
                               ("../escape.json", 400),
                               (f"{episode}/episode/missing.png", 404)):
            refused = None
            try:
                call("GET", f"{base}/artifact?campaign=campaign&path={path}")
            except Exception as exc:  # noqa: BLE001 - the status is the assertion
                refused = exc
            assert refused is not None and refused.status == expected, (path, refused)
    finally:
        host.stop()


def test_playback_drives_the_host_world_from_the_trace(kitchen_root, kitchen_scene, tmp_path):
    campaign = _campaign(tmp_path, kitchen_scene)
    host = StewardHost(kitchen_root, scenario="phase0-kitchen", run_dir=tmp_path / "run",
                       campaigns=[campaign]).start()
    episode = f"runs/dev/{TASK}/{REVISION}-l0-r1-a0"
    try:
        loaded = _pump_while(host, lambda: call("POST", f"{host.wca.url}/episodes/load",
                                               {"campaign": "campaign", "episode": episode}))
        assert loaded["value"]["count"] == 4 and loaded["value"]["index"] == 0
        assert host.sim.paused, "playback must not let the host integrate its own physics"
        first = host.sim.joint_state()
        seek = _pump_while(host, lambda: call("POST", f"{host.wca.url}/episodes/seek", {"index": 3}))
        assert seek["value"]["index"] == 3 and seek["value"]["frame"] == "0003"
        assert host.sim.joint_state() != first, "seek did not move the world"
        played = _pump_while(host, lambda: call("POST", f"{host.wca.url}/episodes/play", {"fps": 60}))
        assert played["value"]["playing"] is True
        paused = _pump_while(host, lambda: call("POST", f"{host.wca.url}/episodes/pause", {}))
        assert paused["value"]["playing"] is False
        # Playing through stops *on* the last row (not one past it), and pressing play again
        # from there restarts the trace instead of doing nothing.
        _pump_while(host, lambda: call("POST", f"{host.wca.url}/episodes/play", {"fps": 60}))
        time.sleep(0.3)
        ended = call("GET", f"{host.wca.url}/episodes/status")
        assert ended["index"] == 3 and ended["playing"] is False and ended["frame"] == "0003"
        again = _pump_while(host, lambda: call("POST", f"{host.wca.url}/episodes/play", {"fps": 60}))
        assert again["value"]["index"] < 3, "play from the last row must replay from the start"
        _pump_while(host, lambda: call("POST", f"{host.wca.url}/episodes/pause", {}))
        assert call("GET", f"{host.wca.url}/episodes/status")["count"] == 4
    finally:
        host.stop()
