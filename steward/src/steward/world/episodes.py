"""Episode records and playback for the console: the operator's view of a learning campaign.

A campaign directory is written by PyRUA-Lean; the only thing this module knows about it is the
one document per episode that project publishes -- ``<episode>/trace/manifest.json`` (schema
``steward-episode-trace-v1``) -- and the campaign's own ``index.json``.  Skills, worker findings
and verdicts all arrive inside those documents, so the console never has to learn a benchmark's
directory layout, and any other recorder that writes the same two files works here too.

Two halves:

* :class:`Campaigns` lists campaigns/episodes/artifacts from disk (read-only, root-confined);
* :class:`Player` puts a recorded episode's world states into this host's WSim, so the console's
  3D view shows the run -- the same state rows the live mirror receives, read from the trace
  instead of the wire.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from ..net import Binary, HttpError

#: The manifest schema this page understands; anything else is listed but not opened.
SCHEMA = "steward-episode-trace-v1"
ARTIFACT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".md": "text/markdown; charset=utf-8",
    ".py": "text/x-python; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".log": "text/plain; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
}
#: Rows a page may fetch in one request (a whole episode is small; this bounds a mistake).
STEPS_PAGE = 2000
EPISODES_HTML = Path(__file__).with_name("episodes.html")


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _newest(path: Path) -> float:
    """The latest mtime among ``path`` and its direct children (0 when it does not exist)."""

    newest = 0.0
    try:
        newest = path.stat().st_mtime
        for child in path.iterdir():
            try:
                newest = max(newest, child.stat().st_mtime)
            except OSError:
                continue
    except OSError:
        return 0.0
    return newest


def _recent(path: Path, seconds: float = 1800.0) -> bool:
    """Whether something under ``path`` was written within ``seconds``.

    An episode without a result.json is only *running* while its files are still growing; one
    that stopped half a day ago (a killed run, a renamed backup) is unfinished, not live.
    """

    newest = _newest(path)
    return newest > 0 and time.time() - newest < seconds


def _rate(rows: list[dict[str, Any]]) -> float:
    """Rows per second of recorded time -- the fps at which playback lasts as long as the trace."""

    times = [row.get("t") for row in rows if isinstance(row.get("t"), (int, float))]
    span = (times[-1] - times[0]) if len(times) > 1 else 0.0
    if span <= 0:
        return 25.0
    return max(1.0, min(60.0, (len(times) - 1) / span))


class Campaigns:
    """The campaign directories this host was told to show (``--campaign``), read-only."""

    def __init__(self, roots: list[Path] | None = None) -> None:
        self.roots: list[Path] = []
        for root in roots or []:
            self.add(root)

    def add(self, path: Path | str) -> dict[str, Any]:
        root = Path(path).expanduser().resolve()
        if not (root / "config.json").is_file() or not (root / "episodes.jsonl").is_file():
            raise HttpError(400, f"{root} is not a campaign (needs config.json and episodes.jsonl)")
        if root not in self.roots:
            self.roots.append(root)
        return {"name": root.name, "root": str(root)}

    def root(self, name: str | None) -> Path:
        if not self.roots:
            raise HttpError(409, "no campaign is configured; start the host with --campaign DIR")
        if name in (None, ""):
            return self.roots[0]
        for root in self.roots:
            if root.name == name:
                return root
        raise HttpError(404, f"no campaign named {name!r}")

    def episode_dir(self, campaign: str | None, episode: str) -> Path:
        """The episode's own directory (the one holding result.json), from any path in the campaign.

        Confined to the campaign, not to ``runs/``: a worker's own replays live under
        ``tasks/<task>/worker-rN/replays/`` and are worth opening the same way.
        """

        root = self.root(campaign)
        run = (root / episode).resolve()
        if root not in run.parents:
            raise HttpError(400, "episode must live inside the campaign")
        target = run / "episode"
        if not target.is_dir():
            raise HttpError(404, f"no episode at {episode}")
        return target

    # ------------------------------------------------------------------ reads
    def manifest(self, campaign: str | None, episode: str) -> dict[str, Any]:
        path = self.episode_dir(campaign, episode) / "trace" / "manifest.json"
        manifest = _read_json(path)
        if not isinstance(manifest, dict):
            raise HttpError(404, f"{episode} has no trace/manifest.json (an episode records one "
                                 "when it ends; a live one shows its records only)")
        if manifest.get("schema") != SCHEMA:
            raise HttpError(422, f"unknown manifest schema {manifest.get('schema')!r}")
        manifest["episode"] = episode
        manifest["campaign"] = self.root(campaign).name
        manifest["rel"] = episode
        return manifest

    def summary(self, root: Path) -> dict[str, Any]:
        """One campaign: its index (rounds, test pairs) plus every episode that has a manifest.

        Episodes still running have no manifest yet; they are listed from their ``request.json``
        so an episode can be watched while it happens, not only after it ends.
        """

        index = _read_json(root / "index.json") or {}
        episodes: dict[str, dict[str, Any]] = {}
        for path in sorted(root.glob("runs/**/episode/trace/manifest.json")):
            manifest = _read_json(path) or {}
            if manifest.get("schema") != SCHEMA:
                continue
            identity, outcome = manifest.get("identity") or {}, manifest.get("outcome") or {}
            run_dir = path.parent.parent.parent
            episodes[run_dir.relative_to(root).as_posix()] = {
                "task": identity.get("task"),
                "revision": identity.get("revision"),
                "round": identity.get("round"),
                "split": identity.get("split"),
                "layout": identity.get("layout"),
                "claim": identity.get("claim"),
                "model": manifest.get("model"),
                "runtime": manifest.get("runtime"),
                "scenario": manifest.get("scenario"),
                "success": outcome.get("environment_success"),
                "end": outcome.get("end"),
                "seconds": outcome.get("seconds"),
                "cells": outcome.get("cells"),
                "steps": (manifest.get("steps") or {}).get("count") or 0,
                "has_worker": bool(manifest.get("worker")),
                "live": False,
                "mtime": _newest(path.parent.parent),
            }
        for path in sorted(root.glob("runs/**/request.json")):
            run_dir = path.parent
            relative = run_dir.relative_to(root).as_posix()
            request = _read_json(path) or {}
            episode_dir = run_dir / "episode"
            result = _read_json(episode_dir / "result.json") or {}
            running = not result and _recent(episode_dir if episode_dir.is_dir() else run_dir)
            if relative in episodes:
                episodes[relative]["live"] = running
                continue
            config = request.get("config") or {}
            episodes[relative] = {
                "task": request.get("task"),
                "revision": request.get("revision"),
                "round": None,
                "split": request.get("split"),
                "layout": request.get("layout"),
                "claim": None,
                "model": (config.get("execution") or {}).get("model"),
                "runtime": None,
                "scenario": next(iter((config.get("scenarios") or {}).get(request.get("split")) or []), None),
                "success": result.get("environment_success") if result else None,
                "end": result.get("end") if result else None,
                "seconds": result.get("total_wall_s") if result else None,
                "cells": None,
                "steps": len(list((episode_dir / "trace" / "steps.jsonl").read_text().splitlines()))
                if (episode_dir / "trace" / "steps.jsonl").is_file() else 0,
                "has_worker": False,
                "live": running,
                "mtime": _newest(episode_dir if episode_dir.is_dir() else run_dir),
            }
        return {
            "name": root.name,
            "root": str(root),
            "config": index.get("config") or _read_json(root / "config.json") or {},
            "report": index.get("report"),
            "tasks": index.get("tasks") or {},
            "episodes": [{"episode": relative, **fields} for relative, fields in sorted(episodes.items())],
            "updated": (root / "index.json").stat().st_mtime if (root / "index.json").is_file() else None,
        }

    def list(self, campaign: str | None = None) -> dict[str, Any]:
        roots = [self.root(campaign)] if campaign else self.roots
        return {"campaigns": [self.summary(root) for root in roots]}

    def steps(self, campaign: str | None, episode: str, start: int = 0, count: int = STEPS_PAGE) -> dict:
        """The recorded world states of an episode: what the 3D view is driven from."""

        manifest = self.manifest(campaign, episode)
        rel = (manifest.get("steps") or {}).get("file")
        if not rel:
            raise HttpError(404, "this episode recorded no world states (the twin did not serve "
                                 "/mirror while it ran)")
        path = self.episode_dir(campaign, episode) / rel
        rows = []
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index < start:
                    continue
                if len(rows) >= min(int(count), STEPS_PAGE):
                    break
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return {"start": start, "count": len(rows), "total": (manifest.get("steps") or {}).get("count"),
                "rows": rows}

    def artifact(self, campaign: str | None, path: str) -> Binary:
        root = self.root(campaign)
        target = (root / path).resolve()
        if root not in target.parents:
            raise HttpError(400, "artifact must live inside the campaign")
        if not target.is_file():
            raise HttpError(404, f"no artifact {path}")
        kind = ARTIFACT_TYPES.get(target.suffix.lower())
        if kind is None:
            raise HttpError(415, f"{target.suffix} is not an artifact type the console serves")
        return Binary(target.read_bytes(), kind)

    def live(self, campaign: str | None, episode: str) -> dict[str, Any]:
        """A running episode's records, re-read from disk so the page can follow it.

        The same files the manifest folds together later: the cells written so far, the call
        log's length, the verdict once ``result.json`` appears.
        """

        directory = self.episode_dir(campaign, episode)
        workspace = sorted(directory.glob("workspace/cell_*.py"))
        calls = directory / "calls.jsonl"
        result = _read_json(directory / "result.json")
        trace = directory / "trace" / "steps.jsonl"
        return {
            "episode": episode,
            "finished": isinstance(result, dict) and bool(result),
            "cells": [{"index": int(p.stem.split("_")[-1]), "file": p.relative_to(directory).as_posix(),
                       "code": p.read_text(encoding="utf-8", errors="replace")[-20000:]}
                      for p in workspace if p.stem.split("_")[-1].isdigit()],
            "calls": sum(1 for _ in calls.open(encoding="utf-8")) if calls.is_file() else 0,
            "steps": sum(1 for _ in trace.open(encoding="utf-8")) if trace.is_file() else 0,
            "result": result if isinstance(result, dict) and result else None,
        }


class Player:
    """Drive this host's world from a recorded episode's state rows, on its own thread."""

    def __init__(self, host) -> None:
        self.host = host
        # Reentrant: the small accessors below call each other while reporting (status inside
        # play), and a plain Lock turned that into a self-deadlock that hung the whole host.
        self.lock = threading.RLock()
        self.episode: str | None = None
        self.campaign: str | None = None
        self.index = 0
        self.count = 0
        self.fps = 25.0
        self.playing = False
        self.error: str | None = None
        self.updated = 0.0
        self._rows: list[dict[str, Any]] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def status(self) -> dict[str, Any]:
        with self.lock:
            row = self._rows[self.index] if 0 <= self.index < len(self._rows) else {}
            return {
                "episode": self.episode, "campaign": self.campaign, "playing": self.playing,
                "index": self.index, "count": len(self._rows), "fps": self.fps,
                "t": row.get("t"), "frame": row.get("frame"), "error": self.error,
                "updated_ago_s": round(time.time() - self.updated, 2) if self.updated else None,
            }

    def load(self, campaign: str, episode: str, rows: list[dict[str, Any]], scenario: str) -> None:
        """Take an episode's rows; builds its scene on the main thread first."""

        self.stop()
        if scenario and (self.host.scenario or {}).get("name") != scenario:
            self.host.on_main(lambda: self.host.load_episode(scenario))
        if self.host.sim is None:
            raise HttpError(409, "the host has no episode loaded")
        self.host.sim.paused = True
        with self.lock:
            self._rows = [row for row in rows if row.get("joints")]
            self.episode, self.campaign = episode, campaign
            self.index = 0
            self.error = None
            self.updated = time.time()
            # 1x playback: as many rows per second as the trace's own clock had, so a run is
            # watched at the pace it happened rather than flashing past at a default fps.
            self.fps = _rate(self._rows)
            first = self._rows[0] if self._rows else None
        if first is not None:
            self._apply(first)

    def seek(self, index: int) -> dict[str, Any]:
        with self.lock:
            if not self._rows:
                raise HttpError(409, "no episode is loaded for playback")
            self.index = max(0, min(int(index), len(self._rows) - 1))
            row = self._rows[self.index]
        self._apply(row)
        return self.status()

    def play(self, fps: float | None = None) -> dict[str, Any]:
        with self.lock:
            if not self._rows:
                raise HttpError(409, "no episode is loaded for playback")
            if fps:
                self.fps = max(1.0, min(60.0, float(fps)))
            if self.playing:
                return self.status()
            first = None
            if self.index >= len(self._rows) - 1:
                # The end was reached: pressing play again replays from the first row.
                self.index = 0
                first = self._rows[0]
            self.playing = True
            self._stop.clear()
            if first is not None:
                self._apply(first)
            self._thread = threading.Thread(target=self._run, name="episode-play", daemon=True)
            self._thread.start()
            return self.status()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self.lock:
            self.playing = False
        return self.status()

    def _run(self) -> None:
        period = 1.0 / self.fps
        while not self._stop.is_set():
            with self.lock:
                if self.index >= len(self._rows) - 1:
                    # Stop *on* the last row, not one past it: the page's counter and scrubber
                    # then rest on the last recorded state instead of an empty row.
                    self.playing = False
                    break
                index = self.index
                self.index = index + 1
                row = self._rows[self.index]
            started = time.monotonic()
            self._apply(row)
            wait = period - (time.monotonic() - started)
            if wait > 0:
                self._stop.wait(wait)

    def _apply(self, row: dict[str, Any]) -> None:
        sim = self.host.sim
        if sim is None:
            return
        try:
            sim.apply_joint_state({name: list(values) for name, values in row["joints"].items()},
                                  time=row.get("t"))
            with self.lock:
                self.updated = time.time()
        except Exception as error:  # noqa: BLE001 - surfaced in the page rather than killing playback
            with self.lock:
                self.error = f"{type(error).__name__}: {error}"
                self.playing = False
            self._stop.set()


def player_of(host) -> Player:
    player = getattr(host, "_episode_player", None)
    if player is None:
        player = Player(host)
        host._episode_player = player
    return player


def campaigns_of(host) -> Campaigns:
    campaigns = getattr(host, "_episode_campaigns", None)
    if campaigns is None:
        campaigns = Campaigns(getattr(host, "episode_roots", []) or [])
        host._episode_campaigns = campaigns
    return campaigns


def episode_routes(host) -> dict[tuple[str, str], Any]:
    """WCA routes of the episode page: list, manifest, artifacts, playback."""

    def page(query: dict, body: dict) -> Binary:
        return Binary(EPISODES_HTML.read_bytes(), "text/html; charset=utf-8")

    def index(query: dict, body: dict) -> dict:
        return campaigns_of(host).list(query.get("campaign"))

    def manifest(query: dict, body: dict) -> dict:
        return campaigns_of(host).manifest(query.get("campaign"), query["episode"])

    def steps(query: dict, body: dict) -> dict:
        return campaigns_of(host).steps(query.get("campaign"), query["episode"],
                                        int(query.get("start", 0)), int(query.get("count", STEPS_PAGE)))

    def artifact(query: dict, body: dict) -> Binary:
        return campaigns_of(host).artifact(query.get("campaign"), query["path"])

    def live(query: dict, body: dict) -> dict:
        return campaigns_of(host).live(query.get("campaign"), query["episode"])

    def add_root(query: dict, body: dict) -> dict:
        return campaigns_of(host).add((body or {}).get("path") or "")

    def load(query: dict, body: dict) -> dict:
        body = body or {}
        campaign = campaigns_of(host).root(body.get("campaign"))
        record = campaigns_of(host).manifest(body["campaign"], body["episode"])
        rows = campaigns_of(host).steps(body["campaign"], body["episode"], 0, STEPS_PAGE)["rows"]
        player = player_of(host)
        player.load(campaign.name, body["episode"], rows, str(record.get("scenario") or ""))
        return {**player.status(), "manifest": record}

    def seek(query: dict, body: dict) -> dict:
        return player_of(host).seek(int((body or {}).get("index", 0)))

    def play(query: dict, body: dict) -> dict:
        return player_of(host).play((body or {}).get("fps"))

    def pause(query: dict, body: dict) -> dict:
        return player_of(host).stop()

    def status(query: dict, body: dict) -> dict:
        return player_of(host).status()

    return {
        ("GET", "/episodes"): page,
        ("GET", "/episodes/index"): index,
        ("GET", "/episodes/manifest"): manifest,
        ("GET", "/episodes/steps"): steps,
        ("GET", "/episodes/artifact"): artifact,
        ("GET", "/episodes/live"): live,
        ("POST", "/episodes/campaign"): add_root,
        ("POST", "/episodes/load"): load,
        ("POST", "/episodes/seek"): seek,
        ("POST", "/episodes/play"): play,
        ("POST", "/episodes/pause"): pause,
        ("GET", "/episodes/status"): status,
    }
