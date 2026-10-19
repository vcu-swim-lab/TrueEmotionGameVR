"""Loading and metrics for the CSV files written by Assets/Scripts/SessionLogger.cs.

The logger writes one row per prediction tick, carrying the argmax class, the full
per-class score vector, and the entire five-frame action-unit window that produced
the prediction. This module turns a directory of those files into numbers: how often
the model agreed with the prompt, which emotions get confused for which, whether the
headset was actually tracking, and which action units ever moved.

Only the standard library is used. These files are read on lab machines and student
laptops that do not all have pandas, and the rows-per-session count is small enough
(a few hundred) that plain `csv` is not the bottleneck.

The schema is read out of the header rather than hardcoded: the class list, the
window length, and the feature count all come from the column names, so a model
change on the Unity side does not silently produce wrong numbers here.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

STATUS_OK = "ok"
STATUS_NO_TRACKING = "no_tracking"

# The model itself failed (Jev unreachable, key rejected ...), as opposed to the headset not
# delivering a face. Written by SessionLogger.cs since the four conditions; `error_code` says why.
STATUS_BACKEND_ERROR = "backend_error"

# Canonical `block_role` values, matching the literals SessionLogger.cs writes.
ROLE_CALIBRATION = "calibration"
ROLE_EVALUATION = "evaluation"

# Action unit values below this are treated as "not moving". The exported floats
# bottom out at denormals (1.4e-45) rather than exactly zero, so a `> 0` test would
# report every channel as active.
AU_EPSILON = 1e-6

_VALID_RE = re.compile(r"^frame_(\d+)_valid$")
_AU_RE = re.compile(r"^frame_(\d+)_au_(\d+)$")

_REQUIRED = (
    "timestamp_utc",
    "session_id",
    "trial_id",
    "prompted_emotion",
    "prediction_index",
    "predicted_emotion",
    "correct",
    "status",
    "generation",
)


class SchemaError(ValueError):
    """Raised when a file does not look like a SessionLogger export."""


@dataclass(frozen=True)
class Schema:
    """Column layout recovered from the CSV header."""

    classes: tuple[str, ...]
    window_frames: int
    features_per_frame: int

    @property
    def newest_frame(self) -> int:
        """Index of the most recent frame in the window (frame 0 is the oldest)."""
        return self.window_frames - 1


@dataclass
class Row:
    """One prediction tick."""

    timestamp: datetime | None
    trial_id: int | None
    prompted: str
    prediction_index: int | None
    predicted: str
    logged_correct: bool
    status: str
    generation: int | None
    scores: list[float] | None
    frames: list[list[float | None]] | None

    # Optional, and last so that positional construction of the columns every recording
    # has keeps working. Both arrive with `session-local-v1`; earlier files do not have
    # them, and the scored game leaves them blank because it has neither block.
    block_role: str | None = None    # "calibration" | "evaluation"
    repeat_index: int | None = None  # 1-based presentation of this state in its block

    # Written since the four conditions (SparseVR/Jev x calibration off/on); older files have neither,
    # and they read as None. `calibrated` is the study's calibrated decision for the tick, blank
    # elsewhere; `error_code` names the library's PredictionErrorCode when the tick failed.
    calibrated: str | None = None
    error_code: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    @property
    def correct(self) -> bool:
        """Recomputed rather than trusted, so a stale `correct` column cannot skew results."""
        return bool(self.prompted) and self.prompted == self.predicted


@dataclass
class Session:
    path: Path
    session_id: str
    schema: Schema
    rows: list[Row]
    metadata: dict | None = None
    warnings: list[str] = field(default_factory=list)

    # Half-open [start, end) row ranges, one per source file. A session read from a
    # single CSV is one span; a pooled session is one span per file, which keeps
    # timing and frame-drop measurements from running across a file boundary where
    # the clock jumps and `generation` restarts.
    spans: list[tuple[int, int]] = field(default_factory=list)

    @property
    def ok_rows(self) -> list[Row]:
        return [r for r in self.rows if r.ok]

    @property
    def row_spans(self) -> list[tuple[int, int]]:
        return self.spans or [(0, len(self.rows))]

    def without_calibration_block(self) -> "Session":
        """The same session minus the study's calibration block, or `self` if it has none.

        The calibration is fitted on that block, so scoring it scores the model on its own
        training data. Accuracy, F1 and the confusion matrix are therefore computed on the rest;
        timing, tracking and action-unit figures still describe the whole recording.
        """
        kept = [r for r in self.rows if r.block_role != ROLE_CALIBRATION]
        if len(kept) == len(self.rows):
            return self
        return Session(path=self.path, session_id=self.session_id, schema=self.schema, rows=kept,
                       metadata=self.metadata, warnings=list(self.warnings))


# --------------------------------------------------------------------------- loading


def find_repo_root(start: Path | None = None) -> Path | None:
    """Walk up from `start` (default: this file) looking for the Unity project root.

    Used to locate session folders relative to the checkout rather than to whoever's
    home directory the tool happens to be running under.
    """
    candidates = [start] if start else []
    candidates += [Path(__file__).resolve().parent, Path.cwd().resolve()]

    for candidate in candidates:
        if candidate is None:
            continue
        for directory in [candidate.resolve(), *candidate.resolve().parents]:
            if (directory / "Assets").is_dir() and (directory / "ProjectSettings").is_dir():
                return directory
    return None


def project_identity(repo_root: Path | None) -> tuple[str | None, str | None]:
    """Read companyName / productName out of ProjectSettings.asset.

    Unity's persistentDataPath is built from those two names, so they are what turn a
    generic AppData path into the one this project actually writes to. The file is YAML
    but these two fields are plain scalars, which is not worth a parser.
    """
    if repo_root is None:
        return None, None
    settings = repo_root / "ProjectSettings" / "ProjectSettings.asset"
    if not settings.is_file():
        return None, None

    found: dict[str, str] = {}
    try:
        text = settings.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None
    for key in ("companyName", "productName"):
        match = re.search(rf"^\s*{key}:\s*(.+?)\s*$", text, re.MULTILINE)
        if match:
            found[key] = match.group(1).strip().strip("'\"")
    return found.get("companyName"), found.get("productName")


def android_package(repo_root: Path | None) -> str | None:
    """The Android application identifier, e.g. com.DefaultCompany.EmotionGatheringTool.

    It names the folder the game writes to on the headset, so pulling recordings needs
    it. `applicationIdentifier` is a map keyed by platform, and `Android:` appears under
    several other keys further down the file, so the value is taken from inside that
    block rather than from the first match in the file.
    """
    if repo_root is None:
        return None
    settings = repo_root / "ProjectSettings" / "ProjectSettings.asset"
    if not settings.is_file():
        return None

    try:
        lines = settings.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None

    inside = False
    for line in lines:
        if re.match(r"^\s*applicationIdentifier:\s*$", line):
            inside = True
            continue
        if inside:
            match = re.match(r"^\s+Android:\s*(\S+)\s*$", line)
            if match:
                return match.group(1)
            if re.match(r"^\s{0,2}\S", line):  # dedented out of the block
                break
    return None


def unity_data_dirs(repo_root: Path | None) -> list[Path]:
    """Where Unity's `Application.persistentDataPath` points on this machine.

    Only the Editor/standalone location: a session recorded on the headset lives inside
    the Quest's own storage and has to be pulled over with adb first.
    """
    company, product = project_identity(repo_root)
    if not company or not product:
        return []

    home = Path.home()
    if sys.platform == "win32":
        roots = [home / "AppData" / "LocalLow"]
    elif sys.platform == "darwin":
        roots = [home / "Library" / "Application Support"]
    else:
        roots = [home / ".config" / "unity3d", home / ".local" / "share" / "unity3d"]

    return [root / company / product for root in roots]


# Directories that never contain session logs but are expensive to walk. Unity's
# Library alone can hold six figures of files.
_SKIP_DIRS = {
    ".git", ".idea", ".utmp", ".venv", ".vs", "__pycache__", "assets", "build",
    "builds", "library", "node_modules", "obj", "packages", "projectsettings",
    "temp", "usersettings", "venv",
}


def scan_for_sessions(root: Path, max_depth: int = 5) -> list[Path]:
    """Find every predictions.csv under `root`, pruning the expensive subtrees."""
    if not root.is_dir():
        return []

    found: list[Path] = []
    root = root.resolve()
    for current, subdirs, files in os.walk(root):
        here = Path(current)
        depth = len(here.relative_to(root).parts)
        if depth >= max_depth:
            subdirs[:] = []
        else:
            subdirs[:] = sorted(d for d in subdirs if d.lower() not in _SKIP_DIRS)
        if "predictions.csv" in files:
            found.append(here / "predictions.csv")
    return sorted(found)


def search_roots(repo_root: Path | None = None) -> list[tuple[str, Path]]:
    """The places checked when no path is given, in priority order.

    Every entry is a (label, directory) pair; directories that do not exist on this
    machine are still returned, so `--where` can show what was looked at and missed.
    """
    repo_root = repo_root or find_repo_root()
    roots: list[tuple[str, Path]] = []

    override = os.environ.get("TRUEEMOTION_SESSIONS", "").strip()
    if override:
        for part in override.split(os.pathsep):
            if part.strip():
                roots.append(("TRUEEMOTION_SESSIONS", Path(part.strip()).expanduser()))

    if repo_root:
        # Anywhere in the checkout: Logs/ is the conventional spot for pulled sessions
        # and is gitignored, but a folder dropped elsewhere is found just as well.
        roots.append(("project folder", repo_root))

    for path in unity_data_dirs(repo_root):
        roots.append(("Unity persistentDataPath", path))

    home = Path.home()
    for name in ("Downloads", "Desktop"):
        roots.append((f"~/{name}", home / name))

    # Keep the first mention of each directory, so a duplicate does not get scanned twice.
    unique: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for label, path in roots:
        try:
            key = path.expanduser().resolve()
        except OSError:
            continue
        if key not in seen:
            seen.add(key)
            unique.append((label, path.expanduser()))
    return unique


def autodiscover() -> tuple[list[Path], list[tuple[str, Path, int]]]:
    """Look for session files without being told where they are.

    Returns the files found plus a per-location tally, so the caller can report where
    it looked when the answer is "nowhere". Scanning stops at the first location that
    yields anything: finding sessions in the checkout should not also drag in every
    unrelated export sitting in Downloads.
    """
    files: list[Path] = []
    searched: list[tuple[str, Path, int]] = []

    for label, root in search_roots():
        # Downloads and Desktop are other people's folders; stay shallow there.
        depth = 3 if label.startswith("~/") else 5
        hits = scan_for_sessions(root, max_depth=depth) if root.is_dir() else []
        searched.append((label, root, len(hits)))
        if hits and not files:
            files = hits

    return files, searched


def discover(paths: Sequence[str | Path]) -> list[Path]:
    """Expand CLI arguments into a list of CSV files.

    A file is used as-is. A directory is searched recursively for `predictions.csv`
    (the name the logger writes); if none are found, every `*.csv` under it is used
    instead, so a folder of renamed exports still works.
    """
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            hits = sorted(path.rglob("predictions.csv")) or sorted(path.rglob("*.csv"))
            if not hits:
                raise FileNotFoundError(f"no CSV files under {path}")
            found.extend(hits)
        else:
            raise FileNotFoundError(f"no such file or directory: {path}")

    # A session directory reachable by two arguments should still be analysed once.
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def parse_schema(header: Sequence[str]) -> Schema:
    missing = [c for c in _REQUIRED if c not in header]
    if missing:
        raise SchemaError(f"missing expected column(s): {', '.join(missing)}")

    classes = tuple(c[len("score_"):] for c in header if c.startswith("score_"))
    if not classes:
        raise SchemaError("no score_<Class> columns found")

    frames = {int(m.group(1)) for c in header if (m := _VALID_RE.match(c))}
    aus: dict[int, set[int]] = {}
    for column in header:
        m = _AU_RE.match(column)
        if m:
            aus.setdefault(int(m.group(1)), set()).add(int(m.group(2)))

    if not frames or not aus:
        raise SchemaError("no frame_<f>_valid / frame_<f>_au_<a> columns found")

    window_frames = max(frames) + 1
    features = max(len(v) for v in aus.values())
    return Schema(classes=classes, window_frames=window_frames, features_per_frame=features)


def _to_float(text: str | None) -> float | None:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _to_int(text: str | None) -> int | None:
    value = _to_float(text)
    return int(value) if value is not None else None


def _to_time(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        # The logger writes "...Z"; fromisoformat only learned to accept that in 3.11.
        return datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def load_session(path: str | Path) -> Session:
    """Read one predictions.csv (or the session directory containing it)."""
    path = Path(path)
    if path.is_dir():
        path = path / "predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"not a file: {path}")

    warnings: list[str] = []

    # utf-8-sig: the logger deliberately writes no BOM, but files that have been
    # through Excel usually come back with one, and it would corrupt the first header.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SchemaError(f"{path} is empty")
        schema = parse_schema(reader.fieldnames)

        au_columns = [
            [f"frame_{f}_au_{a}" for a in range(schema.features_per_frame)]
            for f in range(schema.window_frames)
        ]
        score_columns = [f"score_{c}" for c in schema.classes]

        rows: list[Row] = []
        session_ids: set[str] = set()
        mismatched = 0

        for raw in reader:
            status = (raw.get("status") or "").strip()
            prompted = (raw.get("prompted_emotion") or "").strip()
            predicted = (raw.get("predicted_emotion") or "").strip()
            logged_correct = (raw.get("correct") or "").strip() == "1"

            scores = None
            frames = None
            if status == STATUS_OK:
                values = [_to_float(raw.get(c)) for c in score_columns]
                # A partially blank score vector cannot be ranked, so it is dropped
                # whole rather than silently compared against missing classes.
                scores = values if all(v is not None for v in values) else None
                frames = [[_to_float(raw.get(c)) for c in cols] for cols in au_columns]

            # Both columns arrive with `session-local-v1`. Older recordings do not have
            # them, and the scored game leaves them empty because it has no calibration
            # and no evaluation block, so an absent or blank value reads as None rather
            # than as a role. Lowercased on the way in: the writer emits the canonical
            # spelling, and normalising here means a hand-edited file still loads.
            block_role = (raw.get("block_role") or "").strip().lower() or None

            row = Row(
                timestamp=_to_time(raw.get("timestamp_utc")),
                trial_id=_to_int(raw.get("trial_id")),
                block_role=block_role,
                repeat_index=_to_int(raw.get("repeat_index")),
                prompted=prompted,
                prediction_index=_to_int(raw.get("prediction_index")),
                predicted=predicted,
                logged_correct=logged_correct,
                status=status,
                generation=_to_int(raw.get("generation")),
                scores=scores,
                frames=frames,
                calibrated=(raw.get("calibrated_predicted_emotion") or "").strip() or None,
                error_code=(raw.get("error_code") or "").strip() or None,
            )
            if row.ok and row.logged_correct != row.correct:
                mismatched += 1
            session_ids.add((raw.get("session_id") or "").strip())
            rows.append(row)

    if mismatched:
        warnings.append(
            f"{mismatched} row(s) where the logged `correct` flag disagrees with "
            "predicted_emotion == prompted_emotion; the recomputed value is used"
        )
    session_ids.discard("")
    if len(session_ids) > 1:
        warnings.append(
            f"file mixes {len(session_ids)} session ids; metrics are pooled across them"
        )

    unknown = {r.status for r in rows} - {STATUS_OK, STATUS_NO_TRACKING, STATUS_BACKEND_ERROR, ""}
    if unknown:
        warnings.append(f"unrecognised status value(s): {', '.join(sorted(unknown))}")

    return Session(
        path=path,
        session_id=next(iter(sorted(session_ids)), path.parent.name),
        schema=schema,
        rows=rows,
        metadata=_load_metadata(path.parent / "session_metadata.json"),
        warnings=warnings,
    )


def _load_metadata(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- metrics


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def timing(session: Session) -> dict:
    """Wall-clock span and tick rate, measured within each source file."""
    all_stamps = [r.timestamp for r in session.rows if r.timestamp is not None]
    result: dict = {
        "start_utc": all_stamps[0].isoformat() if all_stamps else None,
        "end_utc": all_stamps[-1].isoformat() if all_stamps else None,
        "duration_s": None,
        "median_tick_s": None,
        "max_tick_s": None,
    }

    duration = 0.0
    deltas: list[float] = []
    for start, end in session.row_spans:
        stamps = [r.timestamp for r in session.rows[start:end] if r.timestamp is not None]
        if len(stamps) < 2:
            continue
        duration += (stamps[-1] - stamps[0]).total_seconds()
        deltas += [
            delta
            for a, b in zip(stamps, stamps[1:])
            if (delta := (b - a).total_seconds()) >= 0
        ]

    if deltas:
        result["duration_s"] = duration
        result["median_tick_s"] = statistics.median(deltas)
        result["max_tick_s"] = max(deltas)
    return result


def generation_health(session: Session) -> dict:
    """Frame-drop diagnostics.

    `generation` counts valid frames pushed into the ring buffer, so the step between
    consecutive ticks is how many frames the tracker actually delivered. A step that
    is short compared with the session's own median means frames were dropped; a step
    of zero means the same window was predicted twice.
    """
    counted = 0
    steps: list[int] = []
    for start, end in session.row_spans:
        gens = [
            r.generation for r in session.rows[start:end] if r.ok and r.generation is not None
        ]
        counted += len(gens)
        steps += [b - a for a, b in zip(gens, gens[1:]) if b >= a]

    result: dict = {
        "ticks_with_generation": counted,
        "median_step": None,
        "min_step": None,
        "max_step": None,
        "stalled_ticks": 0,
        "short_steps": 0,
        "estimated_tracker_fps": None,
    }
    if not steps:
        return result

    median = statistics.median(steps)
    result["median_step"] = median
    result["min_step"] = min(steps)
    result["max_step"] = max(steps)
    result["stalled_ticks"] = sum(1 for s in steps if s == 0)
    result["short_steps"] = sum(1 for s in steps if 0 < s < median * 0.75)

    tick = timing(session).get("median_tick_s")
    if tick:
        result["estimated_tracker_fps"] = median / tick
    return result


def tracking(session: Session) -> dict:
    total = len(session.rows)
    ok = sum(1 for r in session.rows if r.ok)
    return {
        "ticks": total,
        "ok": ok,
        "no_tracking": sum(1 for r in session.rows if r.status == STATUS_NO_TRACKING),
        "backend_error": sum(1 for r in session.rows if r.status == STATUS_BACKEND_ERROR),
        "tracking_rate": _rate(ok, total),
    }


def confusion_counts(truth: Sequence[str], predicted: Sequence[str],
                     classes: Sequence[str]) -> dict[str, dict[str, int]]:
    """truth label -> predicted label -> count.

    Kept separate from `confusion` below so a caller with its own predictions -- a
    calibrated adapter, say -- scores them against the same definition the session-level
    report uses, rather than against a second implementation that can drift.

    Labels outside `classes` are ignored rather than added as rows, so a blank prediction
    from an untracked tick cannot invent a class.
    """
    matrix = {c: {p: 0 for p in classes} for c in classes}
    for t, p in zip(truth, predicted):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1
    return matrix


def macro_f1(truth: Sequence[str], predicted: Sequence[str],
             classes: Sequence[str]) -> dict:
    """Per-class precision/recall/F1 and their unweighted mean.

    A class with no true instance and no prediction is left out of the mean rather than
    scored 0. Scoring it would punish a participant for a class their data does not
    contain, which is precisely the zero-filling the estimand rules forbid.

    That exclusion is also a trap for any caller reporting a headline number: if a class is
    missing because the data is *incomplete* rather than because it does not apply, dropping
    it raises the mean. `classes_scored` is returned so a caller can check it against the
    class set it expected and refuse the figure when they disagree.
    """
    per_class = {}
    f1s = []
    for c in classes:
        tp_ = sum(1 for t, p in zip(truth, predicted) if t == c and p == c)
        fp = sum(1 for t, p in zip(truth, predicted) if t != c and p == c)
        fn = sum(1 for t, p in zip(truth, predicted) if t == c and p != c)
        support = tp_ + fn

        if support == 0 and tp_ + fp == 0:
            per_class[c] = {'precision': None, 'recall': None, 'f1': None, 'support': 0}
            continue

        precision = tp_ / (tp_ + fp) if (tp_ + fp) else 0.0
        recall = tp_ / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[c] = {'precision': precision, 'recall': recall, 'f1': f1,
                        'support': support}
        f1s.append(f1)

    return {'per_class': per_class,
            'macro_f1': statistics.fmean(f1s) if f1s else None,
            'classes_scored': len(f1s)}


def confusion(session: Session) -> dict[str, dict[str, int]]:
    """prompted emotion -> predicted emotion -> count, over tracked ticks."""
    rows = session.ok_rows
    return confusion_counts([r.prompted for r in rows],
                            [r.predicted for r in rows],
                            list(session.schema.classes))


def per_prompted(session: Session) -> dict[str, dict]:
    """Tick accuracy and confidence broken down by the emotion the player was asked for."""
    out: dict[str, dict] = {}
    for emotion in session.schema.classes:
        rows = [r for r in session.ok_rows if r.prompted == emotion]
        if not rows:
            continue
        correct = sum(1 for r in rows if r.correct)
        prompted_scores = [_score_of(session, r, emotion) for r in rows]
        prompted_scores = [s for s in prompted_scores if s is not None]

        wrong = [r.predicted for r in rows if not r.correct and r.predicted]
        top_confusion = None
        if wrong:
            top_confusion = max(set(wrong), key=wrong.count)

        out[emotion] = {
            "ticks": len(rows),
            "correct": correct,
            "accuracy": _rate(correct, len(rows)),
            "mean_prompted_score": _mean(prompted_scores),
            "top_confusion": top_confusion,
            "top_confusion_share": _rate(wrong.count(top_confusion), len(rows))
            if top_confusion
            else None,
        }
    return out


def per_predicted(session: Session) -> dict[str, dict]:
    """How often each class is *emitted*, and how often that was right.

    The mirror image of `per_prompted`, and it answers a different question. A class
    can score badly two ways: the model never emits it (low emission, high precision),
    or it emits it everywhere (high emission, low precision). Those are different
    faults with different fixes, and accuracy alone cannot tell them apart.
    """
    out: dict[str, dict] = {}
    for emotion in session.schema.classes:
        emitted = [r for r in session.ok_rows if r.predicted == emotion]
        correct = sum(1 for r in emitted if r.correct)
        prompted = sum(1 for r in session.ok_rows if r.prompted == emotion)
        out[emotion] = {
            "emitted": len(emitted),
            "correct": correct,
            "precision": _rate(correct, len(emitted)),
            "prompted": prompted,
            # >1 means the model reaches for this class more often than it is asked for.
            "emission_ratio": _rate(len(emitted), prompted),
        }
    return out


def _score_of(session: Session, row: Row, emotion: str) -> float | None:
    if row.scores is None or emotion not in session.schema.classes:
        return None
    return row.scores[session.schema.classes.index(emotion)]


def confidence(session: Session) -> dict:
    tops, correct_tops, wrong_tops, prompted = [], [], [], []
    for row in session.ok_rows:
        if not row.scores:
            continue
        top = max(row.scores)
        tops.append(top)
        (correct_tops if row.correct else wrong_tops).append(top)
        value = _score_of(session, row, row.prompted)
        if value is not None:
            prompted.append(value)
    return {
        "mean_top_score": _mean(tops),
        "mean_top_score_correct": _mean(correct_tops),
        "mean_top_score_incorrect": _mean(wrong_tops),
        "mean_prompted_score": _mean(prompted),
    }


def class_means(session: Session) -> dict[str, float | None]:
    """Average score each class receives, regardless of prompt.

    A class whose mean sits far above the others is the model's resting bias, which
    is usually Neutral and is worth knowing before reading the accuracy numbers.
    """
    out: dict[str, float | None] = {}
    for i, emotion in enumerate(session.schema.classes):
        out[emotion] = _mean(r.scores[i] for r in session.ok_rows if r.scores)
    return out


def trials(session: Session) -> list[dict]:
    """One summary per prompt, in the order the prompts were shown."""
    grouped: dict[int, list[Row]] = {}
    for row in session.rows:
        if row.trial_id is not None:
            grouped.setdefault(row.trial_id, []).append(row)

    summaries = []
    for trial_id in sorted(grouped):
        rows = sorted(
            grouped[trial_id],
            key=lambda r: (r.prediction_index if r.prediction_index is not None else 0),
        )
        ok = [r for r in rows if r.ok]
        prompted = next((r.prompted for r in rows if r.prompted), "")
        correct = [r for r in ok if r.correct]
        predictions = [r.predicted for r in ok if r.predicted]
        scores = [s for s in (_score_of(session, r, prompted) for r in ok) if s is not None]

        stamps = [r.timestamp for r in rows if r.timestamp]
        summaries.append(
            {
                "trial_id": trial_id,
                "prompted": prompted,
                "ticks": len(rows),
                "ok_ticks": len(ok),
                "correct_ticks": len(correct),
                "accuracy": _rate(len(correct), len(ok)),
                "first_correct_index": correct[0].prediction_index if correct else None,
                "final_prediction": predictions[-1] if predictions else None,
                "majority_prediction": max(set(predictions), key=predictions.count)
                if predictions
                else None,
                "mean_prompted_score": _mean(scores),
                "peak_prompted_score": max(scores) if scores else None,
                "duration_s": (stamps[-1] - stamps[0]).total_seconds()
                if len(stamps) >= 2
                else None,
            }
        )
    return summaries


def trial_accuracy(summaries: Sequence[dict]) -> dict:
    """Roll tick-level results up to one hit-or-miss per prompt.

    Three definitions, because they answer different questions: `majority` is how the
    trial mostly looked, `final` is where the player ended up, and `any` is whether
    the expression was ever reached at all.
    """
    scored = [t for t in summaries if t["ok_ticks"]]
    if not scored:
        return {"trials": 0, "majority": None, "final": None, "any": None}
    majority = sum(1 for t in scored if t["majority_prediction"] == t["prompted"])
    final = sum(1 for t in scored if t["final_prediction"] == t["prompted"])
    any_hit = sum(1 for t in scored if t["correct_ticks"] > 0)
    n = len(scored)
    return {
        "trials": n,
        "majority": _rate(majority, n),
        "final": _rate(final, n),
        "any": _rate(any_hit, n),
    }


def tick_curve(session: Session) -> list[dict]:
    """Accuracy and prompted-class score as a function of position within the trial.

    Players need a moment to build an expression after the prompt appears, so this
    usually rises. A flat curve suggests the prompt is not landing at all.
    """
    buckets: dict[int, list[Row]] = {}
    for row in session.ok_rows:
        if row.prediction_index is not None:
            buckets.setdefault(row.prediction_index, []).append(row)

    curve = []
    for index in sorted(buckets):
        rows = buckets[index]
        scores = [s for s in (_score_of(session, r, r.prompted) for r in rows) if s is not None]
        correct = sum(1 for r in rows if r.correct)
        curve.append(
            {
                "prediction_index": index,
                "ticks": len(rows),
                "accuracy": _rate(correct, len(rows)),
                "mean_prompted_score": _mean(scores),
            }
        )
    return curve


def action_units(session: Session) -> dict:
    """Which action-unit channels actually carry signal.

    Only the newest frame of each window is used. Consecutive windows overlap, so
    pooling all five frames would count most samples several times over and make the
    activity fractions meaningless.
    """
    frame = session.schema.newest_frame
    columns: list[list[float]] = [[] for _ in range(session.schema.features_per_frame)]

    for row in session.ok_rows:
        if not row.frames or frame >= len(row.frames):
            continue
        values = row.frames[frame]
        for index, value in enumerate(values):
            if value is not None and index < len(columns):
                columns[index].append(value)

    stats = []
    for index, values in enumerate(columns):
        if not values:
            stats.append(
                {"au": index, "samples": 0, "active_fraction": None, "mean": None, "max": None}
            )
            continue
        active = sum(1 for v in values if abs(v) > AU_EPSILON)
        stats.append(
            {
                "au": index,
                "samples": len(values),
                "active_fraction": active / len(values),
                "mean": statistics.fmean(values),
                "max": max(values),
            }
        )

    dead = [s["au"] for s in stats if s["samples"] and not s["active_fraction"]]
    ranked = sorted(
        (s for s in stats if s["samples"] and s["active_fraction"]),
        key=lambda s: (s["mean"] or 0.0),
        reverse=True,
    )
    return {
        "frame_used": frame,
        "features": session.schema.features_per_frame,
        "dead": dead,
        "dead_count": len(dead),
        "most_active": ranked,
        "all": stats,
    }


def _scores_of_rows(rows: Sequence[Row], classes: Sequence[str]) -> dict:
    """Tick accuracy and macro-F1 of the model's own answers over `rows` (tracked ticks only)."""
    ok = [r for r in rows if r.ok]
    f1 = macro_f1([r.prompted for r in ok], [r.predicted for r in ok], list(classes))
    correct = sum(1 for r in ok if r.correct)
    return {
        "ticks_scored": len(ok),
        "ticks_correct": correct,
        "tick_accuracy": _rate(correct, len(ok)),
        "macro_f1": f1["macro_f1"],
        "classes_scored": f1["classes_scored"],
    }


def calibration_comparison(session: Session) -> dict | None:
    """The calibrated decision against the model's own answer, on the ticks that have a decision.

    Both are scored on exactly the same ticks, so the difference is the calibration alone. None when
    the file has no calibrated decisions (conditions without calibration, the game, older files).
    """
    classes = list(session.schema.classes)
    # Ticks 1 to 9, as the trial votes use. The study never decides on tick 0; the game does, so the
    # filter is what keeps the two programs comparable.
    both = [r for r in session.ok_rows
            if r.calibrated and (r.prediction_index is None or r.prediction_index >= 1)]
    if not both:
        return None

    truth = [r.prompted for r in both]
    result = {"ticks": len(both)}
    for name, predicted in (("model_same_ticks", [r.predicted for r in both]),
                            ("calibrated", [r.calibrated for r in both])):
        correct = sum(1 for t, p in zip(truth, predicted) if t == p)
        f1 = macro_f1(truth, predicted, classes)
        result[name] = {"accuracy": _rate(correct, len(both)), "macro_f1": f1["macro_f1"],
                        "classes_scored": f1["classes_scored"]}
    return result


def analyse(session: Session) -> dict:
    """Everything above, in one JSON-serialisable dictionary.

    Accuracy-type figures (accuracy, F1, confusion, per-trial and per-tick results) are computed on
    the test ticks: the study's calibration block is left out, because the calibration is fitted on it.
    Sessions without a calibration block (the game, uncalibrated study sessions, older recordings) are
    scored whole, exactly as before. Tracking, timing and action-unit figures cover the whole recording.
    """
    scored = session.without_calibration_block()
    summaries = trials(scored)
    ok = scored.ok_rows
    correct = sum(1 for r in ok if r.correct)
    classes = scored.schema.classes
    f1 = macro_f1([r.prompted for r in ok], [r.predicted for r in ok], list(classes))

    left_out = [r for r in session.rows if r.block_role == ROLE_CALIBRATION]
    ticks_1_to_9 = _scores_of_rows(
        [r for r in scored.rows if r.prediction_index is not None and 1 <= r.prediction_index <= 9], classes)

    return {
        "file": str(session.path),
        "session_id": session.session_id,
        "schema": {
            "classes": list(classes),
            "window_frames": session.schema.window_frames,
            "features_per_frame": session.schema.features_per_frame,
        },
        "models": (session.metadata or {}).get("models"),
        "scope": {
            "calibration_block_ticks_excluded": len(left_out),
            "calibration_block_trials_excluded": len({r.trial_id for r in left_out if r.trial_id is not None}),
        },
        "tracking": tracking(session),
        "timing": timing(session),
        "generation": generation_health(session),
        "accuracy": {
            "ticks_scored": len(ok),
            "ticks_correct": correct,
            "tick_accuracy": _rate(correct, len(ok)),
            "chance_level": 1 / len(classes) if classes else None,
            "macro_f1": f1["macro_f1"],
            "classes_scored": f1["classes_scored"],
            "ticks_1_to_9": ticks_1_to_9,
            "trial": trial_accuracy(summaries),
        },
        "calibration": calibration_comparison(scored),
        "confidence": confidence(scored),
        "class_means": class_means(scored),
        "per_prompted": per_prompted(scored),
        "per_predicted": per_predicted(scored),
        "confusion": confusion(scored),
        "tick_curve": tick_curve(scored),
        "action_units": action_units(session),
        "trials": summaries,
        "warnings": list(session.warnings),
    }


def analyse_path(path: str | Path) -> dict:
    return analyse(load_session(path))


def pool(sessions: Sequence[Session]) -> Session:
    """Merge sessions into one pseudo-session for cross-session totals.

    Trial ids restart at zero in every recording, so they are renumbered here; without
    that, trial 0 of six sessions would collapse into a single six-prompt "trial".
    """
    if not sessions:
        raise ValueError("no sessions to pool")

    base = sessions[0].schema
    for other in sessions[1:]:
        if other.schema != base:
            raise ValueError(
                f"cannot pool sessions with different schemas: {sessions[0].path} vs {other.path}"
            )

    rows: list[Row] = []
    spans: list[tuple[int, int]] = []
    offset = 0
    for session in sessions:
        span_start = len(rows)
        highest = -1
        for row in session.rows:
            trial_id = row.trial_id
            if trial_id is not None:
                highest = max(highest, trial_id)
                trial_id += offset
            rows.append(
                Row(
                    timestamp=row.timestamp,
                    trial_id=trial_id,
                    prompted=row.prompted,
                    prediction_index=row.prediction_index,
                    predicted=row.predicted,
                    logged_correct=row.logged_correct,
                    status=row.status,
                    generation=row.generation,
                    scores=row.scores,
                    frames=row.frames,
                    block_role=row.block_role,
                    repeat_index=row.repeat_index,
                    calibrated=row.calibrated,
                    error_code=row.error_code,
                )
            )
        spans.append((span_start, len(rows)))
        offset += highest + 1

    return Session(
        path=Path("<pooled>"),
        session_id=f"{len(sessions)} sessions pooled",
        schema=base,
        rows=rows,
        metadata=None,
        warnings=[
            f"pooled across {len(sessions)} files; duration is their sum, and timing "
            "and frame-drop measurements do not cross a file boundary"
        ],
        spans=spans,
    )
