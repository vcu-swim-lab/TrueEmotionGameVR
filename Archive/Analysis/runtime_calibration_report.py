#!/usr/bin/env python3
"""Report on a session that was calibrated on the headset, and verify it.

    python Analysis/runtime_calibration_report.py
    python Analysis/runtime_calibration_report.py --session Logs/.../session_2026...
    python Analysis/runtime_calibration_report.py --verify        # recompute and compare
    python Analysis/runtime_calibration_report.py --json out.json

What this reports
-----------------
The predictions the *headset* made. `calibrated_predicted_emotion` in the CSV is what the
runtime decided at the time, and every number below is computed from it. This is deliberately
not a recomputation dressed up as a recording: a recomputation would hide precisely the bug
worth finding, which is the runtime and the analysis disagreeing.

`--verify` is where the recomputation happens, and it is a separate claim. It refits the
centroids from the recorded calibration block, reclassifies the evaluation block, and compares
against what the headset wrote. A single differing tick or trial is a parity failure and the
tool exits non-zero.

The recomputation uses the *logged* score columns rather than replaying the action-unit windows
through ONNX. That keeps the check on the calibration arithmetic, which is what was ported to
C#. Movement mode also reads the logged action-unit columns, because its second term is built
from them; they are written with G9 and so are the exact values the runtime used. Whether the headset's Sentis inference agrees with onnxruntime is a different question,
already answered by the `replay_fidelity` block in `calibration_study.py`.

Scope
-----
Evaluation block only. All fourteen D trials, none of the seven K trials: scoring a trial the
centroids were fitted on is the within-trial leak this whole protocol exists to avoid.

Sessions recorded under different calibration modes are never pooled. A personal-only session
and a hybrid session answer different questions, and averaging them answers neither.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibration_study as cs  # noqa: E402
import session_analysis as sa  # noqa: E402

REPORT_SCHEMA_VERSION = "runtime-calibration-report-v1"

MODE_RAW = "raw"
MODE_PERSONAL = "personal"
MODE_POPULATION = "population"
MODE_HYBRID = "hybrid"
MODE_MOVEMENT = "movement"
CALIBRATED_MODES = (MODE_PERSONAL, MODE_POPULATION, MODE_HYBRID, MODE_MOVEMENT)

SCORED_TICKS = frozenset(range(1, 10))
UNITY_ARTIFACT = Path("Assets/StreamingAssets/emojihero_population_centroids_unity.json")

# How close a recomputed similarity has to be before it counts as the same number. The runtime
# computes in double and writes G17, and this recomputation runs through numpy on float32
# score columns, so the two agree to about an ulp of a float32 rather than exactly. Decisions
# are compared exactly; only magnitudes get a tolerance.
SIMILARITY_TOLERANCE = 1e-6


class ParityError(SystemExit):
    """The headset and the recomputation disagree about a tick or a trial."""


# ------------------------------------------------------------------------ loading


def read_session(directory: Path) -> dict:
    """One session folder: the prediction rows, the trial rows, and the metadata."""
    predictions = directory / "predictions.csv"
    if not predictions.is_file():
        raise SystemExit(f"no predictions.csv in\n  {directory}")

    with predictions.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{predictions} is empty")
        columns = list(reader.fieldnames)
        rows = list(reader)

    if "calibration_mode" not in columns:
        raise SystemExit(
            f"{predictions}\nhas no calibration columns, so it was recorded before runtime\n"
            "calibration existed. Use `python Analysis/calibration_study.py` for that file."
        )

    metadata = {}
    metadata_path = directory / "session_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    trials = []
    trial_path = directory / "trials.csv"
    if trial_path.is_file():
        with trial_path.open("r", encoding="utf-8-sig", newline="") as handle:
            trials = list(csv.DictReader(handle))

    classes = [c[len("score_"):] for c in columns if c.startswith("score_")]

    return {
        "directory": directory,
        "rows": rows,
        "trials": trials,
        "metadata": metadata,
        "classes": classes,
    }


def blank(value: str | None) -> bool:
    return value is None or value.strip() == ""


def scored_rows(session: dict, role: str) -> list[dict]:
    """Tracked rows of one block, restricted to the scored tick window."""
    keep = []
    for row in session["rows"]:
        if (row.get("block_role") or "").strip().lower() != role:
            continue
        if (row.get("status") or "").strip() != "ok":
            continue
        if blank(row.get("prediction_index")):
            continue
        if int(row["prediction_index"]) not in SCORED_TICKS:
            continue
        keep.append(row)

    return keep


def session_mode(session: dict) -> str:
    modes = {(r.get("calibration_mode") or "").strip()
             for r in session["rows"] if not blank(r.get("calibration_mode"))}

    if len(modes) != 1:
        raise SystemExit(
            f"{session['directory']}\nrecords {len(modes)} calibration modes "
            f"({', '.join(sorted(modes)) or 'none'}). The mode is fixed at session start, so "
            "this file is either corrupt or hand-edited."
        )

    mode = modes.pop()
    if mode not in (MODE_RAW,) + CALIBRATED_MODES:
        raise SystemExit(f"{session['directory']}: unknown calibration mode {mode!r}")

    return mode


def session_alpha(session: dict) -> float | None:
    values = {(r.get("hybrid_alpha") or "").strip()
              for r in session["rows"] if not blank(r.get("hybrid_alpha"))}

    if not values:
        return None
    if len(values) > 1:
        raise SystemExit(f"{session['directory']}: the file records {len(values)} alphas")

    return float(values.pop())


def calibration_valid(session: dict) -> tuple[bool, str]:
    """Whether the headset sealed a usable profile, and why not if it did not."""
    for row in session["rows"]:
        if (row.get("block_role") or "").strip().lower() != sa.ROLE_EVALUATION:
            continue
        flag = (row.get("calibration_valid") or "").strip().lower()
        if flag:
            return flag == "true", (row.get("calibration_error") or "").strip()

    calibration = (session["metadata"] or {}).get("calibration") or {}
    return (str(calibration.get("calibration_valid", "")).lower() == "true",
            str(calibration.get("calibration_error", "")))


# ------------------------------------------------------------------------ metrics


def majority(rows: list[dict], column: str, classes: list[str]) -> dict[int, dict]:
    """Trial-level majority vote over a prediction column, with the frozen tie rule."""
    by_trial: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if blank(row.get(column)):
            continue
        by_trial[int(row["trial_id"])].append(row)

    votes = {}
    for trial, trial_rows in by_trial.items():
        truth = np.array([r["prompted_emotion"] for r in trial_rows])
        predicted = np.array([r[column].strip() for r in trial_rows])
        ticks = np.array([int(r["prediction_index"]) for r in trial_rows])
        trials = np.zeros(len(trial_rows), dtype=int)

        _, y, p = cs.trial_vote(truth, predicted, trials, ticks, classes)

        counts: dict[str, int] = {}
        for label in predicted:
            counts[label] = counts.get(label, 0) + 1
        best = max(counts.values())

        votes[trial] = {
            "prompted": str(y[0]),
            "predicted": str(p[0]),
            "correct": str(p[0]) == str(y[0]),
            "scored_ticks": len(trial_rows),
            "tie": sum(1 for n in counts.values() if n == best) > 1,
        }

    return votes


def tick_metrics(rows: list[dict], column: str, classes: list[str]) -> dict:
    truth = [r["prompted_emotion"] for r in rows if not blank(r.get(column))]
    predicted = [r[column].strip() for r in rows if not blank(r.get(column))]

    if not truth:
        return {"n": 0, "accuracy": None, "macro_f1": None}

    scored = sa.macro_f1(truth, predicted, classes)
    return {
        "n": len(truth),
        "accuracy": float(np.mean([t == p for t, p in zip(truth, predicted)])),
        "macro_f1": scored["macro_f1"],
        "classes_scored": scored["classes_scored"],
    }


def trial_metrics(votes: dict[int, dict], classes: list[str]) -> dict:
    if not votes:
        return {"n": 0, "accuracy": None, "macro_f1": None}

    order = sorted(votes)
    truth = [votes[t]["prompted"] for t in order]
    predicted = [votes[t]["predicted"] for t in order]
    scored = sa.macro_f1(truth, predicted, classes)

    return {
        "n": len(order),
        "accuracy": float(np.mean([t == p for t, p in zip(truth, predicted)])),
        "macro_f1": scored["macro_f1"],
        "classes_scored": scored["classes_scored"],
        "complete": scored["classes_scored"] == len(classes),
        "per_class": scored["per_class"],
        "confusion": sa.confusion_counts(truth, predicted, classes),
        "ties": sum(1 for t in order if votes[t]["tie"]),
    }


def trial_changes(raw: dict[int, dict], calibrated: dict[int, dict]) -> dict:
    """Which trials calibration moved, and in which direction."""
    shared = sorted(set(raw) & set(calibrated))

    changed, fixed, broken, both_wrong = [], [], [], []
    for trial in shared:
        r, c = raw[trial], calibrated[trial]
        if r["predicted"] == c["predicted"]:
            continue

        entry = {
            "trial": trial,
            "prompted": r["prompted"],
            "raw": r["predicted"],
            "calibrated": c["predicted"],
        }
        changed.append(entry)

        if not r["correct"] and c["correct"]:
            fixed.append(entry)
        elif r["correct"] and not c["correct"]:
            broken.append(entry)
        else:
            both_wrong.append(entry)

    return {
        "n_compared": len(shared),
        "changed": changed,
        "fixed_by_calibration": fixed,
        "broken_by_calibration": broken,
        "changed_but_still_wrong": both_wrong,
        # The headline the rest of the list explains. Reported as a count rather than a rate:
        # fourteen trials cannot support a rate, and writing one invites it to be read as one.
        "net": len(fixed) - len(broken),
    }


# ------------------------------------------------------------------------ verify


def load_artifact(root: Path, session: dict) -> tuple[np.ndarray, dict]:
    """The population profile the session says it used, checked against the one on disk."""
    path = root / UNITY_ARTIFACT
    if not path.is_file():
        raise SystemExit(
            f"--verify needs the population artifact at\n  {path}\n"
            "Export it with `python Analysis/export_population_artifact_for_unity.py`."
        )

    artifact = json.loads(path.read_text(encoding="utf-8"))
    recorded = (session["metadata"] or {}).get("calibration") or {}

    for field, key in (("semantic_sha256", "population_artifact_semantic_sha256"),
                       ("conversion_sha256", "population_artifact_conversion_sha256")):
        want = str(recorded.get(key, ""))
        if want and want != artifact.get(field):
            raise SystemExit(
                f"the session was recorded against a different population artifact.\n"
                f"  session:  {want}\n"
                f"  on disk:  {artifact.get(field)}\n"
                "Recomputing against the wrong profile would report a parity failure that is "
                "really a provenance failure. Check out the artifact the session used."
            )

    matrix = np.array([row["values"] for row in artifact["centroids"]], np.float64)
    return matrix, artifact


def windows(rows: list[dict]) -> np.ndarray:
    """The logged action-unit windows of some rows, `[n, frames, features]`, frame 0 oldest."""
    if not rows:
        return np.zeros((0, 0, 0))

    columns = [c for c in rows[0] if c.startswith("frame_") and "_au_" in c]
    if not columns:
        raise SystemExit(
            "movement mode needs the frame_<f>_au_<a> columns to recompute, and this file has none"
        )

    frames = 1 + max(int(c.split("_")[1]) for c in columns)
    units = 1 + max(int(c.split("_")[3]) for c in columns)
    out = np.empty((len(rows), frames, units), np.float64)
    for i, row in enumerate(rows):
        for f in range(frames):
            for a in range(units):
                out[i, f, a] = float(row[f"frame_{f}_au_{a}"])

    return out


def recompute(session: dict, mode: str, alpha: float | None,
              population: np.ndarray | None) -> dict:
    """Refit and reclassify from the CSV, using the same functions the offline study uses."""
    classes = session["classes"]
    score_columns = [f"score_{c}" for c in classes]

    def matrix(rows: list[dict]) -> np.ndarray:
        return np.array([[float(r[c]) for c in score_columns] for r in rows], np.float32)

    enrol = scored_rows(session, sa.ROLE_CALIBRATION)
    test = scored_rows(session, sa.ROLE_EVALUATION)

    if not test:
        raise SystemExit("the recording has no scored evaluation ticks to verify")

    every_class = list(range(len(classes)))
    test_F = cs.logscore(matrix(test), cs.DEFAULT_EPSILON)

    personal = None
    if mode in (MODE_PERSONAL, MODE_HYBRID, MODE_MOVEMENT):
        enrol_y = np.array([classes.index(r["prompted_emotion"]) for r in enrol])
        counts = {classes[c]: int((enrol_y == c).sum()) for c in every_class}
        short = {c: n for c, n in counts.items() if n != cs.TICKS_REQUIRED_PER_CENTROID}
        if short:
            raise SystemExit(
                "the calibration block does not hold "
                f"{cs.TICKS_REQUIRED_PER_CENTROID} scored ticks for every class: "
                + ", ".join(f"{c} {n}" for c, n in sorted(short.items()))
            )

        personal = cs.centroids(cs.logscore(matrix(enrol), cs.DEFAULT_EPSILON),
                                enrol_y, every_class)

    if mode == MODE_MOVEMENT:
        enrol_y = np.array([classes.index(r["prompted_emotion"]) for r in enrol])
        movement = cs.centroids(cs.movement_feature(windows(enrol)), enrol_y, every_class)
        predicted, top, margin = cs.classify_movement(
            personal, test_F, movement, cs.movement_feature(windows(test)), every_class)
    else:
        if mode == MODE_PERSONAL:
            active = personal
        elif mode == MODE_POPULATION:
            active = population.astype(np.float32)
        elif mode == MODE_HYBRID:
            active = alpha * personal + (1.0 - alpha) * population.astype(np.float32)
        else:
            raise SystemExit(f"nothing to recompute for mode {mode!r}")

        predicted, top, margin = cs.classify(active, every_class, test_F, "cosine")

    return {
        "rows": test,
        "predicted": [classes[i] for i in predicted],
        "winner": [float(v) for v in top],
        "runner_up": [float(v - m) for v, m in zip(top, margin)],
        "margin": [float(v) for v in margin],
    }


def compare(session: dict, recomputed: dict, classes: list[str]) -> dict:
    """Hold the headset's decisions against the recomputation, tick for tick."""
    tick_mismatches = []
    magnitude_mismatches = []

    for index, row in enumerate(recomputed["rows"]):
        trial = int(row["trial_id"])
        tick = int(row["prediction_index"])
        recorded = (row.get("calibrated_predicted_emotion") or "").strip()
        expected = recomputed["predicted"][index]

        if recorded != expected:
            tick_mismatches.append({
                "trial": trial, "tick": tick,
                "runtime": recorded, "recomputed": expected,
            })
            continue

        for column, values, label in (
            ("calibrated_winner_similarity", recomputed["winner"], "winner"),
            ("calibrated_runner_up_similarity", recomputed["runner_up"], "runner_up"),
            ("calibrated_margin", recomputed["margin"], "margin"),
        ):
            if blank(row.get(column)):
                magnitude_mismatches.append({
                    "trial": trial, "tick": tick, "field": label,
                    "runtime": None, "recomputed": values[index],
                })
                continue

            difference = abs(float(row[column]) - values[index])
            if difference > SIMILARITY_TOLERANCE:
                magnitude_mismatches.append({
                    "trial": trial, "tick": tick, "field": label,
                    "runtime": float(row[column]), "recomputed": values[index],
                    "difference": difference,
                })

    # Trial level, from the recomputed ticks, against the runtime's own vote.
    by_trial: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for index, row in enumerate(recomputed["rows"]):
        by_trial[int(row["trial_id"])].append(
            (int(row["prediction_index"]), recomputed["predicted"][index]))

    runtime_votes = majority(scored_rows(session, sa.ROLE_EVALUATION),
                             "calibrated_predicted_emotion", classes)

    trial_mismatches = []
    for trial, entries in sorted(by_trial.items()):
        ticks = np.array([t for t, _ in entries])
        predicted = np.array([p for _, p in entries])
        truth = np.array([runtime_votes[trial]["prompted"]] * len(entries))

        _, _, winner = cs.trial_vote(truth, predicted, np.zeros(len(entries), int),
                                     ticks, classes)

        if str(winner[0]) != runtime_votes[trial]["predicted"]:
            trial_mismatches.append({
                "trial": trial,
                "runtime": runtime_votes[trial]["predicted"],
                "recomputed": str(winner[0]),
            })

    return {
        "n_ticks_compared": len(recomputed["rows"]),
        "n_trials_compared": len(by_trial),
        "tick_mismatches": tick_mismatches,
        "trial_mismatches": trial_mismatches,
        "magnitude_mismatches": magnitude_mismatches,
        "tolerance": SIMILARITY_TOLERANCE,
        "parity": not tick_mismatches and not trial_mismatches and not magnitude_mismatches,
    }


# ------------------------------------------------------------------------ report


def analyse(session: dict, root: Path, verify: bool) -> dict:
    classes = session["classes"]
    mode = session_mode(session)
    alpha = session_alpha(session)
    valid, error = calibration_valid(session)

    evaluation = scored_rows(session, sa.ROLE_EVALUATION)
    if not evaluation:
        raise SystemExit(f"{session['directory']}: no scored evaluation ticks")

    raw_votes = majority(evaluation, "raw_predicted_emotion", classes)
    result = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "session": session["directory"].name,
        "protocol_version": (session["metadata"] or {}).get("protocol_version"),
        "rehearsal": str((session["metadata"] or {}).get("fast_rehearsal", "")).lower() == "true",
        "calibration_mode": mode,
        "hybrid_alpha": alpha,
        "calibration_valid": valid,
        "calibration_error": error,
        "source": "the predictions the headset made; not a recomputation",
        "evaluation_only": (
            f"{len(raw_votes)} evaluation trials; the calibration block is excluded because "
            "the centroids were fitted on it"
        ),
        "raw": {
            "tick": tick_metrics(evaluation, "raw_predicted_emotion", classes),
            "trial": trial_metrics(raw_votes, classes),
        },
    }

    if mode == MODE_RAW:
        # No calibrated columns were ever filled, and inventing a comparison here would be
        # inventing a method. The raw block above is the whole report.
        result["calibrated"] = None
        result["comparison"] = None
        result["note"] = (
            "this session ran with both calibration checkboxes off, so there is no calibrated "
            "prediction to report. The raw figures above are the complete result."
        )
        return result

    if not valid:
        result["calibrated"] = None
        result["comparison"] = None
        result["note"] = (
            f"the headset could not seal a {mode} profile, so no evaluation tick was "
            f"calibrated. Reason recorded: {error or '(none given)'}"
        )
        return result

    calibrated_votes = majority(evaluation, "calibrated_predicted_emotion", classes)
    result["calibrated"] = {
        "tick": tick_metrics(evaluation, "calibrated_predicted_emotion", classes),
        "trial": trial_metrics(calibrated_votes, classes),
    }
    result["comparison"] = trial_changes(raw_votes, calibrated_votes)

    deltas = {}
    for level in ("tick", "trial"):
        for metric in ("accuracy", "macro_f1"):
            a = result["calibrated"][level][metric]
            b = result["raw"][level][metric]
            deltas[f"{level}_{metric}"] = (a - b) if a is not None and b is not None else None
    result["calibrated_minus_raw"] = deltas

    if verify:
        population = None
        if mode in (MODE_POPULATION, MODE_HYBRID):
            population, _ = load_artifact(root, session)

        result["verification"] = compare(
            session, recompute(session, mode, alpha, population), classes)

    return result


def percent(value: float | None) -> str:
    return "  n/a " if value is None else f"{value:6.3f}"


def print_report(result: dict) -> None:
    mode = result["calibration_mode"]
    banner = f"  CALIBRATION MODE: {mode.upper()}"
    if result["hybrid_alpha"] is not None:
        alpha = result["hybrid_alpha"]
        banner += (f"   alpha = {alpha:.2f}"
                   f"  ({alpha:.0%} personal, {1 - alpha:.0%} population)")

    print()
    print("=" * max(72, len(banner) + 2))
    print(banner)
    print("=" * max(72, len(banner) + 2))
    print(f"session            {result['session']}")
    print(f"protocol           {result['protocol_version']}")
    if result["rehearsal"]:
        print("REHEARSAL          this recording is marked as a rehearsal and is not data")
    print(f"scope              {result['evaluation_only']}")
    print(f"figures from       {result['source']}")

    if not result["calibration_valid"] and mode != MODE_RAW:
        print(f"calibration        INVALID -- {result['calibration_error']}")

    print()
    print(f"{'':<12}{'ticks':>22}{'trials':>24}")
    print(f"{'':<12}{'n':>6}{'acc':>8}{'macroF1':>8}{'n':>8}{'acc':>8}{'macroF1':>8}")

    def line(label: str, block: dict | None) -> None:
        if block is None:
            print(f"{label:<12}{'--':>6}{'--':>8}{'--':>8}{'--':>8}{'--':>8}{'--':>8}")
            return
        t, r = block["tick"], block["trial"]
        print(f"{label:<12}{t['n']:>6}{percent(t['accuracy']):>8}{percent(t['macro_f1']):>8}"
              f"{r['n']:>8}{percent(r['accuracy']):>8}{percent(r['macro_f1']):>8}")

    line("raw", result["raw"])
    line(mode, result["calibrated"])

    if result.get("note"):
        print()
        print(f"note: {result['note']}")

    if result["calibrated"] is None:
        return

    delta = result["calibrated_minus_raw"]
    print()
    print(f"{mode} minus raw   "
          f"tick acc {delta['tick_accuracy']:+.4f}   "
          f"trial acc {delta['trial_accuracy']:+.4f}   "
          f"trial macro-F1 {delta['trial_macro_f1']:+.4f}")

    trial = result["calibrated"]["trial"]
    if not trial["complete"]:
        print(f"  macro-F1 covers {trial['classes_scored']} of the seven classes; a class with "
              "no scored trial is left out of the mean, which raises it")

    comparison = result["comparison"]
    print()
    print(f"trials changed by calibration: {len(comparison['changed'])} of "
          f"{comparison['n_compared']}    net {comparison['net']:+d}")
    for label, key in (("fixed", "fixed_by_calibration"),
                       ("broken", "broken_by_calibration"),
                       ("still wrong", "changed_but_still_wrong")):
        for entry in comparison[key]:
            print(f"  {label:<12} trial {entry['trial']:>3}  {entry['prompted']:<10}"
                  f"raw {entry['raw']:<10} -> {entry['calibrated']}")

    print()
    print("per class (trial level, calibrated)")
    for name, scores in sorted(trial["per_class"].items()):
        print(f"  {name:<10} n={scores['support']:<3} "
              f"precision {percent(scores['precision'])}  "
              f"recall {percent(scores['recall'])}  f1 {percent(scores['f1'])}")

    print()
    print("confusion (rows: prompted, columns: predicted)")
    names = sorted(trial["confusion"])
    print("            " + "".join(f"{n[:4]:>6}" for n in names))
    for truth in names:
        row = trial["confusion"][truth]
        print(f"  {truth:<10}" + "".join(f"{row[p]:>6}" for p in names))

    verification = result.get("verification")
    if verification is None:
        print()
        print("not verified. Run again with --verify to recompute this session offline and "
              "compare it tick by tick.")
        return

    print()
    if verification["parity"]:
        print(f"PARITY OK: {verification['n_ticks_compared']} ticks and "
              f"{verification['n_trials_compared']} trials recomputed offline; every "
              "prediction and every similarity matches what the headset wrote.")
        return

    print("PARITY FAILURE")
    for entry in verification["tick_mismatches"]:
        print(f"  tick  trial {entry['trial']:>3} tick {entry['tick']}: "
              f"headset {entry['runtime']} vs recomputed {entry['recomputed']}")
    for entry in verification["trial_mismatches"]:
        print(f"  trial trial {entry['trial']:>3}: "
              f"headset {entry['runtime']} vs recomputed {entry['recomputed']}")
    for entry in verification["magnitude_mismatches"]:
        print(f"  value trial {entry['trial']:>3} tick {entry['tick']} {entry['field']}: "
              f"headset {entry['runtime']} vs recomputed {entry['recomputed']}")


def discover(root: Path) -> list[Path]:
    base = root / "Logs" / "quest-sessions" / "sessions"
    if not base.is_dir():
        return []

    return sorted(p for p in base.iterdir()
                  if p.is_dir() and (p / "predictions.csv").is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", action="append", default=[],
                        help="a session directory; repeatable. Default: every pulled session")
    parser.add_argument("--verify", action="store_true",
                        help="recompute the session offline and compare it to the headset")
    parser.add_argument("--json", default=None, help="write the full result here")
    args = parser.parse_args(argv)

    root = sa.find_repo_root() or Path.cwd()
    directories = [Path(s) for s in args.session] or discover(root)

    if not directories:
        raise SystemExit(
            "no sessions found. Pull them with `python Analysis/pull_sessions.py`, or point at "
            "one with --session."
        )

    results = []
    failures = 0

    for directory in directories:
        session = read_session(directory)
        try:
            result = analyse(session, root, args.verify)
        except SystemExit as exc:
            print(f"\n{directory.name}: {exc}")
            failures += 1
            continue

        print_report(result)
        results.append(result)

        if result.get("verification") and not result["verification"]["parity"]:
            failures += 1

    # Grouped rather than pooled. Two sessions under different modes are measurements of
    # different methods, and a mean over them describes neither.
    by_mode: dict[str, list[str]] = defaultdict(list)
    for result in results:
        by_mode[result["calibration_mode"]].append(result["session"])

    if len(by_mode) > 1:
        print()
        print("These sessions ran under different calibration modes and are reported "
              "separately. They must not be pooled:")
        for mode, sessions in sorted(by_mode.items()):
            print(f"  {mode:<12} {len(sessions)} session(s): {', '.join(sessions)}")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"schema_version": REPORT_SCHEMA_VERSION, "sessions": results}, indent=2),
            encoding="utf-8")
        print(f"\nwrote {args.json}")

    if failures:
        raise ParityError(
            f"\n{failures} session(s) failed. A parity failure means the headset and this "
            "analysis disagree about the same recording; do not report either number until "
            "they agree."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
