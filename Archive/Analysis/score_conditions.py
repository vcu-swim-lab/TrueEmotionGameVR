#!/usr/bin/env python3
"""Accuracy and macro-F1 for each of the four conditions, from the files the game writes.

    python Analysis/score_conditions.py                       # search where analyze_sessions.py searches
    python Analysis/score_conditions.py ~/sessions
    python Analysis/score_conditions.py ~/sessions --per-class --per-session
    python Analysis/score_conditions.py ~/sessions --json scores.json --csv scores.csv
    python Analysis/score_conditions.py ~/sessions --ticks all

The conditions are the ones `ConditionSetup.cs` chooses between and `session_metadata.json` records as
`models.condition`:

    sparsevr_nocal   SparseVR, no calibration        sparsevr_cal   SparseVR, calibration
    jev_nocal        Jev, no calibration             jev_cal        Jev, calibration

What is scored
--------------
The label is `prompted_emotion`: what the player was asked to act, not an independently verified
emotion. The prediction is `predicted_emotion`, the model's own answer (SparseVR's or Jev's, whichever
`backend_id` names). Only test ticks count:

* the study's calibration block is left out. The calibration is fitted on it, so scoring it would score
  the model on its own training data. The game has no `block_role`, and all its ticks are test ticks;
* ticks 1 to 9 of each trial (`--ticks all` for 0 to 9). Tick 0 is taken while the face is still
  changing, and the trial votes and `runtime_calibration_report.py` leave it out too;
* ticks the model did not answer (`no_tracking`, `backend_error`) are not scored, but they are counted.
  `cover` is the share of ticks that were answered, and `acc*` is the accuracy if every unanswered tick
  counted as wrong. Read both: a Jev run that dropped half its ticks is not a good run.

Sessions are pooled tick by tick within a condition, a program (study or game) and a protocol version.
Sessions of different protocol versions are never pooled, as everywhere else in this folder.

Macro-F1 is `session_analysis.macro_f1`, the definition used by the other tools here. A class with no
true tick and no prediction is left out of the mean rather than scored 0, and `classes` says how many
classes the mean covers. The game never prompts Neutral, so for the game that is 6 if the model never says
Neutral and 7 if it does (Neutral then scores F1 0). The study prompts all seven.

The calibrated decision
-----------------------
`calibrated_predicted_emotion` is the calibration's decision for a tick: blank when calibration is off,
was skipped, or the tick failed. Both programs write it (the study since the calibration protocol, the game
since the four conditions); a game file recorded before that has no such column and cannot be scored for
calibration. The calibration table compares the model and the calibration on the very same ticks, so the
difference is the calibration and nothing else. `predicted_emotion` is always the model's own answer in
both programs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import session_analysis as sa  # noqa: E402

REPORT_SCHEMA_VERSION = "score-conditions-v1"

CONDITIONS = ("sparsevr_nocal", "sparsevr_cal", "jev_nocal", "jev_cal")
CALIBRATED_CONDITIONS = ("sparsevr_cal", "jev_cal")
NO_CONDITION = "(none)"

STATUS_BACKEND_ERROR = "backend_error"
ROLES_TESTED = ("", sa.ROLE_EVALUATION)

DEFAULT_TICKS = (1, 9)


# --------------------------------------------------------------------------- reading

def to_int(text: str | None) -> int | None:
    try:
        return int(float((text or "").strip()))
    except ValueError:
        return None


def parse_ticks(text: str) -> tuple[int, int] | None:
    """'1-9' -> (1, 9); '3' -> (3, 3); 'all' -> None (no tick filter)."""
    text = text.strip().lower()
    if text == "all":
        return None
    try:
        if "-" in text:
            low, high = text.split("-", 1)
            ticks = (int(low), int(high))
        else:
            ticks = (int(text), int(text))
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a tick range: {text!r} (use 1-9, 3, or all)")
    if ticks[0] > ticks[1] or ticks[0] < 0:
        raise argparse.ArgumentTypeError(f"not a tick range: {text!r}")
    return ticks


def read_session(csv_path: Path) -> dict:
    """The columns this tool needs from one predictions.csv, plus what the metadata says about it."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise sa.SchemaError(f"{csv_path} is empty")
        schema = sa.parse_schema(reader.fieldnames)
        has_calibrated = "calibrated_predicted_emotion" in reader.fieldnames

        rows = []
        for raw in reader:
            rows.append({
                "role": (raw.get("block_role") or "").strip().lower(),
                "tick": to_int(raw.get("prediction_index")),
                "prompted": (raw.get("prompted_emotion") or "").strip(),
                "predicted": (raw.get("predicted_emotion") or "").strip(),
                "calibrated": (raw.get("calibrated_predicted_emotion") or "").strip(),
                "status": (raw.get("status") or "").strip(),
                "error_code": (raw.get("error_code") or "").strip(),
            })

    meta: dict = {}
    meta_path = csv_path.parent / "session_metadata.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            meta = {}
    models = meta.get("models") or {}

    protocol = str(meta.get("protocol_version") or "")
    is_study = bool(protocol) or any(r["role"] for r in rows)

    return {
        "path": csv_path,
        "session_id": str(meta.get("session_id") or csv_path.parent.name),
        "classes": list(schema.classes),
        "condition": str(models.get("condition") or NO_CONDITION),
        "backend_id": str(models.get("backend_id") or ""),
        "program": "study" if is_study else "game",
        "protocol": protocol,
        "has_calibrated": has_calibrated,
        "rows": rows,
    }


def test_rows(rows: list[dict], ticks: tuple[int, int] | None) -> list[dict]:
    """The ticks that count: not the calibration block, and only the requested tick range."""
    kept = []
    for row in rows:
        if row["role"] not in ROLES_TESTED:
            continue
        if ticks is not None and (row["tick"] is None or not ticks[0] <= row["tick"] <= ticks[1]):
            continue
        kept.append(row)
    return kept


# --------------------------------------------------------------------------- metrics

def answered(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["status"] == sa.STATUS_OK and r["predicted"]]


def score(truth: list[str], predicted: list[str], classes: list[str]) -> dict:
    n = len(truth)
    correct = sum(1 for t, p in zip(truth, predicted) if t == p)
    f1 = sa.macro_f1(truth, predicted, classes)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "macro_f1": f1["macro_f1"],
        "classes_scored": f1["classes_scored"],
        "per_class": f1["per_class"],
    }


def score_rows(rows: list[dict], classes: list[str]) -> dict:
    """Everything one set of test ticks says about the model, and about the calibration if it has one."""
    ok = answered(rows)
    model = score([r["prompted"] for r in ok], [r["predicted"] for r in ok], classes)

    answered_ids = {id(r) for r in ok}
    failures = Counter(r["status"] or "unknown" for r in rows if id(r) not in answered_ids)
    errors = Counter(r["error_code"] or "unspecified" for r in rows if r["status"] == STATUS_BACKEND_ERROR)

    result = {
        "ticks": len(rows),
        "answered": len(ok),
        "coverage": len(ok) / len(rows) if rows else None,
        "failures": dict(failures),
        "backend_errors": dict(errors),
        "model": model,
        "accuracy_failures_wrong": model["correct"] / len(rows) if rows else None,
        "calibration": None,
    }

    both = [r for r in ok if r["calibrated"]]
    if both:
        truth = [r["prompted"] for r in both]
        result["calibration"] = {
            "ticks": len(both),
            "model_same_ticks": score(truth, [r["predicted"] for r in both], classes),
            "calibrated": score(truth, [r["calibrated"] for r in both], classes),
        }
    return result


def delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else b - a


def build_groups(sessions: list[dict], ticks: tuple[int, int] | None) -> list[dict]:
    """Pool the sessions of each (condition, program, protocol version) and score them."""
    pooled: dict[tuple[str, str, str], dict] = {}
    for session in sessions:
        key = (session["condition"], session["program"], session["protocol"])
        group = pooled.setdefault(key, {"sessions": [], "rows": [], "classes": list(session["classes"]),
                                        "has_calibrated": False, "backend_ids": set()})
        group["sessions"].append(session["session_id"])
        group["rows"].extend(test_rows(session["rows"], ticks))
        group["has_calibrated"] |= session["has_calibrated"]
        if session["backend_id"]:
            group["backend_ids"].add(session["backend_id"])
        # A class list can differ between files only if the model changed; keep the widest.
        for name in session["classes"]:
            if name not in group["classes"]:
                group["classes"].append(name)

    groups = []
    for (condition, program, protocol), group in pooled.items():
        entry = {
            "condition": condition,
            "program": program,
            "protocol_version": protocol,
            "sessions": len(group["sessions"]),
            "session_ids": group["sessions"],
            "backend_ids": sorted(group["backend_ids"]),
            "has_calibrated_column": group["has_calibrated"],
        }
        entry.update(score_rows(group["rows"], group["classes"]))
        groups.append(entry)

    order = {name: i for i, name in enumerate(CONDITIONS)}
    groups.sort(key=lambda g: (order.get(g["condition"], len(order)), g["condition"], g["program"], g["protocol_version"]))
    return groups


def build_session_rows(sessions: list[dict], ticks: tuple[int, int] | None) -> list[dict]:
    out = []
    for session in sessions:
        scored = score_rows(test_rows(session["rows"], ticks), session["classes"])
        out.append({"session_id": session["session_id"], "condition": session["condition"],
                    "program": session["program"], "protocol_version": session["protocol"],
                    "path": str(session["path"]), **scored})
    return out


# --------------------------------------------------------------------------- printing

WIDTH = 100


def pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def signed(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}"


def rule(title: str = "") -> str:
    return "-" * WIDTH if not title else f"-- {title} " + "-" * max(0, WIDTH - len(title) - 4)


def label(group: dict) -> str:
    return f"{group['condition']:<15} {group['program']:<6}"


def print_report(groups: list[dict], sessions_rows: list[dict] | None, ticks: tuple[int, int] | None,
                 per_class: bool, skipped: int) -> None:
    span = "all ticks (0-9)" if ticks is None else (f"ticks {ticks[0]}-{ticks[1]}" if ticks[0] != ticks[1] else f"tick {ticks[0]}")
    print(rule("model: the model's own answer vs the prompt"))
    print(f"  test ticks only, {span}; the study's calibration block is left out")
    print(f"  cover = share of ticks answered; acc = accuracy on answered ticks; "
          f"acc* = unanswered ticks counted wrong")
    print()
    print(f"  {'condition':<15} {'prog':<6} {'sess':>4} {'ticks':>6} {'cover':>7} {'acc':>7} {'acc*':>7} "
          f"{'macroF1':>8} {'classes':>7}")
    for g in groups:
        m = g["model"]
        print(f"  {label(g)} {g['sessions']:>4} {g['ticks']:>6} {pct(g['coverage']):>7} {pct(m['accuracy']):>7} "
              f"{pct(g['accuracy_failures_wrong']):>7} {pct(m['macro_f1']):>8} {m['classes_scored']:>7}")
        if g["failures"]:
            parts = [f"{count} {name}" for name, count in sorted(g["failures"].items())]
            if g["backend_errors"]:
                parts.append("(" + ", ".join(f"{c} x {n}" for c, n in sorted(g["backend_errors"].items())) + ")")
            print(f"  {'':<22} unanswered: {' '.join(parts)}")

    calibrated = [g for g in groups if g["calibration"]]
    print()
    print(rule("calibration: the calibrated decision vs the model, on the same ticks"))
    if calibrated:
        print(f"  {'condition':<15} {'prog':<6} {'ticks':>6} | {'model acc':>9} {'F1':>7} | "
              f"{'calib acc':>9} {'F1':>7} | {'d acc':>6} {'d F1':>6}   (differences in points)")
        for g in calibrated:
            c = g["calibration"]
            m, k = c["model_same_ticks"], c["calibrated"]
            print(f"  {label(g)} {c['ticks']:>6} | {pct(m['accuracy']):>9} {pct(m['macro_f1']):>7} | "
                  f"{pct(k['accuracy']):>9} {pct(k['macro_f1']):>7} | "
                  f"{signed(delta(m['accuracy'], k['accuracy'])):>6} {signed(delta(m['macro_f1'], k['macro_f1'])):>6}")
    else:
        print("  no calibrated decisions found")
    for g in groups:
        if g["condition"] in CALIBRATED_CONDITIONS and not g["calibration"]:
            reason = ("these files have no calibrated_predicted_emotion column (recorded before the game wrote "
                      "it)" if not g["has_calibrated_column"]
                      else "every calibrated decision is blank (calibration was skipped, or did not work)")
            print(f"  note: {g['condition']} / {g['program']}: no calibrated decisions to compare: {reason}.")

    if per_class:
        for g in groups:
            print()
            print(rule(f"per class: {g['condition']} / {g['program']}"))
            m = g["model"]["per_class"]
            k = (g["calibration"] or {}).get("calibrated", {}).get("per_class")
            same = (g["calibration"] or {}).get("model_same_ticks", {}).get("per_class")
            head = f"  {'emotion':<10} {'support':>7} {'precision':>9} {'recall':>7} {'F1':>7}"
            if k:
                head += f" | on calibrated ticks: {'model F1':>8} {'calib F1':>8}"
            print(head)
            for name, stats in m.items():
                if stats["support"] == 0 and stats["f1"] is None:
                    continue
                line = (f"  {name:<10} {stats['support']:>7} {pct(stats['precision']):>9} "
                        f"{pct(stats['recall']):>7} {pct(stats['f1']):>7}")
                if k:
                    line += f" | {'':<19}{pct(same[name]['f1']):>8} {pct(k[name]['f1']):>8}"
                print(line)

    if sessions_rows is not None:
        print()
        print(rule("per session"))
        print(f"  {'session':<26} {'condition':<15} {'prog':<6} {'ticks':>5} {'cover':>7} {'acc':>7} {'macroF1':>8}")
        for s in sessions_rows:
            print(f"  {s['session_id']:<26} {s['condition']:<15} {s['program']:<6} {s['ticks']:>5} "
                  f"{pct(s['coverage']):>7} {pct(s['model']['accuracy']):>7} {pct(s['model']['macro_f1']):>8}")

    notes = []
    if any(g["condition"] == NO_CONDITION for g in groups):
        notes.append(f"'{NO_CONDITION}': sessions without models.condition (recorded before the four conditions "
                     "existed); they are scored but belong to none of the four")
    versions = {g["protocol_version"] for g in groups if g["protocol_version"]}
    if len(versions) > 1:
        notes.append("more than one protocol version is present; they are reported separately, never pooled")
    if skipped:
        notes.append(f"{skipped} file(s) could not be read and were skipped")
    if notes:
        print()
        print(rule("notes"))
        for note in notes:
            print(f"  - {note}")


# --------------------------------------------------------------------------- output files

def write_csv(path: Path, groups: list[dict]) -> None:
    columns = ["condition", "program", "protocol_version", "variant", "sessions", "ticks", "answered", "coverage",
               "accuracy", "accuracy_failures_wrong", "macro_f1", "classes_scored"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for g in groups:
            m = g["model"]
            writer.writerow([g["condition"], g["program"], g["protocol_version"], "model", g["sessions"], g["ticks"],
                             g["answered"], g["coverage"], m["accuracy"], g["accuracy_failures_wrong"],
                             m["macro_f1"], m["classes_scored"]])
            if g["calibration"]:
                c = g["calibration"]
                for variant, block in (("model_same_ticks", c["model_same_ticks"]), ("calibrated", c["calibrated"])):
                    writer.writerow([g["condition"], g["program"], g["protocol_version"], variant, g["sessions"],
                                     c["ticks"], c["ticks"], "", block["accuracy"], "",
                                     block["macro_f1"], block["classes_scored"]])


# --------------------------------------------------------------------------- command line

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Accuracy and macro-F1 per condition from predictions.csv files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("paths", nargs="*",
                        help="predictions.csv files, session folders, or a parent folder; "
                             "omit to search this machine as analyze_sessions.py does")
    parser.add_argument("--ticks", type=parse_ticks, default=DEFAULT_TICKS, metavar="A-B|all",
                        help="which ticks of each trial to score (default: 1-9)")
    parser.add_argument("--per-class", action="store_true", help="add precision, recall and F1 per emotion")
    parser.add_argument("--per-session", action="store_true", help="add one line per session")
    parser.add_argument("--json", type=Path, metavar="FILE", help="write every number to a JSON file")
    parser.add_argument("--csv", type=Path, metavar="FILE", help="write the per-condition table to a CSV file")
    args = parser.parse_args(argv)

    if args.paths:
        try:
            files = sa.discover(args.paths)
        except FileNotFoundError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
    else:
        files, _searched = sa.autodiscover()
        if not files:
            print("error: no session logs found on this machine; pass a path, or set TRUEEMOTION_SESSIONS",
                  file=sys.stderr)
            return 2

    sessions, skipped = [], 0
    for path in files:
        try:
            sessions.append(read_session(path))
        except (sa.SchemaError, OSError) as error:
            print(f"error: skipping {path}: {error}", file=sys.stderr)
            skipped += 1

    if not sessions:
        print("error: no readable session files", file=sys.stderr)
        return 1

    groups = build_groups(sessions, args.ticks)
    session_rows = build_session_rows(sessions, args.ticks) if args.per_session else None
    print_report(groups, session_rows, args.ticks, args.per_class, skipped)

    if args.json:
        payload = {"schema": REPORT_SCHEMA_VERSION,
                   "ticks": "all" if args.ticks is None else f"{args.ticks[0]}-{args.ticks[1]}",
                   "groups": groups,
                   "sessions": build_session_rows(sessions, args.ticks)}
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")
    if args.csv:
        write_csv(args.csv, groups)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
