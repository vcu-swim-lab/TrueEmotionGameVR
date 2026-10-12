#!/usr/bin/env python3
"""Summarise the prediction logs written by the VR emotion game.

    python analyze_sessions.py path/to/predictions.csv
    python analyze_sessions.py ~/sessions --pool --json report.json

Point it at a predictions.csv, at a session folder, or at the parent folder holding
many sessions; every file found is reported, and `--pool` adds a combined report over
all of them. See README.md for what the numbers mean.

Output is deliberately plain ASCII: these reports get pasted into notebooks, issues
and emails, and a Windows console using a legacy code page will not render box
drawing or block characters.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import session_analysis as sa

WIDTH = 78


# ------------------------------------------------------------------------ formatting


def pct(value: float | None, digits: int = 1) -> str:
    return "  n/a" if value is None else f"{value * 100:.{digits}f}%"


def num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def rule(title: str = "") -> str:
    if not title:
        return "-" * WIDTH
    return f"-- {title} " + "-" * max(0, WIDTH - len(title) - 4)


def bar(fraction: float | None, width: int = 24) -> str:
    """A fixed-width ASCII meter, so columns of them line up."""
    if fraction is None:
        return " " * width
    filled = max(0, min(width, round(fraction * width)))
    return "#" * filled + "." * (width - filled)


def abbreviations(classes: list[str]) -> dict[str, str]:
    """Short, unique column labels for the confusion matrix."""
    labels: dict[str, str] = {}
    for name in classes:
        size = 3
        while name[:size] in labels.values() and size < len(name):
            size += 1
        labels[name] = name[:size]
    return labels


# --------------------------------------------------------------------------- report


def print_report(report: dict, *, top_aus: int, show_trials: bool) -> None:
    classes = report["schema"]["classes"]

    print()
    print("=" * WIDTH)
    print(f"session {report['session_id']}")
    print(f"{report['file']}")
    print("=" * WIDTH)

    _print_overview(report)
    _print_accuracy(report, classes)
    _print_confusion(report, classes)
    _print_curve(report)
    _print_action_units(report, top_aus)
    if show_trials:
        _print_trials(report)

    for warning in report["warnings"]:
        print(f"\n[warning] {warning}")
    print()


def _print_overview(report: dict) -> None:
    track = report["tracking"]
    time = report["timing"]
    gen = report["generation"]
    schema = report["schema"]

    print(rule("session"))
    print(f"  ticks logged      {track['ticks']}  "
          f"(tracked {track['ok']}, no_tracking {track['no_tracking']})")
    print(f"  tracking rate     {pct(track['tracking_rate'])}")
    print(f"  trials            {len(report['trials'])}")
    print(f"  duration          {num(time['duration_s'], 1)} s"
          f"   median tick {num(time['median_tick_s'], 2)} s"
          f"   slowest {num(time['max_tick_s'], 2)} s")
    print(f"  window            {schema['window_frames']} frames "
          f"x {schema['features_per_frame']} action units")

    fps = gen["estimated_tracker_fps"]
    print(f"  frames per tick   median {num(gen['median_step'], 1)}"
          f"   min {num(gen['min_step'], 1)}   max {num(gen['max_step'], 1)}"
          + (f"   (~{fps:.1f} tracker fps)" if fps else ""))
    if gen["stalled_ticks"] or gen["short_steps"]:
        print(f"  frame dropouts    {gen['short_steps']} short step(s), "
              f"{gen['stalled_ticks']} repeated window(s)")

    models = report.get("models")
    if models:
        print("  models")
        width = max(len(name) for name in models)
        for name, value in models.items():
            print(f"    {name:<{width}}  {value}")


def _print_accuracy(report: dict, classes: list[str]) -> None:
    acc = report["accuracy"]
    conf = report["confidence"]
    trial = acc["trial"]

    print()
    print(rule("accuracy"))
    print(f"  tick accuracy     {pct(acc['tick_accuracy'])}"
          f"   ({acc['ticks_correct']}/{acc['ticks_scored']} tracked ticks,"
          f" chance {pct(acc['chance_level'])})")
    print(f"  per trial         majority {pct(trial['majority'])}"
          f"   final tick {pct(trial['final'])}"
          f"   ever hit {pct(trial['any'])}   (n={trial['trials']})")
    print(f"  confidence        top score {num(conf['mean_top_score'])}"
          f"   when right {num(conf['mean_top_score_correct'])}"
          f"   when wrong {num(conf['mean_top_score_incorrect'])}")
    print(f"  mean score given to the prompted class: "
          f"{num(conf['mean_prompted_score'])}")

    print()
    print("  by prompted emotion")
    print(f"    {'emotion':<10} {'ticks':>5} {'acc':>7}  {'':24}  "
          f"{'p(prompt)':>9}  most confused with")
    for emotion in classes:
        stats = report["per_prompted"].get(emotion)
        if not stats:
            continue
        confusion = ""
        if stats["top_confusion"]:
            confusion = f"{stats['top_confusion']} ({pct(stats['top_confusion_share'], 0)})"
        print(f"    {emotion:<10} {stats['ticks']:>5} {pct(stats['accuracy']):>7}  "
              f"{bar(stats['accuracy'])}  {num(stats['mean_prompted_score']):>9}  {confusion}")

    print()
    print("  mean score per class over all tracked ticks (model resting bias)")
    means = report["class_means"]
    ordered = sorted(classes, key=lambda c: means.get(c) or 0.0, reverse=True)
    for emotion in ordered:
        print(f"    {emotion:<10} {num(means.get(emotion)):>7}  {bar(means.get(emotion))}")


def _print_confusion(report: dict, classes: list[str]) -> None:
    matrix = report["confusion"]
    labels = abbreviations(classes)
    used = [c for c in classes if sum(matrix[c].values())]
    if not used:
        return

    print()
    print(rule("confusion (rows prompted, columns predicted)"))
    header = "    " + f"{'prompted':<10}" + "".join(f"{labels[c]:>6}" for c in classes) + f"{'n':>7}"
    print(header)
    for prompted in used:
        row = matrix[prompted]
        total = sum(row.values())
        cells = "".join(
            (f"{row[p]:>6}" if row[p] else f"{'.':>6}") for p in classes
        )
        print(f"    {prompted:<10}{cells}{total:>7}")

    # Column totals. Reading down a column says how eagerly the model reaches for a
    # class; the precision row says how often that was justified. A class with a low
    # count and high precision is one the model is too shy to use, and a class with a
    # high count and low precision is the one absorbing everything else.
    emitted = report["per_predicted"]
    print("    " + "-" * (10 + 6 * len(classes) + 7))
    print(f"    {'predicted':<10}"
          + "".join(f"{(emitted[c]['emitted'] or '.'):>6}" for c in classes))
    print(f"    {'precision':<10}"
          + "".join(
              f"{(pct(emitted[c]['precision'], 0) if emitted[c]['emitted'] else '-'):>6}"
              for c in classes
          ))


def _print_curve(report: dict) -> None:
    curve = report["tick_curve"]
    if len(curve) < 2:
        return

    print()
    print(rule("within a trial (tick 0 = moment the prompt appeared)"))
    print(f"    {'tick':>4} {'n':>4} {'acc':>7}  {'':24}  p(prompt)")
    for point in curve:
        print(f"    {point['prediction_index']:>4} {point['ticks']:>4} "
              f"{pct(point['accuracy']):>7}  {bar(point['accuracy'])}  "
              f"{num(point['mean_prompted_score'])}")


def _print_action_units(report: dict, top: int) -> None:
    aus = report["action_units"]
    active = aus["most_active"]

    print()
    print(rule("action units"))
    print(f"  channels          {aus['features']} total, {len(active)} ever move, "
          f"{aus['dead_count']} flat"
          f"   (measured on frame {aus['frame_used']}, the newest in each window)")
    if aus["dead_count"] and aus["dead_count"] < aus["features"]:
        flat = ", ".join(str(a) for a in aus["dead"][:20])
        more = " ..." if aus["dead_count"] > 20 else ""
        print(f"  flat channels     {flat}{more}")

    if not active:
        return
    print()
    print(f"  most active (top {min(top, len(active))} by mean value)")
    print(f"    {'au':>4} {'mean':>8} {'max':>8} {'active':>7}")
    for stats in active[:top]:
        print(f"    {stats['au']:>4} {num(stats['mean'], 4):>8} "
              f"{num(stats['max'], 4):>8} {pct(stats['active_fraction'], 0):>7}")


def _print_trials(report: dict) -> None:
    trials = report["trials"]
    if not trials:
        return

    print()
    print(rule("trials"))
    print(f"    {'#':>3} {'prompted':<10} {'ticks':>5} {'acc':>7} {'first hit':>9} "
          f"{'majority':<10} {'final':<10} {'peak p':>7}")
    for trial in trials:
        first = trial["first_correct_index"]
        print(f"    {trial['trial_id']:>3} {trial['prompted']:<10} "
              f"{trial['ok_ticks']:>5} {pct(trial['accuracy']):>7} "
              f"{('-' if first is None else first):>9} "
              f"{(trial['majority_prediction'] or '-'):<10} "
              f"{(trial['final_prediction'] or '-'):<10} "
              f"{num(trial['peak_prompted_score'], 2):>7}")


def print_search_locations(
    searched: list[tuple[str, Path, int]] | None = None, stream=sys.stdout
) -> None:
    """Show the folders checked when no path is given, and what each one holds.

    Machine-specific, by design: the point is that a teammate on another computer can
    run this and see where their own copy should go.
    """
    if searched is None:
        searched = [
            (label, root, len(sa.scan_for_sessions(root, max_depth=3 if label.startswith("~/") else 5)))
            for label, root in sa.search_roots()
        ]

    def say(line: str = "") -> None:
        print(line, file=stream)

    repo = sa.find_repo_root()
    company, product = sa.project_identity(repo)

    say()
    say(f"project root      {repo or 'not found'}")
    if company and product:
        say(f"unity app         {company} / {product}")
    say()
    say("searched, in order:")
    for label, root, count in searched:
        if not root.is_dir():
            state = "missing"
        elif count:
            state = f"{count} session(s)"
        else:
            state = "none found"
        say(f"  [{label:<24}] {state:<16} {root}")
    say()
    say("Override any of this by passing a path, or by setting TRUEEMOTION_SESSIONS.")


def print_index(reports: list[dict]) -> None:
    """One line per file, for when a whole folder of sessions is analysed."""
    print()
    print("=" * WIDTH)
    print(f"{len(reports)} session(s)")
    print("=" * WIDTH)
    print(f"  {'session':<26} {'ticks':>6} {'track':>7} {'tick acc':>9} {'trial acc':>9}")
    for report in reports:
        print(f"  {report['session_id'][:26]:<26} "
              f"{report['tracking']['ticks']:>6} "
              f"{pct(report['tracking']['tracking_rate'], 0):>7} "
              f"{pct(report['accuracy']['tick_accuracy']):>9} "
              f"{pct(report['accuracy']['trial']['majority']):>9}")


# ----------------------------------------------------------------------------- exports


def write_trials_csv(reports: list[dict], path: Path) -> None:
    columns = [
        "session_id", "trial_id", "prompted", "ticks", "ok_ticks", "correct_ticks",
        "accuracy", "first_correct_index", "majority_prediction", "final_prediction",
        "mean_prompted_score", "peak_prompted_score", "duration_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for report in reports:
            for trial in report["trials"]:
                writer.writerow({"session_id": report["session_id"],
                                 **{c: trial.get(c) for c in columns if c != "session_id"}})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarise predictions.csv files written by SessionLogger.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("paths", nargs="*",
                        help="predictions.csv files, session folders, or a parent folder; "
                             "omit to search this machine for them")
    parser.add_argument("--where", action="store_true",
                        help="show where sessions are searched for on this machine, then exit")
    parser.add_argument("--pool", action="store_true",
                        help="also print one combined report over every file")
    parser.add_argument("--pool-only", action="store_true",
                        help="print only the combined report")
    parser.add_argument("--trials", action="store_true",
                        help="include the per-trial table in each report")
    parser.add_argument("--top-aus", type=int, default=12, metavar="N",
                        help="how many action units to list (default: 12)")
    parser.add_argument("--json", type=Path, metavar="FILE",
                        help="write every metric to a JSON file")
    parser.add_argument("--trials-csv", type=Path, metavar="FILE",
                        help="write one row per trial to a CSV file")
    args = parser.parse_args(argv)

    if args.where:
        print_search_locations()
        return 0

    if args.paths:
        try:
            files = sa.discover(args.paths)
        except FileNotFoundError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
    else:
        files, searched = sa.autodiscover()
        if not files:
            print("error: no session logs found on this machine.", file=sys.stderr)
            print_search_locations(searched, stream=sys.stderr)
            print("\nPass a path directly, or set TRUEEMOTION_SESSIONS to the folder "
                  "you pulled the sessions into.", file=sys.stderr)
            return 2
        hit = next(((label, root) for label, root, count in searched if count), None)
        if hit:
            print(f"found {len(files)} session(s) under {hit[1]}  [{hit[0]}]")

    sessions, reports = [], []
    for path in files:
        try:
            session = sa.load_session(path)
        except (sa.SchemaError, OSError) as error:
            print(f"error: skipping {path}: {error}", file=sys.stderr)
            continue
        sessions.append(session)
        reports.append(sa.analyse(session))

    if not reports:
        print("error: no readable session files", file=sys.stderr)
        return 1

    if len(reports) > 1 and not args.pool_only:
        print_index(reports)

    if not args.pool_only:
        for report in reports:
            print_report(report, top_aus=args.top_aus, show_trials=args.trials)

    pooled = None
    if (args.pool or args.pool_only) and sessions:
        try:
            pooled = sa.analyse(sa.pool(sessions))
        except ValueError as error:
            print(f"error: cannot pool: {error}", file=sys.stderr)
        else:
            print_report(pooled, top_aus=args.top_aus, show_trials=args.trials)

    if args.json:
        payload = {"sessions": reports}
        if pooled:
            payload["pooled"] = pooled
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")

    if args.trials_csv:
        write_trials_csv(reports, args.trials_csv)
        print(f"wrote {args.trials_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
