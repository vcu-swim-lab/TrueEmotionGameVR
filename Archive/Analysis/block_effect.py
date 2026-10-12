#!/usr/bin/env python3
"""Split a session-local-v1 recording into its two blocks and compare them.

    python Analysis/block_effect.py
    python Analysis/block_effect.py Logs/quest-sessions/sessions/session_... --json out.json

`calibration_study.py` fits a per-person adapter on the calibration block (K) and scores
it on the evaluation block (D). That comparison silently assumes the two blocks are the
same task performed the same way -- that a profile fitted on K *could* transfer to D if
the model were fixable at all. This script checks that assumption before any adapter is
fitted, by asking one question of the raw log:

    did the deployed model, and the face in front of it, behave the same way
    in both blocks?

Three numbers per block carry it:

  **accuracy**      what the shipped argmax scored, at tick and at trial level
  **p(prompted)**   the score the model gave the class the participant was asked for
  **AU activity**   the sum of the 70 action units in the newest window frame, averaged
                    over ticks of the six acted prompts -- a crude "how much is the face
                    doing" meter, with Neutral left out because sitting still is correct
                    for that one

AU activity is what makes the result readable. If accuracy falls from K to D while the
face keeps moving just as much, the model read the same behaviour differently and the
drop is the model's. If accuracy falls and the face also stops moving, the participant
performed the second block differently, and nothing about the model follows -- including
anything the adapter scores on that block.

The blocks are never counterbalanced: D always follows K, in one sitting. So order,
fatigue and practice live inside every difference printed here and cannot be separated
from the block itself. One participant is one participant; the Fisher test on trial hits
says the split is unlikely under a random relabelling of *these* trials, and says nothing
about the next person.

Stdlib only, like the rest of Analysis/ apart from calibration_study.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import session_analysis as sa                             # noqa: E402
from analyze_sessions import WIDTH, bar, num, pct, rule   # noqa: E402

# Sitting still is the right answer for Neutral, so including it in the activity meter
# would mix "did not perform" with "performed correctly" in the same average.
RESTING_CLASS = "Neutral"

# Reading thresholds for the verdict paragraph. Both are judgement calls stated out loud
# rather than hidden in an if: 10 accuracy points is about the smallest block difference
# worth a sentence at these trial counts, and 0.80 is the activity ratio below which the
# face is doing visibly less rather than jittering around the same amount.
MATERIAL_ACCURACY_DROP = 0.10
MATERIAL_ACTIVITY_DROP = 0.80


class BlockError(RuntimeError):
    """Raised when a recording cannot be split into blocks at all."""


# ------------------------------------------------------------------------- collecting


@dataclass
class Trial:
    """One prompt, rolled up. The trial is the unit any claim here is made in."""

    trial_id: int
    role: str
    repeat: int | None
    prompted: str
    ticks: int
    correct_ticks: int
    majority: str | None
    prompted_score: float | None
    activity: float | None

    @property
    def hit(self) -> bool:
        return self.majority == self.prompted


@dataclass
class Block:
    """A calibration or evaluation block, optionally one repeat of one."""

    label: str
    role: str
    repeat: int | None
    trials: list[Trial] = field(default_factory=list)
    ticks: int = 0
    correct_ticks: int = 0
    prompted_scores: list[float] = field(default_factory=list)
    resting_scores: list[float] = field(default_factory=list)
    activities: list[float] = field(default_factory=list)
    predicted: Counter = field(default_factory=Counter)

    @property
    def tick_accuracy(self) -> float | None:
        return _rate(self.correct_ticks, self.ticks)

    @property
    def trial_hits(self) -> int:
        return sum(1 for t in self.trials if t.hit)

    @property
    def trial_accuracy(self) -> float | None:
        return _rate(self.trial_hits, len(self.trials))

    @property
    def mean_prompted(self) -> float | None:
        return _mean(self.prompted_scores)

    @property
    def mean_resting(self) -> float | None:
        return _mean(self.resting_scores)

    @property
    def activity(self) -> float | None:
        return _mean(self.activities)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _ratio(after: float | None, before: float | None) -> float | None:
    if after is None or not before:
        return None
    return after / before


def parse_ticks(text: str) -> set[int] | None:
    """`--ticks 1-9` or `--ticks 0,2,4`; empty means every tick in the trial.

    Offered because `calibration_study.py` scores ticks 1-9 by default, dropping the
    first while the expression is still forming. Comparing its numbers against these
    means asking for the same window.
    """
    text = (text or "").strip()
    if not text:
        return None
    wanted: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, _, end = part.partition("-")
            wanted.update(range(int(start), int(end) + 1))
        else:
            wanted.add(int(part))
    if not wanted:
        raise argparse.ArgumentTypeError(f"no tick positions in {text!r}")
    return wanted


def activity_of(session: sa.Session, row: sa.Row) -> float | None:
    """Total action-unit movement in the newest frame of this row's window.

    The newest frame is the one `session_analysis.action_units` reports on, so the two
    tools describe the same slice of the window rather than two different ones.
    """
    if not row.frames:
        return None
    frame = row.frames[session.schema.newest_frame]
    values = [v for v in frame if v is not None]
    return sum(values) if values else None


def collect(session: sa.Session, ticks: set[int] | None = None) -> list[Block]:
    """One Block per (role, repeat), plus a merged Block per role that has repeats.

    Rows without a `block_role` cannot be placed, so a recording made before
    session-local-v1 -- or by the scored game, which has no blocks -- is refused here
    rather than silently reported as one undifferentiated lump.
    """
    rows = [r for r in session.ok_rows if r.block_role]
    if not rows:
        raise BlockError(
            "no block_role column values: this is not a session-local-v1 recording, "
            "so it has no calibration and evaluation blocks to compare"
        )
    if ticks is not None:
        rows = [r for r in rows if r.prediction_index in ticks]
        if not rows:
            raise BlockError("no rows left after the --ticks filter")

    grouped: dict[int, list[sa.Row]] = defaultdict(list)
    for row in rows:
        if row.trial_id is not None:
            grouped[row.trial_id].append(row)

    blocks: dict[tuple[str, int | None], Block] = {}
    merged: dict[str, Block] = {}

    for trial_id in sorted(grouped):
        trial_rows = sorted(
            grouped[trial_id],
            key=lambda r: (r.prediction_index if r.prediction_index is not None else 0),
        )
        head = trial_rows[0]
        prompted = next((r.prompted for r in trial_rows if r.prompted), "")
        predictions = [r.predicted for r in trial_rows if r.predicted]
        scores = [sa._score_of(session, r, prompted) for r in trial_rows]
        resting = [sa._score_of(session, r, RESTING_CLASS) for r in trial_rows]
        activities = [activity_of(session, r) for r in trial_rows]

        trial = Trial(
            trial_id=trial_id,
            role=head.block_role,
            repeat=head.repeat_index,
            prompted=prompted,
            ticks=len(trial_rows),
            correct_ticks=sum(1 for r in trial_rows if r.correct),
            # Ties go to the class seen first, which for ten ticks of one expression is
            # near enough arbitrary and is only ever reported as a count, never ranked.
            majority=max(set(predictions), key=predictions.count) if predictions else None,
            prompted_score=_mean(scores),
            activity=_mean(activities),
        )

        targets = [blocks.setdefault(
            (trial.role, trial.repeat),
            Block(label=_label(trial.role, trial.repeat), role=trial.role, repeat=trial.repeat),
        )]
        targets.append(merged.setdefault(
            trial.role, Block(label=f"{trial.role} (all)", role=trial.role, repeat=None)
        ))

        for block in targets:
            block.trials.append(trial)
            block.ticks += trial.ticks
            block.correct_ticks += trial.correct_ticks
            block.prompted_scores.extend(s for s in scores if s is not None)
            block.resting_scores.extend(s for s in resting if s is not None)
            block.predicted.update(predictions)
            # Acted prompts only: see RESTING_CLASS.
            if prompted != RESTING_CLASS:
                block.activities.extend(a for a in activities if a is not None)

    ordered = sorted(blocks.values(), key=_order)
    # A merged row is only informative when the role was actually split into repeats.
    for role, block in sorted(merged.items()):
        if sum(1 for key in blocks if key[0] == role) > 1:
            ordered.append(block)
    return ordered


def _label(role: str, repeat: int | None) -> str:
    return role if repeat is None else f"{role} r{repeat}"


def _order(block: Block) -> tuple:
    # Calibration precedes evaluation because that is the order it was recorded in, and
    # every difference in this file is read in that direction.
    rank = {sa.ROLE_CALIBRATION: 0, sa.ROLE_EVALUATION: 1}.get(block.role, 2)
    return (rank, block.role, block.repeat if block.repeat is not None else -1)


def pick(blocks: list[Block], role: str) -> Block | None:
    """The row to read for a role: the merged one when repeats were split."""
    merged = [b for b in blocks if b.role == role and b.label.endswith("(all)")]
    if merged:
        return merged[0]
    singles = [b for b in blocks if b.role == role]
    return singles[0] if singles else None


# -------------------------------------------------------------------------- statistics


def fisher_exact(hit_a: int, miss_a: int, hit_b: int, miss_b: int) -> float | None:
    """Two-sided Fisher exact p for a 2x2 of trial hits. Exact, so no approximation
    apology is needed at these counts -- seven trials against fourteen is far too few
    for a chi-square.

    It answers a narrow question: given these 21 trials and these totals, how often
    would a random relabelling of which trial fell in which block separate them at
    least this sharply? It is not a test of the model, of calibration, or of anyone
    but this participant on this evening.
    """
    total = hit_a + miss_a + hit_b + miss_b
    row = hit_a + miss_a
    column = hit_a + hit_b
    if not total or not row or not column or row == total or column == total:
        return None

    def probability(k: int) -> float:
        return (math.comb(row, k) * math.comb(total - row, column - k)
                / math.comb(total, column))

    low = max(0, column - (total - row))
    high = min(row, column)
    observed = probability(hit_a)
    # Float slack, so a table with exactly the observed probability is not dropped by a
    # last-bit comparison.
    tolerance = 1 + 1e-9
    return min(1.0, sum(probability(k) for k in range(low, high + 1)
                        if probability(k) <= observed * tolerance))


def per_class(blocks: list[Block]) -> dict[str, dict]:
    """Accuracy and activity for each prompted class, in each role.

    Read at role level rather than per repeat: one trial per class per repeat is too
    thin to say anything, and `pick` already merges the repeats for exactly this.
    """
    table: dict[str, dict] = {}
    for role in (sa.ROLE_CALIBRATION, sa.ROLE_EVALUATION):
        block = pick(blocks, role)
        if not block:
            continue
        for trial in block.trials:
            entry = table.setdefault(trial.prompted, {})
            side = entry.setdefault(role, {"ticks": 0, "correct": 0, "activity": []})
            side["ticks"] += trial.ticks
            side["correct"] += trial.correct_ticks
            if trial.activity is not None:
                side["activity"].append(trial.activity)
    for entry in table.values():
        for side in entry.values():
            side["accuracy"] = _rate(side["correct"], side["ticks"])
            side["mean_activity"] = _mean(side["activity"])
            del side["activity"]
    return table


# ----------------------------------------------------------------------------- report


def print_report(session: sa.Session, blocks: list[Block], classes: list[str],
                 ticks: set[int] | None) -> None:
    calibration = pick(blocks, sa.ROLE_CALIBRATION)
    evaluation = pick(blocks, sa.ROLE_EVALUATION)

    print()
    print("=" * WIDTH)
    print(f"block effect -- session {session.session_id}")
    print(f"{session.path}")
    print("=" * WIDTH)

    protocol = (session.metadata or {}).get("protocol_version", "unknown")
    scope = "every tick" if ticks is None else \
        "ticks " + ",".join(str(t) for t in sorted(ticks))
    print()
    print(f"  protocol          {protocol}")
    print(f"  scored on         {scope}")
    print(f"  activity meter    sum of {session.schema.features_per_frame} action units, "
          f"frame {session.schema.newest_frame} (newest), acted prompts only")

    _print_blocks(blocks)
    _print_effect(calibration, evaluation)
    _print_per_class(blocks, classes)
    _print_destinations(blocks, classes)
    _print_reading(calibration, evaluation)
    print()


def _print_blocks(blocks: list[Block]) -> None:
    print()
    print(rule("blocks"))
    print(f"    {'block':<18}{'trials':>7}{'ticks':>7}{'tick acc':>10}"
          f"{'trial acc':>11}{'p(prompt)':>11}{'p(rest)':>9}{'activity':>10}")
    for block in blocks:
        if block.label.endswith("(all)"):
            print("    " + "-" * (18 + 7 + 7 + 10 + 11 + 11 + 9 + 10 - 4))
        print(f"    {block.label:<18}{len(block.trials):>7}{block.ticks:>7}"
              f"{pct(block.tick_accuracy):>10}"
              f"{f'{block.trial_hits}/{len(block.trials)}':>11}"
              f"{num(block.mean_prompted):>11}{num(block.mean_resting):>9}"
              f"{num(block.activity, 2):>10}")
    print()
    print("    trial acc counts a trial as a hit when the majority of its ticks named")
    print("    the prompted class.  p(rest) is the score given to Neutral, so a block")
    print("    where it rises is one the model increasingly read as a resting face.")


def _print_effect(calibration: Block | None, evaluation: Block | None) -> None:
    if not calibration or not evaluation:
        return

    print()
    print(rule("calibration -> evaluation"))

    for name, before, after, digits in (
        ("tick accuracy", calibration.tick_accuracy, evaluation.tick_accuracy, None),
        ("p(prompted)", calibration.mean_prompted, evaluation.mean_prompted, 3),
        ("AU activity", calibration.activity, evaluation.activity, 2),
    ):
        if before is None or after is None:
            continue
        shown = (f"{pct(before)} -> {pct(after)}" if digits is None
                 else f"{num(before, digits)} -> {num(after, digits)}")
        change = after - before
        relative = f"  ({after / before:.2f}x)" if before else ""
        sign = "+" if change >= 0 else ""
        tail = (f"{sign}{change * 100:.1f} points" if digits is None
                else f"{sign}{change:.{digits}f}{relative}")
        print(f"  {name:<16}{shown:>22}   {tail}")

    hits_c, hits_e = calibration.trial_hits, evaluation.trial_hits
    n_c, n_e = len(calibration.trials), len(evaluation.trials)
    print(f"  {'trial majority':<16}{f'{hits_c}/{n_c} -> {hits_e}/{n_e}':>22}", end="")
    p = fisher_exact(hits_c, n_c - hits_c, hits_e, n_e - hits_e)
    print(f"   Fisher exact p = {p:.4f}" if p is not None else "")
    print()
    print("  The evaluation block always follows the calibration block, so order,")
    print("  fatigue and practice are inside this difference. p is computed over these")
    print("  trials only; it is not evidence about the next participant.")


def _print_per_class(blocks: list[Block], classes: list[str]) -> None:
    table = per_class(blocks)
    if not table:
        return

    print()
    print(rule("per prompted class"))
    print(f"    {'class':<12}{'calib':>8}{'eval':>8}{'delta':>9}   "
          f"{'activity: calib':>15}{'eval':>8}{'ratio':>8}")
    for emotion in classes:
        entry = table.get(emotion)
        if not entry:
            continue
        cal = entry.get(sa.ROLE_CALIBRATION, {})
        ev = entry.get(sa.ROLE_EVALUATION, {})
        delta = (ev.get("accuracy") - cal.get("accuracy")
                 if cal.get("accuracy") is not None and ev.get("accuracy") is not None
                 else None)
        ratio = _ratio(ev.get("mean_activity"), cal.get("mean_activity"))
        note = "" if emotion != RESTING_CLASS else "   (resting prompt)"
        print(f"    {emotion:<12}{pct(cal.get('accuracy'), 0):>8}{pct(ev.get('accuracy'), 0):>8}"
              f"{(f'{delta * 100:+.0f} pts' if delta is not None else '     n/a'):>9}   "
              f"{num(cal.get('mean_activity'), 2):>15}{num(ev.get('mean_activity'), 2):>8}"
              f"{num(ratio, 2):>8}{note}")


def _print_destinations(blocks: list[Block], classes: list[str]) -> None:
    """Where each block's ticks went. A block that collapses onto one class is the
    signature of a face the model stopped distinguishing, and naming that class is
    usually the whole diagnosis."""
    print()
    print(rule("what the model predicted, share of ticks"))
    header = f"    {'block':<18}" + "".join(f"{c[:4]:>7}" for c in classes)
    print(header)
    for block in blocks:
        total = sum(block.predicted.values())
        if not total:
            continue
        cells = "".join(
            f"{(f'{block.predicted[c] / total * 100:.0f}%' if block.predicted[c] else '.'):>7}"
            for c in classes
        )
        print(f"    {block.label:<18}{cells}")


def _print_reading(calibration: Block | None, evaluation: Block | None) -> None:
    """The one paragraph this script exists to print."""
    print()
    print(rule("reading"))
    if not calibration or not evaluation:
        print("  Only one block is present; there is nothing to compare.")
        return

    acc_before, acc_after = calibration.tick_accuracy, evaluation.tick_accuracy
    ratio = _ratio(evaluation.activity, calibration.activity)
    if acc_before is None or acc_after is None:
        print("  Not enough scored ticks in one of the blocks to read anything.")
        return

    drop = acc_before - acc_after
    if drop < MATERIAL_ACCURACY_DROP:
        print(f"  Accuracy holds across the blocks ({pct(acc_before)} -> {pct(acc_after)}).")
        print("  The two blocks look like the same task performed the same way, which is")
        print("  what a calibration fitted on one and scored on the other needs.")
    elif ratio is not None and ratio < MATERIAL_ACTIVITY_DROP:
        print(f"  Accuracy fell {drop * 100:.0f} points AND the face moved "
              f"{(1 - ratio) * 100:.0f}% less")
        print(f"  ({num(calibration.activity, 2)} -> {num(evaluation.activity, 2)} AU "
              "activity). The participant performed the evaluation block")
        print("  differently from the calibration block, so this session cannot separate")
        print("  a calibration effect from a participant effect. An adapter scored on")
        print("  this evaluation block is being asked to predict a weaker performance")
        print("  than it was fitted on, and will look bad for a reason that is not about")
        print("  calibration.")
    elif ratio is not None:
        print(f"  Accuracy fell {drop * 100:.0f} points while the face kept moving about "
              "as much")
        print(f"  (activity {num(calibration.activity, 2)} -> "
              f"{num(evaluation.activity, 2)}, {ratio:.2f}x). Similar behaviour was read")
        print("  differently in the second block, which is the case a per-person")
        print("  calibration exists to address, and the blocks stay comparable enough")
        print("  for an adapter fitted on one to be scored on the other.")
        # Named separately because a difference can be large enough to describe and
        # still be one this many trials cannot separate from chance.
        p = fisher_exact(
            calibration.trial_hits, len(calibration.trials) - calibration.trial_hits,
            evaluation.trial_hits, len(evaluation.trials) - evaluation.trial_hits,
        )
        if p is not None and p > 0.05:
            print(f"  At trial level the two blocks are not distinguishable (p = {p:.3f}),")
            print("  so read the size of this drop as a description, not as a finding.")
    else:
        print(f"  Accuracy fell {drop * 100:.0f} points, but the windows carry no action")
        print("  units, so whether the face also changed cannot be checked from this file.")


def print_index(results: list[tuple[sa.Session, list[Block]]]) -> None:
    print()
    print("=" * WIDTH)
    print(f"{len(results)} session(s)")
    print("=" * WIDTH)
    print(f"  {'session':<26}{'calib':>9}{'eval':>9}{'delta':>10}{'activity':>11}")
    for session, blocks in results:
        calibration = pick(blocks, sa.ROLE_CALIBRATION)
        evaluation = pick(blocks, sa.ROLE_EVALUATION)
        if not calibration or not evaluation:
            print(f"  {session.session_id[:26]:<26}{'one block only':>39}")
            continue
        delta = (evaluation.tick_accuracy - calibration.tick_accuracy
                 if None not in (evaluation.tick_accuracy, calibration.tick_accuracy)
                 else None)
        ratio = _ratio(evaluation.activity, calibration.activity)
        print(f"  {session.session_id[:26]:<26}"
              f"{pct(calibration.tick_accuracy):>9}{pct(evaluation.tick_accuracy):>9}"
              f"{(f'{delta * 100:+.0f} pts' if delta is not None else 'n/a'):>10}"
              f"{(f'{ratio:.2f}x' if ratio is not None else 'n/a'):>11}")


# ------------------------------------------------------------------------------ export


def to_json(session: sa.Session, blocks: list[Block], classes: list[str]) -> dict:
    calibration = pick(blocks, sa.ROLE_CALIBRATION)
    evaluation = pick(blocks, sa.ROLE_EVALUATION)
    effect = None
    if calibration and evaluation:
        effect = {
            "tick_accuracy": [calibration.tick_accuracy, evaluation.tick_accuracy],
            "trial_hits": [[calibration.trial_hits, len(calibration.trials)],
                           [evaluation.trial_hits, len(evaluation.trials)]],
            "fisher_p": fisher_exact(
                calibration.trial_hits, len(calibration.trials) - calibration.trial_hits,
                evaluation.trial_hits, len(evaluation.trials) - evaluation.trial_hits,
            ),
            "mean_prompted": [calibration.mean_prompted, evaluation.mean_prompted],
            "activity": [calibration.activity, evaluation.activity],
            "activity_ratio": _ratio(evaluation.activity, calibration.activity),
        }
    return {
        "session_id": session.session_id,
        "file": str(session.path),
        "protocol_version": (session.metadata or {}).get("protocol_version"),
        "blocks": [
            {
                "label": block.label,
                "role": block.role,
                "repeat": block.repeat,
                "trials": len(block.trials),
                "ticks": block.ticks,
                "tick_accuracy": block.tick_accuracy,
                "trial_hits": block.trial_hits,
                "trial_accuracy": block.trial_accuracy,
                "mean_prompted_score": block.mean_prompted,
                "mean_resting_score": block.mean_resting,
                "au_activity": block.activity,
                "predicted_share": {c: block.predicted[c] / sum(block.predicted.values())
                                    for c in classes if sum(block.predicted.values())},
                "trial_detail": [
                    {"trial_id": t.trial_id, "prompted": t.prompted, "repeat": t.repeat,
                     "majority": t.majority, "hit": t.hit, "ticks": t.ticks,
                     "correct_ticks": t.correct_ticks, "activity": t.activity}
                    for t in block.trials
                ],
            }
            for block in blocks
        ],
        "effect": effect,
        "per_class": per_class(blocks),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare the calibration and evaluation blocks of a session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("paths", nargs="*",
                        help="predictions.csv, session folder, or parent; omit to search")
    parser.add_argument("--ticks", type=parse_ticks, default=None, metavar="SPEC",
                        help="tick positions to score, e.g. 1-9 to match "
                             "calibration_study.py (default: all of them)")
    parser.add_argument("--json", type=Path, metavar="FILE",
                        help="write every number to a JSON file")
    args = parser.parse_args(argv)

    try:
        files = sa.discover(args.paths) if args.paths else sa.autodiscover()[0]
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not files:
        print("no session logs found; see analyze_sessions.py --where", file=sys.stderr)
        return 1

    results: list[tuple[sa.Session, list[Block]]] = []
    skipped = 0
    for path in files:
        try:
            session = sa.load_session(path)
        except (sa.SchemaError, OSError) as error:
            print(f"error: skipping {path}: {error}", file=sys.stderr)
            skipped += 1
            continue

        # A rehearsal records three ticks per trial instead of ten and is marked as such
        # on purpose. Reporting one here would put a number that means nothing next to
        # numbers that do.
        if str((session.metadata or {}).get("fast_rehearsal", "")).lower() == "true":
            print(f"skipping {session.session_id}: recorded with fast_rehearsal on",
                  file=sys.stderr)
            skipped += 1
            continue

        try:
            blocks = collect(session, args.ticks)
        except BlockError as error:
            print(f"skipping {session.session_id}: {error}", file=sys.stderr)
            skipped += 1
            continue
        results.append((session, blocks))

    if not results:
        print("error: no session with calibration and evaluation blocks", file=sys.stderr)
        return 1

    if len(results) > 1:
        print_index(results)
    for session, blocks in results:
        print_report(session, blocks, list(session.schema.classes), args.ticks)

    if args.json:
        payload = {"sessions": [to_json(s, b, list(s.schema.classes)) for s, b in results]}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
