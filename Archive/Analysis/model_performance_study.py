#!/usr/bin/env python3
"""Phase B: how the deployed model performs, participant by participant.

    python Analysis/model_performance_study.py
    python Analysis/model_performance_study.py --json results/model_performance.json

Phase A established that between-participant variance rivals between-emotion variance on
almost every channel, and that the emotions where people diverge most are the ones the model
fails on. This asks the other half directly: does the model's per-participant performance line
up with that, and with the measurement diagnostics?

Three things govern how the numbers are produced, and all three come from earlier mistakes:

**The model is evaluated under three separately named conditions.** The notebook averages six
frames; the headset takes one sample. Those are different inputs and produce different
numbers, so `notebook_exact`, `notebook_ungated` and `runtime_proxy` are reported in separate
tables and never merged. In particular an averaged, ungated result is **not** a runtime
measurement.

**Ten of the fifteen participants are training data.** Their accuracy says nothing about
generalisation. Held-out participants carry every headline; train participants appear in their
own table, labelled, and never in a pooled figure.

**One trial per participant per emotion.** A subject-level seven-class macro-F1 is therefore
computed from seven single-recording classes, which makes it granular to about 1/7 and nearly
a restatement of accuracy-out-of-seven. Both are printed, and the report says so rather than
letting macro-F1 imply a precision it does not have.

Needs numpy and onnxruntime.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import acting_dataset as ad
import au_names
import calibration_study as cs
import session_analysis as sa
import teacher_preprocessing as tp
from analyze_sessions import WIDTH, num, pct, rule

SEED = 42
CLASS_ORDER = list(ad.EMOTIONS)   # Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise


# ------------------------------------------------------------------ fidelity


def fidelity_check(model_path: Path) -> dict:
    """Prove the ONNX path is right before trusting a single number from it.

    The acted corpus carries no model output of its own, so there is nothing in it to check
    against. The live session logs do: every row stores the seven scores the headset computed
    at capture time alongside the exact window that produced them. Replaying those through
    this code and matching the logged scores shows the graph, the input layout and the class
    order are all correct — which is what licenses running it on the acted corpus.

    Skipped, loudly, when no live session is on the machine.
    """
    try:
        files = sa.autodiscover()[0]
    except Exception:
        files = []
    if not files:
        return {'ran': False, 'reason': 'no live session logs on this machine'}

    import onnxruntime as ort
    runner = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    name = runner.get_inputs()[0].name

    results = []
    for path in files:
        session = sa.load_session(path)
        rows = [r for r in session.ok_rows if r.frames and r.scores]
        if not rows:
            continue
        windows = np.array([r.frames for r in rows], np.float32)
        logged = np.array([r.scores for r in rows], np.float32)
        replayed = runner.run(None, {name: windows})[0]
        results.append({
            'session': session.session_id,
            'rows': len(rows),
            'agreement': float((replayed.argmax(1) == logged.argmax(1)).mean()),
            'max_diff': float(np.abs(replayed - logged).max()),
        })

    passed = all(r['agreement'] == 1.0 and r['max_diff'] < cs.FIDELITY_TOLERANCE
                 for r in results)
    return {'ran': True, 'passed': passed, 'sessions': results,
            'tolerance': cs.FIDELITY_TOLERANCE}


# ---------------------------------------------------------------- prediction


def predict_recordings(corpus: ad.Corpus, condition: str, model_path: Path) -> list[dict]:
    """One row per recording: its windows, their predictions, and the trial verdict.

    The trial verdict is a majority vote over the recording's windows, because a trial is the
    unit a person actually performed and consecutive windows within it are not independent.
    Ties go to the class with the highest mean probability.

    A recording that yields no windows gets `predicted = None`. It is counted as unscored, not
    as wrong: under `notebook_exact` two recordings produce nothing at all, and scoring those
    as errors would blame the model for a gate that refused to emit an input.
    """
    import onnxruntime as ort
    runner = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    name = runner.get_inputs()[0].name

    rows = []
    for recording in corpus.recordings:
        windows = tp.windows_for_condition(recording, condition)
        offered = len(windows)

        # `fixed_windows_gated` keeps the boundaries of `notebook_ungated` and refuses the
        # windows the activity rule rejects. Abstention is the policy chosen here: a refused
        # window produces no answer and leaves the denominator, rather than being scored as
        # Neutral or as a held-over previous answer. Those are different runtime policies
        # with different costs, and none of them is measured by this condition.
        if condition in tp.ABSTAINING_CONDITIONS and offered:
            keep = tp.gate_mask(windows, recording.emotion)
            windows = windows[keep]

        if len(windows) == 0:
            rows.append({
                'participant': recording.participant,
                'emotion': recording.emotion,
                'split': recording.split,
                'excluded': recording.excluded,
                'windows': 0,
                'windows_offered': offered,
                'predicted': None,
                'correct': None,
                'window_correct': 0,
                'mean_confidence': None,
                'mean_prompted_score': None,
            })
            continue

        scores = runner.run(None, {name: windows})[0]
        argmax = scores.argmax(1)
        votes = Counter(argmax.tolist())
        best = max(votes.values())
        tied = [c for c, n in votes.items() if n == best]
        winner = (tied[0] if len(tied) == 1
                  else max(tied, key=lambda c: scores[:, c].mean()))

        truth = CLASS_ORDER.index(recording.emotion)
        rows.append({
            'participant': recording.participant,
            'emotion': recording.emotion,
            'split': recording.split,
            'excluded': recording.excluded,
            'windows': int(len(windows)),
            'windows_offered': offered,
            'predicted': CLASS_ORDER[winner],
            'correct': bool(winner == truth),
            'window_correct': int((argmax == truth).sum()),
            'mean_confidence': float(scores.max(axis=1).mean()),
            'mean_prompted_score': float(scores[:, truth].mean()),
        })

    return rows


# ------------------------------------------------------------------- metrics


# Lives in `session_analysis` so the calibration study can use the same definition without
# importing this module -- which would be circular, since this one imports
# `calibration_study`. Re-exported here because that is where it was first written and where
# `test_subject_variability.py` reaches for it.
macro_f1 = sa.macro_f1


def per_participant(rows: list[dict], classes: list[str]) -> list[dict]:
    out = []
    for participant in sorted({r['participant'] for r in rows}):
        mine = [r for r in rows if r['participant'] == participant]
        scored = [r for r in mine if r['predicted'] is not None]

        truth = [r['emotion'] for r in scored]
        predicted = [r['predicted'] for r in scored]
        metrics = macro_f1(truth, predicted, classes)

        windows = sum(r['windows'] for r in mine)
        window_correct = sum(r['window_correct'] for r in mine)
        confidences = [r['mean_confidence'] for r in scored if r['mean_confidence'] is not None]

        out.append({
            'participant': participant,
            'split': mine[0]['split'],
            'recordings': len(mine),
            'unscored': len(mine) - len(scored),
            'trial_correct': sum(1 for r in scored if r['correct']),
            'trial_accuracy': (sum(1 for r in scored if r['correct']) / len(scored)
                               if scored else None),
            'macro_f1': metrics['macro_f1'],
            'classes_scored': metrics['classes_scored'],
            'windows': windows,
            'window_accuracy': window_correct / windows if windows else None,
            'mean_confidence': float(np.mean(confidences)) if confidences else None,
            'per_class': metrics['per_class'],
        })
    return out


def pooled_confusion(rows: list[dict], classes: list[str]) -> dict[str, dict[str, int]]:
    matrix = {t: {p: 0 for p in classes} for t in classes}
    for r in rows:
        if r['predicted'] is not None:
            matrix[r['emotion']][r['predicted']] += 1
    return matrix


# ------------------------------------------------------------------ printing


def print_fidelity(check: dict) -> None:
    print(rule('replay fidelity (live sessions, where logged scores exist)'))
    if not check['ran']:
        print(f"  NOT RUN - {check['reason']}")
        print('  Nothing downstream has been checked against a known-good output.')
        print()
        return
    for s in check['sessions']:
        flag = '' if (s['agreement'] == 1.0 and s['max_diff'] < check['tolerance']) \
               else '   <-- MISMATCH'
        print(f"  {s['session'][:30]:<32} {s['rows']:>4} rows  "
              f"agreement {num(s['agreement'])}  max diff {s['max_diff']:.2e}{flag}")
    print(f"  {'passed' if check['passed'] else 'FAILED'}")
    print()


def print_condition(name: str, rows: list[dict], classes: list[str], splits: tuple[str, ...],
                    heading: str) -> None:
    subset = [r for r in rows if r['split'] in splits]
    stats = per_participant(subset, classes)

    total_windows = sum(r['windows'] for r in subset)
    offered = sum(r.get('windows_offered', r['windows']) for r in subset)
    unscored = sum(1 for r in subset if r['predicted'] is None)

    print(rule(f'{name} - {heading}'))
    line = f"  {len(subset)} recordings, {total_windows} windows scored"
    if offered != total_windows:
        line += f" of {offered} offered  (coverage {pct(total_windows / offered)})"
    if unscored:
        line += f", {unscored} recording(s) unscored"
    print(line)
    if offered != total_windows:
        print('  Accuracy below is measured only on the windows that were answered for.')
        print('  It is not comparable to a condition that answers for everything.')
    print()
    print(f"  {'participant':<16} {'split':<11} {'trials':>7} {'acc':>6} "
          f"{'macroF1':>8} {'windows':>8} {'win acc':>8} {'conf':>6}")
    for s in stats:
        trials = f"{s['trial_correct']}/{s['recordings'] - s['unscored']}"
        print(f"  {s['participant']:<16} {s['split']:<11} {trials:>7} "
              f"{num(s['trial_accuracy'], 2):>6} {num(s['macro_f1'], 3):>8} "
              f"{s['windows']:>8} {num(s['window_accuracy'], 3):>8} "
              f"{num(s['mean_confidence'], 2):>6}")

    accuracies = [s['trial_accuracy'] for s in stats if s['trial_accuracy'] is not None]
    if accuracies:
        print(f"  {'':<16} {'':<11} {'':>7} {np.mean(accuracies):>6.2f}  <- mean over "
              f"{len(accuracies)} participants, not a pooled score")
    print()


def print_per_class(rows: list[dict], classes: list[str], splits: tuple[str, ...]) -> None:
    subset = [r for r in rows if r['split'] in splits and r['predicted'] is not None]
    metrics = macro_f1([r['emotion'] for r in subset],
                       [r['predicted'] for r in subset], classes)
    print(rule('per class, pooled over held-out participants'))
    print(f"  {'class':<12} {'support':>8} {'recall':>8} {'precision':>10} {'f1':>7}")
    for c in classes:
        m = metrics['per_class'][c]
        print(f"  {c:<12} {m['support']:>8} {num(m['recall'], 2):>8} "
              f"{num(m['precision'], 2):>10} {num(m['f1'], 2):>7}")
    print(f"  macro-F1 over {metrics['classes_scored']} scored classes: "
          f"{num(metrics['macro_f1'], 3)}")
    print()


def print_confusion(matrix: dict, classes: list[str]) -> None:
    short = {c: c[:3] for c in classes}
    print(rule('confusion, held-out participants (rows prompted, columns predicted)'))
    print(f"  {'prompted':<12}" + ''.join(f"{short[c]:>6}" for c in classes) + f"{'n':>6}")
    for t in classes:
        row = matrix[t]
        n = sum(row.values())
        cells = ''.join(f"{(row[p] if row[p] else '.'):>6}" for p in classes)
        print(f"  {t:<12}{cells}{n:>6}")
    print()


def print_limits() -> None:
    print(rule('limits'))
    print('  Held-out results rest on 5 participants and 35 recordings. Everything here is')
    print('  exploratory; none of it is a confirmatory test.')
    print()
    print('  One trial per participant per emotion, so a subject-level 7-class macro-F1 is')
    print('  built from seven single-recording classes. It is granular to about 1/7 and is')
    print('  close to a restatement of trial accuracy. Read them together.')
    print()
    print('  notebook_ungated is NOT a runtime measurement: it averages six frames, which')
    print('  the headset never does. runtime_proxy is the closest stand-in, and is still a')
    print('  proxy - the device polls against live tracking and skips rejected frames.')
    print()
    print('  Train participants are in the model\'s training data. Their numbers describe')
    print('  memorisation, not generalisation, and are never pooled with held-out numbers.')
    print()


# ---------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', help='path to ActingRecordings')
    parser.add_argument('--model', help='path to sparseVR_teacher_model.onnx')
    parser.add_argument('--conditions', default=','.join(tp.CONDITIONS),
                        help='comma-separated subset of ' + ', '.join(tp.CONDITIONS))
    parser.add_argument('--sensitivity', action='store_true',
                        help='also run the self-report-consistent subset (a natural-data '
                             'filter, shown only as a check)')
    parser.add_argument('--json', help='write every metric to this file')
    args = parser.parse_args(argv)

    try:
        corpus = ad.load_corpus(ad.find_corpus_root(args.data))
        model_path = cs.find_model(args.model)
    except (ad.CorpusError, SystemExit) as e:
        print(f'error: {e}', file=sys.stderr)
        return 2

    print('=' * WIDTH)
    print('phase B - per-participant model performance')
    print(f'model  {model_path}')
    print('=' * WIDTH)
    print()

    check = fidelity_check(model_path)
    print_fidelity(check)
    if check['ran'] and not check['passed']:
        print('refusing to report performance: the ONNX path does not reproduce the '
              'scores the headset logged.', file=sys.stderr)
        return 2

    conditions = [c.strip() for c in args.conditions.split(',') if c.strip()]
    for condition in conditions:
        if condition not in tp.CONDITIONS:
            print(f'error: unknown condition {condition!r}', file=sys.stderr)
            return 2

    results = {}
    for condition in conditions:
        rows = predict_recordings(corpus, condition, model_path)

        print('=' * WIDTH)
        print(f'condition: {condition}')
        print('=' * WIDTH)
        print()

        print_condition(condition, rows, CLASS_ORDER, ('validation', 'test'),
                        'HELD OUT - the only participants whose numbers describe generalisation')
        print_per_class(rows, CLASS_ORDER, ('validation', 'test'))
        print_confusion(pooled_confusion([r for r in rows if r['split'] in ('validation', 'test')],
                                         CLASS_ORDER), CLASS_ORDER)
        print_condition(condition, rows, CLASS_ORDER, ('train',),
                        'TRAINING DATA - memorisation, shown for contrast only')

        if args.sensitivity:
            kept = [r for r in rows if not r['excluded']]
            print_condition(condition + ' [sensitivity]', kept, CLASS_ORDER,
                            ('validation', 'test'),
                            'self-report-consistent subset, not the headline')

        results[condition] = {
            'recordings': rows,
            'held_out': per_participant(
                [r for r in rows if r['split'] in ('validation', 'test')], CLASS_ORDER),
            'train': per_participant([r for r in rows if r['split'] == 'train'], CLASS_ORDER),
            'confusion_held_out': pooled_confusion(
                [r for r in rows if r['split'] in ('validation', 'test')], CLASS_ORDER),
        }

    print_limits()

    if args.json:
        Path(args.json).write_text(json.dumps({
            'seed': SEED,
            'model': str(model_path),
            'fidelity': check,
            'class_order': CLASS_ORDER,
            'conditions': results,
        }, indent=2, default=str), encoding='utf-8')
        print(f'wrote {args.json}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
