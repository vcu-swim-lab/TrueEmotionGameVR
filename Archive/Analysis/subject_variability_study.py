#!/usr/bin/env python3
"""Why does the model work for some people and not others?

    python Analysis/subject_variability_study.py
    python Analysis/subject_variability_study.py --json study.json

Phase A of the study described in the approved plan: an inventory of the acted corpus and a
descriptive analysis of how much action-unit behaviour varies between people, between
emotions, and within a single trial.

The question behind it. Three live sessions gave 61.7% and 65.0% for one subject and 20.0%
for another, against 14.3% chance -- and the failing subject was not under-expressing. Is
that a **measurement** difference (neutral offset, gain, headset fit, dead channels) or an
**expression-style** difference (people using different muscles for the same emotion)? Only
the first is safe to normalise away. The second is the signal.

This phase deliberately reports no model accuracy and fits no calibration. It establishes
what the data can support before anything is fitted to it. Two limits are load-bearing and
are printed on every run:

- **One trial per participant per emotion.** No within-person repeatability, no ICC. Nothing
  here substitutes for that; within-trial stability is measured and is named as such.
- **The primary set is all 105 prompted recordings.** The teacher notebook applies its
  self-report filter to the natural recordings only, and asserts that it changes no acted
  recording -- "Acted recordings are instructed poses, so they are kept unfiltered." Someone
  asked to pose anger was not asked to feel it. The filtered subset is still computed, as a
  clearly labelled sensitivity analysis, because it is a fair question whether a participant
  who did not feel an emotion also posed it badly. It is never the headline.

Needs numpy. See SUBJECT_VARIABILITY_REPORT.md for what the numbers mean.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

import acting_dataset as ad
import au_names
from analyze_sessions import WIDTH, num, pct, rule

SEED = 42

# A channel counts as moving when it exceeds this. The exported floats bottom out at
# denormals rather than at zero, so a bare `> 0` would call every channel active.
ACTIVE_THRESHOLD = 0.05

# Matches the live-session analysis, where anything under this is treated as flat.
FLAT_EPSILON = 1e-6


# ------------------------------------------------------------------ inventory


def inventory(corpus: ad.Corpus) -> dict:
    """What is actually in the corpus, measured rather than assumed."""
    rates = [r.sample_rate for r in corpus.recordings]
    jitter = [r.median_interval_rate for r in corpus.recordings]
    durations = [r.duration for r in corpus.recordings]
    frames = [r.frames for r in corpus.recordings]
    tracking = [r.tracking_rate for r in corpus.recordings]

    excluded_by_emotion: dict[str, int] = {e: 0 for e in ad.EMOTIONS}
    for r in corpus.recordings:
        if r.excluded:
            excluded_by_emotion[r.emotion] += 1

    splits: dict[str, list[str]] = {}
    for r in corpus.recordings:
        splits.setdefault(r.split, [])
        if r.participant not in splits[r.split]:
            splits[r.split].append(r.participant)

    return {
        'domain': corpus.domain,
        'root': str(corpus.root),
        'participants': len(corpus.participants),
        'recordings': len(corpus.recordings),
        'emotions': list(ad.EMOTIONS),
        'neutral_present': any(r.emotion == 'Neutral' for r in corpus.recordings),
        'channels': au_names.CHANNEL_COUNT,
        'participant_id_source': 'directory name (capture timestamp)',
        'label_source': 'directory name, cross-checked against the Emotion column',
        'sample_rate_hz': {
            'median': float(np.median(rates)),
            'min': float(np.min(rates)),
            'max': float(np.max(rates)),
        },
        'median_interval_rate_hz': {
            'median': float(np.median(jitter)),
            'note': 'jitter diagnostic only - the capture is bursty, so this overstates '
                    'the rate by ~18%. The teacher pipeline averages 6 rows to reach 10 Hz, '
                    'which works because the rate above really is ~60.',
        },
        'duration_s': {
            'median': float(np.median(durations)),
            'min': float(np.min(durations)),
            'max': float(np.max(durations)),
        },
        'frames': {
            'median': int(np.median(frames)),
            'min': int(np.min(frames)),
            'max': int(np.max(frames)),
        },
        'tracking_rate': {
            'median': float(np.median(tracking)),
            'min': float(np.min(tracking)),
        },
        'trials_per_participant_emotion': 1,
        'downsampling_applied_here': 'none - raw capture rate is analysed',
        'splits': {k: sorted(v) for k, v in splits.items()},
        'excluded_total': sum(excluded_by_emotion.values()),
        'excluded_by_emotion': excluded_by_emotion,
        'warnings': corpus.warnings,
    }


def coverage_table(corpus: ad.Corpus) -> dict:
    """Which emotions survive per participant, under each estimand."""
    e1 = corpus.coverage()
    e2 = corpus.coverage(self_report_consistent=True)
    rows = []
    for p in corpus.participants:
        recordings = [r for r in corpus.recordings if r.participant == p]
        rows.append({
            'participant': p,
            'split': recordings[0].split,
            'split_source': recordings[0].split_source,
            'e1_count': len(e1.get(p, ())),
            'e2_count': len(e2.get(p, ())),
            'e2_missing': sorted(set(ad.EMOTIONS) - e2.get(p, set())),
        })
    return {'rows': rows}


# ----------------------------------------------------------------- profiles


def _baseline(corpus: ad.Corpus, participant: str) -> np.ndarray | None:
    """A participant's own resting channel vector, from their Neutral recording.

    Used only descriptively here. It is NOT a validated calibration: the Neutral trial is the
    only resting data a participant has, so subtracting it and then scoring Neutral on that
    same trial would be circular. That test belongs to the calibration-v1 protocol.
    """
    for r in corpus.recordings:
        if r.participant == participant and r.emotion == 'Neutral':
            valid = r.valid_au
            return valid.mean(axis=0) if len(valid) else None
    return None


def profile(recording: ad.Recording, baseline: np.ndarray | None) -> dict:
    """Static and dynamic per-channel description of one trial."""
    au = recording.valid_au
    if len(au) < 3:
        return {'usable': False, 'frames': len(au)}

    times = recording.times[recording.face_valid]
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0

    peak = au.max(axis=0)
    mean = au.mean(axis=0)

    # Velocity and acceleration in per-second units, so recordings captured at slightly
    # different rates stay comparable -- the capture runs on Unity's update loop and the
    # median rate varies between recordings.
    velocity = np.diff(au, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt

    peak_frame = au.argmax(axis=0)
    time_to_peak = times[peak_frame] - times[0]

    # Stability across the held portion of the trial. This is a within-trial measure and is
    # never to be read as repeatability: the segments are consecutive moments of one
    # continuous expression, not independent attempts.
    half = len(au) // 2
    stability = np.where(
        peak > ACTIVE_THRESHOLD,
        1.0 - np.abs(au[half:].mean(axis=0) - au[:half].mean(axis=0)) / np.maximum(peak, 1e-6),
        np.nan,
    )

    result = {
        'usable': True,
        'frames': int(len(au)),
        'peak': peak,
        'mean': mean,
        'median': np.median(au, axis=0),
        'std': au.std(axis=0),
        'max_velocity': velocity.max(axis=0),
        'max_acceleration': acceleration.max(axis=0),
        'time_to_peak': time_to_peak,
        'auc': np.trapezoid(au, times, axis=0) if hasattr(np, 'trapezoid')
               else np.trapz(au, times, axis=0),
        'within_trial_stability': stability,
        'active_channels': int((peak > ACTIVE_THRESHOLD).sum()),
        'flat_channels': int((peak <= FLAT_EPSILON).sum()),
    }

    if baseline is not None:
        result['minus_neutral'] = peak - baseline
        # Scaled to the person's own range, so a participant whose channels simply register
        # weaker is not counted as less expressive than one whose register stronger.
        span = np.maximum(peak.max(), 1e-6)
        result['self_normalised'] = peak / span

    return result


def build_profiles(corpus: ad.Corpus, *, self_report_consistent: bool) -> dict:
    baselines = {p: _baseline(corpus, p) for p in corpus.participants}
    out = {}
    for r in corpus.subset(self_report_consistent=self_report_consistent):
        p = profile(r, baselines.get(r.participant))
        if p['usable']:
            out[(r.participant, r.emotion)] = p
    return out


# ----------------------------------------------------------------- variance


def variance_comparison(profiles: dict, metric: str = 'peak',
                        rng: np.random.Generator | None = None,
                        bootstrap: int = 2000) -> dict:
    """Between-participant against between-emotion variance, per channel.

    The comparison the whole study turns on. For each channel:

    - **between-emotion** variance is how much the channel separates one emotion from
      another, averaged within participants. This is the signal the model needs.
    - **between-participant** variance is how much the same emotion looks different from one
      person to the next, averaged within emotions. This is what a calibration would try to
      remove.

    A channel whose between-participant variance dominates is one where identity drowns out
    emotion. A channel where the reverse holds is one the model can rely on.

    The bootstrap resamples **participants**, not frames or windows. Frames within a trial
    are consecutive samples of one continuous expression and resampling them would report a
    confidence interval many times narrower than the data earns.
    """
    rng = rng or np.random.default_rng(SEED)

    participants = sorted({p for p, _ in profiles})
    emotions = sorted({e for _, e in profiles})
    n_ch = au_names.CHANNEL_COUNT

    # [participant, emotion, channel], NaN where that recording is absent
    grid = np.full((len(participants), len(emotions), n_ch), np.nan)
    for (p, e), prof in profiles.items():
        grid[participants.index(p), emotions.index(e)] = prof[metric]

    def decompose(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # An all-NaN slice -- an emotion no remaining participant has, which happens both
        # under E2 and in a bootstrap draw -- yields NaN and is then skipped by the outer
        # nanmean. That is the intended behaviour: the sample shrinks rather than gaining a
        # fabricated zero. numpy announces it on every such slice, which would bury the
        # output, so the warning is silenced here rather than the condition avoided.
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            # Spread across people, for a fixed emotion, then averaged over emotions.
            between_participant = np.nanmean(np.nanvar(g, axis=0), axis=0)
            # Spread across emotions, for a fixed person, then averaged over people.
            between_emotion = np.nanmean(np.nanvar(g, axis=1), axis=0)
        return between_participant, between_emotion

    bp, be = decompose(grid)

    samples_bp = np.empty((bootstrap, n_ch))
    samples_be = np.empty((bootstrap, n_ch))
    for i in range(bootstrap):
        pick = rng.integers(0, len(participants), len(participants))
        s_bp, s_be = decompose(grid[pick])
        samples_bp[i], samples_be[i] = s_bp, s_be

    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = be / np.maximum(bp, 1e-12)
        ratio_samples = samples_be / np.maximum(samples_bp, 1e-12)

    return {
        'metric': metric,
        'participants': participants,
        'emotions': emotions,
        'bootstrap': bootstrap,
        'resampled': 'participants',
        'between_participant': bp,
        'between_emotion': be,
        'ratio': ratio,
        'ratio_ci': (np.nanpercentile(ratio_samples, 2.5, axis=0),
                     np.nanpercentile(ratio_samples, 97.5, axis=0)),
        'coverage': (~np.isnan(grid)).sum(axis=(0, 1)),
    }


def profile_similarity(profiles: dict, metric: str = 'peak') -> dict:
    """How alike are two people's profiles for the same emotion?

    Cosine between participants' per-channel profiles, one emotion at a time. A low value
    means two people produced the same prompted emotion with different channels -- the
    "expression idiolect" reading. It cannot on its own distinguish that from a measurement
    difference, which is what the bilateral-asymmetry and dead-channel diagnostics are for.
    """
    emotions = sorted({e for _, e in profiles})
    expression = list(au_names.EXPRESSION_INDICES)
    out = {}

    for emotion in emotions:
        people = sorted(p for p, e in profiles if e == emotion)
        vectors = {p: profiles[(p, emotion)][metric][expression] for p in people}
        pairs = []
        for i, a in enumerate(people):
            for b in people[i + 1:]:
                va, vb = vectors[a], vectors[b]
                denom = np.linalg.norm(va) * np.linalg.norm(vb)
                pairs.append(float(va @ vb / denom) if denom > 0 else np.nan)
        out[emotion] = {
            'n_participants': len(people),
            'n_pairs': len(pairs),
            'mean_cosine': float(np.nanmean(pairs)) if pairs else None,
            'min_cosine': float(np.nanmin(pairs)) if pairs else None,
            'max_cosine': float(np.nanmax(pairs)) if pairs else None,
        }
    return out


def measurement_diagnostics(corpus: ad.Corpus, profiles: dict) -> dict:
    """Signals that point at the sensor rather than at the person.

    Bilateral asymmetry is the most informative of these. Faces are roughly symmetric, so a
    channel pair that stays lopsided across every emotion is more readily explained by
    headset fit or camera coverage than by how someone expresses anything.
    """
    rows = []
    for participant in corpus.participants:
        peaks = [prof['peak'] for (p, _), prof in profiles.items() if p == participant]
        if not peaks:
            continue
        stacked = np.stack(peaks)
        session_peak = stacked.max(axis=0)

        asymmetry = []
        for left, right in au_names.BILATERAL_PAIRS:
            a, b = session_peak[left], session_peak[right]
            if max(a, b) > ACTIVE_THRESHOLD:
                asymmetry.append(abs(a - b) / max(a, b))

        recordings = [r for r in corpus.recordings if r.participant == participant]
        rows.append({
            'participant': participant,
            'split': recordings[0].split,
            'dead_channels': int((session_peak[list(au_names.EXPRESSION_INDICES)]
                                  <= FLAT_EPSILON).sum()),
            'active_channels': int((session_peak[list(au_names.EXPRESSION_INDICES)]
                                    > ACTIVE_THRESHOLD).sum()),
            'overall_peak': float(session_peak[list(au_names.EXPRESSION_INDICES)].max()),
            'mean_bilateral_asymmetry': float(np.mean(asymmetry)) if asymmetry else None,
            'tracking_rate': float(np.mean([r.tracking_rate for r in recordings])),
        })
    return {'rows': rows}


# ----------------------------------------------------------------- printing


def print_inventory(inv: dict) -> None:
    print(rule('inventory'))
    print(f"  domain                {inv['domain']}")
    print(f"  root                  {inv['root']}")
    print(f"  participants          {inv['participants']}")
    print(f"  recordings            {inv['recordings']}  "
          f"({inv['trials_per_participant_emotion']} trial per participant per emotion)")
    print(f"  emotions              {len(inv['emotions'])}  "
          f"Neutral present: {'yes' if inv['neutral_present'] else 'no'}")
    print(f"  channels              {inv['channels']}")
    print(f"  participant id from   {inv['participant_id_source']}")
    print(f"  label from            {inv['label_source']}")
    r = inv['sample_rate_hz']
    print(f"  sample rate           median {r['median']:.2f} Hz  "
          f"(min {r['min']:.2f}, max {r['max']:.2f})   frames / elapsed time")
    j = inv['median_interval_rate_hz']
    print(f"  median frame gap      {j['median']:.2f} Hz equivalent  "
          f"-- jitter only, not the rate")
    d = inv['duration_s']
    print(f"  duration              median {d['median']:.2f} s  "
          f"(min {d['min']:.2f}, max {d['max']:.2f})")
    t = inv['tracking_rate']
    print(f"  face tracking         median {pct(t['median'])}  worst {pct(t['min'])}")
    print(f"  downsampling          {inv['downsampling_applied_here']}")
    print()
    print('  splits')
    for split in ('train', 'validation', 'test', 'unknown'):
        if split in inv['splits']:
            names = inv['splits'][split]
            print(f"    {split:<12} {len(names):>2}  {', '.join(names)}")
    print()
    print(f"  self-report exclusions  {inv['excluded_total']} of {inv['recordings']}")
    for emotion, count in sorted(inv['excluded_by_emotion'].items(),
                                 key=lambda kv: -kv[1]):
        if count:
            print(f"    {emotion:<12} {count:>2}  {'#' * count}")
    if inv['warnings']:
        print()
        for w in inv['warnings']:
            print(f"  ! {w}")
    print()


def print_coverage(cov: dict) -> None:
    print(rule('coverage per participant (E1 prompted / E2 self-report consistent)'))
    print(f"  {'participant':<16} {'split':<11} {'src':<8} {'E1':>3} {'E2':>3}  missing from E2")
    for row in cov['rows']:
        missing = ', '.join(row['e2_missing']) if row['e2_missing'] else ''
        flag = '  <-- lost entirely' if row['e2_count'] == 0 else ''
        print(f"  {row['participant']:<16} {row['split']:<11} {row['split_source']:<8} "
              f"{row['e1_count']:>3} {row['e2_count']:>3}  {missing}{flag}")
    print()


def print_variance(var: dict, top: int) -> None:
    print(rule(f"variance, {var['metric']} per channel "
               f"(bootstrap {var['bootstrap']} over {var['resampled']})"))
    print('  ratio = between-emotion / between-participant variance.')
    print('  Above 1: the channel separates emotions more than it separates people.')
    print('  Below 1: identity dominates, and the model cannot lean on it across subjects.')
    print()
    order = np.argsort(-np.nan_to_num(var['ratio'], nan=-np.inf))
    lo, hi = var['ratio_ci']

    print(f"  {'channel':<26} {'ratio':>7}  {'95% CI':>17}  {'betw-emo':>9} {'betw-part':>10}")
    for i in list(order[:top]):
        print(f"  {au_names.NAMES[i]:<26} {var['ratio'][i]:>7.2f}  "
              f"[{lo[i]:>6.2f}, {hi[i]:>6.2f}]  "
              f"{var['between_emotion'][i]:>9.4f} {var['between_participant'][i]:>10.4f}")
    print(f"  {'...':<26}")
    for i in list(order[-top:]):
        print(f"  {au_names.NAMES[i]:<26} {var['ratio'][i]:>7.2f}  "
              f"[{lo[i]:>6.2f}, {hi[i]:>6.2f}]  "
              f"{var['between_emotion'][i]:>9.4f} {var['between_participant'][i]:>10.4f}")
    print()


def print_similarity(sim: dict) -> None:
    print(rule('cross-participant profile similarity (cosine, expression channels)'))
    print('  How alike two people look when acting the same emotion. 1.00 = identical')
    print('  channel pattern. Low values are consistent with different expression styles,')
    print('  but a measurement difference produces them too - see the diagnostics below.')
    print()
    print(f"  {'emotion':<12} {'n':>3} {'pairs':>6} {'mean':>7} {'min':>7} {'max':>7}")
    for emotion, s in sorted(sim.items(), key=lambda kv: kv[1]['mean_cosine'] or 0):
        print(f"  {emotion:<12} {s['n_participants']:>3} {s['n_pairs']:>6} "
              f"{num(s['mean_cosine'], 3):>7} {num(s['min_cosine'], 3):>7} "
              f"{num(s['max_cosine'], 3):>7}")
    print()


def print_diagnostics(diag: dict) -> None:
    print(rule('measurement diagnostics per participant'))
    print('  Bilateral asymmetry is the mean relative left/right gap over 29 mirrored')
    print('  channel pairs, taken at each channel\'s session peak. Persistently lopsided')
    print('  channels point at headset fit rather than at expression.')
    print()
    print(f"  {'participant':<16} {'split':<11} {'active':>7} {'dead':>5} "
          f"{'peak':>6} {'asym':>6} {'track':>7}")
    for row in sorted(diag['rows'], key=lambda r: -(r['mean_bilateral_asymmetry'] or 0)):
        print(f"  {row['participant']:<16} {row['split']:<11} "
              f"{row['active_channels']:>7} {row['dead_channels']:>5} "
              f"{row['overall_peak']:>6.3f} "
              f"{num(row['mean_bilateral_asymmetry'], 3):>6} "
              f"{pct(row['tracking_rate']):>7}")
    print()


def print_limits() -> None:
    print(rule('limits - read before quoting anything above'))
    print('  One trial per participant per emotion. No within-person repeatability and no')
    print('  ICC is computable from this corpus, and nothing here substitutes for one.')
    print('  Within-trial stability measures one continuous expression, not repeats.')
    print()
    print('  The primary set is all 105 prompted recordings, matching the teacher pipeline,')
    print('  which filters on self-report only for the natural recordings. Every held-out')
    print('  participant therefore keeps all seven emotions. The sensitivity subset drops 29')
    print('  recordings very unevenly and leaves no held-out participant with seven classes,')
    print('  so it carries no seven-class subject-level score and must not be given one.')
    print()
    print('  This phase reports no model accuracy and fits no calibration.')
    print()


# ---------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', help='path to ActingRecordings')
    parser.add_argument('--metric', default='peak',
                        choices=['peak', 'mean', 'auc', 'max_velocity'],
                        help='per-channel statistic the variance analysis uses')
    parser.add_argument('--estimand', default='both', choices=['E1', 'E2', 'both'])
    parser.add_argument('--top', type=int, default=8, help='channels listed per table')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--json', help='write every metric to this file')
    args = parser.parse_args(argv)

    try:
        root = ad.find_corpus_root(args.data)
        corpus = ad.load_corpus(root)
    except ad.CorpusError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2

    print('=' * WIDTH)
    print('subject variability study - phase A (inventory and description)')
    print('=' * WIDTH)
    print()

    inv = inventory(corpus)
    print_inventory(inv)
    cov = coverage_table(corpus)
    print_coverage(cov)

    estimands = ['E1', 'E2'] if args.estimand == 'both' else [args.estimand]
    results = {}

    for estimand in estimands:
        consistent = estimand == 'E2'
        profiles = build_profiles(corpus, self_report_consistent=consistent)

        label = ('PRIMARY - every prompted recording, as the teacher pipeline uses them'
                 if estimand == 'E1'
                 else 'SENSITIVITY - self-report consistent subset '
                      '(a natural-data filter, applied here only as a check)')
        print('=' * WIDTH)
        print(f'{estimand}: {label}   ({len(profiles)} recordings)')
        print('=' * WIDTH)
        print()

        var = variance_comparison(profiles, args.metric,
                                  np.random.default_rng(SEED), args.bootstrap)
        sim = profile_similarity(profiles, args.metric)
        diag = measurement_diagnostics(corpus, profiles)

        print_variance(var, args.top)
        print_similarity(sim)
        print_diagnostics(diag)

        results[estimand] = {
            'recordings': len(profiles),
            'variance': {
                'metric': var['metric'],
                'bootstrap': var['bootstrap'],
                'resampled': var['resampled'],
                'channels': {
                    au_names.NAMES[i]: {
                        'between_participant': float(var['between_participant'][i]),
                        'between_emotion': float(var['between_emotion'][i]),
                        'ratio': float(var['ratio'][i]),
                        'ratio_ci': [float(var['ratio_ci'][0][i]),
                                     float(var['ratio_ci'][1][i])],
                        'coverage': int(var['coverage'][i]),
                    }
                    for i in range(au_names.CHANNEL_COUNT)
                },
            },
            'similarity': sim,
            'diagnostics': diag,
        }

    print_limits()

    if args.json:
        Path(args.json).write_text(json.dumps({
            'seed': SEED,
            'inventory': inv,
            'coverage': cov,
            'estimands': results,
        }, indent=2, default=str), encoding='utf-8')
        print(f'wrote {args.json}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
