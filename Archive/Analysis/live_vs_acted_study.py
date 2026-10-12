#!/usr/bin/env python3
"""Are the live players' expression profiles near the acted training distribution?

Three live sessions exist. Two are one person who scores well against the prompt, one is a
second person who scores near chance. The obvious hypothesis, and the only one this script
tests, is:

    the model works on A and not on B because A's per-emotion channel profiles sit closer to
    the acted participants the teacher was fitted on, and B's sit further away.

This is a *descriptive* comparison. Nothing is fitted, no calibration is applied, and no
model is re-run: live predictions come from the logs the headset wrote, and acted predictions
are read back out of `results/model_performance.json`. The output is a position in a
distribution, not a decision.

Three things decide whether the comparison means anything, and each is enforced rather than
assumed:

**Two live people, not three.** `session_20260912_213907_eec228` and
`session_20260913_005336_07c050` are the same person on two evenings. Counting them as two
subjects would double that person's weight and manufacture the very contrast being tested.
They are profiled separately, then averaged with equal weight per session -- not by pooling
frames, which would hand the longer session more say.

**"Training distribution" means the ten `train` participants.** Validation and test
participants are acted data the model was *not* fitted on, so including them would answer a
different question. They are carried as secondary context under their own name.

**The live subject never enters a centroid it is then scored against.** Centroids are built
from acted participants only. The reference distribution for "how far out is this person" is
built by leave-one-out over the training participants, so a training participant is compared
against a centroid of the other nine -- and, so the two numbers are actually comparable, the
live subject is compared against those same nine-member centroids.

Reproduce:

    python Analysis/live_vs_acted_study.py --json results/live_vs_acted.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import acting_dataset as ad          # noqa: E402
import au_names                       # noqa: E402
import session_analysis as sa         # noqa: E402
import subject_variability_study as sv  # noqa: E402
import teacher_preprocessing as tp    # noqa: E402

# The game prompts six emotions; the acted corpus has seven. Neutral is the odd one out and
# is dropped from every comparison here rather than half-handled. It is not merely absent
# from the live data -- `teacher_preprocessing.activity_labels` *inverts* the activity rule
# for it, so an acted Neutral profile is built under a different definition of a usable
# frame. Nothing in this file supports a conclusion about Neutral.
COMMON_EMOTIONS: tuple[str, ...] = (
    'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise',
)

TRAINING_SPLIT = 'train'

# Tick 0 is caught at the instant the prompt appears, before the player has begun to act, and
# it carries whatever the face was doing on the way out of the previous trial. Phase A's
# calibration pre-experiment found including it actively harmful (see README, "A result worth
# not repeating"), so it is excluded from every live profile here.
TRANSITION_TICK = 0

# Which live sessions belong to which person. This mapping is the single place the
# two-sessions-one-person fact is recorded, and `test_live_vs_acted_study.py` asserts against
# it: if it is ever wrong, every number in the report is wrong in the same direction.
LIVE_SUBJECTS: dict[str, tuple[str, ...]] = {
    'A': ('20260912_213907_eec228', '20260913_005336_07c050'),
    'B': ('20260914_175512_8ccb69',),
}

# Which Phase B condition to read acted per-recording verdicts from. `notebook_exact` is the
# training distribution -- the condition under which the acted recordings were actually seen.
DEFAULT_CONDITION = 'notebook_exact'

EXPRESSION = list(au_names.EXPRESSION_INDICES)


class StudyError(RuntimeError):
    """Raised when the inputs are not shaped the way this comparison requires.

    Every path that could otherwise return a quiet zero -- a missing emotion, a missing
    session, an empty profile -- raises this instead. A cosine of 0.0 and "this person has no
    Fear recording" are the same number and opposite findings.
    """


# --------------------------------------------------------------------- live side


def resolve_live_sessions(root: str | Path | None) -> dict[str, list[Path]]:
    """Locate each subject's session files, by the ids in `LIVE_SUBJECTS`.

    Reuses `session_analysis` discovery rather than hardcoding a path, so this runs on
    another machine that has pulled the same sessions somewhere else.
    """
    files = sa.discover([root]) if root else sa.autodiscover()[0]

    out: dict[str, list[Path]] = {}
    for subject, ids in LIVE_SUBJECTS.items():
        resolved = []
        for session_id in ids:
            matches = [f for f in files if session_id in str(f)]
            if not matches:
                raise StudyError(
                    f'live session {session_id} (subject {subject}) not found. '
                    f'Searched {len(files)} file(s). Pass --sessions PATH.')
            if len(matches) > 1:
                raise StudyError(
                    f'live session {session_id} matched {len(matches)} files: '
                    + ', '.join(str(m) for m in matches))
            resolved.append(matches[0])
        out[subject] = resolved
    return out


def live_profiles(session: sa.Session) -> dict[str, dict]:
    """Per-emotion peak channel profile for one live session.

    Built from the exact 5x70 windows the headset logged, which are the frames the model
    actually saw -- not a re-read of raw tracking. `peak` is the per-channel maximum over
    every frame of every scored tick, which is the same statistic
    `subject_variability_study.profile` takes over an acted recording's valid frames.

    Ticks are selected on the `prediction_index` column, never by enumerating rows. Only
    `ok` rows carry a window, so a mid-trial `no_tracking` tick would shift an enumerated
    timeline and silently move ticks across the tick-0 boundary.
    """
    if session.schema.features_per_frame != au_names.CHANNEL_COUNT:
        raise StudyError(
            f'{session.path}: {session.schema.features_per_frame} channels per frame, '
            f'expected {au_names.CHANNEL_COUNT}')

    missing_index = [r for r in session.rows if r.ok and r.prediction_index is None]
    if missing_index:
        raise StudyError(
            f'{session.path}: {len(missing_index)} scored row(s) have no prediction_index, '
            'so the transition tick cannot be identified. Counting rows instead would be '
            'wrong wherever a tick was dropped.')

    out: dict[str, dict] = {}
    for emotion in COMMON_EMOTIONS:
        prompted = [r for r in session.rows if r.prompted == emotion]
        if not prompted:
            raise StudyError(f'{session.path}: no {emotion} trial')

        kept = [r for r in prompted if r.ok and r.prediction_index > TRANSITION_TICK]
        if not kept:
            raise StudyError(
                f'{session.path}: {emotion} has no scored tick after the transition tick')

        frames = np.array([f for r in kept for f in r.frames], dtype=np.float64)
        if not np.isfinite(frames).all():
            raise StudyError(f'{session.path}: {emotion} window contains a non-finite value')

        peak = frames.max(axis=0)
        out[emotion] = {
            'peak': peak,
            'mean': frames.mean(axis=0),
            'ticks': len(kept),
            'ticks_prompted': len(prompted),
            'frames': int(len(frames)),
            'active_channels': int((peak[EXPRESSION] > sv.ACTIVE_THRESHOLD).sum()),
            'flat_channels': int((peak[EXPRESSION] <= sv.FLAT_EPSILON).sum()),
        }
    return out


def combine_sessions(per_session: list[dict[str, dict]]) -> dict[str, dict]:
    """One profile per emotion for a subject, equal weight per session.

    Averaging the two session *profiles* rather than pooling their frames is the point. A
    pooled maximum would be dominated by whichever session happened to contain the single
    largest reading, and a pooled mean would weight the longer session more. Equal weight per
    session is what "this is one person, sampled twice" means.
    """
    if not per_session:
        raise StudyError('no sessions to combine')

    out: dict[str, dict] = {}
    for emotion in COMMON_EMOTIONS:
        present = [p for p in per_session if emotion in p]
        if len(present) != len(per_session):
            raise StudyError(
                f'{emotion} is missing from {len(per_session) - len(present)} of this '
                f"subject's {len(per_session)} session(s); refusing to average over a hole")
        out[emotion] = {
            'peak': np.mean([p[emotion]['peak'] for p in present], axis=0),
            'mean': np.mean([p[emotion]['mean'] for p in present], axis=0),
            'sessions': len(present),
            'ticks': sum(p[emotion]['ticks'] for p in present),
            'frames': sum(p[emotion]['frames'] for p in present),
        }
    return out


# -------------------------------------------------------------------- acted side


def acted_profiles(corpus: ad.Corpus, sampling: str) -> dict[tuple[str, str], np.ndarray]:
    """Peak channel profile per (participant, emotion), over the six common emotions.

    `sampling` exists because the two corpora are not sampled alike, and a per-channel
    *maximum* is exactly the statistic that cares:

    - `full` reuses `subject_variability_study.profile`, i.e. the peak over every frame the
      tracker vouched for. An acted recording holds ~602 such frames.
    - `decimated` takes the peak over `teacher_preprocessing.decimate`, every 6th frame,
      which is the 10 Hz spacing the headset itself samples at. ~100 frames.

    A live profile is built from ~45 frames. A maximum taken over 602 draws sits higher than
    one taken over 45 on any channel with noise in it, which inflates noisy channels relative
    to consistently-used ones and can rotate the profile vector. `decimated` does not remove
    that gap, it narrows it from ~13x to ~2x, and the report carries both so the reader can
    see whether any conclusion depends on the choice.
    """
    if sampling not in ('full', 'decimated'):
        raise StudyError(f'unknown sampling {sampling!r}; expected full or decimated')

    out: dict[tuple[str, str], np.ndarray] = {}
    for recording in corpus.recordings:
        if recording.emotion not in COMMON_EMOTIONS:
            continue

        if sampling == 'full':
            prof = sv.profile(recording, baseline=None)
            if not prof['usable']:
                continue
            out[recording.key()] = prof['peak']
        else:
            valid = recording.valid_au
            if len(valid) < 3:
                continue
            sampled = tp.decimate(valid)
            if len(sampled) == 0:
                continue
            out[recording.key()] = sampled.max(axis=0)

    return out


def participants_in_split(corpus: ad.Corpus, split: str) -> list[str]:
    return sorted({r.participant for r in corpus.recordings if r.split == split})


def centroid(profiles: dict[tuple[str, str], np.ndarray],
             participants: list[str], emotion: str) -> np.ndarray:
    """Equal-weight mean over participants' profiles for one emotion.

    Equal weight per *participant*, which is what makes this a population profile rather than
    a profile of whoever produced the most data. Raises rather than dropping a participant
    silently: a centroid quietly built from eight of ten people is not the thing the
    leave-one-out reference below assumes it is.
    """
    if not participants:
        raise StudyError(f'no participants for the {emotion} centroid')

    missing = [p for p in participants if (p, emotion) not in profiles]
    if missing:
        raise StudyError(
            f'{emotion}: no usable profile for {", ".join(missing)}. '
            'A centroid over the rest would silently answer a different question.')

    return np.mean([profiles[(p, emotion)] for p in participants], axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine over the expression channels only.

    `au_names.EXPRESSION_INDICES` drops gaze direction and the tongue channels. Gaze is where
    someone was looking, not how their face was set; leaving it in would let a player who
    looked around the scene register as expressively different.
    """
    va, vb = np.asarray(a)[EXPRESSION], np.asarray(b)[EXPRESSION]
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 0:
        raise StudyError('cosine against an all-zero profile is undefined')
    return float(va @ vb / denom)


# ------------------------------------------------------- position in the distribution


def loo_reference(profiles: dict[tuple[str, str], np.ndarray],
                  train: list[str], emotion: str) -> list[dict]:
    """Each training participant against a centroid of the other nine.

    This is the reference distribution. It answers "how close to the training centroid does a
    person who belongs to the training set normally sit?", and it has to be leave-one-out: a
    participant compared against a centroid they are inside is compared partly against
    themselves, and would look closer than any outsider could ever be.
    """
    out = []
    for held in train:
        others = [p for p in train if p != held]
        out.append({
            'participant': held,
            'cosine': cosine(profiles[(held, emotion)], centroid(profiles, others, emotion)),
        })
    return sorted(out, key=lambda r: r['cosine'])


def position_in_training(live: np.ndarray,
                         profiles: dict[tuple[str, str], np.ndarray],
                         train: list[str], emotion: str) -> dict:
    """Where the live profile falls against the leave-one-out reference.

    The live subject is scored against the *same* nine-member centroids the reference is
    built from -- once per held-out training participant, then averaged. Comparing it against
    the full ten-member centroid instead would put a ten-person centroid on one side of the
    comparison and a nine-person centroid on the other, and the difference goes the wrong way
    (a larger centroid is smoother, so it scores higher) exactly where the answer matters.

    The full-centroid number is reported too, because it is the one a reader expects, but the
    percentile is computed from the matched one.

    Ten reference values means the percentile is granular to 10 points and cannot be read more
    finely than "inside the range" or "below everyone".
    """
    reference = loo_reference(profiles, train, emotion)
    values = [r['cosine'] for r in reference]

    matched = [cosine(live, centroid(profiles, [p for p in train if p != held], emotion))
               for held in train]
    matched_mean = float(np.mean(matched))

    below = sum(1 for v in values if v < matched_mean)
    return {
        'cosine_to_full_centroid': cosine(live, centroid(profiles, train, emotion)),
        'cosine_matched_loo': matched_mean,
        'reference': reference,
        'reference_min': float(min(values)),
        'reference_median': float(np.median(values)),
        'reference_max': float(max(values)),
        'n_reference_below': below,
        'n_reference': len(values),
        'percentile': 100.0 * below / len(values),
        'inside_reference_range': bool(min(values) <= matched_mean <= max(values)),
    }


def nearest_participants(live: np.ndarray,
                         profiles: dict[tuple[str, str], np.ndarray],
                         participants: list[str], emotion: str, k: int = 3) -> list[dict]:
    scored = [{'participant': p, 'cosine': cosine(live, profiles[(p, emotion)])}
              for p in participants if (p, emotion) in profiles]
    return sorted(scored, key=lambda r: -r['cosine'])[:k]


def nearest_emotion(live: np.ndarray,
                    centroids: dict[str, np.ndarray], prompted: str) -> dict:
    """Which acted emotion centroid this live profile actually looks most like.

    The question behind it: when the model fails on B's Sadness, is B's Sadness simply far
    from everything, or is it sitting on top of some other emotion's training profile? Those
    are different problems -- the first is a coverage gap, the second is a collision.
    """
    ranked = sorted(((e, cosine(live, c)) for e, c in centroids.items()),
                    key=lambda item: -item[1])
    names = [e for e, _ in ranked]
    if prompted not in names:
        raise StudyError(f'{prompted} has no centroid to rank against')

    wrong = [(e, c) for e, c in ranked if e != prompted]
    return {
        'ranking': [{'emotion': e, 'cosine': c} for e, c in ranked],
        'nearest': names[0],
        'nearest_is_prompted': names[0] == prompted,
        'prompted_rank': names.index(prompted) + 1,
        'prompted_cosine': dict(ranked)[prompted],
        'nearest_wrong': wrong[0][0],
        'nearest_wrong_cosine': wrong[0][1],
        'margin_over_nearest_wrong': dict(ranked)[prompted] - wrong[0][1],
    }


# ----------------------------------------------------------------- model results


def live_model_results(sessions: list[sa.Session]) -> dict[str, dict]:
    """What the model did on this subject's live trials, per emotion.

    Two accuracies. `acc_scored` is over the ticks this study profiles -- everything after
    the transition tick -- and is the one that belongs next to a profile built from those
    same ticks. `acc_all` is over all ten ticks, which is what the in-game score and the
    earlier reports in this folder count, and is carried so the numbers here can be tied back
    to those without arithmetic.
    """
    out: dict[str, dict] = {}
    for emotion in COMMON_EMOTIONS:
        rows = [r for s in sessions for r in s.rows if r.prompted == emotion]
        scored = [r for r in rows if r.ok and r.prediction_index > TRANSITION_TICK]
        ok_all = [r for r in rows if r.ok]
        if not scored:
            raise StudyError(f'no scored {emotion} tick in this subject\'s sessions')

        wrong: dict[str, int] = {}
        for r in scored:
            if not r.correct:
                wrong[r.predicted] = wrong.get(r.predicted, 0) + 1

        out[emotion] = {
            'ticks_scored': len(scored),
            'ticks_all': len(ok_all),
            'hits_scored': sum(1 for r in scored if r.correct),
            'acc_scored': sum(1 for r in scored if r.correct) / len(scored),
            'acc_all': (sum(1 for r in ok_all if r.correct) / len(ok_all)) if ok_all else None,
            'dominant_wrong': max(wrong, key=wrong.get) if wrong else None,
            'dominant_wrong_ticks': max(wrong.values()) if wrong else 0,
            'wrong_breakdown': dict(sorted(wrong.items(), key=lambda kv: -kv[1])),
        }
    return out


def acted_model_results(path: Path, condition: str) -> dict[tuple[str, str], dict]:
    """Per-recording verdicts from Phase B, read rather than recomputed.

    `model_performance_study.py` already scored every acted recording through the ONNX graph
    and checked that replay against the headset's own logged scores. Re-running it here would
    add a second copy of that decision with no second check on it.
    """
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    rows = data.get('conditions', {}).get(condition, {}).get('recordings', [])
    return {(r['participant'], r['emotion']): r for r in rows}


# ----------------------------------------------------------------- diagnostics


def diagnostics_from_peaks(peaks: list[np.ndarray], tracking_rate: float | None) -> dict:
    """Sensor-facing summary from a set of per-emotion peak profiles.

    Same definitions as `subject_variability_study.measurement_diagnostics`, restricted to
    the six common emotions so the live and acted sides are computed over the same emotion
    set. Bilateral asymmetry is the one with diagnostic weight: a face is roughly symmetric,
    so a channel pair that stays lopsided across every emotion is more readily explained by
    headset fit than by how someone expresses anything.
    """
    session_peak = np.stack(peaks).max(axis=0)

    asymmetry = []
    for left, right in au_names.BILATERAL_PAIRS:
        a, b = session_peak[left], session_peak[right]
        if max(a, b) > sv.ACTIVE_THRESHOLD:
            asymmetry.append(abs(a - b) / max(a, b))

    return {
        'active_channels': int((session_peak[EXPRESSION] > sv.ACTIVE_THRESHOLD).sum()),
        'dead_channels': int((session_peak[EXPRESSION] <= sv.FLAT_EPSILON).sum()),
        'overall_peak': float(session_peak[EXPRESSION].max()),
        'mean_bilateral_asymmetry': float(np.mean(asymmetry)) if asymmetry else None,
        'n_asymmetry_pairs': len(asymmetry),
        'tracking_rate': tracking_rate,
    }


# ------------------------------------------------------------------- the study


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared. Spelled out rather than pulled from scipy, which is not a
    dependency of this folder."""
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    for value in np.unique(values):
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def run(live_root: str | Path | None, acted_root: str | Path | None,
        sampling: str, condition: str) -> dict:
    corpus = ad.load_corpus(ad.find_corpus_root(acted_root))
    profiles = acted_profiles(corpus, sampling)

    train = participants_in_split(corpus, TRAINING_SPLIT)
    everyone = sorted(corpus.participants)
    centroids = {e: centroid(profiles, train, e) for e in COMMON_EMOTIONS}
    centroids_all = {e: centroid(profiles, everyone, e) for e in COMMON_EMOTIONS}

    live_files = resolve_live_sessions(live_root)
    acted_verdicts = acted_model_results(
        Path(__file__).resolve().parent / 'results' / 'model_performance.json', condition)

    subjects: dict[str, dict] = {}
    for subject, paths in live_files.items():
        sessions = [sa.load_session(p) for p in paths]
        per_session = [live_profiles(s) for s in sessions]
        combined = combine_sessions(per_session)
        model = live_model_results(sessions)

        emotions: dict[str, dict] = {}
        for emotion in COMMON_EMOTIONS:
            live = combined[emotion]['peak']
            nearest = nearest_participants(live, profiles, train, emotion)
            for row in nearest:
                verdict = acted_verdicts.get((row['participant'], emotion))
                row['acted_model_correct'] = verdict['correct'] if verdict else None
                row['acted_model_predicted'] = verdict['predicted'] if verdict else None

            emotions[emotion] = {
                'position': position_in_training(live, profiles, train, emotion),
                'cosine_to_all15_centroid': cosine(live, centroids_all[emotion]),
                'nearest_training_participants': nearest,
                'nearest_emotion': nearest_emotion(live, centroids, emotion),
                'model': model[emotion],
                'ticks': combined[emotion]['ticks'],
                'frames': combined[emotion]['frames'],
            }

        # Within-subject, between-session agreement. For A this is the reference the
        # A-versus-B gap has to be read against: a difference smaller than one person's own
        # variation between two evenings is not a difference between people.
        between_session = None
        if len(per_session) > 1:
            between_session = {
                e: cosine(per_session[0][e]['peak'], per_session[1][e]['peak'])
                for e in COMMON_EMOTIONS
            }

        ok = sum(1 for s in sessions for r in s.rows if r.ok)
        total = sum(len(s.rows) for s in sessions)
        subjects[subject] = {
            'sessions': [str(p.parent.name) for p in paths],
            'n_sessions': len(sessions),
            'emotions': emotions,
            'between_session_cosine': between_session,
            'diagnostics': diagnostics_from_peaks(
                [combined[e]['peak'] for e in COMMON_EMOTIONS], ok / total if total else None),
            'overall_acc_scored': float(np.mean(
                [emotions[e]['model']['acc_scored'] for e in COMMON_EMOTIONS])),
            'overall_acc_all': float(np.mean(
                [emotions[e]['model']['acc_all'] for e in COMMON_EMOTIONS])),
        }

    # The hypothesis, reduced to one number. If closeness to the training distribution is why
    # the model works on one player and not the other, then across these twelve
    # subject-emotion cells the two should move together.
    pairs = [
        {'subject': s, 'emotion': e,
         'cosine': data['emotions'][e]['position']['cosine_matched_loo'],
         'accuracy': data['emotions'][e]['model']['acc_scored']}
        for s, data in subjects.items() for e in COMMON_EMOTIONS
    ]
    cos = np.array([p['cosine'] for p in pairs])
    acc = np.array([p['accuracy'] for p in pairs])
    association = {
        'n_cells': len(pairs),
        'pairs': pairs,
        # Spearman by hand: Pearson on the ranks. Rank-based because accuracy over nine ticks
        # is coarse and bounded, and one cell at 0.00 or 1.00 would drag a raw correlation.
        'spearman': float(np.corrcoef(_rank(cos), _rank(acc))[0, 1]),
        'pearson': float(np.corrcoef(cos, acc)[0, 1]),
    }

    acted_diagnostics = []
    for participant in everyone:
        peaks = [profiles[(participant, e)] for e in COMMON_EMOTIONS
                 if (participant, e) in profiles]
        if not peaks:
            continue
        recordings = [r for r in corpus.recordings
                      if r.participant == participant and r.emotion in COMMON_EMOTIONS]
        row = diagnostics_from_peaks(
            peaks, float(np.mean([r.tracking_rate for r in recordings])))
        row['participant'] = participant
        row['split'] = recordings[0].split
        acted_diagnostics.append(row)

    return {
        'sampling': sampling,
        'condition': condition,
        'emotions': list(COMMON_EMOTIONS),
        'acted_root': str(corpus.root),
        'training_participants': train,
        'all_participants': everyone,
        'subjects': subjects,
        'association': association,
        'acted_diagnostics': acted_diagnostics,
        'corpus_warnings': corpus.warnings,
    }


# -------------------------------------------------------------------- printing


def _bar(value: float, lo: float, hi: float, width: int = 20) -> str:
    """Where `value` sits between `lo` and `hi`, as a single-character marker."""
    if hi <= lo:
        return ' ' * width
    pos = int(round((value - lo) / (hi - lo) * (width - 1)))
    pos = max(0, min(width - 1, pos))
    return ' ' * pos + '|' + ' ' * (width - 1 - pos)


def print_header(result: dict) -> None:
    print('=' * 78)
    print('live players against the acted training distribution')
    print(f'  acted corpus   {result["acted_root"]}')
    print(f'  sampling       {result["sampling"]}')
    print(f'  training set   {len(result["training_participants"])} participants '
          f'(of {len(result["all_participants"])})')
    print(f'  emotions       {", ".join(result["emotions"])}  (Neutral excluded throughout)')
    for subject, data in result['subjects'].items():
        print(f'  subject {subject}      {data["n_sessions"]} session(s): '
              f'{", ".join(data["sessions"])}')
    print('=' * 78)


def print_positions(result: dict) -> None:
    print()
    print('-- distance to the acted TRAINING centroid ' + '-' * 34)
    print('   cos = cosine over expression channels, against nine-member leave-one-out')
    print('   centroids. "ref" is where the ten training participants themselves fall.')
    print()
    print(f'  {"emotion":<11} {"subj":<5} {"cos":>6} {"ref min":>8} {"ref med":>8} '
          f'{"ref max":>8} {"below":>6}  position in ref range')

    for emotion in result['emotions']:
        for subject, data in result['subjects'].items():
            pos = data['emotions'][emotion]['position']
            lo, hi = pos['reference_min'], pos['reference_max']
            marker = _bar(pos['cosine_matched_loo'], lo, hi)
            flag = '' if pos['inside_reference_range'] else '  OUTSIDE'
            print(f'  {emotion:<11} {subject:<5} {pos["cosine_matched_loo"]:>6.3f} '
                  f'{lo:>8.3f} {pos["reference_median"]:>8.3f} {hi:>8.3f} '
                  f'{pos["n_reference_below"]:>3}/{pos["n_reference"]:<2} [{marker}]{flag}')
        print()


def print_ab_gap(result: dict) -> None:
    subjects = list(result['subjects'])
    if len(subjects) != 2:
        return
    a, b = subjects
    between = result['subjects'][a]['between_session_cosine']

    print('-- A against B, with A\'s own between-session variation as the yardstick ' + '-' * 5)
    print('   A-B gap is the difference in cosine to the training centroid.')
    print('   A self is cosine between A\'s two sessions for the same emotion: how much one')
    print('   person varies between two evenings. A gap smaller than that is not a finding.')
    print()
    print(f'  {"emotion":<11} {"A cos":>7} {"B cos":>7} {"A-B gap":>9} {"A self":>8}   verdict')
    for emotion in result['emotions']:
        ca = result['subjects'][a]['emotions'][emotion]['position']['cosine_matched_loo']
        cb = result['subjects'][b]['emotions'][emotion]['position']['cosine_matched_loo']
        gap = ca - cb
        self_cos = between[emotion] if between else float('nan')
        # A between-session cosine of `self_cos` means 1 - self_cos of A's profile direction
        # is not stable across evenings. A gap has to clear that to be about people.
        noise = 1.0 - self_cos
        verdict = ('A closer' if gap > noise else
                   'B closer' if -gap > noise else
                   'within A\'s own session-to-session spread')
        print(f'  {emotion:<11} {ca:>7.3f} {cb:>7.3f} {gap:>+9.3f} {self_cos:>8.3f}   {verdict}')
    print()


def print_association(result: dict) -> None:
    """The hypothesis as a scatter: does being near the training centroid predict scoring?"""
    assoc = result['association']
    print('-- does closeness to the training centroid predict the model getting it right? ' + '-')
    print('   Twelve subject-emotion cells from two people. EXPLORATORY. Twelve cells drawn')
    print('   from two participants are not twelve independent observations, and no p-value')
    print('   or confidence interval is computed or implied.')
    print()
    print(f'  {"subj":<5} {"emotion":<11} {"cos to centroid":>16} {"acc (ticks 1-9)":>16}')
    for pair in sorted(assoc['pairs'], key=lambda p: -p['cosine']):
        print(f'  {pair["subject"]:<5} {pair["emotion"]:<11} {pair["cosine"]:>16.3f} '
              f'{pair["accuracy"]:>16.2f}')
    print()
    print(f'  Spearman {assoc["spearman"]:+.3f}   Pearson {assoc["pearson"]:+.3f}   '
          f'n = {assoc["n_cells"]} cells')
    print()


def print_nearest_emotion(result: dict) -> None:
    print('-- what each live profile actually looks like ' + '-' * 31)
    print('   Nearest acted training emotion centroid to the live profile, and where the')
    print('   prompted emotion ranked among the six.')
    print()
    print(f'  {"emotion":<11} {"subj":<5} {"nearest":<11} {"rank":>5} {"margin":>8}   '
          f'model acc (ticks 1-9), dominant miss')
    for emotion in result['emotions']:
        for subject, data in result['subjects'].items():
            ne = data['emotions'][emotion]['nearest_emotion']
            m = data['emotions'][emotion]['model']
            miss = (f'{m["dominant_wrong"]} x{m["dominant_wrong_ticks"]}'
                    if m['dominant_wrong'] else '-')
            mark = ' ' if ne['nearest_is_prompted'] else '*'
            print(f'  {emotion:<11} {subject:<5} {ne["nearest"]:<11}{mark}{ne["prompted_rank"]:>4} '
                  f'{ne["margin_over_nearest_wrong"]:>+8.3f}   {m["acc_scored"]:.2f}  {miss}')
        print()
    print('   * the live profile was nearer some other emotion\'s training centroid than the')
    print('     one the player was asked for.')
    print()


def print_neighbours(result: dict) -> None:
    print('-- nearest training participants ' + '-' * 44)
    print('   The three acted training participants whose profile for this emotion is')
    print('   closest, and whether the model got that participant\'s own trial right.')
    print('   Those verdicts are on training data and are not evidence of generalisation.')
    print()
    for subject, data in result['subjects'].items():
        print(f'  subject {subject}')
        for emotion in result['emotions']:
            parts = []
            for row in data['emotions'][emotion]['nearest_training_participants']:
                ok = ('ok' if row['acted_model_correct']
                      else '--' if row['acted_model_correct'] is False else '??')
                parts.append(f'{row["participant"]} {row["cosine"]:.3f} {ok}')
            print(f'    {emotion:<11} ' + '   '.join(parts))
        print()


def print_diagnostics(result: dict) -> None:
    print('-- sensor-facing diagnostics ' + '-' * 48)
    print('   Described against the acted spread, not diagnosed. The live and acted')
    print('   protocols differ in sampling, duration and task, and each of these numbers')
    print('   moves with all three.')
    print()
    acted = result['acted_diagnostics']
    train = [r for r in acted if r['split'] == 'train']

    def spread(key: str) -> str:
        vals = [r[key] for r in train if r[key] is not None]
        return f'{min(vals):.3f} .. {max(vals):.3f} (med {np.median(vals):.3f})'

    print(f'  {"":<16} {"active ch":>10} {"dead ch":>9} {"asymmetry":>10} {"tracking":>9}')
    for subject, data in result['subjects'].items():
        d = data['diagnostics']
        asym = f'{d["mean_bilateral_asymmetry"]:.3f}' if d['mean_bilateral_asymmetry'] else '-'
        track = f'{d["tracking_rate"]:.3f}' if d['tracking_rate'] is not None else '-'
        print(f'  live {subject:<11} {d["active_channels"]:>10} {d["dead_channels"]:>9} '
              f'{asym:>10} {track:>9}')
    print()
    print(f'  acted train (n={len(train)}):')
    for key, label in (('active_channels', 'active channels'),
                       ('dead_channels', 'dead channels'),
                       ('mean_bilateral_asymmetry', 'bilateral asymmetry'),
                       ('tracking_rate', 'tracking rate')):
        vals = [r[key] for r in train if r[key] is not None]
        if key in ('active_channels', 'dead_channels'):
            print(f'    {label:<22} {min(vals):.0f} .. {max(vals):.0f} '
                  f'(med {np.median(vals):.0f})')
        else:
            print(f'    {label:<22} {spread(key)}')
    print()


def print_limits(result: dict) -> None:
    n_live = len(result['subjects'])
    print('-- limits ' + '-' * 67)
    print(f'  Two live people, not three. N={n_live} on the live side. Nothing here supports a')
    print('  p-value, an effect size, or a claim that generalises to a third player.')
    print()
    print('  Frames and windows are not sample size. One live emotion is one continuous')
    print('  performance sampled 45 times, not 45 observations.')
    print()
    print('  A low cosine does not identify its cause. Different muscle use, a different')
    print('  headset fit, a different face shape and a different tracker confidence all')
    print('  lower it, and this data cannot separate them.')
    print()
    print('  The two protocols differ: the acted corpus is a continuous ~10 s capture at')
    print('  ~60 Hz with no feedback; the live sessions are ten 5-frame windows sampled at')
    print('  10 Hz while the player watches a score bar. Peak-based profiles are sensitive')
    print('  to that, which is why --sampling decimated exists.')
    print()
    print('  Neutral is excluded throughout and no conclusion about it can be drawn here.')
    print()
    print('  Acted model verdicts shown next to neighbours are on the model\'s own training')
    print('  participants. They describe memorisation, not generalisation.')
    print()


def to_json(result: dict) -> dict:
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return [float(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj
    return clean(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--sessions', help='live session file or folder')
    parser.add_argument('--data', help='ActingRecordings root')
    parser.add_argument('--sampling', choices=('full', 'decimated'), default='full',
                        help='how acted peak profiles are sampled (default: full)')
    parser.add_argument('--condition', default=DEFAULT_CONDITION,
                        choices=tp.CONDITIONS,
                        help='Phase B condition to read acted verdicts from')
    parser.add_argument('--json', help='write the full result here')
    args = parser.parse_args(argv)

    try:
        result = run(args.sessions, args.data, args.sampling, args.condition)
    except (StudyError, ad.CorpusError, sa.SchemaError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1

    print_header(result)
    print_positions(result)
    print_ab_gap(result)
    print_association(result)
    print_nearest_emotion(result)
    print_neighbours(result)
    print_diagnostics(result)
    print_limits(result)

    if args.json:
        path = Path(args.json)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_json(result), indent=2), encoding='utf-8')
        print(f'wrote {path.relative_to(Path(__file__).resolve().parent)}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
