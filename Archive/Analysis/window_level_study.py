#!/usr/bin/env python3
"""The 5x70 windows the model actually saw, live against acted.

`LIVE_VS_ACTED_REPORT.md` reduced each ten-second recording to one 70-channel peak profile
and found that distance from the acted training profiles does not predict whether the model
gets the trial right (Spearman +0.007 over twelve subject-emotion cells). A peak profile
throws away everything the model is actually given: the ordering of five frames, the level
each channel sat at rather than the highest it reached, and the fact that one trial is ten
separate decisions rather than one.

This study puts that back. The unit here is **one window** -- the exact [5, 70] block the
network consumed and answered for -- and the question is:

    are the live windows A and B produce actually different inputs, or are they similar
    inputs the model maps to different answers?

Those are opposite findings with opposite fixes, and a peak profile cannot separate them.
Both spaces are measured, and kept apart:

  **input space**   the window flattened to 350 numbers, frame-major, raw. The model was
                    trained on unscaled action units (`teacher_preprocessing`, note 2), so
                    the comparison is on raw values too.
  **score space**   the seven-dimensional output. Live scores come from the headset's own
                    log; acted scores come from running the graph, after a fidelity check.

Two windows can be close in the first and far apart in the second. That gap is the whole
point of the file.

Reproduce:

    python Analysis/window_level_study.py --json results/window_level.json

This is a diagnostic, not a solution. Nothing is fitted and no calibration is proposed; a
nearest-neighbour result is not a design.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import acting_dataset as ad              # noqa: E402
import au_names                          # noqa: E402
import calibration_study as cs           # noqa: E402
import live_vs_acted_study as lva        # noqa: E402
import model_performance_study as mp     # noqa: E402
import session_analysis as sa            # noqa: E402
import teacher_preprocessing as tp       # noqa: E402

# Reused rather than restated. `live_vs_acted_study` owns which sessions belong to which
# person, which six emotions are comparable, and that tick 0 is a transition sample; this
# study must not be able to drift from those.
COMMON_EMOTIONS = lva.COMMON_EMOTIONS
LIVE_SUBJECTS = lva.LIVE_SUBJECTS
TRANSITION_TICK = lva.TRANSITION_TICK
TRAINING_SPLIT = lva.TRAINING_SPLIT

CLASS_ORDER = mp.CLASS_ORDER            # seven classes, model output order

# `runtime_proxy` is the primary because it is the only acted condition whose windows are
# built the way the headset builds one: instantaneous samples 100 ms apart, no averaging, no
# gate. `notebook_exact` is the distribution the model was fitted on, and is reported beside
# it as a training-distribution control -- never as a runtime number.
PRIMARY_CONDITION = 'runtime_proxy'
CONTROL_CONDITION = 'notebook_exact'

# Two channel sets, reported separately and never substituted for one another. The model is
# given all seventy, so `all70` is what it saw and is the primary. `expression55` drops gaze
# and tongue and exists only to show whether a finding survives without them.
CHANNEL_SETS: dict[str, list[int]] = {
    'all70': list(range(au_names.CHANNEL_COUNT)),
    'expression55': list(au_names.EXPRESSION_INDICES),
}
PRIMARY_CHANNELS = 'all70'

TOP_K = 5


class StudyError(RuntimeError):
    """Raised when the inputs are not shaped the way this comparison requires."""


# ------------------------------------------------------------------- windows


def flatten(windows: np.ndarray) -> np.ndarray:
    """[n, 5, 70] -> [n, 350], frame-major, time order preserved.

    Frame 0's seventy channels come first, then frame 1's, and so on, so position `f*70 + c`
    is channel `c` at frame `f`. Cosine on this representation is sensitive to *when* a
    channel moved, which is the entire reason for preferring it to a peak profile: two
    windows holding the same expression and two windows where it rises and falls are far
    apart here and identical under a maximum.
    """
    if windows.ndim != 3:
        raise StudyError(f'expected [n, frames, channels], got shape {windows.shape}')
    if windows.shape[1] != tp.SEQ_LENGTH or windows.shape[2] != au_names.CHANNEL_COUNT:
        raise StudyError(
            f'expected [n, {tp.SEQ_LENGTH}, {au_names.CHANNEL_COUNT}] windows, '
            f'got {windows.shape}')
    return windows.reshape(len(windows), -1)


def live_windows(session: sa.Session) -> dict[str, dict]:
    """Every scored window one live session produced, per emotion.

    These are not reconstructed from raw tracking: they are the exact frames the logger wrote
    alongside the prediction they caused, so a window here and the score beside it are the
    same event.

    `no_tracking` rows carry no window and are dropped -- there is nothing to drop them *from*,
    since `Row.frames` is None whenever the status is not `ok`. Tick 0 is excluded as the
    transition sample, selected on `prediction_index` rather than by counting rows.
    """
    if session.schema.window_frames != tp.SEQ_LENGTH:
        raise StudyError(f'{session.path}: {session.schema.window_frames} frames per window, '
                         f'expected {tp.SEQ_LENGTH}')
    if session.schema.features_per_frame != au_names.CHANNEL_COUNT:
        raise StudyError(f'{session.path}: {session.schema.features_per_frame} channels, '
                         f'expected {au_names.CHANNEL_COUNT}')
    if list(session.schema.classes) != CLASS_ORDER:
        raise StudyError(f'{session.path}: class order {session.schema.classes} '
                         f'does not match the model order {CLASS_ORDER}')

    out: dict[str, dict] = {}
    for emotion in COMMON_EMOTIONS:
        prompted = [r for r in session.rows if r.prompted == emotion]
        if not prompted:
            raise StudyError(f'{session.path}: no {emotion} trial')

        kept = [r for r in prompted
                if r.ok and r.frames is not None and r.scores is not None
                and r.prediction_index is not None
                and r.prediction_index > TRANSITION_TICK]
        if not kept:
            raise StudyError(
                f'{session.path}: {emotion} has no scored window after the transition tick')

        windows = np.array([r.frames for r in kept], dtype=np.float32)
        if windows.shape[1:] != (tp.SEQ_LENGTH, au_names.CHANNEL_COUNT):
            raise StudyError(f'{session.path}: {emotion} windows are {windows.shape[1:]}')
        if not np.isfinite(windows).all():
            raise StudyError(f'{session.path}: {emotion} window has a non-finite value')

        out[emotion] = {
            'windows': windows,
            'scores': np.array([r.scores for r in kept], dtype=np.float32),
            'predicted': [r.predicted for r in kept],
            'tick': [r.prediction_index for r in kept],
            'n_prompted_ticks': len(prompted),
        }
    return out


def acted_windows(corpus: ad.Corpus, condition: str,
                  participants: list[str]) -> dict[tuple[str, str], np.ndarray]:
    """Model-ready windows per (participant, emotion), built by the shared pipeline.

    `teacher_preprocessing.windows_for_condition` is the only place window construction
    lives; rebuilding it here is how the two sides would silently stop matching.

    A recording that yields no window is kept out rather than represented by an empty array.
    Under `notebook_exact` that happens -- the activity gate can refuse a whole recording --
    and it is a fact about the gate, reported as a count rather than smoothed over.
    """
    out: dict[tuple[str, str], np.ndarray] = {}
    for recording in corpus.recordings:
        if recording.participant not in participants:
            continue
        if recording.emotion not in COMMON_EMOTIONS:
            continue
        windows = tp.windows_for_condition(recording, condition)
        if condition in tp.ABSTAINING_CONDITIONS and len(windows):
            windows = windows[tp.gate_mask(windows, recording.emotion)]
        if len(windows) == 0:
            continue
        out[(recording.participant, recording.emotion)] = windows
    return out


# ---------------------------------------------------------------- similarity


def unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise StudyError('cannot normalise an all-zero window; cosine is undefined for it')
    return matrix / norms


def cosine_matrix(live: np.ndarray, acted: np.ndarray) -> np.ndarray:
    """[n_live, n_acted] cosine similarities."""
    return unit(live) @ unit(acted).T


def balanced_by_emotion(per_participant: dict[tuple[str, str], np.ndarray],
                        participants: list[str]) -> dict[str, np.ndarray]:
    """Collapse per-participant similarities to one number per emotion, per live window.

    Each acted participant contributes their **single best-matching window** for that
    emotion, and the ten participants are then averaged with equal weight. Taking a plain
    maximum over the pooled acted set instead would let one participant who happens to
    contribute more windows -- which `notebook_exact` does, from 0 to 19 per recording --
    decide the answer on their own.

    A participant with no window for an emotion is left out of that emotion's mean rather
    than counted as zero: absent and dissimilar are opposite findings.
    """
    out: dict[str, np.ndarray] = {}
    for emotion in COMMON_EMOTIONS:
        columns = [per_participant[(p, emotion)].max(axis=1)
                   for p in participants if (p, emotion) in per_participant]
        if not columns:
            raise StudyError(f'no acted training window for {emotion}')
        out[emotion] = np.mean(np.stack(columns), axis=0)
    return out


def describe_windows(live: np.ndarray, acted: dict[tuple[str, str], np.ndarray],
                     participants: list[str], prompted: str,
                     channels: list[int]) -> dict:
    """Per-window nearest-neighbour description of one live trial against the acted set."""
    live_flat = flatten(live)[:, _channel_columns(channels)]

    keys = [k for k in acted if k[0] in participants]
    per_participant = {
        key: cosine_matrix(live_flat, flatten(acted[key])[:, _channel_columns(channels)])
        for key in keys
    }

    balanced = balanced_by_emotion(per_participant, participants)
    stacked = np.stack([balanced[e] for e in COMMON_EMOTIONS], axis=1)   # [n_live, 6]
    prompted_col = COMMON_EMOTIONS.index(prompted)
    surprise_col = COMMON_EMOTIONS.index('Surprise')

    # Rank of the prompted emotion, 1 = the balanced-nearest of the six.
    order = np.argsort(-stacked, axis=1)
    ranks = np.empty(len(live_flat), dtype=int)
    for i, row in enumerate(order):
        ranks[i] = int(np.where(row == prompted_col)[0][0]) + 1

    # Global nearest windows. Deliberately not participant-balanced: "which single acted
    # window is this most like" is a question about the pooled acted set, and it is reported
    # alongside the balanced numbers rather than instead of them.
    all_sims = np.concatenate([per_participant[k] for k in keys], axis=1)
    labels = np.array([k[1] for k in keys for _ in range(per_participant[k].shape[1])])
    people = np.array([k[0] for k in keys for _ in range(per_participant[k].shape[1])])

    nearest = all_sims.argmax(axis=1)
    top_k = np.argsort(-all_sims, axis=1)[:, :TOP_K]

    return {
        'n_windows': int(len(live_flat)),
        'balanced': {e: balanced[e] for e in COMMON_EMOTIONS},
        'balanced_to_prompted': stacked[:, prompted_col],
        'balanced_to_surprise': stacked[:, surprise_col],
        'balanced_best_emotion': [COMMON_EMOTIONS[c] for c in stacked.argmax(axis=1)],
        'prompted_rank': ranks,
        'nearest_emotion': labels[nearest].tolist(),
        'nearest_participant': people[nearest].tolist(),
        'nearest_cosine': all_sims[np.arange(len(live_flat)), nearest],
        'top_k_emotions': [labels[row].tolist() for row in top_k],
    }


def _channel_columns(channels: list[int]) -> np.ndarray:
    """Flattened-vector positions for a channel subset, across all five frames."""
    return np.array([f * au_names.CHANNEL_COUNT + c
                     for f in range(tp.SEQ_LENGTH) for c in channels])


# -------------------------------------------------------------------- scores


def acted_scores(acted: dict[tuple[str, str], np.ndarray],
                 model_path: Path) -> dict[tuple[str, str], np.ndarray]:
    import onnxruntime as ort
    runner = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    name = runner.get_inputs()[0].name
    return {key: runner.run(None, {name: windows})[0] for key, windows in acted.items()}


def acted_self_similarity(acted: dict[tuple[str, str], np.ndarray],
                          participants: list[str], channels: list[int]) -> dict[str, dict]:
    """How well raw-window cosine separates the acted emotions from each other.

    The control that bounds every nearest-neighbour claim in this file. Each participant's
    windows for one emotion are compared against the *other nine* participants' windows for
    every emotion, balanced the same way the live comparison is, and averaged over the ten.
    Leave-one-out because a participant compared against their own windows would be compared
    against themselves.

    If an acted Fear window is already nearer acted Surprise than acted Fear, then a live Fear
    window landing nearest Surprise says nothing about the player -- it is what this feature
    space does to Fear. Without this table the live findings would be read as facts about A
    and B when they are facts about the representation.
    """
    out: dict[str, dict] = {}
    for source in COMMON_EMOTIONS:
        rows: list[dict[str, float]] = []
        for held in participants:
            if (held, source) not in acted:
                continue
            others = [p for p in participants if p != held]
            live = flatten(acted[(held, source)])[:, _channel_columns(channels)]
            per_participant = {
                key: cosine_matrix(live, flatten(acted[key])[:, _channel_columns(channels)])
                for key in acted if key[0] in others
            }
            balanced = balanced_by_emotion(per_participant, others)
            rows.append({e: float(v.mean()) for e, v in balanced.items()})

        if not rows:
            raise StudyError(f'no acted window for {source}')
        means = {e: float(np.mean([r[e] for r in rows])) for e in COMMON_EMOTIONS}
        nearest = max(means, key=means.get)
        out[source] = {
            'n_participants': len(rows),
            'cosine': means,
            'nearest': nearest,
            'self_is_nearest': nearest == source,
            'margin': means[source] - max(v for e, v in means.items() if e != source),
        }
    return out


def score_mass(acted: dict[tuple[str, str], np.ndarray],
               participants: list[str]) -> dict[str, dict]:
    """Where the model's output mass sits on the acted windows, per emotion.

    Printed before any score-space cosine, because it decides how one may be read. The graph
    ends in Softmax, so a cosine between two probability vectors is dominated by whichever
    class already holds most of the mass. If Neutral is large on every acted emotion, then
    every pair of score vectors is similar for a reason that has nothing to do with the
    emotions involved, and a high `cos_out` means much less than it looks like it does.

    `calibration_study.features` solves this for its own purposes with a log transform. This
    file does not: it reports the raw cosine the brief asks for, and prints the mass table
    next to it so the inflation is visible rather than argued about.
    """
    out: dict[str, dict] = {}
    for emotion in COMMON_EMOTIONS:
        stacked = np.concatenate(
            [v for k, v in acted.items() if k[1] == emotion and k[0] in participants])
        argmax = Counter(CLASS_ORDER[i] for i in stacked.argmax(axis=1))
        out[emotion] = {
            'n_windows': int(len(stacked)),
            'mean_scores': {c: float(stacked[:, i].mean()) for i, c in enumerate(CLASS_ORDER)},
            'mean_confidence': float(stacked.max(axis=1).mean()),
            'argmax_counts': dict(argmax.most_common()),
        }
    return out


def describe_scores(live_scores: np.ndarray,
                    acted: dict[tuple[str, str], np.ndarray],
                    participants: list[str], prompted: str) -> dict:
    """The same nearest-neighbour description, in the model's seven-dimensional output.

    Kept strictly apart from the input-space description. The contrast between the two is the
    finding this study exists to produce: similar inputs with dissimilar outputs means the
    model separates these people, not the sensor.

    Read alongside `score_mass`: these are cosines between Softmax outputs, and a shared
    Neutral component lifts all of them.
    """
    per_participant = {
        key: cosine_matrix(live_scores.astype(np.float64), acted[key].astype(np.float64))
        for key in acted if key[0] in participants
    }
    balanced = balanced_by_emotion(per_participant, participants)
    stacked = np.stack([balanced[e] for e in COMMON_EMOTIONS], axis=1)
    return {
        'balanced': balanced,
        'balanced_to_prompted': stacked[:, COMMON_EMOTIONS.index(prompted)],
        'balanced_to_surprise': stacked[:, COMMON_EMOTIONS.index('Surprise')],
        'balanced_best_emotion': [COMMON_EMOTIONS[c] for c in stacked.argmax(axis=1)],
        'live_mean_scores': {c: float(live_scores[:, i].mean())
                             for i, c in enumerate(CLASS_ORDER)},
        'live_mean_confidence': float(live_scores.max(axis=1).mean()),
    }


# ------------------------------------------------------------------- the study


def summarise_cell(inputs: dict, scores: dict, predicted: list[str], prompted: str) -> dict:
    """One live trial, reduced to the numbers the report quotes."""
    n = inputs['n_windows']
    nearest = inputs['nearest_emotion']
    pred = np.array(predicted)

    nearest_is_surprise = np.array([e == 'Surprise' for e in nearest])
    predicted_surprise = pred == 'Surprise'

    top_k_counts = Counter(e for row in inputs['top_k_emotions'] for e in row)

    return {
        'n_windows': n,
        'model_accuracy': float((pred == prompted).mean()),
        'prediction_counts': dict(Counter(predicted).most_common()),

        'input_cos_to_prompted': float(inputs['balanced_to_prompted'].mean()),
        'input_cos_to_surprise': float(inputs['balanced_to_surprise'].mean()),
        'input_cos_by_emotion': {e: float(v.mean()) for e, v in inputs['balanced'].items()},
        'input_prompted_rank_mean': float(inputs['prompted_rank'].mean()),
        'input_prompted_rank_1_frac': float((inputs['prompted_rank'] == 1).mean()),
        'input_balanced_best_counts': dict(Counter(inputs['balanced_best_emotion'])),

        'nearest_emotion_counts': dict(Counter(nearest).most_common()),
        'nearest_is_prompted_frac': float(np.mean([e == prompted for e in nearest])),
        'nearest_is_surprise_frac': float(nearest_is_surprise.mean()),
        'nearest_cosine_mean': float(inputs['nearest_cosine'].mean()),
        'nearest_participant_counts': dict(Counter(inputs['nearest_participant']).most_common(3)),
        'top_k_emotion_counts': dict(top_k_counts.most_common()),

        'score_cos_to_prompted': float(scores['balanced_to_prompted'].mean()),
        'score_cos_to_surprise': float(scores['balanced_to_surprise'].mean()),
        'score_cos_by_emotion': {e: float(v.mean()) for e, v in scores['balanced'].items()},
        'score_balanced_best_counts': dict(Counter(scores['balanced_best_emotion'])),
        'live_mean_scores': scores['live_mean_scores'],
        'live_mean_confidence': scores['live_mean_confidence'],

        # Does the model answer Surprise on exactly the windows whose input is nearest acted
        # Surprise? Reported as an overlap rather than a correlation: with nine windows per
        # trial a correlation coefficient would be noise with a decimal point on it.
        'predicted_surprise_frac': float(predicted_surprise.mean()),
        'surprise_overlap': {
            'both': int((nearest_is_surprise & predicted_surprise).sum()),
            'nearest_only': int((nearest_is_surprise & ~predicted_surprise).sum()),
            'predicted_only': int((~nearest_is_surprise & predicted_surprise).sum()),
            'neither': int((~nearest_is_surprise & ~predicted_surprise).sum()),
        },
    }


def mean_of_summaries(summaries: list[dict]) -> dict:
    """Combine a subject's sessions with equal weight per session.

    Not by pooling windows. A's two evenings happen to contribute nine windows each, so the
    two agree today -- but the rule has to be the one that stays right when they do not, and
    the alternative silently reweights a person by how long they played.
    """
    if not summaries:
        raise StudyError('nothing to combine')
    if len(summaries) == 1:
        return dict(summaries[0])

    out: dict = {}
    for key, value in summaries[0].items():
        values = [s[key] for s in summaries]
        if isinstance(value, float):
            out[key] = float(np.mean(values))
        elif isinstance(value, int):
            out[key] = int(sum(values))
        elif isinstance(value, dict) and all(isinstance(v, float) for v in value.values()):
            out[key] = {k: float(np.mean([v[k] for v in values])) for k in value}
        elif isinstance(value, dict):
            merged: Counter = Counter()
            for v in values:
                merged.update(v)
            out[key] = dict(merged.most_common())
        else:
            out[key] = value
    out['n_sessions'] = len(summaries)
    return out


def run(live_root: str | Path | None, acted_root: str | Path | None,
        model: str | None, channels: str, conditions: tuple[str, ...]) -> dict:
    if channels not in CHANNEL_SETS:
        raise StudyError(f'unknown channel set {channels!r}')

    model_path = cs.find_model(model)
    fidelity = mp.fidelity_check(model_path)
    if not fidelity.get('ran'):
        raise StudyError(f'fidelity check could not run: {fidelity.get("reason")}. '
                         'No ONNX result may be reported without it.')
    if not fidelity.get('passed'):
        raise StudyError(
            'fidelity check FAILED: the ONNX file on disk does not reproduce the scores the '
            'headset logged. Every score-space number would describe a different model. '
            f'{fidelity["sessions"]}')

    corpus = ad.load_corpus(ad.find_corpus_root(acted_root))
    train = lva.participants_in_split(corpus, TRAINING_SPLIT)

    live_files = lva.resolve_live_sessions(live_root)

    result: dict = {
        'model': str(model_path),
        'fidelity': fidelity,
        'acted_root': str(corpus.root),
        'channel_set': channels,
        'channel_count': len(CHANNEL_SETS[channels]),
        'primary_condition': PRIMARY_CONDITION,
        'control_condition': CONTROL_CONDITION,
        'emotions': list(COMMON_EMOTIONS),
        'training_participants': train,
        'conditions': {},
    }

    for condition in conditions:
        acted = acted_windows(corpus, condition, train)
        scores = acted_scores(acted, model_path)

        missing = [f'{p}/{e}' for p in train for e in COMMON_EMOTIONS
                   if (p, e) not in acted]
        subjects: dict[str, dict] = {}

        for subject, paths in live_files.items():
            sessions = [sa.load_session(p) for p in paths]
            per_session_cells: dict[str, list[dict]] = {e: [] for e in COMMON_EMOTIONS}

            for session in sessions:
                windows = live_windows(session)
                for emotion in COMMON_EMOTIONS:
                    data = windows[emotion]
                    per_session_cells[emotion].append(summarise_cell(
                        describe_windows(data['windows'], acted, train, emotion,
                                         CHANNEL_SETS[channels]),
                        describe_scores(data['scores'], scores, train, emotion),
                        data['predicted'], emotion))

            subjects[subject] = {
                'sessions': [p.parent.name for p in paths],
                'n_sessions': len(sessions),
                'emotions': {e: mean_of_summaries(per_session_cells[e])
                             for e in COMMON_EMOTIONS},
            }

        result['conditions'][condition] = {
            'acted_windows': int(sum(len(w) for w in acted.values())),
            'acted_recordings': len(acted),
            'missing_participant_emotions': missing,
            'acted_score_mass': score_mass(scores, train),
            'acted_self_similarity': acted_self_similarity(
                acted, train, CHANNEL_SETS[channels]),
            'subjects': subjects,
        }

    return result


# ------------------------------------------------------------------- printing


def print_header(result: dict) -> None:
    print('=' * 78)
    print('window-level comparison: the 5x70 inputs the model actually saw')
    print(f'  model          {result["model"]}')
    print(f'  channels       {result["channel_set"]} ({result["channel_count"]} of 70)')
    print(f'  acted corpus   {result["acted_root"]}')
    print(f'  reference      {len(result["training_participants"])} acted TRAINING '
          'participants')
    print(f'  emotions       {", ".join(result["emotions"])}  (Neutral excluded from the '
          'AU comparison)')
    print('=' * 78)

    f = result['fidelity']
    print()
    print('-- replay fidelity (required before any ONNX number) ' + '-' * 24)
    for row in f['sessions']:
        print(f'  {row["session"]:<34} {row["rows"]:>4} rows  '
              f'agreement {row["agreement"]:.3f}  max diff {row["max_diff"]:.2e}')
    print(f'  {"passed" if f["passed"] else "FAILED"}  (tolerance {f["tolerance"]})')


def print_condition_header(name: str, data: dict, primary: bool) -> None:
    print()
    print('=' * 78)
    label = 'PRIMARY - closest offline stand-in for the headset' if primary else \
            'CONTROL - the training distribution, NOT a runtime measurement'
    print(f'condition: {name}   [{label}]')
    print(f'  {data["acted_windows"]} acted windows from {data["acted_recordings"]} '
          'recordings')
    if data['missing_participant_emotions']:
        print('  no window at all for: ' + ', '.join(data['missing_participant_emotions']))
    print('=' * 78)


def print_self_similarity(data: dict, emotions: list[str]) -> None:
    """How much separation the representation has, before asking what the live data does."""
    print()
    print('-- CONTROL: does raw-window cosine separate the acted emotions at all? ' + '-' * 6)
    print('   Acted against acted, leave-one-participant-out, balanced the same way. Rows are')
    print('   the emotion acted, columns the emotion compared against. Read the diagonal: if')
    print('   a class is not nearest itself here, a live window landing elsewhere is a fact')
    print('   about this feature space, not about the player.')
    print()
    print(f'  {"acted":<11} ' + ' '.join(f'{e[:5]:>7}' for e in emotions) + '   self-nearest?')
    for source in emotions:
        row = data['acted_self_similarity'][source]
        cells = []
        for e in emotions:
            mark = '<' if e == row['nearest'] else ('*' if e == source else ' ')
            cells.append(f'{row["cosine"][e]:>6.3f}{mark}')
        verdict = (f'yes, by {row["margin"]:+.3f}' if row['self_is_nearest']
                   else f'NO - nearest {row["nearest"]} ({row["margin"]:+.3f})')
        print(f'  {source:<11} ' + ' '.join(cells) + f'   {verdict}')
    print()
    print('   * = the same emotion,  < = the nearest')
    print()


def print_cells(data: dict, emotions: list[str]) -> None:
    print()
    print('-- input space: the flattened 5x70 window ' + '-' * 35)
    print('   cos(prompted) and cos(Surprise) are participant-balanced: each acted training')
    print('   participant contributes their best-matching window, then the ten are averaged.')
    print('   rank is where the prompted emotion placed among the six. NN = the single')
    print('   nearest acted window, pooled.')
    print()
    print(f'  {"subj":<5} {"emotion":<11} {"n":>3} {"acc":>5} {"cos(prm)":>9} {"cos(Sur)":>9} '
          f'{"rank":>5} {"NN=prm":>7} {"NN=Sur":>7}  model predictions')

    for emotion in emotions:
        for subject, sdata in data['subjects'].items():
            c = sdata['emotions'][emotion]
            preds = ', '.join(f'{k} x{v}' for k, v in list(c['prediction_counts'].items())[:3])
            print(f'  {subject:<5} {emotion:<11} {c["n_windows"]:>3} '
                  f'{c["model_accuracy"]:>5.2f} {c["input_cos_to_prompted"]:>9.3f} '
                  f'{c["input_cos_to_surprise"]:>9.3f} {c["input_prompted_rank_mean"]:>5.2f} '
                  f'{c["nearest_is_prompted_frac"]:>7.2f} {c["nearest_is_surprise_frac"]:>7.2f}'
                  f'  {preds}')
        print()


def print_score_mass(data: dict, emotions: list[str]) -> None:
    """Where the output mass sits, before any score-space cosine is read."""
    print('-- where the model puts its output mass ' + '-' * 38)
    print('   Read this before the score-space cosines below. These are Softmax vectors, so')
    print('   a component every emotion shares lifts every cosine between them.')
    print()
    print(f'  {"":<22} ' + ' '.join(f'{c[:4]:>6}' for c in CLASS_ORDER) + f' {"conf":>7}')
    for emotion in emotions:
        m = data['acted_score_mass'][emotion]
        top = next(iter(m['argmax_counts']))
        print(f'  acted {emotion:<16} '
              + ' '.join(f'{m["mean_scores"][c]:>6.3f}' for c in CLASS_ORDER)
              + f' {m["mean_confidence"]:>7.3f}   argmax mostly {top}'
              f' ({m["argmax_counts"][top]}/{m["n_windows"]})')
    print()
    for emotion in emotions:
        for subject, sdata in data['subjects'].items():
            c = sdata['emotions'][emotion]
            print(f'  live {subject} {emotion:<16} '
                  + ' '.join(f'{c["live_mean_scores"][k]:>6.3f}' for k in CLASS_ORDER)
                  + f' {c["live_mean_confidence"]:>7.3f}')
    print()


def print_input_vs_score(data: dict, emotions: list[str]) -> None:
    print('-- input space against score space ' + '-' * 42)
    print('   The contrast this study exists for. cos_in is how alike the raw windows are;')
    print('   cos_out is how alike the seven-dimensional outputs are. A pair of cells with')
    print('   similar cos_in and different cos_out means the model separates these people,')
    print('   not the sensor.')
    print()
    print(f'  {"emotion":<11} {"subj":<5} {"cos_in(prm)":>12} {"cos_out(prm)":>13} '
          f'{"cos_in(Sur)":>12} {"cos_out(Sur)":>13} {"acc":>6}')
    for emotion in emotions:
        for subject, sdata in data['subjects'].items():
            c = sdata['emotions'][emotion]
            print(f'  {emotion:<11} {subject:<5} {c["input_cos_to_prompted"]:>12.3f} '
                  f'{c["score_cos_to_prompted"]:>13.3f} {c["input_cos_to_surprise"]:>12.3f} '
                  f'{c["score_cos_to_surprise"]:>13.3f} {c["model_accuracy"]:>6.2f}')
        print()


def print_surprise_overlap(data: dict, emotions: list[str]) -> None:
    print('-- Surprise: does the input explain the answer? ' + '-' * 30)
    print('   "near" = the nearest acted window was a Surprise window.')
    print('   "said" = the model answered Surprise for that window.')
    print()
    print(f'  {"subj":<5} {"emotion":<11} {"near&said":>10} {"near only":>10} '
          f'{"said only":>10} {"neither":>9}')
    for emotion in emotions:
        for subject, sdata in data['subjects'].items():
            o = sdata['emotions'][emotion]['surprise_overlap']
            print(f'  {subject:<5} {emotion:<11} {o["both"]:>10} {o["nearest_only"]:>10} '
                  f'{o["predicted_only"]:>10} {o["neither"]:>9}')
        print()


def print_per_emotion_similarity(data: dict, emotions: list[str]) -> None:
    print('-- balanced input similarity to every acted emotion ' + '-' * 26)
    print('   Each row is one live trial. The prompted emotion is marked [ ]; the highest')
    print('   value in the row is marked <.')
    print()
    header = ' '.join(f'{e[:5]:>7}' for e in emotions)
    print(f'  {"subj":<5} {"prompted":<11} {header}')
    for emotion in emotions:
        for subject, sdata in data['subjects'].items():
            c = sdata['emotions'][emotion]['input_cos_by_emotion']
            best = max(c, key=c.get)
            cells = []
            for e in emotions:
                mark = '<' if e == best else ('*' if e == emotion else ' ')
                cells.append(f'{c[e]:>6.3f}{mark}')
            print(f'  {subject:<5} {emotion:<11} ' + ' '.join(cells))
        print()
    print('   * = the prompted emotion,  < = the nearest')
    print()


def print_limits() -> None:
    print('-- limits ' + '-' * 67)
    print('  Two live people. Not three: two of the sessions are one person on two evenings.')
    print()
    print('  Windows are not observations. Nine windows from one trial are nine samples of')
    print('  one continuous performance, and the acted windows from one recording are the')
    print('  same. No p-value is computed and none would be meaningful.')
    print()
    print('  Nearest-neighbour is a diagnostic, not a design. That a live window sits near an')
    print('  acted Surprise window does not license a calibration built on that fact.')
    print()
    print('  Neutral is outside the AU comparison entirely -- the game never prompts it and')
    print('  the teacher pipeline inverts its activity rule, so an acted Neutral window is')
    print('  built under a different definition. Where Neutral appears below it is only as a')
    print('  model output that was observed, never as an AU similarity.')
    print()
    print('  runtime_proxy is still a proxy. The device polls against live tracking and skips')
    print('  frames the tracker rejects, so a real window can span more than 500 ms.')
    print()
    print('  notebook_exact numbers describe the training distribution and are a control.')
    print('  They are not a runtime measurement and must not be quoted as one.')
    print()


def to_json(obj):
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_json(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--sessions', help='live session file or folder')
    parser.add_argument('--data', help='ActingRecordings root')
    parser.add_argument('--model', help='path to sparseVR_teacher_model.onnx')
    parser.add_argument('--channels', choices=sorted(CHANNEL_SETS), default=PRIMARY_CHANNELS,
                        help=f'channel set (default: {PRIMARY_CHANNELS}, what the model sees)')
    parser.add_argument('--condition', action='append', choices=tp.CONDITIONS,
                        help='repeatable; default is the primary plus the control')
    parser.add_argument('--json', help='write the full result here')
    args = parser.parse_args(argv)

    conditions = tuple(args.condition or (PRIMARY_CONDITION, CONTROL_CONDITION))

    try:
        result = run(args.sessions, args.data, args.model, args.channels, conditions)
    except (StudyError, ad.CorpusError, sa.SchemaError, lva.StudyError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1

    print_header(result)
    for name, data in result['conditions'].items():
        print_condition_header(name, data, name == PRIMARY_CONDITION)
        print_self_similarity(data, result['emotions'])
        print_cells(data, result['emotions'])
        print_per_emotion_similarity(data, result['emotions'])
        print_score_mass(data, result['emotions'])
        print_input_vs_score(data, result['emotions'])
        print_surprise_overlap(data, result['emotions'])
    print_limits()

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
