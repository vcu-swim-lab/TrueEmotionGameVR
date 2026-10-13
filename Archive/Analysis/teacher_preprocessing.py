#!/usr/bin/env python3
"""Faithful reproduction of the teacher's input pipeline.

Transcribed from `final_teacher_notebook (4).ipynb`, cells 9, 12, 15, 17 and 23. Every
choice here exists because the notebook makes it, not because it is a good idea, and where
the notebook contradicts the project's own documentation the notebook wins -- it is what
produced the weights in `sparseVR_teacher_model.onnx`.

Four things the documentation got wrong, each of which would have made Phase B's accuracy
numbers meaningless:

**1. Downsampling is by frame count, not by time.** `DOWNSAMPLE_FACTOR = 6` with the comment
`60 Hz -> 10 Hz`, implemented as `df.groupby(arange(len(df)) // 6).mean()`. It never looks at
a timestamp, so a capture at a different rate is silently downsampled to a different
frequency. On ActingRecordings it does land on 10.0 Hz, because the capture really is ~60 Hz
(frames divided by elapsed time; the median inter-frame gap reads ~71 Hz because the capture
is bursty, and an earlier draft of this file wrongly quoted that as the rate). The point
stands for any future corpus: the operation is mean-of-six-rows, not resample-to-10-Hz.

**2. No scaling is applied.** `mean_centered_max_scale` is defined and then commented out at
its only call site:

    df = downsample_60hz_to_10hz_face_au(raw, AU_NAMES)   # raw downsampled
    #df_scaled = mean_centered_max_scale(df)              # model input only

That single `#` is the whole meaning of `NOT-normalized` in the checkpoint name. Raw
action-unit values go into the model.

**3. `FaceValid` is never consulted.** The pipeline selects the 70 AU columns by name and
ignores the tracker's own validity flag, so frames the headset did not vouch for were
trained on as if they were readings. Descriptive analysis in `subject_variability_study.py`
does drop them, and should; reproducing model input must not.

**4. All 70 channels are used, eye channels included.** The notebook contains a leftover cell
asserting the eye channels have been removed, which would leave 60. It is dead: the cell that
ran asserts the opposite and its saved output reads `Number of AU features (including eye
AUs): 70`.

One consequence worth carrying into Phase B. The activity gate below is not a filter applied
to a set of windows -- it is the only way a window is ever created. The model has therefore
never seen a window without three consecutive active frames in it. Evaluating the deployed
runtime, which applies no gate, means feeding it inputs of a kind it was never trained on.
That is the right thing to measure, because it is what actually happens on the headset, but
the gated condition is not merely a stricter subset: it is the training distribution.
"""

from __future__ import annotations

import numpy as np

import au_names

SEQ_LENGTH = 5
DOWNSAMPLE_FACTOR = 6
CONSECUTIVE = 3
ACTIVITY_THRESHOLD = 0.25

NEUTRAL = 'Neutral'


def downsample(au: np.ndarray, factor: int = DOWNSAMPLE_FACTOR) -> np.ndarray:
    """Mean of every `factor` consecutive rows, trailing remainder dropped.

    Deliberately index-based. The notebook groups on `arange(len(df)) // 6` and never reads a
    timestamp, so a recording captured at a different rate is silently downsampled to a
    different frequency. Reproducing the model's input means reproducing that.
    """
    usable = (len(au) // factor) * factor
    if usable == 0:
        return np.empty((0, au.shape[1]), au.dtype)
    trimmed = au[:usable]
    return trimmed.reshape(usable // factor, factor, au.shape[1]).mean(axis=1)


def clean(au: np.ndarray) -> np.ndarray:
    """Replace non-finite values the way the notebook's ffill/bfill/fillna chain does."""
    out = np.array(au, dtype=np.float64, copy=True)
    out[~np.isfinite(out)] = np.nan

    # Forward fill, then back fill, then zero anything still missing.
    for column in range(out.shape[1]):
        values = out[:, column]
        missing = np.isnan(values)
        if not missing.any():
            continue
        idx = np.where(~missing, np.arange(len(values)), 0)
        np.maximum.accumulate(idx, out=idx)
        values = values[idx]
        still = np.isnan(values)
        if still.any():
            good = np.where(~still)[0]
            values[still] = values[good[0]] if len(good) else 0.0
        out[:, column] = values

    return np.nan_to_num(out, nan=0.0)


def activity_labels(au: np.ndarray, emotion: str,
                    threshold: float = ACTIVITY_THRESHOLD) -> np.ndarray:
    """Per-frame activity flag, with Neutral inverted.

    A non-Neutral frame counts as active when **any** channel exceeds the threshold. A
    Neutral frame counts when **every** channel stays at or below it -- Neutral is defined as
    the absence of movement rather than the presence of it, which is why a Neutral trial in
    which the participant fidgets yields no windows at all.
    """
    if emotion == NEUTRAL:
        return (au <= threshold).all(axis=1)
    return (au > threshold).any(axis=1)


def make_windows(au: np.ndarray, emotion: str,
                 seq_length: int = SEQ_LENGTH,
                 consecutive: int = CONSECUTIVE,
                 threshold: float = ACTIVITY_THRESHOLD,
                 gate: bool = True) -> np.ndarray:
    """Windows of `seq_length` downsampled frames, gated as the notebook gates them.

    The scan accepts a window when it contains `consecutive` consecutive active frames, then
    jumps a whole window forward so accepted windows never overlap; on rejection it advances
    a single frame. The loop stops at `len(au) - seq_length`, which -- as in the notebook --
    leaves the final full window unexamined.

    `gate=False` takes every non-overlapping window instead. It exists to isolate the gate's
    contribution with the sampling held fixed -- it is **not** the headset's input, because
    the frames here are still means of six. Only `runtime_proxy` approximates that.

    Note that the two paths do not produce the same window boundaries. Ungated windows start
    at 0, 5, 10, ...; the gate advances one frame on rejection, so an accepted window can
    start anywhere. Comparing the two therefore mixes the gate's selection with a shift in
    where windows begin. `fixed_windows_gated` exists to separate those.
    """
    if len(au) < seq_length:
        return np.empty((0, seq_length, au.shape[1]), np.float32)

    if not gate:
        count = len(au) // seq_length
        return au[:count * seq_length].reshape(count, seq_length, au.shape[1]).astype(np.float32)

    labels = activity_labels(au, emotion, threshold)

    windows = []
    i = 0
    n = len(au) - seq_length          # notebook's bound, exclusive
    while i < n:
        has_run = any(
            labels[i + j:i + j + consecutive].all()
            for j in range(seq_length - consecutive + 1)
        )
        if has_run:
            windows.append(au[i:i + seq_length])
            i += seq_length
        else:
            i += 1

    if not windows:
        return np.empty((0, seq_length, au.shape[1]), np.float32)
    return np.stack(windows).astype(np.float32)


def decimate(au: np.ndarray, factor: int = DOWNSAMPLE_FACTOR) -> np.ndarray:
    """Every `factor`-th row, kept as-is. The runtime's sampling, not the notebook's.

    The headset never averages. `AUDevice` polls `OVRFaceExpressions` on a 10 Hz timer
    (`DeviceManager.cs`, `AUDevice(go, 10)`) and `NewFaceAuModel.AwaitWindow` clears the ring
    buffer and waits for five fresh readings, so the model receives five *instantaneous*
    samples 100 ms apart. Taking every 6th row of a 60 Hz capture reproduces that spacing
    exactly.

    The difference from `downsample` is not cosmetic. A mean of six frames smooths; a single
    sample does not. `ANGER_POST_HOC_LOGIT_ADJUSTMENT.md` §8.2 measured this directly as
    temporal roughness: mean-of-6 acted data scores 0.966 while live headset data scores
    1.18-1.25. Feeding averaged windows and calling the result a runtime measurement would
    understate exactly that gap.
    """
    return np.array(au[::factor], copy=True)


# The conditions Phase B reports, under these names and never merged.
#
#   notebook_exact      - the training distribution: mean-of-6 frames, activity gate on,
#                         window starts wherever the scan accepts. What the model was fitted
#                         to.
#   fixed_windows_gated - the same fixed windows as notebook_ungated, with the activity rule
#                         applied as an accept/abstain decision on each. Isolates the gate's
#                         selection effect alone.
#   notebook_ungated    - mean-of-6 frames, every fixed window, no gate.
#   runtime_proxy       - instantaneous 10 Hz samples, every fixed window, no gate. The
#                         closest offline stand-in for what the headset feeds the model.
#
# Why the fourth exists. Comparing notebook_exact against notebook_ungated does not measure
# the gate: the two do not share window boundaries. Ungated windows start at 0, 5, 10, ...,
# while the gated scan advances a single frame on rejection, so its accepted windows are
# shifted onto active stretches and it emits a different number of predictions per trial.
# The difference between those two therefore folds together (a) not scoring quiet windows,
# (b) starting windows in better places, (c) a different vote count per trial, and (d) one
# recording producing nothing at all. `fixed_windows_gated` holds the boundaries fixed and
# varies only the accept/abstain decision, which is the part that could be ported to the
# runtime.
#
# Only runtime_proxy speaks to deployed behaviour, and even it is a proxy: the real device
# polls on a timer against live tracking, and a frame the tracker rejects is dropped rather
# than sampled, so a live window can span more than 500 ms of wall clock.
CONDITIONS = ('notebook_exact', 'fixed_windows_gated', 'notebook_ungated', 'runtime_proxy')

# Conditions where some windows are refused rather than scored. Their accuracy is meaningless
# without the coverage alongside it.
ABSTAINING_CONDITIONS = ('fixed_windows_gated',)


def window_passes_gate(window: np.ndarray, emotion: str,
                       consecutive: int = CONSECUTIVE,
                       threshold: float = ACTIVITY_THRESHOLD) -> bool:
    """Whether one already-cut window would satisfy the training activity rule.

    The same test the notebook's scan applies, asked of a window whose boundaries are already
    decided. This is the form a runtime gate would take: the device has a window in hand and
    must decide whether to answer for it.
    """
    labels = activity_labels(window, emotion, threshold)
    return any(labels[j:j + consecutive].all()
               for j in range(len(labels) - consecutive + 1))


def windows_for_condition(recording, condition: str) -> np.ndarray:
    """Model-ready windows for one `acting_dataset.Recording` under a named condition.

    Takes `recording.au` rather than `recording.valid_au` throughout: the pipeline never
    consulted `FaceValid`, so dropping invalid frames would feed the model something it was
    not trained on and would quietly shift every window boundary.
    """
    if recording.au.shape[1] != au_names.CHANNEL_COUNT:
        raise ValueError(
            f'expected {au_names.CHANNEL_COUNT} channels, got {recording.au.shape[1]}')

    cleaned = clean(recording.au)

    if condition == 'notebook_exact':
        return make_windows(downsample(cleaned), recording.emotion, gate=True)
    if condition in ('notebook_ungated', 'fixed_windows_gated'):
        # Identical windows. `fixed_windows_gated` differs only in that the caller consults
        # `gate_mask` and abstains on the ones that fail, which is why it can be compared
        # against `notebook_ungated` window for window.
        return make_windows(downsample(cleaned), recording.emotion, gate=False)
    if condition == 'runtime_proxy':
        return make_windows(decimate(cleaned), recording.emotion, gate=False)

    raise ValueError(f'unknown condition {condition!r}; expected one of {CONDITIONS}')


def gate_mask(windows: np.ndarray, emotion: str) -> np.ndarray:
    """Which of these windows the activity rule would answer for. [n] of bool."""
    if len(windows) == 0:
        return np.zeros(0, bool)
    return np.array([window_passes_gate(w, emotion) for w in windows])


def teacher_windows(recording, *, gate: bool = True) -> np.ndarray:
    """Shorthand for the two notebook conditions."""
    return windows_for_condition(
        recording, 'notebook_exact' if gate else 'notebook_ungated')
