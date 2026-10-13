#!/usr/bin/env python3
"""Does a short per-person enrolment recover a subject the model fails on?

    python Analysis/calibration_study.py
    python Analysis/calibration_study.py ~/sessions --json study.json

Background. Three logged sessions -- two from one subject, one from a second --
produced 61.7%, 65.0% and 20.0% tick accuracy against 14.3% chance. The third
subject is close to chance. This script asks whether a supervised per-person
remapping of the model's own output, fitted on part of a session, recovers the
rest of that session.

It is a *pre*-experiment. It cannot answer the question, and the validity block it
prints on every run says why: enrolment and test windows come from the same
continuous expression, there is one trial per emotion per session, and Neutral was
never prompted. What it does is make the measurement reproducible and state its
knobs -- the log transform, the epsilon, the class set, the distance metric -- as
parameters rather than as choices buried in a scratch file. Every one of those
changes the headline number, and three of them were found by review to have been
doing silent work in an earlier version of this analysis.

The model is re-run rather than trusted: `--check` compares replayed scores against
the `score_<Class>` columns the headset logged, which catches a wrong or stale ONNX
file before any of the numbers below mean anything.

Unlike the rest of `Analysis/`, this needs numpy and onnxruntime:

    pip install numpy onnxruntime

See README.md ("Calibration study") for what the numbers mean and do not mean.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import session_analysis as sa
from analyze_sessions import WIDTH, num, pct, rule

try:
    import numpy as np
except ImportError:  # pragma: no cover - environment guard
    raise SystemExit(
        "calibration_study.py needs numpy and onnxruntime, which the other scripts in\n"
        "Analysis/ deliberately do not:  pip install numpy onnxruntime"
    )


# The package is normally a local checkout, either linked into Packages/ or resolved
# into Library/PackageCache by the Unity package manager. Both are searched, so this
# works from a clone without anyone editing a path into the script.
MODEL_RELATIVE = "Runtime/Models/sparseVR_teacher_model.onnx"
PACKAGE_NAME = "com.vcu-swim-lab.vr-emotion-detection"

# A guard against log(0), not a tunable. The old default of 1e-6 sat inside the live dynamic
# range -- the smallest score in any recording so far is 6.95e-06 -- so it could act as a
# smoother, and on one session sweeping it moved adapter accuracy from 0.7407 to 0.8333. At
# 1e-12 it is below every observed score, so its inertness is demonstrable rather than hoped
# for, and a test asserts it. Verified not to change any previously reported number: no score
# in any recording falls below 1e-6, so 1e-12 and 1e-6 give identical results on all of them.
DEFAULT_EPSILON = 1e-12

# Largest probability difference tolerated between a replayed score and the one the
# headset logged. Observed across the current sessions is ~4e-07, which is ordinary
# float32 drift between Unity Inference Engine and onnxruntime. Anything materially
# larger means the ONNX file on disk is not the one that produced the recordings,
# and every number downstream would be measuring the wrong model.
FIDELITY_TOLERANCE = 1e-4


def find_model(explicit: str | None) -> Path:
    """Locate the deployed ONNX teacher, or explain every place that was tried."""
    tried: list[Path] = []

    def candidate(path: Path) -> Path | None:
        tried.append(path)
        return path if path.is_file() else None

    if explicit:
        found = candidate(Path(explicit).expanduser())
        if found:
            return found
        raise SystemExit(f"--model does not exist: {explicit}")

    env = os.environ.get("TRUEEMOTION_MODEL")
    if env and (found := candidate(Path(env).expanduser())):
        return found

    root = sa.find_repo_root()
    if root:
        for base in (root / "Packages" / PACKAGE_NAME, *sorted(
            (root / "Library" / "PackageCache").glob(f"{PACKAGE_NAME}*")
        )):
            if found := candidate(base / MODEL_RELATIVE):
                return found
        # A sibling checkout, which is how the package is usually developed.
        if found := candidate(root.parent / "VREmotionDetection" / MODEL_RELATIVE):
            return found

    listing = "\n".join(f"  {p}" for p in tried) or "  (nowhere -- no repo root found)"
    raise SystemExit(
        "could not find sparseVR_teacher_model.onnx. Looked in:\n"
        f"{listing}\n"
        "Pass --model PATH, or set TRUEEMOTION_MODEL."
    )


# ------------------------------------------------------------------- loading


@dataclass
class Replayed:
    """One session, with the model re-run over its logged windows."""

    name: str
    classes: list[str]
    windows: "np.ndarray"        # [n, frames, features]
    prompted: "np.ndarray"       # [n] class index
    tick: "np.ndarray"           # [n] position within its trial, 0 = prompt onset
    trial: "np.ndarray"          # [n] trial id
    role: "np.ndarray"           # [n] block_role, "" before session-local-v1
    repeat: "np.ndarray"         # [n] repeat index within the block, -1 when absent
    scores: "np.ndarray"         # [n, classes] replayed
    logged: "np.ndarray"         # [n, classes] as recorded on the headset
    frame_reuse: int             # frames shared between consecutive windows
    tick_source: str             # how tick positions were determined

    # Trials the file says existed, by role, including ones every tick of which was
    # untracked. Taken from all rows rather than from the usable ones, so a trial that
    # contributed no window is still counted as attempted rather than vanishing.
    attempted: dict              # role -> {trial_id: prompted class name}

    protocol_version: str | None
    rehearsal: bool

    @property
    def has_roles(self) -> bool:
        return bool(set(self.role.tolist()) - {""})

    @property
    def fidelity(self) -> tuple[float, float]:
        """Agreement and largest probability difference against the logged scores."""
        agree = float((self.scores.argmax(1) == self.logged.argmax(1)).mean())
        return agree, float(np.abs(self.scores - self.logged).max())

    @property
    def faithful(self) -> bool:
        agree, diff = self.fidelity
        return agree == 1.0 and diff < FIDELITY_TOLERANCE


def tick_positions(rows: list[sa.Row]) -> tuple["np.ndarray", str]:
    """Position of each row within its own trial.

    `SessionLogger.cs` documents `prediction_index` as "0-based tick within the
    trial", so it is used whenever every selected row carries one. Counting
    positions by enumeration instead is only correct when no row has been dropped
    -- and rows *are* dropped here, because windows only exist on `ok` rows. A
    future session with a `no_tracking` tick in the middle of a trial would have
    its timeline silently compressed, shifting rows between the enrolment and test
    splits. The fallback is kept for files written before the column existed, and
    the source is reported so a run can say which one it used.
    """
    recorded = [r.prediction_index for r in rows]
    if all(v is not None for v in recorded):
        return np.array(recorded), "prediction_index"

    seen: dict[int, int] = {}
    out = []
    for r in rows:
        key = r.trial_id if r.trial_id is not None else -1
        seen[key] = seen.get(key, -1) + 1
        out.append(seen[key])
    return np.array(out), "enumerated (prediction_index missing)"


def count_frame_reuse(windows: "np.ndarray") -> int:
    """Frames shared between consecutive windows.

    `NewFaceAuModel.AwaitWindow` clears the ring buffer and waits for a full window
    of fresh frames, and at one tick per second roughly nine frames arrive per tick,
    so at the logged rate this is normally zero: the windows are disjoint. That does
    not make the rows independent -- adjacent ticks still sample one continuous held
    expression -- but it does mean the usual "overlapping windows" caveat is the
    wrong description of why.
    """
    shared = 0
    for i in range(len(windows) - 1):
        for a in windows[i]:
            for b in windows[i + 1]:
                if np.array_equal(a, b):
                    shared += 1
    return shared


def replay(session: sa.Session, model_path: Path) -> Replayed:
    import onnxruntime as ort

    rows = [r for r in session.ok_rows if r.frames and r.scores]
    if not rows:
        raise SystemExit(f"{session.path}: no rows with a usable window")
    if any(v is None for r in rows for f in r.frames for v in f):
        raise SystemExit(f"{session.path}: window contains blank action-unit cells")

    windows = np.array([r.frames for r in rows], np.float32)
    logged = np.array([r.scores for r in rows], np.float32)
    classes = list(session.schema.classes)
    prompted = np.array([classes.index(r.prompted) for r in rows])
    trial = np.array([r.trial_id if r.trial_id is not None else -1 for r in rows])
    role = np.array([r.block_role or "" for r in rows])
    repeat = np.array([r.repeat_index if r.repeat_index is not None else -1 for r in rows])
    tick, tick_source = tick_positions(rows)

    # Every trial the recording attempted, from *all* rows. A trial whose every tick was
    # `no_tracking` carries no window and so is absent from `rows`; counting it here is what
    # lets the evaluation report it as unscored instead of silently shrinking the
    # denominator.
    attempted: dict[str, dict[int, str]] = {}
    for r in session.rows:
        if r.trial_id is None:
            continue
        attempted.setdefault(r.block_role or "", {})[r.trial_id] = r.prompted

    metadata = session.metadata or {}

    # A file that is not a usable graph fails here, before fidelity can be measured,
    # so it gets its own message rather than an onnxruntime traceback. A file that
    # loads but is the *wrong* model is caught later, by the fidelity check.
    try:
        runner = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        scores = runner.run(None, {runner.get_inputs()[0].name: windows})[0]
    except Exception as exc:
        raise SystemExit(
            f"could not run the model at\n  {model_path}\n"
            f"onnxruntime said: {exc}\n"
            "Expected a graph taking [N, frames, features] and returning [N, classes]."
        ) from exc

    return Replayed(
        name=session.session_id,
        classes=classes,
        windows=windows,
        prompted=prompted,
        tick=tick,
        trial=trial,
        role=role,
        repeat=repeat,
        scores=np.asarray(scores, np.float32),
        logged=logged,
        frame_reuse=count_frame_reuse(windows),
        tick_source=tick_source,
        attempted=attempted,
        protocol_version=metadata.get("protocol_version"),
        rehearsal=str(metadata.get("fast_rehearsal", "")).lower() == "true",
    )


# ------------------------------------------------------------------ features


def logscore(scores: "np.ndarray", epsilon: float = DEFAULT_EPSILON) -> "np.ndarray":
    """The `logscore` feature transform, on a bare `[n, classes]` score array.

    Split out of `features` so a study working on scores that did not come from a logged
    headset session -- `emojihero_calibration_study.py` replays a corpus instead -- fits its
    centroids in exactly this space rather than in a second copy of the same two lines.
    `features` below calls it, so the two can never drift apart.
    """
    return np.log(np.clip(scores, epsilon, None))


def features(data: Replayed, space: str, epsilon: float) -> "np.ndarray":
    """Feature vectors the adapter is fitted in.

    `logscore` is not a cosmetic choice. Cosine similarity on raw probabilities is
    dominated by whichever class already has most of the mass, so a subject whose
    output collapses onto one class has near-identical directions for every trial.
    The log puts the classes on an additive scale, where a consistent *ratio* shift
    -- which is what a mis-mapped subject produces -- becomes a consistent offset.
    Measured difference on the third subject: 0.792 with the log, 0.583 without.

    `epsilon` clamps the log. The exported graph ends in Softmax and can emit values
    small enough to underflow; without a floor a single 0.0 sends a whole feature
    vector to -inf and the similarity becomes NaN.
    """
    if space == "logscore":
        return logscore(data.scores, epsilon)
    if space == "score":
        return data.scores
    if space == "aumax":
        return data.windows.max(axis=1)
    if space == "auflat":
        return data.windows.reshape(len(data.windows), -1)
    raise ValueError(f"unknown feature space: {space}")


def centroids(F: "np.ndarray", y: "np.ndarray", classes: list[int]) -> "np.ndarray":
    return np.stack([F[y == c].mean(0) for c in classes])


def classify(M: "np.ndarray", classes: list[int], F: "np.ndarray", metric: str):
    """Nearest centroid. Returns (predicted class, score of the winner, margin)."""
    if metric == "cosine":
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
        sim = Fn @ Mn.T
    elif metric == "euclidean":
        sim = -np.linalg.norm(F[:, None, :] - M[None, :, :], axis=2)
    else:
        raise ValueError(f"unknown metric: {metric}")

    order = np.argsort(-sim, axis=1)
    best = order[:, 0]
    top = sim[np.arange(len(F)), best]
    second = sim[np.arange(len(F)), order[:, 1]] if sim.shape[1] > 1 else top
    return np.array(classes)[best], top, top - second


# ------------------------------------------------------------ movement calibration

# The face action units of a window are its first 63 features; the last 7 are tongue values.
FACE_AU_COUNT = 63


def movement_feature(windows: "np.ndarray") -> "np.ndarray":
    """Movement-calibration feature: per window, the mean over its frames of the 63 face AUs.

    `windows` is `[n, frames, features]`. The tongue values are left out because they say
    nothing about the expression.
    """
    return np.asarray(windows)[..., :FACE_AU_COUNT].mean(axis=1)


def classify_movement(score_M: "np.ndarray", score_F: "np.ndarray",
                      movement_M: "np.ndarray", movement_F: "np.ndarray", classes: list[int]):
    """Movement calibration's decision. Returns (predicted class, winner total, margin).

    Per class: cosine of the log-score feature to the personal score centroid, plus cosine of
    the movement feature to the personal movement centroid. The highest sum wins. Ranked
    exactly as `classify` ranks, so the tie rule is the same.

    Tested offline in `Analysis/exploratory/au_calibration.py` (method `fuse-sim`).
    """
    def unit(X):
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    sim = unit(score_F) @ unit(score_M).T + unit(movement_F) @ unit(movement_M).T

    order = np.argsort(-sim, axis=1)
    best = order[:, 0]
    top = sim[np.arange(len(sim)), best]
    second = sim[np.arange(len(sim)), order[:, 1]] if sim.shape[1] > 1 else top
    return np.array(classes)[best], top, top - second


# --------------------------------------------------------------- evaluation


@dataclass
class Condition:
    space: str = "logscore"
    metric: str = "cosine"
    neutral: str = "none"
    epsilon: float = DEFAULT_EPSILON

    def label(self) -> str:
        extra = "" if self.neutral == "none" else f", neutral={self.neutral}"
        return f"{self.space}/{self.metric}{extra}"


def split_masks(data: Replayed, enrol: set[int], test: set[int]):
    return (
        np.isin(data.tick, list(enrol)),
        np.isin(data.tick, list(test)),
    )


# Tick 0 is a transition sample, caught as the prompt appears and before the participant has
# begun; every block drops it. A complete hold therefore contributes nine ticks.
SCORED_TICKS = frozenset(range(1, 10))
TICKS_REQUIRED_PER_CENTROID = 9


def role_masks(data: Replayed, enrol_role: str, test_role: str,
               ticks: frozenset = SCORED_TICKS):
    """Masks for a trial-disjoint split, keyed on `block_role`.

    This is the split the study is about. `split_masks` above divides one held expression
    into early and late ticks, which measures consistency within a single performance; this
    one fits on the calibration block and scores on separate trials of the same states.
    """
    if not data.has_roles:
        raise SystemExit(
            f"{data.name}: no block_role column, so a calibration/evaluation split cannot\n"
            "be made. Only recordings from session-local-v1 onwards carry one; for an older\n"
            "file use the tick split (--enrol-ticks/--test-ticks), which is a within-trial\n"
            "reference condition and not a validation."
        )

    usable = np.isin(data.tick, list(ticks))
    return usable & (data.role == enrol_role), usable & (data.role == test_role)


def trial_vote(truth: "np.ndarray", predicted: "np.ndarray", trials: "np.ndarray",
               ticks: "np.ndarray", class_order: list[str]):
    """Collapse ticks to one prediction per trial by majority vote.

    The tie rule is frozen before collection and applied identically to every method being
    compared, because a rule chosen per method is a way of choosing the winner:

      1. the class winning the most ticks in the trial;
      2. on a tie, the prediction of the trial's final tracked tick;
      3. if that still does not decide it, the first tied class in `class_order`.

    Step 2 is not arbitrary: a tie is usually a trial that changed its mind partway through,
    and the end of a held expression is the part the participant had most settled into.
    Step 3 is arbitrary on purpose -- it has to be fixed, decided in advance, and not read
    off the scores.

    Returns (trial ids, truth per trial, prediction per trial).
    """
    order = {c: i for i, c in enumerate(class_order)}
    ids, y, p = [], [], []

    for t in sorted(set(trials.tolist())):
        sel = trials == t
        votes = {}
        for label in predicted[sel]:
            votes[label] = votes.get(label, 0) + 1
        best = max(votes.values())
        tied = [c for c, n in votes.items() if n == best]

        if len(tied) == 1:
            winner = tied[0]
        else:
            last = predicted[sel][np.argmax(ticks[sel])]
            winner = last if last in tied else min(tied, key=lambda c: order.get(c, len(order)))

        ids.append(int(t))
        y.append(truth[sel][0])
        p.append(winner)

    return ids, np.array(y), np.array(p)


def _names(indices: "np.ndarray", classes: list[str]) -> "np.ndarray":
    return np.array([classes[i] for i in indices])


def evaluate(data: Replayed, cond: Condition,
             enrol_mask: "np.ndarray", test_mask: "np.ndarray", *,
             required_ticks: int | None = None,
             test_role: str | None = None,
             population_M: "np.ndarray | None" = None,
             hybrid_alpha: float | None = None) -> dict:
    """Adapter and both baselines on one session, under one condition.

    `population_M` and `hybrid_alpha` are the VR shadow evaluation. When both are given,
    two further methods are scored on the *same* test ticks: `population`, the nearest of
    seven centroids estimated from the EmojiHero training participants, and `hybrid`,
    the nearest of `alpha * personal + (1 - alpha) * population`. When either is absent
    nothing about this function's output changes, which a test asserts -- the personal-only
    result is a published number and must not move because a new option exists.

    The restricted baseline is the one that matters. The adapter can only ever emit
    a class it was enrolled on; comparing it against an argmax that ranges over all
    seven classes credits it for excluding Neutral rather than for calibrating
    anything. Both are reported, and they differ a lot. Under `session-local-v1` every
    state is prompted, so the two coincide and that trap closes.

    `required_ticks`, when given, is how many scored ticks a state must contribute
    before a centroid is fitted for it. A partial hold is refused rather than
    averaged: a centroid built from two of nine ticks is not a weaker version of the
    same estimate, it is a different one, and producing it quietly is how a bad
    recording turns into a result.
    """
    if not enrol_mask.any() or not test_mask.any():
        raise SystemExit(f"{data.name}: empty enrolment or test split")

    neutral_index = data.classes.index("Neutral") if "Neutral" in data.classes else None
    present = sorted(set(data.prompted[enrol_mask].tolist()))
    classes = list(data.classes)

    ticks_per_centroid = {
        classes[c]: int(((data.prompted == c) & enrol_mask).sum()) for c in present
    }
    incomplete = ([c for c in present
                   if ticks_per_centroid[classes[c]] < required_ticks]
                  if required_ticks is not None else [])
    enrolled = [c for c in present if c not in incomplete]
    if not enrolled:
        raise SystemExit(
            f"{data.name}: no state contributed {required_ticks} tracked calibration "
            "ticks, so nothing can be enrolled"
        )

    F = features(data, cond.space, cond.epsilon)
    M = centroids(F[enrol_mask], data.prompted[enrol_mask], enrolled)

    if cond.neutral == "rest":
        # Tick 0 is the face at the moment the prompt appeared, before the subject
        # has begun to act. It is the only resting sample the pre-session-local
        # protocols recorded. Measured to be a poor substitute for a prompted
        # Neutral -- see the README -- and kept only so that result stays
        # reproducible. `session-local-v1` prompts Neutral properly instead.
        if neutral_index is None:
            raise SystemExit("--neutral rest: no Neutral class in this schema")
        rest = data.tick == 0
        if not rest.any():
            raise SystemExit(f"{data.name}: no tick-0 rows to build a rest centroid")
        M = np.vstack([M, F[rest].mean(0)])
        enrolled = enrolled + [neutral_index]
    elif cond.neutral == "prompted":
        if neutral_index not in enrolled:
            raise SystemExit(
                "--neutral prompted: no Neutral trial was recorded in this session."
            )

    y = data.prompted[test_mask]
    predicted, top, margin = classify(M, enrolled, F[test_mask], cond.metric)

    full = data.scores[test_mask].argmax(1)
    restricted = np.array(enrolled)[data.scores[test_mask][:, enrolled].argmax(1)]

    # The two shadow methods, on the identical test ticks. `population_M` is indexed by the
    # schema's class order, so it is sliced to whatever this session enrolled -- which under
    # session-local-v1 is all seven, but the slice is what keeps the three methods ranging
    # over one class set rather than three.
    shadow: dict[str, "np.ndarray"] = {}
    if population_M is not None and hybrid_alpha is not None:
        if population_M.shape[0] != len(classes):
            raise SystemExit(
                f"population centroids have {population_M.shape[0]} rows for a "
                f"{len(classes)}-class schema"
            )
        if population_M.shape[1] != data.scores.shape[1]:
            raise SystemExit(
                f"population centroids are {population_M.shape[1]}-dimensional but the "
                f"model emits {data.scores.shape[1]} scores"
            )
        if not 0.0 <= hybrid_alpha <= 1.0:
            raise SystemExit(f"--hybrid-alpha must lie in [0, 1], got {hybrid_alpha}")

        population_rows = population_M[enrolled]
        blended = hybrid_alpha * M + (1.0 - hybrid_alpha) * population_rows
        shadow["population"] = classify(
            population_rows, enrolled, F[test_mask], cond.metric)[0]
        shadow["hybrid"] = classify(blended, enrolled, F[test_mask], cond.metric)[0]

    # Enrolment-only diagnostics: computable from the calibration block alone, so looking
    # at them commits no leakage. Reported with no threshold attached -- the single
    # low-quality recording on hand is nowhere near enough to set one, and a gate fitted to
    # it would be fitted to noise.
    separation = None
    if len(M) > 1:
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        off = (Mn @ Mn.T)[~np.eye(len(M), dtype=bool)]
        separation = {
            "mean_pairwise_cosine": float(off.mean()),
            "min": float(off.min()),
            "max": float(off.max()),
        }

    # Drift: the same states re-estimated from the test block and compared against the
    # enrolled centroids. This is the quantity that killed the cross-session design, so it
    # is measured here rather than assumed away. It needs the test block to hold its own
    # trials of those states, which is why the evaluation block has two repeats.
    drift = {}
    for c in enrolled:
        sel = test_mask & (data.prompted == c)
        if not sel.any():
            continue
        a = M[enrolled.index(c)]
        b = F[sel].mean(0)
        drift[classes[c]] = float(
            a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12)
        )

    # Labels rather than indices from here on, so the shared metric helpers in
    # `session_analysis` are reused as they stand instead of reimplemented.
    y_n = _names(y, classes)
    pred_n = _names(predicted, classes)
    full_n = _names(full, classes)
    restr_n = _names(restricted, classes)

    trials = data.trial[test_mask]
    ticks = data.tick[test_mask]
    ids, ty, tp_adapter = trial_vote(y_n, pred_n, trials, ticks, classes)
    _, _, tp_full = trial_vote(y_n, full_n, trials, ticks, classes)
    _, _, tp_restricted = trial_vote(y_n, restr_n, trials, ticks, classes)

    shadow_trial = {
        name: trial_vote(y_n, _names(p, classes), trials, ticks, classes)[2]
        for name, p in shadow.items()
    }

    # Trials the recording attempted that contributed no tracked tick. Reported rather
    # than dropped: shrinking the denominator quietly is how a tracking failure becomes
    # a better-looking number.
    attempted = data.attempted.get(test_role, {}) if test_role is not None else {}
    unscored = sorted(set(attempted) - set(ids))

    # `macro_f1` leaves a class with no support out of the mean. That is right when the
    # class does not apply and wrong when it is absent because the recording is
    # incomplete -- there it raises the figure. So completeness is checked and reported
    # rather than assumed.
    scored_classes = sorted(set(ty.tolist()))
    complete = len(scored_classes) == len(classes)

    trial_majority = {}
    for key, votes in (("argmax_full", tp_full),
                       ("argmax_restricted", tp_restricted),
                       ("adapter", tp_adapter),
                       *sorted(shadow_trial.items())):
        m = sa.macro_f1(ty, votes, classes)
        trial_majority[key] = {
            "accuracy": float((votes == ty).mean()),
            "macro_f1": m["macro_f1"],
            "classes_scored": m["classes_scored"],
            "per_class": m["per_class"],
        }

    # `adapter` is the personal-only method under its published name. The shadow deltas are
    # named for what they compare so that "hybrid beat the model" and "the population term
    # earned its place" cannot be confused for one another: only the second is
    # hybrid - personal.
    shadow_deltas = {}
    if shadow_trial:
        for better, worse in (("hybrid", "adapter"), ("hybrid", "argmax_full"),
                              ("population", "argmax_full"), ("adapter", "argmax_full")):
            a, b = trial_majority[better], trial_majority[worse]
            label = f"{'personal' if better == 'adapter' else better}_minus_" \
                    f"{'personal' if worse == 'adapter' else 'raw'}"
            shadow_deltas[label] = {
                "accuracy": a["accuracy"] - b["accuracy"],
                "macro_f1": ((a["macro_f1"] - b["macro_f1"])
                             if a["macro_f1"] is not None and b["macro_f1"] is not None
                             else None),
            }

    return {
        "session": data.name,
        "protocol_version": data.protocol_version,
        "rehearsal": data.rehearsal,
        "condition": cond.label(),
        "enrolled_classes": [classes[c] for c in enrolled],
        "ticks_per_centroid": ticks_per_centroid,
        "incomplete_centroids": [classes[c] for c in incomplete],
        "calibration_valid": not incomplete,
        "n_enrol": int(enrol_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_test_trials": len(ids),
        "unscored_trials": {str(t): attempted[t] for t in unscored},
        "classes_with_a_scored_trial": scored_classes,
        "schema_classes": classes,
        "centroid_separation": separation,
        "centroid_drift_enrol_to_test": drift,
        "replay_fidelity": {
            "argmax_agreement": data.fidelity[0],
            "max_probability_difference": data.fidelity[1],
            "faithful": data.faithful,
        },
        "primary_macro_f1_valid": bool(complete and not incomplete and not unscored),
        "accuracy": {
            "argmax_full": float((full == y).mean()),
            "argmax_restricted": float((restricted == y).mean()),
            "adapter": float((predicted == y).mean()),
            **{name: float((p == y).mean()) for name, p in sorted(shadow.items())},
        },
        "trial_majority": trial_majority,
        # Present only in shadow mode. A personal-only run must produce the same result
        # file it produced before these options existed, down to the key set.
        **({"shadow_deltas": shadow_deltas, "shadow_alpha": float(hybrid_alpha)}
           if shadow else {}),
        "confusion": {
            "argmax_restricted": sa.confusion_counts(ty, tp_restricted, classes),
            "adapter": sa.confusion_counts(ty, tp_adapter, classes),
            **{name: sa.confusion_counts(ty, votes, classes)
               for name, votes in sorted(shadow_trial.items())},
        },
        "per_class": {
            classes[c]: {
                "n": int((y == c).sum()),
                "argmax_full": int((full == y)[y == c].sum()),
                "argmax_restricted": int((restricted == y)[y == c].sum()),
                "adapter": int((predicted == y)[y == c].sum()),
                **{name: int((p == y)[y == c].sum()) for name, p in sorted(shadow.items())},
            }
            for c in sorted(set(y.tolist()))
        },
        "_arrays": {"y": y, "predicted": predicted, "top": top, "margin": margin},
    }


def abstention_sweep(result: dict, thresholds: list[float], criterion: str) -> list[dict]:
    """Accuracy when the adapter is allowed to decline, by one of two criteria.

    Reported because raw confidence cannot serve as the gate here: on the third
    subject the deployed model's top score was *higher* when it was wrong (0.572)
    than when it was right (0.494).

    The two criteria answer different questions and neither subsumes the other:

    - `margin` -- how much the winning centroid beat the runner-up. Detects a sample
      caught between two classes. It does *not* detect a sample that is far from
      every centroid: something unlike anything enrolled can still win by a wide
      margin and be confidently wrong.
    - `distance` -- absolute similarity to the winning centroid. Detects exactly that
      case, and is the one an "I don't recognise this expression" gate would need.

    Thresholds are not comparable between criteria, nor between `--metric` choices:
    cosine similarity lies in [-1, 1] while euclidean reports a negated distance.

    This sweep is exploratory. It is fitted and evaluated on the same correlated
    test ticks as everything else, and no threshold so far helps every session at
    once, so it must not be read as a recommended operating point.
    """
    a = result["_arrays"]
    statistic = a["margin"] if criterion == "margin" else a["top"]
    out = []
    for t in thresholds:
        keep = statistic >= t
        out.append(
            {
                "criterion": criterion,
                "threshold": t,
                "coverage": float(keep.mean()),
                "accuracy": float((a["predicted"][keep] == a["y"][keep]).mean())
                if keep.any()
                else None,
            }
        )
    return out


# ------------------------------------------------------------------ printing


def parse_ticks(text: str) -> set[int]:
    """`1-5`, `0,2,4`, or a mix."""
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        elif part:
            out.add(int(part))
    if not out:
        raise argparse.ArgumentTypeError(f"no ticks in {text!r}")
    return out


# Exactly what a result file may hold. A whitelist rather than a blocklist, because the
# output of this analysis must not quietly become a second copy of the face signal: no
# centroid vectors, no score vectors, no AU windows. The consented data record is the
# research CSV, and adding a field to `evaluate` should require naming it here before it
# reaches disk.
def load_population_profile(path: str, model_path: Path) -> tuple["np.ndarray", dict]:
    """The EmojiHero population centroids, validated against the model on disk.

    Imported lazily from `emojihero_calibration_study` rather than reimplemented, and
    lazily rather than at module scope because that module imports *this* one: the two
    share the log transform, the centroids, the metric and the tie rule, and the direction
    of the dependency is emojihero -> calibration. Loading inside the function keeps that
    direction true while still reusing the loader and its schema, model-hash, source-split,
    epsilon and tamper checks.

    One caveat this cannot check, and the report must state: these centroids describe how
    the teacher responds to *EmojiHero* input -- 30-frame captures mean-pooled three at a
    time. A live window is five instantaneous 10 Hz samples. The blend is well defined
    because both live in the model's seven-dimensional output space, but the population
    term is being applied off the distribution it was estimated on.
    """
    import emojihero_calibration_study as eh

    model_sha = __import__("hashlib").sha256(model_path.read_bytes()).hexdigest()
    return eh.load_population_centroids(Path(path).expanduser(), model_sha=model_sha)


RESULT_FIELDS = (
    "session",
    "protocol_version",
    "rehearsal",
    "condition",
    "enrolled_classes",
    "ticks_per_centroid",
    "incomplete_centroids",
    "calibration_valid",
    "n_enrol",
    "n_test",
    "n_test_trials",
    "unscored_trials",
    "classes_with_a_scored_trial",
    "schema_classes",
    "primary_macro_f1_valid",
    "centroid_separation",
    "centroid_drift_enrol_to_test",
    "replay_fidelity",
    "accuracy",
    "trial_majority",
    "shadow_deltas",
    "shadow_alpha",
    "confusion",
    "per_class",
)

_PRIVATE_KEYS = ("_arrays", "_abstention")


def write_results(results: list[dict], settings: dict, results_dir: str | None) -> list[Path]:
    """One file per session, named after it, and never overwritten.

    A single shared filename invites a second run to silently replace the first, which is
    how a result stops being attributable to the recording and the settings that produced
    it. An existing file is an error: rename or remove it deliberately.
    """
    if results_dir is not None:
        out_dir = Path(results_dir)
    else:
        root = sa.find_repo_root()
        out_dir = (root / "Analysis" / "results") if root else Path("Analysis/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    planned = [(out_dir / f"calibration_{r['session']}.json", r) for r in results]
    clashes = [p for p, _ in planned if p.exists()]
    if clashes:
        listing = "\n".join(f"  {p}" for p in clashes)
        raise SystemExit(
            "refusing to overwrite an existing result file:\n"
            f"{listing}\n"
            "Remove or rename it if the previous run should be replaced."
        )

    for path, r in planned:
        body = {k: r[k] for k in RESULT_FIELDS if k in r}
        path.write_text(
            json.dumps({**settings, **body}, indent=2), encoding="utf-8"
        )

    return [p for p, _ in planned]


def print_trial_majority(results: list[dict], sessions: list[str]) -> None:
    by_session = {r["session"]: r for r in results}
    print(rule("trial-majority vote (primary) -- accuracy / macro-F1"))
    print("  tie rule: most ticks, then the final tracked tick, then class order")
    label_width = 34
    print(" " * label_width + "".join(f"{s[:13]:>15}" for s in sessions))
    for key, label in (
        ("argmax_full", "argmax, all classes (deployed)"),
        ("argmax_restricted", "argmax, enrolled classes only"),
        ("adapter", "per-person adapter"),
    ):
        line = f"  {label:<{label_width - 2}}"
        for s in sessions:
            cell = by_session[s]["trial_majority"][key]
            line += f"{num(cell['accuracy']):>7}/{num(cell['macro_f1']):<7}"
        print(line)
    print()
    for s in sessions:
        r = by_session[s]
        if not r["primary_macro_f1_valid"]:
            missing = sorted(set(r.get("schema_classes", r["enrolled_classes"]))
                             - set(r["classes_with_a_scored_trial"]))
            print(f"  {s[:30]}: macro-F1 above is NOT the primary figure --")
            if missing:
                print(f"    no scored trial for {', '.join(missing)}. A class with no")
                print("    support is left out of the mean, which RAISES it, so this")
                print("    number is not comparable with a complete seven-class one.")
            if r["incomplete_centroids"]:
                print(f"    incomplete centroids: {', '.join(r['incomplete_centroids'])}")
            if r["unscored_trials"]:
                print(f"    unscored trials: {len(r['unscored_trials'])}")
    print()


def print_coverage(results: list[dict], sessions: list[str]) -> None:
    by_session = {r["session"]: r for r in results}
    print(rule("coverage and enrolment quality (diagnostic -- no thresholds)"))
    for s in sessions:
        r = by_session[s]
        sep = r.get("centroid_separation")
        drift = r.get("centroid_drift_enrol_to_test") or {}
        print(f"  {s[:30]}")
        print(f"    ticks per centroid   {r['ticks_per_centroid']}")
        print(f"    calibration valid    {r['calibration_valid']}"
              + (f"  (incomplete: {', '.join(r['incomplete_centroids'])})"
                 if r["incomplete_centroids"] else ""))
        print(f"    test trials scored   {r['n_test_trials']}"
              + (f"  unscored: {len(r['unscored_trials'])}"
                 if r["unscored_trials"] else ""))
        if sep:
            print(f"    centroid separation  mean {sep['mean_pairwise_cosine']:.5f}  "
                  f"min {sep['min']:.5f}  max {sep['max']:.5f}")
            print("                         (nearer 1 means the centroids are harder to")
            print("                          tell apart, so the decision is more fragile)")
        if drift:
            values = sorted(drift.items(), key=lambda kv: kv[1])
            worst = ", ".join(f"{k} {v:.4f}" for k, v in values[:3])
            print(f"    drift enrol->test    mean {sum(drift.values())/len(drift):.5f}"
                  f"   lowest: {worst}")
    print()


def print_validity(datasets: list[Replayed], enrol: set[int], test: set[int],
                   *, role_split: bool = False) -> None:
    print(rule("validity -- read before the numbers"))
    trials = {len(set(d.trial.tolist())) for d in datasets}
    per_class = {
        len(set(d.trial[d.prompted == c].tolist()))
        for d in datasets
        for c in set(d.prompted.tolist())
    }
    reuse = {d.frame_reuse for d in datasets}
    prompted_classes = sorted({d.classes[c] for d in datasets for c in set(d.prompted.tolist())})
    missing = sorted({c for d in datasets for c in d.classes} - set(prompted_classes))

    roles = sorted({r for d in datasets for r in set(d.role.tolist()) if r})

    print(f"  trials per session      {sorted(trials)}")
    print(f"  trials per class        {sorted(per_class)}")
    print(f"  frames shared between   {sorted(reuse)}  <- 0 means the windows are disjoint;")
    print("  consecutive windows             adjacent ticks are still correlated")
    print(f"  never prompted          {', '.join(missing) or '(none)'}")
    print(f"  tick positions from     {', '.join(sorted({d.tick_source for d in datasets}))}")
    print(f"  block roles present     {', '.join(roles) or '(none -- pre-session-local-v1)'}")
    print()

    if role_split:
        print("  Enrolment and test are SEPARATE TRIALS of the same states, so the")
        print("  within-trial leak is gone. What remains, and must be said with every")
        print("  number below:")
        print("    - both blocks are one sitting, so this says nothing about another day;")
        print("    - the evaluation block always follows the calibration block, so order,")
        print("      fatigue and learning are inside the effect, not controlled for;")
        print("    - ticks within a trial are strongly correlated (measured design effect")
        print("      of roughly 5-9x), so 9 ticks are not 9 observations and the trial is")
        print("      the unit worth reading;")
        print("    - the unit of any decision is the PARTICIPANT, not the trial.")
    else:
        print(f"  enrolment ticks {sorted(enrol)}   test ticks {sorted(test)}")
        print("  Enrolment and test come from the SAME continuous expression when trials")
        print("  per class is 1. This is a within-trial reference condition, optimistically")
        print("  biased, and NOT a validation. The gain it reports was measured decaying to")
        print("  nothing as the two tick sets are moved apart. For the real split pass")
        print("  --enrol-role calibration --test-role evaluation on a session-local-v1 file.")
    print()


def print_results(results: list[dict], sessions: list[str]) -> None:
    by_session = {r["session"]: r for r in results}
    label_width = 34
    print(rule("accuracy on held-out ticks"))
    print(" " * label_width + "".join(f"{s[:13]:>15}" for s in sessions))
    for key, label in (
        ("argmax_full", "argmax, all classes (deployed)"),
        ("argmax_restricted", "argmax, enrolled classes only"),
        ("adapter", "per-person adapter"),
    ):
        line = f"  {label:<{label_width - 2}}"
        for s in sessions:
            line += f"{num(by_session[s]['accuracy'][key]):>15}"
        print(line)
    print()
    print("  `argmax, enrolled classes only` is the fair comparison for the adapter:")
    print("  both are restricted to the same class set. The deployed row is included")
    print("  because it is what actually runs on the headset.")
    print()


def print_shadow_comparison(results: list[dict], sessions: list[str]) -> None:
    """The four methods and the deltas that separate two different questions.

    `hybrid - personal` says whether the population term earned its place.
    `hybrid - raw` is the deployment gate, and the two are not interchangeable: a hybrid
    that beats the raw model may be doing it entirely with the personal term.
    """
    by_session = {r["session"]: r for r in results}
    print(rule("four methods, trial-majority vote -- accuracy / macro-F1"))
    label_width = 34
    print(" " * label_width + "".join(f"{s[:13]:>15}" for s in sessions))
    for key, label in (
        ("argmax_full", "raw (deployed argmax)"),
        ("population", "population-only"),
        ("adapter", "personal-only"),
        ("hybrid", "hybrid"),
    ):
        line = f"  {label:<{label_width - 2}}"
        for s in sessions:
            cell = by_session[s]["trial_majority"].get(key)
            line += (f"{num(cell['accuracy']):>7}/{num(cell['macro_f1']):<7}"
                     if cell else f"{'-':>15}")
        print(line)
    print()

    print(rule("shadow deltas -- the two of these answer different questions"))
    for key, label in (
        ("hybrid_minus_personal", "hybrid - personal   (did the population term help?)"),
        ("hybrid_minus_raw", "hybrid - raw        (DEPLOYMENT GATE)"),
        ("population_minus_raw", "population - raw"),
        ("personal_minus_raw", "personal - raw"),
    ):
        line = f"  {label:<52}"
        for s in sessions:
            cell = (by_session[s].get("shadow_deltas") or {}).get(key)
            if not cell or cell["macro_f1"] is None:
                line += f"{'-':>12}"
            else:
                line += f"{cell['macro_f1']:>+12.4f}"
        print(line)
    print()
    print("  Beating the raw model is not evidence that the population centroid did any")
    print("  of the work -- only hybrid - personal is. And a positive hybrid - raw on one")
    print("  session is a case study, not a gate: the gate is defined over participants.")
    print()


def print_per_class(results: list[dict], sessions: list[str]) -> None:
    by_session = {r["session"]: r for r in results}
    print(rule("per class, held-out ticks (hits / n)"))
    header = f"  {'class':<12}"
    for s in sessions:
        header += f"{s[:13]:>26}"
    print(header)
    print(f"  {'':<12}" + "".join(f"{'full  restr  adapter':>26}" for _ in sessions))
    names = sorted({c for r in results for c in r["per_class"]})
    for name in names:
        line = f"  {name:<12}"
        for s in sessions:
            cell = by_session[s]["per_class"].get(name)
            if cell is None:
                line += f"{'-':>26}"
                continue
            n = cell["n"]
            for key, width in (
                ("argmax_full", 10),
                ("argmax_restricted", 7),
                ("adapter", 9),
            ):
                line += "{:>{w}}".format(f"{cell[key]}/{n}", w=width)
        print(line)
    print()


def print_abstention(
    results: list[dict], sessions: list[str], thresholds: list[float], criterion: str
) -> None:
    by_session = {r["session"]: r for r in results}
    described = (
        "margin to runner-up centroid"
        if criterion == "margin"
        else "absolute similarity to winning centroid"
    )
    print(rule(f"adapter with abstention ({described})"))
    print("  exploratory -- swept on the same correlated ticks, no threshold helps")
    print("  every session at once. Not an operating point.")
    print(f"  {'threshold':<12}" + "".join(f"{s[:13]:>24}" for s in sessions))
    print(f"  {'':<12}" + "".join(f"{'coverage  accuracy':>24}" for _ in sessions))
    for i, t in enumerate(thresholds):
        line = f"  {t:<12.3f}"
        for s in sessions:
            row = by_session[s]["_abstention"][i]
            line += f"{pct(row['coverage']):>13}{num(row['accuracy']):>11}"
        print(line)
    print()


# ---------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="*", help="predictions.csv, session folder, or parent")
    parser.add_argument("--model", help="path to sparseVR_teacher_model.onnx")
    parser.add_argument(
        "--space",
        default="logscore",
        choices=["logscore", "score", "aumax", "auflat"],
        help="feature space the adapter is fitted in (default: logscore)",
    )
    parser.add_argument(
        "--metric", default="cosine", choices=["cosine", "euclidean"],
        help="nearest-centroid distance (default: cosine)",
    )
    parser.add_argument(
        "--neutral", default="none", choices=["none", "rest", "prompted"],
        help="how Neutral enters the adapter's class set (default: none -- it cannot)",
    )
    parser.add_argument(
        "--epsilon", type=float, default=DEFAULT_EPSILON,
        help=f"floor under the log transform (default: {DEFAULT_EPSILON:g})",
    )
    parser.add_argument(
        "--enrol-role", choices=[sa.ROLE_CALIBRATION, sa.ROLE_EVALUATION],
        help="fit the centroids on this block. With --test-role this is the "
             "trial-disjoint split the study is about, and it needs a session-local-v1 "
             "recording. Without it the tick split below is used instead.",
    )
    parser.add_argument(
        "--test-role", choices=[sa.ROLE_CALIBRATION, sa.ROLE_EVALUATION],
        help="score on this block. Must differ from --enrol-role.",
    )
    parser.add_argument("--enrol-ticks", type=parse_ticks, default=parse_ticks("1-5"))
    parser.add_argument("--test-ticks", type=parse_ticks, default=parse_ticks("6-9"))
    parser.add_argument(
        "--abstain", default="0,0.01,0.02,0.05,0.1",
        help="comma-separated thresholds to sweep (exploratory; see README)",
    )
    parser.add_argument(
        "--abstain-on", default="margin", choices=["margin", "distance"],
        help="margin to the runner-up centroid, or absolute similarity to the winner "
             "(default: margin). Thresholds are not comparable between the two.",
    )
    parser.add_argument(
        "--population-centroids",
        help="EmojiHero population centroid artifact. With --hybrid-alpha this adds the "
             "population-only and hybrid methods to the same test ticks (the VR shadow "
             "evaluation). Without both, nothing about the output changes.",
    )
    parser.add_argument(
        "--hybrid-alpha", type=float,
        help="weight on the personal centroid in alpha*personal + (1-alpha)*population. "
             "Frozen at 0.40 by the EmojiHero validation phase; see "
             "Docs/CALIBRATION_STUDY_PROTOCOL.md section 9.",
    )
    parser.add_argument(
        "--compare-all", action="store_true",
        help="print the four-method comparison table. Requires the two options above.",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="show the replay-fidelity table (it is always verified, and a mismatch "
             "always fails, with or without this flag)",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="directory for calibration_<session_id>.json (default: Analysis/results). "
             "One file per session, and an existing file is never overwritten.",
    )
    parser.add_argument("--json", help="write every metric to this file as well")
    args = parser.parse_args(argv)

    if bool(args.population_centroids) != (args.hybrid_alpha is not None):
        raise SystemExit(
            "--population-centroids and --hybrid-alpha must be given together: a "
            "population profile with no weight scores nothing, and a weight with no "
            "profile has nothing to blend."
        )
    if args.compare_all and not args.population_centroids:
        raise SystemExit(
            "--compare-all needs --population-centroids and --hybrid-alpha; without them "
            "only the raw model and the personal adapter exist to compare."
        )
    if bool(args.enrol_role) != bool(args.test_role):
        raise SystemExit("--enrol-role and --test-role must be given together")
    if args.enrol_role and args.enrol_role == args.test_role:
        raise SystemExit(
            "--enrol-role and --test-role must differ; fitting and scoring on the same "
            "block is the leakage this split exists to remove"
        )

    files = sa.discover(args.paths) if args.paths else sa.autodiscover()[0]
    if not files:
        print("no session logs found; see analyze_sessions.py --where", file=sys.stderr)
        return 1

    model_path = find_model(args.model)
    datasets = [replay(sa.load_session(p), model_path) for p in files]

    print("=" * WIDTH)
    print(f"calibration study -- {len(datasets)} session(s)")
    print(f"model  {model_path}")
    print("=" * WIDTH)
    print()

    # Fidelity is never optional: if the replayed scores do not match what the headset
    # logged, the ONNX file on disk is not the model that produced these recordings and
    # every number below describes the wrong thing. `--check` only controls whether the
    # table is shown; the failure is enforced either way.
    failed = [d for d in datasets if not d.faithful]
    if args.check or failed:
        print(rule("replay fidelity against the logged scores"))
        for d in datasets:
            agree, diff = d.fidelity
            flag = "" if d.faithful else "   <-- MISMATCH"
            print(f"  {d.name[:30]:<32} agreement {num(agree)}  max prob diff {diff:.2e}{flag}")
        print()

    if failed:
        names = ", ".join(d.name for d in failed)
        print(
            f"replay does not reproduce the logged scores for: {names}\n"
            f"(tolerance {FIDELITY_TOLERANCE:g}). The model at\n"
            f"  {model_path}\n"
            "is not the one these sessions were recorded with. Refusing to report "
            "numbers measured\non a different model; pass --model to point at the "
            "right file.",
            file=sys.stderr,
        )
        return 2

    rehearsals = [d for d in datasets if d.rehearsal]
    if rehearsals:
        print(rule("NON-PRODUCTION REHEARSAL"))
        for d in rehearsals:
            print(f"  {d.name} was recorded with fast_rehearsal on.")
        print("  A rehearsal records three ticks per trial, so after dropping tick 0 a")
        print("  centroid rests on two samples. These numbers check that the pipeline")
        print("  runs; they are not evidence about calibration and must not be reported")
        print("  as such.")
        print()

    role_split = bool(args.enrol_role)
    print_validity(datasets, args.enrol_ticks, args.test_ticks, role_split=role_split)

    cond = Condition(args.space, args.metric, args.neutral, args.epsilon)
    thresholds = [float(t) for t in args.abstain.split(",") if t.strip()]
    print(f"  condition: {cond.label()}, epsilon {cond.epsilon:g}")
    if role_split:
        print(f"  split: fit on {args.enrol_role}, score on {args.test_role} "
              f"(trial-disjoint), ticks {min(SCORED_TICKS)}-{max(SCORED_TICKS)}")
    else:
        print(f"  split: ticks {sorted(args.enrol_ticks)} -> {sorted(args.test_ticks)} "
              "within the same trial (reference condition, not a validation)")
    print()

    population_M, population_meta = (None, None)
    if args.population_centroids:
        population_M, population_meta = load_population_profile(
            args.population_centroids, model_path)
        print(rule("VR shadow evaluation -- population and hybrid on the same ticks"))
        print(f"  artifact        {args.population_centroids}")
        print(f"  semantic hash   {population_meta['semantic_sha256']}")
        print(f"  built from      the EmojiHero {population_meta['source_split']} split, "
              f"{population_meta['n_participants']} participants")
        print(f"  alpha           {args.hybrid_alpha:.2f}  "
              "(hybrid = alpha*personal + (1-alpha)*population)")
        print()
        print("  The population centroids describe how this model answers EmojiHero")
        print("  input -- 30-frame captures mean-pooled three at a time. A live window is")
        print("  five instantaneous 10 Hz samples. The blend is well defined, because both")
        print("  live in the model's seven-dimensional output space, but the population")
        print("  term is applied off the distribution it was estimated on.")
        print()

    results = []
    for d in datasets:
        if role_split:
            enrol_mask, test_mask = role_masks(d, args.enrol_role, args.test_role)
            r = evaluate(d, cond, enrol_mask, test_mask,
                         required_ticks=TICKS_REQUIRED_PER_CENTROID,
                         test_role=args.test_role,
                         population_M=population_M, hybrid_alpha=args.hybrid_alpha)
        else:
            enrol_mask, test_mask = split_masks(d, args.enrol_ticks, args.test_ticks)
            r = evaluate(d, cond, enrol_mask, test_mask,
                         population_M=population_M, hybrid_alpha=args.hybrid_alpha)
        r["_abstention"] = abstention_sweep(r, thresholds, args.abstain_on)
        results.append(r)

    sessions = [r["session"] for r in results]
    print_results(results, sessions)
    print_trial_majority(results, sessions)
    if args.compare_all:
        print_shadow_comparison(results, sessions)
    print_per_class(results, sessions)
    print_coverage(results, sessions)
    print_abstention(results, sessions, thresholds, args.abstain_on)

    settings = {
        "model": str(model_path),
        "condition": cond.label(),
        "space": cond.space,
        "metric": cond.metric,
        "neutral": cond.neutral,
        "epsilon": cond.epsilon,
        "transform": "log(clip(score, epsilon))",
        "centroid": "mean of the per-tick transformed vectors",
        "tie_rule": "most_ticks -> final_tracked_tick -> class_order",
        "abstain_on": args.abstain_on,
        **({"shadow": {
            "population_artifact": args.population_centroids,
            "population_semantic_sha256": population_meta["semantic_sha256"],
            "population_source_split": population_meta["source_split"],
            "population_participants": population_meta["n_participants"],
            "hybrid_alpha": args.hybrid_alpha,
            "hybrid_formula": "alpha*personal + (1-alpha)*population",
            "domain_caveat": "the population centroids were estimated on EmojiHero input "
                             "(30 frames mean-pooled three at a time); a live window is "
                             "five instantaneous 10 Hz samples",
        }} if population_meta else {}),
        "split": ({"kind": "role", "enrol_role": args.enrol_role,
                   "test_role": args.test_role, "ticks": sorted(SCORED_TICKS)}
                  if role_split else
                  {"kind": "tick", "enrol_ticks": sorted(args.enrol_ticks),
                   "test_ticks": sorted(args.test_ticks)}),
        "ticks_required_per_centroid": (TICKS_REQUIRED_PER_CENTROID
                                        if role_split else None),
        "tick_sources": sorted({d.tick_source for d in datasets}),
    }

    written = write_results(results, settings, args.results_dir)
    for path in written:
        print(f"wrote {path}")

    if args.json:
        payload = [{k: v for k, v in r.items() if k not in _PRIVATE_KEYS}
                   for r in results]
        Path(args.json).write_text(
            json.dumps({**settings, "sessions": payload}, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
