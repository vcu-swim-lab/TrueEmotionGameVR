#!/usr/bin/env python3
"""Personal, population and hybrid recalibration of the teacher's output on EmojiHeroVR.

    python Analysis/emojihero_calibration_study.py --build-population
    python Analysis/emojihero_calibration_study.py --split validation --population-centroids ...
    python Analysis/emojihero_calibration_study.py --split test --frozen-config ...

Four methods are scored on exactly the same evaluation sequences:

    raw           argmax of the model's seven softmax scores, unchanged
    population    nearest of seven centroids built from the EmojiHero TRAINING split,
                  participant-balanced -- one profile for everybody
    personal      nearest of seven centroids built from one labelled sequence per
                  emotion from the participant being scored
    hybrid        nearest of  H_e = alpha * P_e + (1 - alpha) * G_e,  one global alpha
                  chosen on the validation split and frozen before test is opened

The questions, stated so they can only be answered one way:

    Do personal log-score centroids, built from one complete labelled EmojiHero sequence
    per emotion, improve on the raw ONNX argmax when scoring *separate* sequences from the
    same participant?

    And does blending in a population centroid built from participants the person has
    never met add anything on top of that? The comparison that answers the second question
    is `hybrid - personal`, not `hybrid - raw`: beating the raw model is not evidence that
    the population term did any of the work.

alpha = 1 is personal-only and alpha = 0 is population-only, so the grid contains both
baselines as special cases. If validation selects alpha = 1, the honest reading is that
the EmojiHero population adds nothing over personal enrolment, and the report says so.

This is the **offline analogue** of the live `session-local-v1` protocol, not a repeat of
it. The adapter arithmetic is identical -- the same log transform, the same centroids, the
same cosine rule, the same tie rule, imported from `calibration_study.py` rather than
copied -- but the data underneath is not:

  live session-local-v1                     EmojiHero
  ---------------------------------------   ----------------------------------------
  ~10 s trial, 9 scored ticks               30-frame sequence, 6 scored windows
  ring buffer cleared between predictions   stride-1 windows share 4 of their 5 frames
  K then D in one sitting, one headset      pre-recorded sequences across emoji/levels
  our prompt protocol                       the EmojiHero game protocol

So a gain here is **not** a bound on the live gain in either direction: the familiar
capture domain could inflate it, and a higher baseline could compress it through a
ceiling. It is a mechanism control on 16 participants, which is 15 more than the live
study has. The live 6-8 participant pilot is still required, and nothing here replaces it.

Provenance, which limits what the two splits can carry (see `--split`):

  - the teacher was trained by gradient on EmojiHero's *training* split;
  - EmojiHero's *validation* split contributed 2,310 of the 3,279 windows in the tensor
    `model.fit` received as `validation_data`, and early stopping monitored `val_accuracy`
    with best-weight restore -- so ~70.4% of the model-selection signal came from these
    participants, and this split is suitable for developing the adapter but is not
    independent evidence of anything about the base model;
  - EmojiHero's *test* split entered neither gradients nor model selection, but has been
    opened before when reporting teacher performance. It is reused here as a confirmatory
    split for a new, pre-frozen hypothesis, which is legitimate only because the adapter
    has never been tuned on it.

The model is never retrained, fine-tuned or re-exported. Only the interpretation of its
seven output scores changes.

Needs numpy, onnxruntime and openpyxl (the corpus ships as .xlsx):

    pip install numpy onnxruntime openpyxl
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import au_names
import calibration_study as cs
import session_analysis as sa
import teacher_preprocessing as tp
from analyze_sessions import WIDTH, num, rule

try:
    import numpy as np
except ImportError:  # pragma: no cover - environment guard
    raise SystemExit(
        "emojihero_calibration_study.py needs numpy, onnxruntime and openpyxl:\n"
        "  pip install numpy onnxruntime openpyxl"
    )


# ------------------------------------------------------------------ constants

# Class order. Index is the emotion_id carried in the last field of a sequence ID and is
# also the `Label` column's integer, which the integrity gate asserts against each other.
EMOTIONS: tuple[str, ...] = (
    "Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise",
)
N_EMOTIONS = len(EMOTIONS)

# Preprocessing, transcribed from the teacher notebook's EmojiHero path. Note that the
# pooling factor is 3 here and 6 in `teacher_preprocessing.DOWNSAMPLE_FACTOR`: that module
# reproduces the 60 Hz ActingRecordings path, while EmojiHero arrives as 30 frames which the
# notebook pools three at a time down to 10. The operation -- mean of N consecutive rows,
# never a resample against a timestamp -- is the same, so `tp.downsample` is reused with a
# different factor rather than reimplemented.
FRAMES_PER_SEQUENCE = 30
POOL_FACTOR = 3
POOLED_FRAMES = FRAMES_PER_SEQUENCE // POOL_FACTOR      # 10
WINDOW_LENGTH = 5
WINDOW_STRIDE = 1
WINDOWS_PER_SEQUENCE = (POOLED_FRAMES - WINDOW_LENGTH) // WINDOW_STRIDE + 1   # 6

# The two windows that share no frame: [0:5] and [5:10]. Used only as a secondary
# sensitivity check -- six stride-1 windows are six views of one performance, and this pair
# is the most independent subset the sequence admits.
NON_OVERLAPPING_WINDOWS: tuple[int, ...] = (0, 5)
ALL_WINDOWS: tuple[int, ...] = tuple(range(WINDOWS_PER_SEQUENCE))

# Deterministic profile selection. `MASTER_SEED` and the profile indices are the whole of
# the randomness in this study; everything downstream is a pure function of them, so a run
# is reproducible from the config alone without shipping the selections.
MASTER_SEED = 42
VALIDATION_PROFILE_COUNT = 1000
PRIMARY_TEST_PROFILE = 0
TEST_ROBUSTNESS_PROFILES = 1000

# The four methods, scored on identical evaluation sequences. `raw` carries no centroids.
METHODS: tuple[str, ...] = ("raw", "population", "personal", "hybrid")

# One global blending weight, swept at 0.05. Deliberately not per class, not per
# participant and not quality-dependent: each of those is a second thing fitted on the same
# validation participants, and the grid already contains both endpoints as special cases.
ALPHA_GRID: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(21))

# Population centroids come from the training split and from nowhere else. A validation or
# test sequence entering them would make the population term partly personal and the
# comparison meaningless.
POPULATION_SOURCE_SPLIT = "training"
POPULATION_SCHEMA_VERSION = "emojihero-population-centroids-v1"
RESULT_SCHEMA_VERSION = "emojihero-hybrid-calibration-v1"

# Every emotion needs support from several training participants before a population
# centroid means anything. Well below the 20 participants the corpus has, so this fires
# only if the corpus is not the one described.
POPULATION_MIN_PARTICIPANTS_PER_EMOTION = 5

# A participant needs, for every one of the seven emotions, one sequence to enrol and at
# least one left over to score. Two sequences per emotion, i.e. fourteen in the right
# places. Expressed as a rule rather than as a list of participant IDs so that a corpus
# change moves the eligibility set by itself.
SEQUENCES_REQUIRED_PER_EMOTION = 2

# What the corpus must look like before any number is produced.
EXPECTED_SEQUENCES = {"training": 964, "validation": 385, "test": 378}
EXPECTED_PARTICIPANTS = {"training": 20, "validation": 8, "test": 8}

SPLITS = ("training", "validation", "test")
ANALYSABLE_SPLITS = ("validation", "test")

# Softmax rows must sum to 1. Float32 over seven classes drifts by ~1e-7; anything larger
# means the graph does not end where this study assumes it does.
SOFTMAX_TOLERANCE = 1e-4

FROZEN_SETTINGS = {
    "feature_space": "logscore",
    "transform": "log(clip(score, epsilon))",
    "epsilon": cs.DEFAULT_EPSILON,
    "centroid": "arithmetic mean of the per-window log-score vectors of one sequence",
    "metric": "cosine",
    "class_set": list(EMOTIONS),
    "calibration_unit": "one complete sequence per emotion",
    "evaluation_unit": "the participant's remaining complete sequences",
    "sequence_split": "sequence-disjoint; no sequence contributes to both sides",
    "methods": list(METHODS),
    "population_centroid": {
        "source_split": POPULATION_SOURCE_SPLIT,
        "excluded": "validation and test sequences never enter a population centroid",
        "estimator": "hierarchical and participant-balanced: window log-scores -> "
                     "sequence centroid -> participant-emotion centroid -> unweighted "
                     "mean over training participants",
        "why_balanced": "a participant with many sequences of an emotion would otherwise "
                        "dominate that emotion's population centroid",
        "min_participants_per_emotion": POPULATION_MIN_PARTICIPANTS_PER_EMOTION,
    },
    "hybrid": {
        "formula": "H_e = alpha * P_e + (1 - alpha) * G_e",
        "alpha_scope": "one global scalar; not per class, per participant or "
                       "quality-dependent",
        "alpha_grid": list(ALPHA_GRID),
        "alpha_selected_on": "the EmojiHero validation split only",
        "alpha_objective": "highest participant-equal mean sequence-level seven-class "
                           "macro-F1: mean over profiles within a participant first, then "
                           "an unweighted mean over eligible participants",
        "alpha_tie_break": "the larger alpha, i.e. the more personal and less "
                           "population-dependent of two equal options",
        "alpha_endpoints": "alpha=0 is population-only, alpha=1 is personal-only",
    },
    "preprocessing": {
        "tongue_channels": "zero-filled to 70, model-input representation only",
        "pooling": f"mean of {POOL_FACTOR} consecutive frames, {FRAMES_PER_SEQUENCE} -> "
                   f"{POOLED_FRAMES}",
        "normalization": "none",
        "mean_centering_or_max_scaling": "none",
        "window_length": WINDOW_LENGTH,
        "window_stride": WINDOW_STRIDE,
        "windows_per_sequence": WINDOWS_PER_SEQUENCE,
    },
    "aggregation": "majority vote over the sequence's windows",
    "tie_rule": "most windows -> chronologically last window -> class order",
    "primary_unit": "sequence, then participant",
    # The primary analysis, named before the test split was opened. Exactly one comparison
    # is primary; everything else in this study is secondary and cannot be promoted after
    # the fact. `hybrid - personal` is the primary one because it is the only comparison
    # that isolates what the population term contributed -- `hybrid - raw` moves when the
    # personal term alone does the work.
    "primary_analysis": {
        "metric": "participant-level seven-class sequence macro-F1",
        "comparison": "hybrid minus personal",
        "inference_unit": "participant",
        "statistical_test": "exact two-sided paired sign test over participants",
        "alpha_source": "the frozen config's selected_alpha; never the command line, and "
                        "never re-chosen on the split being scored",
        "profiles_are_not_a_sample_size":
            "the 1000 calibration profiles are repeated re-draws of the same participants' "
            "sequences, not independent observations. They are averaged within a "
            "participant to give ONE value per participant, and n is the number of "
            "eligible participants. No test is ever computed over profiles.",
        "secondary_only": [
            "hybrid minus raw", "personal minus raw", "population minus raw",
            "sequence-majority accuracy for every method and delta",
            "window-level metrics",
            "per-class sequence recall and its deltas",
            "profile sensitivity distributions",
            "the non-overlapping two-window condition",
            "centroid separation, drift and personal-to-population similarity",
            "level and emoji diagnostics",
        ],
    },
    "headline_comparison_for_the_population_term": "hybrid minus personal-only; beating "
        "raw is not evidence that the population centroid contributed anything",
    "missing_class_rule": "a participant missing a scored sequence for any of the seven "
                          "classes yields no valid seven-class macro-F1 and is excluded "
                          "from the primary population",
    "eligibility_rule": f"at least {SEQUENCES_REQUIRED_PER_EMOTION} sequences for every "
                        "emotion, so that one can be enrolled and one scored",
    "master_seed": MASTER_SEED,
    "primary_test_profile_index": PRIMARY_TEST_PROFILE,
    "test_robustness_profile_indices": f"0..{TEST_ROBUSTNESS_PROFILES - 1}",
    "validation_profile_count": VALIDATION_PROFILE_COUNT,
    "profile_selection": "sha256(master_seed|split|participant|emotion|profile_index) "
                         "indexed into that participant-emotion's sequences sorted by "
                         "(timestamp, sequence_id)",
    "level_and_emoji_id": "diagnostic only; never used to select sequences, drop "
                          "sequences, or choose a profile",
}


# -------------------------------------------------------------------- corpus


@dataclass(frozen=True)
class Sequence:
    """One EmojiHero performance: 30 frames of 63 measured action units."""

    sequence_id: str
    timestamp: int
    set_id: int
    participant: int
    level: int
    emoji: int
    emotion: int                 # 0-6, index into EMOTIONS
    label: int                   # the `Label` column, asserted equal to `emotion`
    au: "np.ndarray"             # [30, 63]

    @property
    def emotion_name(self) -> str:
        return EMOTIONS[self.emotion]


def parse_sequence_id(sequence_id: str) -> tuple[int, int, int, int, int, int]:
    """`<timestamp>-<set>-<participant>-<level>-<emoji>-<emotion>` into its six integers.

    Raises rather than guessing. A sequence whose ID does not carry six integer fields is
    not a sequence this study knows how to attribute to a participant, and silently
    skipping it would shrink a denominator without saying so.
    """
    parts = sequence_id.split("-")
    if len(parts) != 6:
        raise ValueError(
            f"sequence id {sequence_id!r} has {len(parts)} fields, expected 6: "
            "<timestamp>-<set-id>-<participant-id>-<level-id>-<emoji-id>-<emotion-id>"
        )
    try:
        timestamp, set_id, participant, level, emoji, emotion = (int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"sequence id {sequence_id!r} has a non-integer field") from exc
    if not 0 <= emotion < N_EMOTIONS:
        raise ValueError(
            f"sequence id {sequence_id!r} ends in emotion id {emotion}, "
            f"outside 0..{N_EMOTIONS - 1}"
        )
    return timestamp, set_id, participant, level, emoji, emotion


def _workbook_bytes(data: Path, split: str) -> tuple[bytes, str]:
    """The `<split>_set.xlsx` payload, from a directory or from inside a zip."""
    wanted = f"{split}_set.xlsx"

    if data.is_dir():
        path = data / wanted
        if not path.is_file():
            raise SystemExit(f"{data}: no {wanted}")
        return path.read_bytes(), str(path)

    if data.suffix.lower() == ".zip":
        with zipfile.ZipFile(data) as archive:
            matches = [n for n in archive.namelist()
                       if Path(n).name.lower() == wanted and not n.endswith("/")]
            if not matches:
                raise SystemExit(
                    f"{data}: no {wanted} inside the archive. It holds:\n  "
                    + "\n  ".join(sorted(archive.namelist())[:20])
                )
            if len(matches) > 1:
                raise SystemExit(
                    f"{data}: {len(matches)} copies of {wanted} inside the archive "
                    f"({', '.join(matches)}); cannot tell which one is meant"
                )
            return archive.read(matches[0]), f"{data}!{matches[0]}"

    if data.is_file() and data.suffix.lower() in (".xlsx", ".xlsm"):
        raise SystemExit(
            f"--data points at a single workbook ({data.name}). Pass the directory or the "
            "zip holding all three splits instead: the integrity gate checks participant "
            "disjointness across training, validation and test, which needs all of them."
        )

    raise SystemExit(f"--data: expected a directory or a .zip, got {data}")


def load_split(data: Path, split: str) -> tuple[list[Sequence], dict]:
    """Every sequence of one split, plus the provenance of the file it came from."""
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - environment guard
        raise SystemExit("this study needs pandas and openpyxl: pip install pandas openpyxl")

    payload, origin = _workbook_bytes(data, split)
    digest = hashlib.sha256(payload).hexdigest()

    try:
        frame = pd.read_excel(io.BytesIO(payload))
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(f"could not read {origin} (openpyxl missing?): {exc}") from exc

    columns = list(frame.columns)
    for required in ("sequence_id", "timestamp", "Label"):
        if required not in columns:
            raise SystemExit(f"{origin}: no {required!r} column; found {columns[:8]}...")

    au_columns = [c for c in columns if c not in ("sequence_id", "timestamp", "Label")]

    sequences: list[Sequence] = []
    for sequence_id, group in frame.groupby("sequence_id", sort=False):
        timestamp, set_id, participant, level, emoji, emotion = parse_sequence_id(
            str(sequence_id))
        ordered = group.sort_values("timestamp", kind="stable")
        labels = set(int(v) for v in ordered["Label"].unique())
        if len(labels) != 1:
            raise SystemExit(
                f"{origin}: sequence {sequence_id} carries {len(labels)} different labels "
                f"({sorted(labels)}); a sequence must hold exactly one emotion"
            )
        sequences.append(Sequence(
            sequence_id=str(sequence_id),
            timestamp=timestamp,
            set_id=set_id,
            participant=participant,
            level=level,
            emoji=emoji,
            emotion=emotion,
            label=labels.pop(),
            au=np.asarray(ordered[au_columns].to_numpy(), dtype=np.float64),
        ))

    sequences.sort(key=lambda s: (s.timestamp, s.sequence_id))
    provenance = {
        "split": split,
        "origin": origin,
        "sha256": digest,
        "bytes": len(payload),
        "rows": int(len(frame)),
        "au_columns": au_columns,
    }
    return sequences, provenance


# ------------------------------------------------------------- integrity gate


class IntegrityError(SystemExit):
    """A corpus check failed. Nothing downstream may run."""


def check_corpus(loaded: dict[str, list[Sequence]],
                 provenance: dict[str, dict]) -> list[dict]:
    """Every structural claim this study rests on, checked before it is relied upon.

    Returns one record per check. Raises on the first failure rather than reporting a
    number measured on a corpus that is not the one described: each of these is load
    bearing somewhere downstream, so a soft warning would just move the failure to a place
    where it is harder to see.
    """
    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        if not ok:
            listing = "\n".join(
                f"  {'ok  ' if c['passed'] else 'FAIL'}  {c['check']}: {c['detail']}"
                for c in checks
            )
            raise IntegrityError(
                "corpus integrity check failed; refusing to produce any result.\n"
                f"{listing}\n"
                "The counts this study asserts come from the EmojiHeroVR dataset summary. "
                "A mismatch means the data on disk is not that corpus, and every number "
                "downstream would describe something else."
            )

    for split in SPLITS:
        sequences = loaded[split]
        expected = EXPECTED_SEQUENCES[split]
        record(f"{split}: sequence count", len(sequences) == expected,
               f"{len(sequences)} sequences, expected {expected}")

        participants = sorted({s.participant for s in sequences})
        expected_participants = EXPECTED_PARTICIPANTS[split]
        record(f"{split}: participant count",
               len(participants) == expected_participants,
               f"{len(participants)} participants {participants}, "
               f"expected {expected_participants}")

        wrong_length = [s.sequence_id for s in sequences
                        if s.au.shape[0] != FRAMES_PER_SEQUENCE]
        record(f"{split}: {FRAMES_PER_SEQUENCE} frames per sequence", not wrong_length,
               "every sequence" if not wrong_length
               else f"{len(wrong_length)} sequences differ, e.g. {wrong_length[:3]}")

        wrong_width = [s.sequence_id for s in sequences
                       if s.au.shape[1] != au_names.V1_CHANNEL_COUNT]
        record(f"{split}: {au_names.V1_CHANNEL_COUNT} measured action units",
               not wrong_width,
               "every sequence" if not wrong_width
               else f"{len(wrong_width)} sequences differ, e.g. {wrong_width[:3]}")

        mismatched = [s.sequence_id for s in sequences if s.label != s.emotion]
        record(f"{split}: Label equals the sequence id's emotion field", not mismatched,
               "all agree" if not mismatched
               else f"{len(mismatched)} disagree, e.g. {mismatched[:3]}")

        columns = provenance[split]["au_columns"]
        expected_columns = list(au_names.NAMES[:au_names.V1_CHANNEL_COUNT])
        record(f"{split}: action-unit columns match the teacher's first 63, in order",
               columns == expected_columns,
               "identical" if columns == expected_columns
               else f"first difference at index "
                    f"{next((i for i, (a, b) in enumerate(zip(columns, expected_columns)) if a != b), len(columns))}")

        non_finite = sum(int((~np.isfinite(s.au)).sum()) for s in sequences)
        record(f"{split}: action units are finite", non_finite == 0,
               "no NaN or infinity" if non_finite == 0
               else f"{non_finite} non-finite cells (the notebook's ffill/bfill would "
                    "have filled these; reproduce it before trusting a window)")

    # The seven channels the teacher expects but EmojiHero never measured. Asserted here so
    # that the zero-fill below is demonstrably filling the tongue and nothing else.
    record("the seven unmeasured channels are exactly the tongue channels",
           au_names.TONGUE_INDICES == tuple(range(au_names.V1_CHANNEL_COUNT,
                                                  au_names.CHANNEL_COUNT)),
           f"indices {au_names.TONGUE_INDICES} = "
           f"{[au_names.NAMES[i] for i in au_names.TONGUE_INDICES]}")

    by_split = {split: {s.participant for s in loaded[split]} for split in SPLITS}
    for a, b in (("training", "validation"), ("training", "test"), ("validation", "test")):
        overlap = sorted(by_split[a] & by_split[b])
        record(f"no participant in both {a} and {b}", not overlap,
               "disjoint" if not overlap else f"shared: {overlap}")

    ids = [s.sequence_id for split in SPLITS for s in loaded[split]]
    record("sequence ids are unique across the corpus", len(ids) == len(set(ids)),
           f"{len(ids)} ids, {len(set(ids))} distinct")

    # Preprocessing is a claim about shape, so it is asserted on the real data rather than
    # assumed from the arithmetic.
    sample = loaded["validation"][0]
    windows = to_model_windows(sample.au)
    record("preprocessing yields six (5, 70) windows per sequence",
           windows.shape == (WINDOWS_PER_SEQUENCE, WINDOW_LENGTH, au_names.CHANNEL_COUNT),
           f"{windows.shape} from {sample.sequence_id}")
    record("the zero-filled channels are the tongue channels and only those",
           bool((windows[:, :, list(au_names.TONGUE_INDICES)] == 0.0).all()),
           "all tongue cells are 0.0 (model-input representation; scientifically these "
           "are unobserved, not measured zeros)")

    # The structural exception the protocol pre-registers. Recorded as an observation
    # either way -- if a future corpus fixes it, the run says so instead of quietly
    # changing the primary population.
    thin = thin_cells(loaded["test"])
    record("test: participant-emotion cells too thin to enrol and score", True,
           ", ".join(f"p{p} {EMOTIONS[e]}={n}" for p, e, n in thin) or "none")

    return checks


def thin_cells(sequences: list[Sequence]) -> list[tuple[int, int, int]]:
    """(participant, emotion, count) for every cell below the eligibility threshold."""
    counts = cell_counts(sequences)
    return sorted(
        (p, e, counts.get((p, e), 0))
        for p in sorted({s.participant for s in sequences})
        for e in range(N_EMOTIONS)
        if counts.get((p, e), 0) < SEQUENCES_REQUIRED_PER_EMOTION
    )


def cell_counts(sequences: list[Sequence]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for s in sequences:
        counts[(s.participant, s.emotion)] = counts.get((s.participant, s.emotion), 0) + 1
    return counts


# ------------------------------------------------------------- preprocessing


def to_model_windows(au63: "np.ndarray") -> "np.ndarray":
    """One sequence's 30x63 measured frames into the six (5, 70) windows the model takes.

    Reproduces the teacher notebook's EmojiHero path exactly, in its order:

      1. widen 63 measured channels to the 70 the graph expects, the seven tongue channels
         filled with 0.0;
      2. mean-pool every three consecutive frames, 30 -> 10 (`tp.downsample`, the same
         mean-of-N-rows operation the 60 Hz path uses with a different factor);
      3. no normalization, no mean-centering, no max-scaling -- the notebook defines
         `mean_centered_max_scale` and leaves the call commented out, which is the whole
         meaning of `NOT-normalized` in the checkpoint name;
      4. length-5 windows at stride 1 over those 10 frames, none crossing the sequence
         boundary, giving exactly six.

    On the zero fill. `au_names` records that EmojiHero's missing tongue channels must be
    treated as *absent, not zero*, and for descriptive action-unit comparison that still
    holds. This function is not descriptive: it builds model input, and the weights in
    `sparseVR_teacher_model.onnx` were fitted with those channels at 0.0. Feeding anything
    else would be replaying a different model. The two readings are kept apart in the
    report rather than reconciled.
    """
    if au63.ndim != 2 or au63.shape[1] != au_names.V1_CHANNEL_COUNT:
        raise ValueError(
            f"expected [frames, {au_names.V1_CHANNEL_COUNT}], got {au63.shape}")
    if au63.shape[0] != FRAMES_PER_SEQUENCE:
        raise ValueError(
            f"expected {FRAMES_PER_SEQUENCE} frames, got {au63.shape[0]}")

    widened = np.zeros((au63.shape[0], au_names.CHANNEL_COUNT), dtype=np.float64)
    widened[:, :au_names.V1_CHANNEL_COUNT] = au63

    pooled = tp.downsample(widened, POOL_FACTOR)
    if pooled.shape[0] != POOLED_FRAMES:
        raise ValueError(f"pooling gave {pooled.shape[0]} frames, expected {POOLED_FRAMES}")

    windows = np.stack([
        pooled[i:i + WINDOW_LENGTH]
        for i in range(0, POOLED_FRAMES - WINDOW_LENGTH + 1, WINDOW_STRIDE)
    ])
    if len(windows) != WINDOWS_PER_SEQUENCE:
        raise ValueError(
            f"got {len(windows)} windows, expected {WINDOWS_PER_SEQUENCE}")
    return windows.astype(np.float32)


# ---------------------------------------------------------------- inference


def run_model(sequences: list[Sequence], model_path: Path) -> "np.ndarray":
    """Softmax scores for every window of every sequence. [n_sequences, 6, 7].

    One `InferenceSession.run` over the whole split. Every profile below re-slices these
    cached scores, so the model is executed once per run no matter how many calibration
    profiles are sampled -- and, more to the point, every profile is scored on numerically
    identical model output.
    """
    import onnxruntime as ort

    stacked = np.concatenate([to_model_windows(s.au) for s in sequences])

    try:
        runner = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        scores = runner.run(None, {runner.get_inputs()[0].name: stacked})[0]
    except Exception as exc:
        raise SystemExit(
            f"could not run the model at\n  {model_path}\n"
            f"onnxruntime said: {exc}\n"
            "Expected a graph taking [N, 5, 70] and returning [N, 7]."
        ) from exc

    scores = np.asarray(scores, np.float64)
    if scores.shape != (len(stacked), N_EMOTIONS):
        raise SystemExit(
            f"model returned {scores.shape}, expected ({len(stacked)}, {N_EMOTIONS})")
    if not np.isfinite(scores).all():
        raise SystemExit("model returned non-finite scores; refusing to continue")

    sums = scores.sum(axis=1)
    worst = float(np.abs(sums - 1.0).max())
    if worst > SOFTMAX_TOLERANCE:
        raise SystemExit(
            f"model output rows do not sum to 1 (worst deviation {worst:.2e}, tolerance "
            f"{SOFTMAX_TOLERANCE:g}). This study assumes the graph ends in Softmax."
        )

    return scores.reshape(len(sequences), WINDOWS_PER_SEQUENCE, N_EMOTIONS)


# -------------------------------------------------------- profile selection


def _stable_index(split: str, participant: int, emotion: int, profile_index: int,
                  modulus: int, master_seed: int = MASTER_SEED) -> int:
    """A deterministic index into a participant-emotion's sequences.

    SHA-256 rather than `random.Random` or `hash()`. `hash()` is salted per process and
    would give a different profile on every run; `random.Random` would make each draw
    depend on the order of every draw before it, so adding a participant would silently
    move everyone else's selection. Hashing the coordinates makes each choice a pure
    function of (seed, split, participant, emotion, profile index) and of nothing else.
    """
    key = f"{master_seed}|{split}|{participant}|{emotion}|{profile_index}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % modulus


def calibration_profile(sequences_by_cell: dict[tuple[int, int], list[Sequence]],
                        split: str, participant: int, profile_index: int,
                        master_seed: int = MASTER_SEED) -> dict[int, Sequence]:
    """One calibration sequence per emotion, chosen jointly and deterministically.

    Jointly because a prediction compares all seven centroids at once: a profile is the
    whole set, not seven independent choices that happen to be reported together.
    """
    profile: dict[int, Sequence] = {}
    for emotion in range(N_EMOTIONS):
        candidates = sequences_by_cell.get((participant, emotion), [])
        if not candidates:
            continue
        index = _stable_index(split, participant, emotion, profile_index, len(candidates),
                              master_seed)
        profile[emotion] = candidates[index]
    return profile


# ----------------------------------------------------- population centroids


def build_population_centroids(sequences: list[Sequence], logf: "np.ndarray",
                               index_of: dict[str, int]) -> dict:
    """Seven population centroids from the training split, participant-balanced.

    Hierarchical on purpose, in three steps rather than one mean over every window:

        window log-scores  -> sequence centroid       (mean over that sequence's windows)
        sequence centroids -> participant-emotion     (mean over that person's sequences)
        participant-emotion -> population centroid    (unweighted mean over participants)

    A flat mean over all windows would weight each participant by how many sequences of
    that emotion they happen to have recorded, and the corpus is not balanced -- Anger
    ranges from 3 to 10 sequences per person. The population profile would then be mostly
    the most prolific participants, which is a per-person profile wearing a population's
    name. Averaging participants equally at the last step is what makes "population" true.

    Returns the matrix and the counts behind it, so the manifest can show what each
    centroid actually rests on.
    """
    by_emotion: dict[int, dict[int, list[np.ndarray]]] = {}
    for s in sequences:
        sequence_centroid = logf[index_of[s.sequence_id]].mean(axis=0)
        by_emotion.setdefault(s.emotion, {}).setdefault(
            s.participant, []).append(sequence_centroid)

    matrix = np.zeros((N_EMOTIONS, N_EMOTIONS), dtype=np.float64)
    support: dict[str, dict] = {}
    missing: list[str] = []

    for emotion in range(N_EMOTIONS):
        per_participant = by_emotion.get(emotion, {})
        if not per_participant:
            missing.append(EMOTIONS[emotion])
            continue

        participant_centroids = np.stack([
            np.stack(rows).mean(axis=0)
            for _, rows in sorted(per_participant.items())
        ])
        matrix[emotion] = participant_centroids.mean(axis=0)

        support[EMOTIONS[emotion]] = {
            "participants": len(per_participant),
            "sequences": sum(len(rows) for rows in per_participant.values()),
            "sequences_per_participant": {
                str(p): len(rows) for p, rows in sorted(per_participant.items())
            },
        }

    if missing:
        raise SystemExit(
            "no training sequence for: " + ", ".join(missing)
            + ". A population centroid cannot be built for a class the training split "
              "does not contain."
        )

    thin = {c: s["participants"] for c, s in support.items()
            if s["participants"] < POPULATION_MIN_PARTICIPANTS_PER_EMOTION}
    if thin:
        raise SystemExit(
            "insufficient population support: "
            + ", ".join(f"{c} from only {n} participant(s)" for c, n in sorted(thin.items()))
            + f". At least {POPULATION_MIN_PARTICIPANTS_PER_EMOTION} training participants "
              "are required per emotion before a population centroid is reported."
        )

    return {"matrix": matrix, "support": support}


def corpus_fingerprint(provenance: dict, sequences: list[Sequence]) -> dict:
    """What the population centroids were estimated from, in a form that cannot drift.

    The workbook's own SHA-256 plus the shape of what was read out of it. The file hash
    alone would be enough to detect a changed corpus, but it says nothing about what the
    loader made of it, and the counts are what the estimator actually used.
    """
    return {
        "split": POPULATION_SOURCE_SPLIT,
        "workbook_sha256": provenance["sha256"],
        "rows": provenance["rows"],
        "n_sequences": len(sequences),
        "n_participants": len({s.participant for s in sequences}),
        "participants": sorted({s.participant for s in sequences}),
    }


def population_semantic_hash(*, centroids: dict, class_order: list, schema_version: str,
                             preprocessing: dict, epsilon: float, feature_space: str,
                             transform: str, estimator: str, model_sha: str,
                             fingerprint: dict) -> str:
    """A content hash over everything that determines the centroids, and nothing else.

    The file's byte hash is useless as a preregistration pin: the artifact records when it
    was created, so rebuilding it from identical inputs yields identical centroids inside a
    file with a different hash, and the frozen config would then reject a correct rebuild.
    This hashes the *meaning* instead -- the centroid values, the class order, the schema
    version, the whole preprocessing and feature configuration, the ONNX model hash and the
    training corpus fingerprint. Timestamps, file paths and git commits are deliberately
    excluded; none of them can change a centroid.

    `sort_keys` with the default separators gives one canonical serialisation, and Python's
    float repr round-trips exactly, so the same inputs always give the same digest.
    """
    payload = {
        "schema_version": schema_version,
        "class_order": list(class_order),
        "epsilon": epsilon,
        "feature_space": feature_space,
        "transform": transform,
        "estimator": estimator,
        "preprocessing": preprocessing,
        "model_sha256": model_sha,
        "corpus_fingerprint": fingerprint,
        "centroids": centroids,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def population_artifact(matrix: "np.ndarray", support: dict, *, model_sha: str,
                        model_path: Path, provenance: dict,
                        sequences: list[Sequence]) -> dict:
    """The saved population profile, with everything needed to refuse a wrong reuse.

    This artifact is aggregate model *output* -- seven seven-dimensional vectors averaged
    over twenty people. It is the only centroid this study writes to disk, and it carries
    no action units, no window and no individual's centroid. A personal centroid describes
    one identifiable participant and is never written.
    """
    centroids = {EMOTIONS[e]: [float(v) for v in matrix[e]] for e in range(N_EMOTIONS)}
    fingerprint = corpus_fingerprint(provenance, sequences)

    semantic = population_semantic_hash(
        centroids=centroids, class_order=list(EMOTIONS),
        schema_version=POPULATION_SCHEMA_VERSION,
        preprocessing=FROZEN_SETTINGS["preprocessing"],
        epsilon=cs.DEFAULT_EPSILON,
        feature_space=FROZEN_SETTINGS["feature_space"],
        transform=FROZEN_SETTINGS["transform"],
        estimator=FROZEN_SETTINGS["population_centroid"]["estimator"],
        model_sha=model_sha, fingerprint=fingerprint)

    return {
        "schema_version": POPULATION_SCHEMA_VERSION,
        # The pin. Everything below that is not in the hash -- the timestamp, the paths,
        # the git commit -- is provenance for a human, and a rebuild from identical inputs
        # produces a byte-different file with this same value.
        "semantic_sha256": semantic,
        "semantic_hash_covers": [
            "schema_version", "class_order", "epsilon", "feature_space", "transform",
            "estimator", "preprocessing", "model_sha256", "corpus_fingerprint",
            "centroids",
        ],
        "semantic_hash_excludes": ["created_utc", "git_commit", "paths", "support detail"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "source_split": POPULATION_SOURCE_SPLIT,
        "model": {"path": str(model_path), "relative": cs.MODEL_RELATIVE,
                  "sha256": model_sha, "retrained": False},
        "corpus_fingerprint": fingerprint,
        "data_provenance": {k: v for k, v in provenance.items() if k != "au_columns"},
        "class_order": list(EMOTIONS),
        "epsilon": cs.DEFAULT_EPSILON,
        "feature_space": FROZEN_SETTINGS["feature_space"],
        "transform": FROZEN_SETTINGS["transform"],
        "estimator": FROZEN_SETTINGS["population_centroid"]["estimator"],
        "preprocessing": FROZEN_SETTINGS["preprocessing"],
        "n_participants": len({s.participant for s in sequences}),
        "n_sequences": len(sequences),
        "support_per_emotion": support,
        "centroids": centroids,
        "note": "aggregate model-output representation over the training participants; "
                "carries no action units, no face windows and no per-person centroid",
    }


def load_population_centroids(path: Path, *, model_sha: str) -> tuple["np.ndarray", dict]:
    """Read a population artifact, refusing one built against a different model.

    The centroids live in the model's own output space, so a profile built from a
    different graph describes a different space entirely. Silently mixing the two would
    produce numbers that look ordinary and mean nothing, which is why the hash is checked
    here rather than trusted to the person typing the command.
    """
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))

    version = artifact.get("schema_version")
    if version != POPULATION_SCHEMA_VERSION:
        raise SystemExit(
            f"{path}: schema_version is {version!r}, expected "
            f"{POPULATION_SCHEMA_VERSION!r}"
        )

    recorded = (artifact.get("model") or {}).get("sha256")
    if recorded != model_sha:
        raise SystemExit(
            f"{path}: built against a different model.\n"
            f"  artifact: {recorded}\n"
            f"  on disk:  {model_sha}\n"
            "Population centroids live in that model's output space, so they cannot be "
            "reused across models. Rebuild them with --build-population."
        )

    if artifact.get("source_split") != POPULATION_SOURCE_SPLIT:
        raise SystemExit(
            f"{path}: source_split is {artifact.get('source_split')!r}, expected "
            f"{POPULATION_SOURCE_SPLIT!r}. Population centroids must come from the "
            "training split; anything else leaks the data being scored."
        )

    if artifact.get("class_order") != list(EMOTIONS):
        raise SystemExit(
            f"{path}: class order {artifact.get('class_order')} does not match "
            f"{list(EMOTIONS)}"
        )

    if artifact.get("epsilon") != cs.DEFAULT_EPSILON:
        raise SystemExit(
            f"{path}: built with epsilon {artifact.get('epsilon')}, this build uses "
            f"{cs.DEFAULT_EPSILON}. The feature spaces would not match."
        )

    centroids = artifact.get("centroids") or {}
    missing = [c for c in EMOTIONS if c not in centroids]
    if missing:
        raise SystemExit(f"{path}: no centroid for {', '.join(missing)}")

    matrix = np.array([centroids[c] for c in EMOTIONS], dtype=np.float64)
    if matrix.shape != (N_EMOTIONS, N_EMOTIONS):
        raise SystemExit(f"{path}: centroid matrix is {matrix.shape}, expected "
                         f"({N_EMOTIONS}, {N_EMOTIONS})")
    if not np.isfinite(matrix).all():
        raise SystemExit(f"{path}: centroid matrix holds non-finite values")

    for required in ("semantic_sha256", "corpus_fingerprint", "preprocessing",
                     "estimator", "feature_space", "transform"):
        if required not in artifact:
            raise SystemExit(
                f"{path}: no {required!r}. This artifact predates the semantic hash and "
                "cannot be used as a preregistration pin; rebuild it with "
                "--build-population."
            )

    # Recompute the pin from the artifact's own contents. A file whose centroids were
    # edited after it was written, or whose recorded configuration no longer matches the
    # centroids beside it, fails here rather than at the point where it would have quietly
    # changed a result.
    recomputed = population_semantic_hash(
        centroids=centroids, class_order=artifact["class_order"],
        schema_version=artifact["schema_version"],
        preprocessing=artifact["preprocessing"], epsilon=artifact["epsilon"],
        feature_space=artifact["feature_space"], transform=artifact["transform"],
        estimator=artifact["estimator"], model_sha=recorded,
        fingerprint=artifact["corpus_fingerprint"])
    stored = artifact.get("semantic_sha256")
    if stored != recomputed:
        raise SystemExit(
            f"{path}: the artifact's semantic hash does not match its contents.\n"
            f"  recorded:   {stored}\n"
            f"  recomputed: {recomputed}\n"
            "The file has been edited since it was written. Rebuild it with "
            "--build-population."
        )

    return matrix, artifact


def hybrid_centroids(personal: "np.ndarray", population: "np.ndarray",
                     alpha: float) -> "np.ndarray":
    """H_e = alpha * P_e + (1 - alpha) * G_e, per emotion.

    The blend happens in log-score space, before any normalisation: cosine normalises the
    centroid and the sample at classification time, exactly as it does for the other
    methods, so the four are compared under one rule.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
    if personal.shape != population.shape:
        raise ValueError(
            f"shape mismatch: personal {personal.shape}, population {population.shape}")
    return alpha * personal + (1.0 - alpha) * population


# ------------------------------------------------- fast paths for the sweep
#
# The alpha grid is 21 values over 1000 profiles over 8 participants, which is 168,000
# scorings. `calibration_study.trial_vote` and `session_analysis.macro_f1` are Python
# loops -- correct, shared and authoritative, but far too slow at that count. The two
# functions below are vectorised equivalents used *only* to choose alpha; everything that
# is finally reported goes back through the shared implementations. `test_..._matches_...`
# asserts the two agree on random input, so the fast path cannot drift into a second
# definition of the tie rule.


def fast_majority_vote(window_predictions: "np.ndarray") -> "np.ndarray":
    """[n_sequences, n_windows] window classes -> [n_sequences] sequence classes.

    The frozen tie rule, vectorised: most windows, then the chronologically last window,
    then the first tied class in the fixed class order. `window_predictions` must be in
    chronological order along axis 1, which is how the caller slices it.
    """
    n_sequences, n_windows = window_predictions.shape
    counts = np.zeros((n_sequences, N_EMOTIONS), dtype=np.int32)
    for emotion in range(N_EMOTIONS):
        counts[:, emotion] = (window_predictions == emotion).sum(axis=1)

    best = counts.max(axis=1)
    tied = counts == best[:, None]
    n_tied = tied.sum(axis=1)

    outright = counts.argmax(axis=1)              # first maximum == the sole winner
    first_tied = tied.argmax(axis=1)              # lowest class index among the tied
    last = window_predictions[:, -1]
    last_is_tied = tied[np.arange(n_sequences), last]

    return np.where(n_tied == 1, outright,
                    np.where(last_is_tied, last, first_tied))


def fast_macro_f1(truth: "np.ndarray", predicted: "np.ndarray") -> float:
    """Unweighted mean F1 over the classes with support or predictions.

    Matches `session_analysis.macro_f1`, including its treatment of a class that is
    neither present nor predicted: left out of the mean rather than scored zero.
    """
    f1s = []
    for emotion in range(N_EMOTIONS):
        tp = int(((truth == emotion) & (predicted == emotion)).sum())
        fp = int(((truth != emotion) & (predicted == emotion)).sum())
        fn = int(((truth == emotion) & (predicted != emotion)).sum())
        if tp + fn == 0 and tp + fp == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * precision * recall / (precision + recall)
                   if (precision + recall) else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


# --------------------------------------------------------------- evaluation


def _sign_test(deltas: list[float]) -> dict:
    """Two-sided exact sign test over participants, ties dropped.

    The participant is the unit of a decision, so this is computed over participants and
    nothing else. Seven of them can at best reach p = 0.0156, which is worth knowing
    before the number is read: this design cannot produce a small p, and a large one is
    not evidence of absence.
    """
    positive = sum(1 for d in deltas if d > 0)
    negative = sum(1 for d in deltas if d < 0)
    n = positive + negative
    if n == 0:
        return {"positive": 0, "negative": 0, "ties": len(deltas), "n": 0, "p_two_sided": None}
    k = min(positive, negative)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {
        "positive": positive,
        "negative": negative,
        "ties": len(deltas) - n,
        "n": n,
        "p_two_sided": float(min(1.0, 2 * tail)),
        "smallest_reachable_p": float(min(1.0, 2 / (2 ** n))),
    }


def personal_centroids(profile: dict[int, Sequence], index_of: dict[str, int],
                       logf: "np.ndarray",
                       window_indices: tuple[int, ...] = ALL_WINDOWS
                       ) -> tuple["np.ndarray", list[int]]:
    """Seven personal centroids, each the mean of one calibration sequence's windows.

    The whole of the personal side of every method: `personal` uses these directly and
    `hybrid` blends them with the population profile. Returned with the emotion list they
    are in row order of, because a participant missing an emotion entirely would return
    fewer than seven and the caller must not assume otherwise.
    """
    enrolled: list[int] = []
    rows: list[np.ndarray] = []
    for emotion in sorted(profile):
        window_rows = logf[index_of[profile[emotion].sequence_id]][list(window_indices)]
        enrolled.append(emotion)
        rows.append(window_rows.mean(axis=0))
    if not rows:
        return np.zeros((0, N_EMOTIONS)), []
    return np.stack(rows), enrolled


def split_calibration_evaluation(sequences: list[Sequence],
                                 profile: dict[int, Sequence]
                                 ) -> tuple[list[Sequence], set[str]]:
    """Evaluation sequences and the calibration ids, asserted disjoint.

    A calibration sequence is removed whole -- never partially, which is the leak the live
    study measured decaying a +0.150 gain to -0.010 once enrolment and test were moved
    apart within one performance.
    """
    calibration_ids = {s.sequence_id for s in profile.values()}
    evaluation = [s for s in sequences if s.sequence_id not in calibration_ids]
    overlap = calibration_ids & {s.sequence_id for s in evaluation}
    if overlap:  # pragma: no cover - structurally impossible, asserted anyway
        raise AssertionError(f"sequence on both sides of the split: {sorted(overlap)}")
    return evaluation, calibration_ids


def _cosine_separation(M: "np.ndarray") -> dict | None:
    if len(M) < 2:
        return None
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    off = (Mn @ Mn.T)[~np.eye(len(M), dtype=bool)]
    return {"mean_pairwise_cosine": float(off.mean()),
            "min": float(off.min()), "max": float(off.max())}


def method_window_predictions(F: "np.ndarray", raw_scores: "np.ndarray",
                              personal_M: "np.ndarray", enrolled: list[int],
                              population_M: "np.ndarray | None",
                              alpha: float) -> dict[str, "np.ndarray"]:
    """Window-level class indices for every method, on one shared evaluation set.

    All four read the same `F`, so any difference between them is the decision rule and
    nothing else. `population` ranges over all seven classes because the training split
    supplies all seven; `personal` and `hybrid` range over whatever the participant
    enrolled, which under the eligibility rule is also all seven.
    """
    metric = FROZEN_SETTINGS["metric"]
    out = {"raw": raw_scores.argmax(axis=1)}

    if population_M is not None:
        out["population"] = cs.classify(
            population_M, list(range(N_EMOTIONS)), F, metric)[0]

    if len(personal_M):
        out["personal"] = cs.classify(personal_M, enrolled, F, metric)[0]
        if population_M is not None:
            blended = hybrid_centroids(personal_M, population_M[enrolled], alpha)
            out["hybrid"] = cs.classify(blended, enrolled, F, metric)[0]

    return out


def evaluate_participant(sequences: list[Sequence],
                         index_of: dict[str, int],
                         logf: "np.ndarray",
                         scores: "np.ndarray",
                         profile: dict[int, Sequence],
                         window_indices: tuple[int, ...] = ALL_WINDOWS,
                         population_M: "np.ndarray | None" = None,
                         alpha: float = 1.0) -> dict:
    """One participant, one calibration profile, all four methods on the same sequences.

    `sequences` is that participant's sequences and nothing else. Every method is scored
    on the identical evaluation set, with the identical tie rule, so the deltas between
    them are differences of decision rule rather than of what was asked.

    The reported numbers go through the shared `session_analysis.macro_f1`,
    `session_analysis.confusion_counts` and `calibration_study.trial_vote`, not through the
    vectorised sweep helpers -- those exist only to make the alpha grid affordable.
    """
    evaluation, _ = split_calibration_evaluation(sequences, profile)
    personal_M, enrolled = personal_centroids(profile, index_of, logf, window_indices)

    if not evaluation or not len(personal_M):
        return {"scorable": False,
                "reason": "no evaluation sequence" if not evaluation else "nothing enrolled"}

    rows = [index_of[s.sequence_id] for s in evaluation]
    window_count = len(window_indices)
    F = logf[np.ix_(rows, list(window_indices))].reshape(-1, N_EMOTIONS)
    raw_scores = scores[np.ix_(rows, list(window_indices))].reshape(-1, N_EMOTIONS)

    truth = np.repeat([s.emotion for s in evaluation], window_count)
    trials = np.repeat(np.arange(len(evaluation)), window_count)
    positions = np.tile(np.arange(window_count), len(evaluation))

    window_predictions = method_window_predictions(
        F, raw_scores, personal_M, enrolled, population_M, alpha)

    names = np.array(EMOTIONS)
    truth_n = names[truth]

    sequence_predictions: dict[str, np.ndarray] = {}
    seq_truth = None
    for method, predicted in window_predictions.items():
        _, seq_truth, votes = cs.trial_vote(
            truth_n, names[predicted], trials, positions, list(EMOTIONS))
        sequence_predictions[method] = votes

    scored_classes = sorted(set(seq_truth.tolist()))
    complete = len(scored_classes) == N_EMOTIONS
    enrolled_complete = len(enrolled) == N_EMOTIONS

    def block(predictions: "np.ndarray") -> dict:
        m = sa.macro_f1(seq_truth, predictions, list(EMOTIONS))
        return {
            "accuracy": float((predictions == seq_truth).mean()),
            "macro_f1": m["macro_f1"],
            "classes_scored": m["classes_scored"],
            "per_class_recall": {
                c: (m["per_class"][c]["recall"] if m["per_class"][c]["support"] else None)
                for c in EMOTIONS
            },
        }

    sequence_level = {method: block(votes)
                      for method, votes in sequence_predictions.items()}

    def delta(better: str, worse: str) -> dict | None:
        if better not in sequence_level or worse not in sequence_level:
            return None
        a, b = sequence_level[better], sequence_level[worse]
        return {
            "accuracy": a["accuracy"] - b["accuracy"],
            "macro_f1": ((a["macro_f1"] - b["macro_f1"])
                         if a["macro_f1"] is not None and b["macro_f1"] is not None
                         else None),
            "per_class_recall": {
                c: (a["per_class_recall"][c] - b["per_class_recall"][c]
                    if a["per_class_recall"][c] is not None
                    and b["per_class_recall"][c] is not None else None)
                for c in EMOTIONS
            },
        }

    # `hybrid_minus_personal` is the comparison that isolates what the population term
    # contributed. `hybrid_minus_raw` cannot: the personal term alone would move it.
    deltas = {
        "personal_minus_raw": delta("personal", "raw"),
        "population_minus_raw": delta("population", "raw"),
        "hybrid_minus_raw": delta("hybrid", "raw"),
        "hybrid_minus_personal": delta("hybrid", "personal"),
    }

    def paired(better: str, worse: str) -> dict | None:
        if better not in sequence_predictions or worse not in sequence_predictions:
            return None
        a = sequence_predictions[better] == seq_truth
        b = sequence_predictions[worse] == seq_truth
        return {
            "both_correct": int((a & b).sum()),
            f"{worse}_only": int((~a & b).sum()),
            f"{better}_only": int((a & ~b).sum()),
            "neither": int((~a & ~b).sum()),
        }

    separation = {"personal": _cosine_separation(personal_M)}
    if population_M is not None:
        separation["population"] = _cosine_separation(population_M)
        separation["hybrid"] = _cosine_separation(
            hybrid_centroids(personal_M, population_M[enrolled], alpha))

    windowed = F.reshape(len(evaluation), window_count, N_EMOTIONS)
    drift = {}
    for position, emotion in enumerate(enrolled):
        sel = [i for i, s in enumerate(evaluation) if s.emotion == emotion]
        if not sel:
            continue
        a = personal_M[position]
        b = windowed[sel].reshape(-1, N_EMOTIONS).mean(axis=0)
        drift[EMOTIONS[emotion]] = float(
            a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))

    population_similarity = {}
    if population_M is not None:
        for position, emotion in enumerate(enrolled):
            a, b = personal_M[position], population_M[emotion]
            population_similarity[EMOTIONS[emotion]] = float(
                a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))

    calibration_levels = [s.level for s in profile.values()]
    evaluation_levels = [s.level for s in evaluation]

    return {
        "scorable": True,
        "alpha": float(alpha),
        "methods": sorted(sequence_level),
        "n_calibration_sequences": len(profile),
        "n_evaluation_sequences": len(evaluation),
        "n_evaluation_windows": int(len(F)),
        "enrolled_classes": [EMOTIONS[e] for e in enrolled],
        "classes_with_a_scored_sequence": scored_classes,
        "primary_macro_f1_valid": bool(complete and enrolled_complete),
        "sequence_level": sequence_level,
        "delta": deltas,
        "paired_sequences": {
            "personal_vs_raw": paired("personal", "raw"),
            "population_vs_raw": paired("population", "raw"),
            "hybrid_vs_raw": paired("hybrid", "raw"),
            "hybrid_vs_personal": paired("hybrid", "personal"),
        },
        "window_level": {
            method: {
                "accuracy": float((names[predicted] == truth_n).mean()),
                "macro_f1": sa.macro_f1(truth_n, names[predicted],
                                        list(EMOTIONS))["macro_f1"],
            }
            for method, predicted in window_predictions.items()
        },
        "centroid_separation": separation,
        "centroid_drift_calibration_to_evaluation": drift,
        "personal_to_population_cosine": population_similarity,
        "level_diagnostic": {
            "calibration_levels": sorted(calibration_levels),
            "calibration_mean_level": float(np.mean(calibration_levels)),
            "evaluation_mean_level": float(np.mean(evaluation_levels)),
            "level_gap_calibration_minus_evaluation":
                float(np.mean(calibration_levels) - np.mean(evaluation_levels)),
        },
        "calibration_sequence_ids": {EMOTIONS[e]: profile[e].sequence_id
                                     for e in sorted(profile)},
        "evaluation_sequence_ids": [s.sequence_id for s in evaluation],
        "confusion": {
            method: sa.confusion_counts(seq_truth, votes, list(EMOTIONS))
            for method, votes in sequence_predictions.items()
        },
    }


def sweep_alpha(evaluation: list[Sequence], index_of: dict[str, int],
                logf: "np.ndarray",
                personal_M: "np.ndarray", enrolled: list[int],
                population_M: "np.ndarray", window_indices: tuple[int, ...],
                alphas: tuple[float, ...]) -> dict[float, float]:
    """Sequence-level macro-F1 for every alpha, on one participant-profile.

    The vectorised path. Used only to choose alpha; every reported number is recomputed
    through the shared implementations afterwards.
    """
    rows = [index_of[s.sequence_id] for s in evaluation]
    F = logf[np.ix_(rows, list(window_indices))]
    truth = np.array([s.emotion for s in evaluation])
    flat = F.reshape(-1, N_EMOTIONS)
    n_windows = len(window_indices)

    enrolled_array = np.array(enrolled)
    population_rows = population_M[enrolled]

    out: dict[float, float] = {}
    for alpha in alphas:
        blended = hybrid_centroids(personal_M, population_rows, alpha)
        Mn = blended / (np.linalg.norm(blended, axis=1, keepdims=True) + 1e-12)
        Fn = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12)
        predicted = enrolled_array[(Fn @ Mn.T).argmax(axis=1)]
        votes = fast_majority_vote(predicted.reshape(len(evaluation), n_windows))
        out[alpha] = fast_macro_f1(truth, votes)
    return out


# --------------------------------------------------------------- aggregation


def _summary(values: list[float]) -> dict | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "fraction_positive": float((array > 0).mean()),
    }


DELTA_KEYS = ("personal_minus_raw", "population_minus_raw",
              "hybrid_minus_raw", "hybrid_minus_personal")

# Two alphas whose participant-equal macro-F1 differ by less than this are a tie, and the
# tie goes to the larger alpha -- the less population-dependent of the two.
ALPHA_TIE_TOLERANCE = 1e-12


def eligibility_table(sequences: list[Sequence]) -> dict[int, dict]:
    """Which participants can carry a seven-class primary figure, and why not.

    A general completeness rule over participant-emotion cells: every emotion needs one
    sequence to enrol and one left to score. No participant id appears anywhere in it, so
    a corpus change moves the eligible set by itself.
    """
    counts = cell_counts(sequences)
    table: dict[int, dict] = {}
    for participant in sorted({s.participant for s in sequences}):
        thin = {EMOTIONS[e]: counts.get((participant, e), 0)
                for e in range(N_EMOTIONS)
                if counts.get((participant, e), 0) < SEQUENCES_REQUIRED_PER_EMOTION}
        table[participant] = {
            "eligible": not thin,
            "sequences_per_emotion": {EMOTIONS[e]: counts.get((participant, e), 0)
                                      for e in range(N_EMOTIONS)},
            "reason": (None if not thin else
                       f"fewer than {SEQUENCES_REQUIRED_PER_EMOTION} sequences for "
                       + ", ".join(f"{c} ({n})" for c, n in sorted(thin.items()))
                       + "; enrolling one would leave nothing of that class to score, and "
                         "the sequence must not be split"),
        }
    return table


def _sequences_by_cell(sequences: list[Sequence]) -> dict[tuple[int, int], list[Sequence]]:
    by_cell: dict[tuple[int, int], list[Sequence]] = {}
    for s in sequences:
        by_cell.setdefault((s.participant, s.emotion), []).append(s)
    for cell in by_cell.values():
        cell.sort(key=lambda s: (s.timestamp, s.sequence_id))
    return by_cell


def select_alpha(sequences: list[Sequence], split: str, index_of: dict[str, int],
                 logf: "np.ndarray", scores: "np.ndarray", population_M: "np.ndarray",
                 profile_indices: list[int], *,
                 window_indices: tuple[int, ...] = ALL_WINDOWS,
                 master_seed: int = MASTER_SEED,
                 alphas: tuple[float, ...] = ALPHA_GRID,
                 progress: bool = True) -> dict:
    """Choose one global alpha on this split, by the objective frozen before it ran.

    Participant-equal by construction: profiles are averaged within a participant first,
    then participants are averaged with equal weight. Averaging over all
    participant-profile pairs at once would weight participants by how many evaluation
    sequences they happen to have, which is the same mistake the population centroid's
    hierarchy exists to avoid.

    Only eligible participants contribute. A participant who cannot carry a seven-class
    figure cannot help choose the weight that a seven-class figure will be read from.
    """
    by_cell = _sequences_by_cell(sequences)
    eligibility = eligibility_table(sequences)
    eligible = [p for p, record in eligibility.items() if record["eligible"]]

    per_participant: dict[int, dict[float, float]] = {}
    for position, participant in enumerate(eligible, 1):
        if progress:
            print(f"  alpha sweep: participant {participant} "
                  f"({position}/{len(eligible)})", flush=True)
        own = [s for s in sequences if s.participant == participant]
        totals = {alpha: 0.0 for alpha in alphas}
        scored = 0
        for profile_index in profile_indices:
            profile = calibration_profile(by_cell, split, participant, profile_index,
                                          master_seed)
            evaluation, _ = split_calibration_evaluation(own, profile)
            personal_M, enrolled = personal_centroids(
                profile, index_of, logf, window_indices)
            if not evaluation or not len(personal_M):
                continue
            for alpha, value in sweep_alpha(
                    evaluation, index_of, logf, personal_M, enrolled,
                    population_M, window_indices, alphas).items():
                totals[alpha] += value
            scored += 1
        if scored:
            per_participant[participant] = {a: totals[a] / scored for a in alphas}

    if not per_participant:
        raise SystemExit("no eligible participant could be scored; cannot choose alpha")

    objective = {
        alpha: float(np.mean([per_participant[p][alpha] for p in per_participant]))
        for alpha in alphas
    }
    best = max(objective.values())
    tied = [a for a in alphas if objective[a] >= best - ALPHA_TIE_TOLERANCE]
    selected = max(tied)

    return {
        "alpha_grid": list(alphas),
        "objective": "participant-equal mean sequence-level seven-class macro-F1",
        "objective_by_alpha": {str(a): objective[a] for a in alphas},
        "per_participant_by_alpha": {
            str(p): {str(a): per_participant[p][a] for a in alphas}
            for p in sorted(per_participant)
        },
        "participants_used": sorted(per_participant),
        "n_profiles": len(profile_indices),
        "best_objective": float(best),
        "tied_alphas": [float(a) for a in tied],
        "tie_break": "the larger alpha, i.e. the more personal option",
        "selected_alpha": float(selected),
        "personal_only_objective": float(objective[1.0]) if 1.0 in objective else None,
        "population_only_objective": float(objective[0.0]) if 0.0 in objective else None,
        "population_adds_nothing": bool(selected == 1.0),
    }


def run_split(sequences: list[Sequence], split: str, model_path: Path,
              profile_indices: list[int], *, primary_profile: int,
              population_M: "np.ndarray | None" = None,
              alpha: float = 1.0,
              window_indices: tuple[int, ...] = ALL_WINDOWS,
              master_seed: int = MASTER_SEED,
              epsilon: float = cs.DEFAULT_EPSILON,
              precomputed: tuple | None = None,
              progress: bool = True) -> dict:
    """All four methods on one split, over every requested calibration profile."""
    if precomputed is not None:
        index_of, scores, logf = precomputed
    else:
        index_of = {s.sequence_id: i for i, s in enumerate(sequences)}
        scores = run_model(sequences, model_path)
        logf = cs.logscore(scores, epsilon)

    by_cell = _sequences_by_cell(sequences)
    eligibility = eligibility_table(sequences)
    participants = sorted({s.participant for s in sequences})

    per_participant: dict[int, dict] = {}
    selection_audit = hashlib.sha256()

    for position, participant in enumerate(participants, 1):
        if progress:
            print(f"  scoring: participant {participant} "
                  f"({position}/{len(participants)})", flush=True)
        own = [s for s in sequences if s.participant == participant]
        by_profile: list[dict] = []
        for profile_index in profile_indices:
            profile = calibration_profile(by_cell, split, participant, profile_index,
                                          master_seed)
            for emotion in sorted(profile):
                selection_audit.update(
                    f"{participant}|{profile_index}|{emotion}|"
                    f"{profile[emotion].sequence_id}\n".encode())
            result = evaluate_participant(
                own, index_of, logf, scores, profile, window_indices,
                population_M, alpha)
            result["profile_index"] = profile_index
            by_profile.append(result)

        scorable = [r for r in by_profile if r["scorable"]]
        primary = next((r for r in by_profile if r["profile_index"] == primary_profile),
                       None)
        distinct = len({tuple(sorted(r["calibration_sequence_ids"].items()))
                        for r in scorable})

        methods_present = sorted({m for r in scorable for m in r["sequence_level"]})

        across = {
            "methods": {
                method: {
                    "accuracy": _summary([r["sequence_level"][method]["accuracy"]
                                          for r in scorable
                                          if method in r["sequence_level"]]),
                    "macro_f1": _summary([r["sequence_level"][method]["macro_f1"]
                                          for r in scorable
                                          if method in r["sequence_level"]
                                          and r["sequence_level"][method]["macro_f1"]
                                          is not None]),
                    "window_accuracy": _summary([r["window_level"][method]["accuracy"]
                                                 for r in scorable
                                                 if method in r["window_level"]]),
                }
                for method in methods_present
            },
            "deltas": {
                key: {
                    "accuracy": _summary([r["delta"][key]["accuracy"] for r in scorable
                                          if r["delta"].get(key)]),
                    "macro_f1": _summary([r["delta"][key]["macro_f1"] for r in scorable
                                          if r["delta"].get(key)
                                          and r["delta"][key]["macro_f1"] is not None]),
                    "per_class_recall": {
                        c: _summary([r["delta"][key]["per_class_recall"][c]
                                     for r in scorable
                                     if r["delta"].get(key)
                                     and r["delta"][key]["per_class_recall"][c] is not None])
                        for c in EMOTIONS
                    },
                }
                for key in DELTA_KEYS
            },
            "level_gap": _summary([r["level_diagnostic"][
                "level_gap_calibration_minus_evaluation"] for r in scorable]),
            "centroid_separation_mean": {
                method: _summary([r["centroid_separation"][method]["mean_pairwise_cosine"]
                                  for r in scorable
                                  if r["centroid_separation"].get(method)])
                for method in ("personal", "population", "hybrid")
            },
            "personal_to_population_cosine": _summary(
                [v for r in scorable
                 for v in r["personal_to_population_cosine"].values()]),
        }

        per_participant[participant] = {
            "participant": participant,
            "eligible": eligibility[participant]["eligible"],
            "eligibility": eligibility[participant],
            "n_profiles": len(by_profile),
            "n_distinct_profiles_sampled": distinct,
            "primary_profile": primary,
            "across_profiles": across,
        }

    eligible = [p for p in participants if eligibility[p]["eligible"]]

    def aggregate(key: str, statistic: str) -> dict:
        """Participant-level aggregate of one delta, over eligible participants only."""
        primary_values, mean_values = [], []
        for p in eligible:
            record = per_participant[p]
            primary = record["primary_profile"]
            if primary and primary["scorable"] and primary["delta"].get(key):
                value = primary["delta"][key][statistic]
                if value is not None:
                    primary_values.append(value)
            summary = record["across_profiles"]["deltas"][key][statistic]
            if summary:
                mean_values.append(summary["mean"])
        return {
            "at_primary_profile": {
                "per_participant": primary_values,
                "summary": _summary(primary_values),
                "sign_test": _sign_test(primary_values),
            },
            "mean_over_profiles": {
                "per_participant": mean_values,
                "summary": _summary(mean_values),
                "sign_test": _sign_test(mean_values),
            },
        }

    def method_level(statistic: str) -> dict:
        out = {}
        for method in METHODS:
            values = []
            for p in eligible:
                summary = per_participant[p]["across_profiles"]["methods"].get(
                    method, {}).get(statistic)
                if summary:
                    values.append(summary["mean"])
            out[method] = _summary(values)
        return out

    return {
        "split": split,
        "alpha": float(alpha),
        "participants": participants,
        "eligible_participants": eligible,
        "ineligible_participants": {
            str(p): eligibility[p]["reason"] for p in participants
            if not eligibility[p]["eligible"]
        },
        "n_profiles": len(profile_indices),
        "profile_indices": (profile_indices if len(profile_indices) <= 16
                            else [profile_indices[0], "...", profile_indices[-1]]),
        "primary_profile_index": primary_profile,
        "window_indices": list(window_indices),
        "selection_audit_sha256": selection_audit.hexdigest(),
        "per_participant": {str(p): per_participant[p] for p in participants},
        "population": {
            "by_method": {"macro_f1": method_level("macro_f1"),
                          "accuracy": method_level("accuracy")},
            "deltas": {
                key: {"macro_f1": aggregate(key, "macro_f1"),
                      "accuracy": aggregate(key, "accuracy")}
                for key in DELTA_KEYS
            },
        },
    }


# ------------------------------------------------------------------ printing


def _fmt(value, digits: int = 4) -> str:
    return "-" if value is None else f"{value:+.{digits}f}"


def print_report(payload: dict) -> None:
    result = payload["result"]
    alpha_selection = payload.get("alpha_selection")
    split = result["split"]

    print("=" * WIDTH)
    print(f"EmojiHero hybrid calibration study -- {split} split")
    print("=" * WIDTH)
    print()

    print(rule("what this is, and is not"))
    print("  The offline analogue of session-local-v1, not a repeat of it. The adapter")
    print("  arithmetic is identical; the data is not. EmojiHero windows share 4 of their")
    print("  5 frames where live windows share none, a sequence is 30 frames where a live")
    print("  trial is ~10 seconds, and the prompts are a different protocol.")
    print()
    print("  A gain here is NOT an upper or lower bound on the live gain. The live 6-8")
    print("  participant pilot is still required.")
    print()
    if split == "validation":
        print("  This split contributed ~70.4% of the tensor the teacher early-stopped on")
        print("  (2,310 of 3,279 windows), so it is where the method is DEVELOPED and it")
        print("  is not independent evidence about the base model.")
    else:
        print("  This split entered neither gradients nor model selection, but has been")
        print("  opened before when reporting teacher performance. It is confirmatory")
        print("  here only because the adapter was frozen before it was run.")
    print()
    print("  Four methods on identical evaluation sequences:")
    print("    raw         the model's own argmax")
    print("    population  seven centroids from the EmojiHero TRAINING participants")
    print("    personal    seven centroids from one sequence per emotion, this person")
    print(f"    hybrid      alpha * personal + (1 - alpha) * population, "
          f"alpha = {result['alpha']:.2f}")
    print()
    print("  `hybrid - personal` is what isolates the population term's contribution.")
    print("  `hybrid - raw` cannot: the personal term alone would move it.")
    print()

    if alpha_selection:
        print(rule("alpha selection (validation only)"))
        print(f"  objective   {alpha_selection['objective']}")
        print(f"  grid        {alpha_selection['alpha_grid'][0]} .. "
              f"{alpha_selection['alpha_grid'][-1]} in steps of 0.05")
        print(f"  profiles    {alpha_selection['n_profiles']} per participant")
        print(f"  participants{'':<1} {alpha_selection['participants_used']}")
        print()
        objective = alpha_selection["objective_by_alpha"]
        ordered = sorted(objective.items(), key=lambda kv: float(kv[0]))
        for start in range(0, len(ordered), 7):
            chunk = ordered[start:start + 7]
            print("    " + "".join(f"{float(a):>10.2f}" for a, _ in chunk))
            print("    " + "".join(f"{v:>10.4f}" for _, v in chunk))
        print()
        print(f"  selected alpha        {alpha_selection['selected_alpha']:.2f}"
              f"   (tie-break: {alpha_selection['tie_break']})")
        print(f"  personal-only (a=1)   {num(alpha_selection['personal_only_objective'])}")
        print(f"  population-only (a=0) {num(alpha_selection['population_only_objective'])}")
        if alpha_selection["population_adds_nothing"]:
            print()
            print("  alpha = 1.00 was selected. On this validation split the EmojiHero")
            print("  population centroid adds NOTHING over personal enrolment. That is the")
            print("  result, not a failure to be tuned away.")
        print()

    print(rule("eligibility"))
    print(f"  participants          {result['participants']}")
    print(f"  eligible (primary)    {result['eligible_participants']}")
    for participant, reason in result["ineligible_participants"].items():
        print(f"  participant {participant} EXCLUDED from the seven-class primary:")
        print(f"    {reason}")
    if not result["ineligible_participants"]:
        print("  none excluded")
    print()

    print(rule("per participant -- sequence-level macro-F1 at the primary profile"))
    header = (f"  {'pid':<5}{'elig':<6}{'raw':>8}{'popul':>8}{'pers':>8}{'hybr':>8}"
              f"{'pers-raw':>10}{'hybr-raw':>10}{'hybr-pers':>11}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key, record in result["per_participant"].items():
        primary = record["primary_profile"]
        if not primary or not primary["scorable"]:
            print(f"  {key:<5}{'-':<6}  (not scorable)")
            continue
        level = primary["sequence_level"]
        delta = primary["delta"]

        def cell(method: str) -> str:
            value = level.get(method, {}).get("macro_f1")
            return "-" if value is None else f"{value:.4f}"

        def d(key_name: str) -> str:
            block = delta.get(key_name)
            if not block or block["macro_f1"] is None:
                return "-"
            return f"{block['macro_f1']:+.4f}"

        print(f"  {key:<5}{'yes' if record['eligible'] else 'NO':<6}"
              f"{cell('raw'):>8}{cell('population'):>8}{cell('personal'):>8}"
              f"{cell('hybrid'):>8}"
              f"{d('personal_minus_raw'):>10}{d('hybrid_minus_raw'):>10}"
              f"{d('hybrid_minus_personal'):>11}")
    print()

    print(rule("per participant -- mean over all sampled profiles, macro-F1"))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key, record in result["per_participant"].items():
        across = record["across_profiles"]

        def cell(method: str) -> str:
            summary = across["methods"].get(method, {}).get("macro_f1")
            return "-" if not summary else f"{summary['mean']:.4f}"

        def d(key_name: str) -> str:
            summary = across["deltas"][key_name]["macro_f1"]
            return "-" if not summary else f"{summary['mean']:+.4f}"

        print(f"  {key:<5}{'yes' if record['eligible'] else 'NO':<6}"
              f"{cell('raw'):>8}{cell('population'):>8}{cell('personal'):>8}"
              f"{cell('hybrid'):>8}"
              f"{d('personal_minus_raw'):>10}{d('hybrid_minus_raw'):>10}"
              f"{d('hybrid_minus_personal'):>11}")
    print()

    for key in DELTA_KEYS:
        block = result["population"]["deltas"][key]["macro_f1"]
        print(rule(f"population -- {key.replace('_', ' ')}, macro-F1, "
                   "eligible participants"))
        for name, cell_key in (("at the primary profile", "at_primary_profile"),
                               ("mean over all profiles", "mean_over_profiles")):
            summary = block[cell_key]["summary"]
            sign = block[cell_key]["sign_test"]
            if not summary:
                print(f"  {name:<24} (nothing scorable)")
                continue
            print(f"  {name:<24} mean {summary['mean']:+.4f}   "
                  f"median {summary['median']:+.4f}   "
                  f"min {summary['min']:+.4f}   max {summary['max']:+.4f}")
            print(f"  {'':<24} improved {sign['positive']}/{sign['n'] or 0}"
                  f"   sign test p = {num(sign['p_two_sided'])}"
                  f"   (smallest reachable {num(sign.get('smallest_reachable_p'))})")
        print()

    print(rule("per class -- sequence recall delta, mean over profiles and participants"))
    print(f"  {'class':<12}" + "".join(f"{k.replace('_minus_', '-'):>20}"
                                       for k in ("personal_minus_raw",
                                                 "hybrid_minus_personal")))
    for emotion in EMOTIONS:
        line = f"  {emotion:<12}"
        for key in ("personal_minus_raw", "hybrid_minus_personal"):
            values = [
                record["across_profiles"]["deltas"][key]["per_class_recall"][emotion]["mean"]
                for record in result["per_participant"].values()
                if record["eligible"]
                and record["across_profiles"]["deltas"][key]["per_class_recall"][emotion]
            ]
            line += f"{np.mean(values):>+20.4f}" if values else f"{'-':>20}"
        print(line)
    print()

    print(rule("diagnostics (never used to select anything)"))
    for key, record in result["per_participant"].items():
        across = record["across_profiles"]
        gap = across["level_gap"]
        if not gap:
            continue
        separation = across["centroid_separation_mean"]
        similarity = across["personal_to_population_cosine"]
        personal_sep = separation.get("personal")
        print(f"  p{key:<4} level gap K-D {gap['mean']:+.3f}   "
              f"personal centroid separation "
              f"{personal_sep['mean']:.4f}" if personal_sep else "")
        if similarity:
            print(f"        personal-to-population cosine {similarity['mean']:.4f}")
    print()
    print("  A positive level gap means calibration sequences came from harder levels")
    print("  than the sequences scored, which would confound personalisation with")
    print("  expression-intensity drift. Reported, never acted on.")
    print()


# ------------------------------------------------------------------- output


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


# Field names a per-participant record may carry into the result file. A whitelist, for the
# same reason `calibration_study.RESULT_FIELDS` is one: the output of this analysis must not
# become a second copy of the face signal. No AU frames, no score vectors, and no PERSONAL
# centroid vector -- only counts, metrics and identifiers. The population centroid is the
# single exception and lives in its own artifact, because it is an average over twenty
# training participants and describes nobody.
PARTICIPANT_FIELDS = (
    "scorable", "reason", "alpha", "methods", "profile_index",
    "n_calibration_sequences", "n_evaluation_sequences", "n_evaluation_windows",
    "enrolled_classes", "classes_with_a_scored_sequence", "primary_macro_f1_valid",
    "sequence_level", "delta", "paired_sequences", "window_level",
    "centroid_separation", "centroid_drift_calibration_to_evaluation",
    "personal_to_population_cosine", "level_diagnostic", "calibration_sequence_ids",
    "evaluation_sequence_ids", "confusion",
)

FORBIDDEN_SUBSTRINGS = ("centroid_vector", "personal_centroid", "score_vector",
                        "au_window", "frames", "windows_raw")


def _check_no_biometrics(payload: dict) -> None:
    """Refuse to write anything that looks like a copy of the face signal.

    Asserted against the serialised text rather than against the structure, so a field
    added anywhere in the tree has to get past it. The consented record is the corpus; this
    file is a set of measurements about it.
    """
    text = json.dumps(payload)
    for needle in FORBIDDEN_SUBSTRINGS:
        if f'"{needle}"' in text:
            raise SystemExit(
                f"refusing to write a result file containing a {needle!r} field. "
                "Result files hold metrics and identifiers, never action units, score "
                "vectors or per-person centroid vectors."
            )


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=str(Path(__file__).resolve().parent), timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def package_versions() -> dict:
    versions = {"python": sys.version.split()[0]}
    for name in ("numpy", "onnxruntime", "pandas", "openpyxl"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = None
    return versions


def write_json(path: Path, payload: dict) -> None:
    """Never overwrite. A result belongs to the settings and corpus that produced it."""
    if path.exists():
        raise SystemExit(
            f"refusing to overwrite an existing file:\n  {path}\n"
            "Remove or rename it if the previous run should be replaced."
        )
    _check_no_biometrics(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def prune_participants(result: dict, keep_profiles: int) -> dict:
    """Trim per-profile detail before writing.

    With a thousand profiles the full per-profile records run to hundreds of megabytes and
    say nothing the summaries do not. The primary profile is kept whole, the rest collapse
    to their summary statistics -- and because selection is a pure function of the seed, any
    profile can be regenerated exactly from the config in this same file.
    """
    trimmed = json.loads(json.dumps(_jsonable(result)))
    for record in trimmed["per_participant"].values():
        primary = record.get("primary_profile")
        if primary and primary.get("scorable"):
            record["primary_profile"] = {k: v for k, v in primary.items()
                                         if k in PARTICIPANT_FIELDS}
    trimmed["per_profile_detail_retained"] = keep_profiles
    return trimmed


# ---------------------------------------------------------------------- main


def print_dry_run(split: str, sequences: list[Sequence], frozen_config: dict | None,
                  population_meta: dict, profile_count: int, window_condition: str,
                  primary_profile: int) -> None:
    """Everything that can be checked without scoring the split.

    Reads sequence identifiers and counts -- which the integrity gate above has already
    read -- and runs no model, produces no prediction and computes no metric. Safe to run
    against the test split before the test phase is approved: it can tell you the run
    would be valid without telling you anything about the answer.
    """
    print(rule(f"DRY RUN -- nothing was scored on the {split} split"))
    print("  The model was not run over this split. No prediction, no metric, no result.")
    print()

    table = eligibility_table(sequences)
    eligible = [p for p, record in table.items() if record["eligible"]]
    print(f"  participants         {sorted(table)}")
    print(f"  eligible for primary {eligible}  (n = {len(eligible)})")
    for participant, record in table.items():
        if not record["eligible"]:
            print(f"  participant {participant} would be EXCLUDED: {record['reason']}")
    print()

    print(f"  sequences            {len(sequences)}")
    print(f"  profiles             {profile_count}, seed {MASTER_SEED}")
    print(f"  primary profile      {primary_profile}")
    print(f"  window condition     {window_condition}")
    print()

    print(rule("population artifact"))
    print(f"  semantic hash        {population_meta['semantic_sha256']}")
    print(f"  source split         {population_meta['source_split']}")
    print(f"  training participants{population_meta['n_participants']:>4}, "
          f"{population_meta['n_sequences']} sequences")
    print(f"  model sha256         {population_meta['model']['sha256']}")
    print()

    primary = FROZEN_SETTINGS["primary_analysis"]
    print(rule("frozen primary analysis"))
    if frozen_config:
        print(f"  fixed alpha          {frozen_config['selected_alpha']}")
        print(f"  frozen at            {frozen_config.get('frozen_utc')}")
        print(f"  frozen after         {frozen_config.get('frozen_after')}")
    else:
        print("  alpha                (not frozen -- no --frozen-config given)")
    print(f"  metric               {primary['metric']}")
    print(f"  comparison           {primary['comparison']}")
    print(f"  inference unit       {primary['inference_unit']}, n = {len(eligible)}")
    print(f"  test                 {primary['statistical_test']}")
    print()
    print("  profiles are NOT a sample size:")
    for line in _wrap(primary["profiles_are_not_a_sample_size"], 72):
        print(f"    {line}")
    print()
    print("  secondary only:")
    for item in primary["secondary_only"]:
        print(f"    - {item}")
    print()


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def load_frozen_config(path: Path) -> dict:
    """Read the specification frozen after validation, refusing any drift from it.

    Every setting that could move a number is compared, and so is the selected alpha: the
    test split may only be scored with the weight chosen before it was opened.
    """
    config = json.loads(Path(path).read_text(encoding="utf-8"))

    frozen = config.get("frozen_settings")
    if frozen is None:
        raise SystemExit(f"{path}: no 'frozen_settings' block")

    differences = [
        f"  {key}: frozen {frozen.get(key)!r}, this build {FROZEN_SETTINGS.get(key)!r}"
        for key in sorted(set(frozen) | set(FROZEN_SETTINGS))
        if frozen.get(key) != _jsonable(FROZEN_SETTINGS).get(key)
    ]
    if differences:
        raise SystemExit(
            "the frozen specification does not match this build of the study:\n"
            + "\n".join(differences)
            + "\nThe test split may only be scored with the settings frozen before it was "
              "opened. Restore the code, or run the validation phase again."
        )

    alpha = config.get("selected_alpha")
    if alpha is None:
        raise SystemExit(
            f"{path}: no 'selected_alpha'. The hybrid weight must be chosen on validation "
            "and written down before the test split is opened."
        )
    if not 0.0 <= float(alpha) <= 1.0:
        raise SystemExit(f"{path}: selected_alpha {alpha} is outside [0, 1]")

    # The config states the fixed alpha twice -- once as the value the run uses and once
    # inside the primary-analysis definition a reader will quote. If those two ever
    # disagree, one of them was edited, and there is no way to tell which is the
    # preregistration.
    primary = config.get("primary_analysis") or {}
    fixed = primary.get("fixed_alpha")
    if fixed is None:
        raise SystemExit(
            f"{path}: no 'primary_analysis.fixed_alpha'. The primary analysis must be "
            "stated in the config, not only in the code."
        )
    if float(fixed) != float(alpha):
        raise SystemExit(
            f"{path}: primary_analysis.fixed_alpha ({fixed}) disagrees with "
            f"selected_alpha ({alpha}). The config has been edited."
        )

    plan = config.get("test_plan") or {}
    for required in ("primary_profile_index", "robustness_profile_count",
                     "window_condition"):
        if required not in plan:
            raise SystemExit(
                f"{path}: no 'test_plan.{required}'. Everything that could move a number "
                "has to be pinned in the config, because the command line cannot supply "
                "it on the test split."
            )

    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data", default=None,
        help="directory or .zip holding training_set.xlsx, validation_set.xlsx and "
             "test_set.xlsx (default: Analysis/data)")
    parser.add_argument(
        "--split", choices=ANALYSABLE_SPLITS,
        help="validation develops the method and chooses alpha; test is confirmatory and "
             "needs --frozen-config")
    parser.add_argument(
        "--build-population", action="store_true",
        help="build the population centroids from the TRAINING split and exit")
    parser.add_argument(
        "--population-output", default=None,
        help="where --build-population writes the artifact "
             "(default: Analysis/results/emojihero_population_centroids.json)")
    parser.add_argument(
        "--population-centroids", default=None,
        help="the population artifact to score with. Required unless --build-population.")
    parser.add_argument("--model", help="path to sparseVR_teacher_model.onnx")
    parser.add_argument(
        "--profile-count", type=int, default=None,
        help=f"calibration profiles per participant (default: "
             f"{VALIDATION_PROFILE_COUNT} on validation, {TEST_ROBUSTNESS_PROFILES} on test)")
    parser.add_argument(
        "--hybrid-alpha", type=float, default=None,
        help="score at this alpha instead of choosing one. On validation this skips the "
             "sweep; on test the frozen config supplies it and this is refused.")
    parser.add_argument(
        "--frozen-config", default=None,
        help="the frozen specification written after validation. Required for --split test.")
    parser.add_argument(
        "--frozen-config-output", default=None,
        help="after a validation run, write the frozen specification here")
    parser.add_argument(
        "--windows", default="stride1", choices=["stride1", "non-overlapping"],
        help="stride1 is the frozen primary (six windows). non-overlapping scores only "
             "the two windows that share no frame, as a sensitivity check.")
    parser.add_argument("--output", default=None, help="result JSON (never overwritten)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="load the corpus, run the integrity gate, validate the frozen config and the "
             "population artifact, and print the frozen primary-analysis definition -- "
             "then stop. Runs no model over the split and produces no prediction, so it "
             "is safe to run against test before the test phase is approved.")
    parser.add_argument("--quiet", action="store_true", help="no per-participant progress")
    args = parser.parse_args(argv)

    if not args.build_population and not args.split:
        raise SystemExit("pass --split validation, --split test, or --build-population")
    if args.build_population and args.split:
        raise SystemExit("--build-population builds from the training split; it takes no "
                         "--split")
    if args.split == "test":
        if not args.frozen_config:
            raise SystemExit(
                "--split test requires --frozen-config.\n"
                "The test split is confirmatory: it may only be scored with a "
                "specification written down before it was opened. Run --split validation "
                "first, write the config with --frozen-config-output, then pass it here."
            )
        # Nothing that could move a primary number may come from the command line on the
        # confirmatory split. Each of these is pinned in the frozen config instead, and a
        # run that silently accepted an override here would turn a preregistration into a
        # suggestion. There is deliberately no --epsilon, --metric, --seed or
        # --primary-profile flag anywhere in this parser, for the same reason.
        overrides = [
            name for name, given in (
                ("--hybrid-alpha", args.hybrid_alpha is not None),
                ("--profile-count", args.profile_count is not None),
                ("--windows", args.windows != parser.get_default("windows")),
            ) if given
        ]
        if overrides:
            raise SystemExit(
                f"{', '.join(overrides)} cannot be given with --split test.\n"
                "The alpha, the profile count and the window condition are pinned in the "
                "frozen config. Choosing any of them now would be choosing it after "
                "seeing the split it is scored on.\n"
                "Run the sensitivity conditions on validation, where they belong."
            )
        if args.frozen_config_output:
            raise SystemExit(
                "--frozen-config-output cannot be given with --split test: the "
                "specification is frozen before the test split is opened, not after."
            )

    root = sa.find_repo_root()
    results_dir = (root / "Analysis" / "results") if root else Path("Analysis/results")
    data = Path(args.data).expanduser() if args.data else (
        (root / "Analysis" / "data") if root else Path("Analysis/data"))
    if not data.exists():
        raise SystemExit(f"--data does not exist: {data}")

    model_path = cs.find_model(args.model)
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()

    frozen_config = load_frozen_config(Path(args.frozen_config)) if args.frozen_config else None
    if frozen_config and frozen_config.get("model_sha256") not in (None, model_sha):
        raise SystemExit(
            f"the model on disk is not the one the specification was frozen against.\n"
            f"  frozen: {frozen_config['model_sha256']}\n"
            f"  found:  {model_sha}\n"
            "Refusing to score the test split with a different model."
        )

    print("=" * WIDTH)
    print(f"loading the corpus from {data}")
    print("=" * WIDTH)
    loaded, provenance = {}, {}
    for split in SPLITS:
        sequences, meta = load_split(data, split)
        loaded[split], provenance[split] = sequences, meta
        print(f"  {split:<11} {len(sequences):>4} sequences   sha256 {meta['sha256'][:16]}...")
    print()

    checks = check_corpus(loaded, provenance)
    print(rule("corpus integrity -- all checks passed"))
    for check in checks:
        print(f"  ok   {check['check']}: {check['detail']}")
    print()
    print(f"  model  {model_path}")
    print(f"  sha256 {model_sha}")
    print()

    # ------------------------------------------------------ population centroids

    if args.build_population:
        training = loaded[POPULATION_SOURCE_SPLIT]
        print(rule(f"building population centroids from the {POPULATION_SOURCE_SPLIT} split"))
        index_of = {s.sequence_id: i for i, s in enumerate(training)}
        scores = run_model(training, model_path)
        logf = cs.logscore(scores, cs.DEFAULT_EPSILON)
        built = build_population_centroids(training, logf, index_of)

        for emotion, support in sorted(built["support"].items()):
            print(f"  {emotion:<12} {support['participants']:>3} participants, "
                  f"{support['sequences']:>4} sequences")
        separation = _cosine_separation(built["matrix"])
        print(f"  centroid separation  mean {separation['mean_pairwise_cosine']:.5f}  "
              f"min {separation['min']:.5f}  max {separation['max']:.5f}")
        print()

        artifact = population_artifact(
            built["matrix"], built["support"], model_sha=model_sha,
            model_path=model_path, provenance=provenance[POPULATION_SOURCE_SPLIT],
            sequences=training)
        output = Path(args.population_output).expanduser() if args.population_output else (
            results_dir / "emojihero_population_centroids.json")
        write_json(output, artifact)
        print(f"wrote {output}")
        return 0

    # ------------------------------------------------------------- scoring runs

    if not args.population_centroids:
        raise SystemExit(
            "--population-centroids is required. Build the artifact first:\n"
            "  python Analysis/emojihero_calibration_study.py --build-population"
        )
    population_M, population_meta = load_population_centroids(
        Path(args.population_centroids).expanduser(), model_sha=model_sha)

    if frozen_config:
        frozen_population = (frozen_config.get("population_artifact") or {}).get(
            "semantic_sha256")
        actual = population_meta["semantic_sha256"]
        if frozen_population is None:
            raise SystemExit(
                f"{args.frozen_config}: the population artifact is pinned by file bytes "
                "rather than by its semantic hash. Re-freeze the config: a byte hash "
                "rejects a correct rebuild, because the artifact records when it was "
                "created."
            )
        if frozen_population != actual:
            raise SystemExit(
                "the population artifact is not the one frozen after validation.\n"
                f"  frozen semantic hash: {frozen_population}\n"
                f"  artifact on disk:     {actual}\n"
                "The centroids, the model, the preprocessing or the training corpus "
                "differ. Restore the frozen artifact."
            )

    # With a frozen config, every one of these comes from the config rather than from the
    # command line or from this build's defaults. The CLI guard above refuses the flags;
    # this is the other half of the same rule, so that a config written against different
    # defaults still governs the run.
    if frozen_config:
        plan = frozen_config.get("test_plan") or {}
        profile_count = int(plan.get("robustness_profile_count", TEST_ROBUSTNESS_PROFILES))
        window_condition = str(plan.get("window_condition", "stride1"))
        primary_profile = int(plan.get("primary_profile_index", PRIMARY_TEST_PROFILE))
    else:
        profile_count = args.profile_count if args.profile_count is not None else (
            VALIDATION_PROFILE_COUNT if args.split == "validation"
            else TEST_ROBUSTNESS_PROFILES)
        window_condition = args.windows
        primary_profile = PRIMARY_TEST_PROFILE

    if profile_count < 1:
        raise SystemExit("--profile-count must be at least 1")
    if window_condition not in ("stride1", "non-overlapping"):
        raise SystemExit(f"unknown window condition {window_condition!r}")
    profile_indices = list(range(profile_count))
    window_indices = (ALL_WINDOWS if window_condition == "stride1"
                      else NON_OVERLAPPING_WINDOWS)

    sequences = loaded[args.split]

    print(f"  profiles {profile_count}, seed {MASTER_SEED}, "
          f"windows {window_condition} {list(window_indices)}")
    print(f"  primary profile index {primary_profile}")
    print(f"  population artifact {args.population_centroids}")
    print(f"  population semantic hash {population_meta['semantic_sha256']}")
    print(f"  settings source {'frozen config' if frozen_config else 'command line'}")
    print()

    if args.dry_run:
        print_dry_run(args.split, sequences, frozen_config, population_meta,
                      profile_count, window_condition, primary_profile)
        return 0

    index_of = {s.sequence_id: i for i, s in enumerate(sequences)}
    scores = run_model(sequences, model_path)
    logf = cs.logscore(scores, cs.DEFAULT_EPSILON)

    alpha_selection = None
    if frozen_config:
        alpha = float(frozen_config["selected_alpha"])
        print(rule(f"alpha = {alpha:.2f}, taken from the frozen config (not re-chosen)"))
        print()
    elif args.hybrid_alpha is not None:
        alpha = float(args.hybrid_alpha)
        if not 0.0 <= alpha <= 1.0:
            raise SystemExit("--hybrid-alpha must lie in [0, 1]")
        print(rule(f"alpha = {alpha:.2f}, given on the command line (no sweep)"))
        print()
    else:
        print(rule("choosing alpha on this split"))
        alpha_selection = select_alpha(
            sequences, args.split, index_of, logf, scores, population_M,
            profile_indices, window_indices=window_indices,
            progress=not args.quiet)
        alpha = alpha_selection["selected_alpha"]
        print(f"  selected alpha {alpha:.2f}")
        print()

    result = run_split(
        sequences, args.split, model_path, profile_indices,
        primary_profile=primary_profile, population_M=population_M, alpha=alpha,
        window_indices=window_indices,
        precomputed=(index_of, scores, logf), progress=not args.quiet)

    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "analysis": "emojihero_hybrid_calibration",
        "split": args.split,
        "window_condition": args.windows,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "model": {
            "path": str(model_path),
            "relative": cs.MODEL_RELATIVE,
            "sha256": model_sha,
            "retrained": False,
            "note": "loaded read-only and executed; never retrained, fine-tuned or "
                    "re-exported",
        },
        "data_provenance": {split: {k: v for k, v in provenance[split].items()
                                    if k != "au_columns"} for split in SPLITS},
        "population_artifact": {
            "path": str(args.population_centroids),
            # Both, and named for what they are. The semantic hash is the pin the frozen
            # config enforces; the file hash identifies the exact bytes that were read and
            # changes on every rebuild, so it is provenance rather than an identity. An
            # earlier version recorded only the file hash here under the bare name
            # `sha256`, which invited a reader to compare it against the config's pin and
            # conclude the artifact had changed when it had not.
            "semantic_sha256": population_meta["semantic_sha256"],
            "file_sha256": hashlib.sha256(
                Path(args.population_centroids).expanduser().read_bytes()).hexdigest(),
            "source_split": population_meta.get("source_split"),
            "created_utc": population_meta.get("created_utc"),
            "n_participants": population_meta.get("n_participants"),
            "n_sequences": population_meta.get("n_sequences"),
            "support_per_emotion": population_meta.get("support_per_emotion"),
        },
        "corpus_integrity": checks,
        "frozen_settings": FROZEN_SETTINGS,
        "alpha_selection": alpha_selection,
        "alpha_used": float(alpha),
        "alpha_source": ("frozen_config" if frozen_config else
                         "command_line" if args.hybrid_alpha is not None else
                         "selected_on_this_split"),
        "package_versions": package_versions(),
        "limitations": [
            "This is the offline analogue of session-local-v1, not a repeat of it: "
            "EmojiHero stride-1 windows share 4 of their 5 frames where live windows share "
            "none, a sequence is 30 frames where a live trial is ~10 seconds, and the "
            "prompt protocol differs.",
            "A gain here is not an upper or lower bound on the live effect in either "
            "direction: the familiar capture domain could inflate it, a higher baseline "
            "could compress it through a ceiling.",
            "The teacher was trained by gradient on the EmojiHero training split, which is "
            "also where the population centroids come from. The population term therefore "
            "describes participants the model was fitted on.",
            "The EmojiHero validation split contributed 2,310 of the 3,279 windows in the "
            "tensor used for early stopping and best-weight selection (~70.4%), so it is "
            "suitable for developing the adapter and choosing alpha but is not independent "
            "evidence of base model generalisation.",
            "The EmojiHero test split entered neither gradients nor model selection but has "
            "been opened before when reporting teacher performance; it is not untouched.",
            "Six stride-1 windows per sequence are not six independent observations. The "
            "sequence is the unit of measurement and the participant the unit of decision.",
            "Profile-to-profile spread mixes the calibration choice with the evaluation "
            "composition, because the enrolled sequence leaves the evaluation set. It is "
            "full protocol sensitivity, not calibration-sequence sensitivity.",
            "The live 6-8 participant session-local-v1 pilot remains necessary; nothing "
            "here substitutes for it.",
            "Tongue channels are zero-filled because the teacher was trained that way. For "
            "descriptive action-unit comparison they remain unobserved, not measured zeros.",
        ],
        "result": prune_participants(result, keep_profiles=1),
    }

    print_report({"result": result, "alpha_selection": alpha_selection})

    output = Path(args.output).expanduser() if args.output else (
        results_dir / f"emojihero_hybrid_{args.split}.json")
    write_json(output, payload)
    print(f"wrote {output}")

    text_path = output.with_suffix(".txt")
    if not text_path.exists():
        import contextlib
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            print_report({"result": result, "alpha_selection": alpha_selection})
        text_path.write_text(buffer.getvalue(), encoding="utf-8")
        print(f"wrote {text_path}")

    if args.frozen_config_output:
        config_path = Path(args.frozen_config_output).expanduser()
        write_json(config_path, {
            "schema_version": RESULT_SCHEMA_VERSION,
            "analysis": "emojihero_hybrid_calibration",
            "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "frozen_after": f"the {args.split} split, git {git_commit()}",
            "git_commit": git_commit(),
            "model_sha256": model_sha,
            "data_sha256": {split: provenance[split]["sha256"] for split in SPLITS},
            "population_artifact": {
                "path": str(args.population_centroids),
                # The pin is the semantic hash, not the file's bytes: the artifact records
                # when it was built, so a correct rebuild from identical inputs is
                # byte-different and a byte pin would reject it.
                "semantic_sha256": population_meta["semantic_sha256"],
                "file_sha256_at_freeze_time": hashlib.sha256(
                    Path(args.population_centroids).expanduser().read_bytes()).hexdigest(),
                "corpus_fingerprint": population_meta["corpus_fingerprint"],
            },
            "selected_alpha": float(alpha),
            "primary_analysis": {
                **_jsonable(FROZEN_SETTINGS["primary_analysis"]),
                "fixed_alpha": float(alpha),
                "expected_primary_n": "the eligible participants of the split being "
                                      "scored; 8 on validation, 7 on test",
            },
            "alpha_selection": alpha_selection,
            "frozen_settings": FROZEN_SETTINGS,
            "test_plan": {
                "primary_profile_index": PRIMARY_TEST_PROFILE,
                "robustness_profile_count": TEST_ROBUSTNESS_PROFILES,
                "robustness_profile_indices": f"0..{TEST_ROBUSTNESS_PROFILES - 1}",
                "window_condition": "stride1",
                "primary_population": "participants with at least "
                                      f"{SEQUENCES_REQUIRED_PER_EMOTION} sequences for "
                                      "every emotion",
                "secondary": ["non-overlapping window condition",
                              "level and emoji diagnostics",
                              "window-level metrics",
                              "centroid separation, drift and personal-to-population "
                              "similarity"],
                "forbidden_after_seeing_the_test_result": [
                    "changing alpha, or re-running the sweep on test",
                    "changing the seed, the primary profile index, or the profile count",
                    "changing preprocessing, epsilon, metric, centroid or tie rule",
                    "changing the eligibility rule or the primary population",
                    "promoting a secondary metric to primary",
                    "rebuilding the population centroids",
                ],
            },
        })
        print(f"wrote {config_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
