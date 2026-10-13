#!/usr/bin/env python3
"""Generate the golden fixture the C# runtime calibration is held to.

    python Analysis/make_runtime_parity_fixture.py

Every expected value in the output is produced by the *same* functions the offline analysis
uses -- `calibration_study.logscore`, `.centroids`, `.classify`, `.trial_vote` -- called on
synthetic scores. Nothing is reimplemented here, which is the point: if someone changes the
offline algorithm, this fixture changes, and the C# tests fail until the runtime is changed
to match. The two implementations cannot drift silently.

Two precisions, two claims
--------------------------
The real offline pipeline carries `float32` from the ONNX output through `np.log` and the
matmul. The C# runtime uses `double`. So each case is computed twice:

  `expected_f64`  -- the same operations in float64. The C# must match these numerically,
                     within a tolerance that accounts for numpy's vectorised `log` differing
                     from .NET's `Math.Log` by an ulp or so. This is the "is the arithmetic
                     the same arithmetic" claim.

  `expected_f32`  -- the pipeline exactly as the analysis runs it. The C# must reach the
                     *same predictions*, tick for tick and trial for trial. This is the "does
                     the precision difference ever change a decision" claim, and it is the one
                     that matters for a result.

Bit-exact equality across numpy and .NET is neither achievable nor claimed: the two use
different `log` implementations. Decisions are asserted exactly; magnitudes to 1e-12.

The synthetic scores never touch a corpus or a headset recording. Nothing here is participant
data.
"""

from __future__ import annotations

import json
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibration_study as cs  # noqa: E402
import session_analysis as sa  # noqa: E402

SCHEMA_VERSION = "runtime-calibration-parity-v2"   # v2: movement mode and action-unit windows
FIXTURE_SEED = 20260921
WINDOW_FRAMES = 5
WINDOW_FEATURES = 70                                # 63 face action units + 7 tongue values
FROZEN_ALPHA = 0.40

CLASSES = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
N = len(CLASSES)

SCORED_TICKS = list(range(1, 10))   # tick 0 is a transition sample and is dropped
DEFAULT_DEST = Path("Assets/Tests/EditMode/Fixtures/runtime_calibration_parity.json")
UNITY_ARTIFACT = Path("Assets/StreamingAssets/emojihero_population_centroids_unity.json")


def bits(value: float) -> str:
    return struct.pack(">d", float(value)).hex()


# ----------------------------------------------------------------- synthetic sessions


def softmax_row(rng: np.random.Generator, peak: int, sharpness: float) -> list[float]:
    """A plausible seven-class score vector that favours `peak`.

    Built through an actual softmax so the values look like model output -- summing to one,
    no exact zeros, a realistic dynamic range -- rather than like a hand-written one-hot,
    which would make the log transform trivially separable and test nothing.
    """
    logits = rng.normal(0.0, 1.0, N)
    logits[peak] += sharpness
    shifted = np.exp(logits - logits.max())
    row = shifted / shifted.sum()

    # Rounded to float32 before being written. A score *is* a float32 -- the ONNX graph emits
    # one and the C# runtime takes one -- so a fixture carrying float64 literals would ask the
    # runtime to reproduce a number it could never have been given, and the parity check
    # would fail on the width of a float rather than on the calibration.
    return [float(np.float32(v)) for v in row]


def session(rng: np.random.Generator, *, confusion: dict[int, int] | None = None,
            sharpness: float = 3.0, evaluation_repeats: int = 2) -> dict:
    """One synthetic `session-local-v1` recording: 7 K trials then 14 D trials.

    `confusion` maps a prompted class to the class the model actually favours, which is the
    consistent mis-mapping personal calibration exists to undo.
    """
    confusion = confusion or {}
    calibration, evaluation = [], []
    trial_id = 0

    for cls in range(N):
        ticks = []
        for tick in range(10):                      # tick 0 included; the reader drops it
            peak = confusion.get(cls, cls)
            ticks.append({"tick": tick, "scores": softmax_row(rng, peak, sharpness)})
        calibration.append({"trial_id": trial_id, "prompted": cls, "ticks": ticks})
        trial_id += 1

    for _ in range(evaluation_repeats):
        for cls in range(N):
            ticks = []
            for tick in range(10):
                peak = confusion.get(cls, cls)
                ticks.append({"tick": tick, "scores": softmax_row(rng, peak, sharpness)})
            evaluation.append({"trial_id": trial_id, "prompted": cls, "ticks": ticks})
            trial_id += 1

    return {"calibration": calibration, "evaluation": evaluation}


def add_windows(recording: dict, rng: np.random.Generator) -> dict:
    """Give every tick an action-unit window shaped by the *prompted* class.

    Each class gets a sparse template over the 63 face action units; a tick's window is that
    template at a random strength with a little frame-to-frame noise. The 7 tongue values are
    filled with large random numbers on purpose: if an implementation forgot to drop them,
    the movement centroids would change and the parity check would fail.

    The windows follow the prompted class even where the scores are confused, which is the
    situation movement calibration is for.
    """
    templates = np.zeros((N, cs.FACE_AU_COUNT))
    for cls in range(N):
        active = rng.choice(cs.FACE_AU_COUNT, size=6, replace=False)
        templates[cls, active] = rng.uniform(0.2, 0.8, size=6)

    for trial in recording["calibration"] + recording["evaluation"]:
        for tick in trial["ticks"]:
            strength = rng.uniform(0.3, 1.2)
            face = templates[trial["prompted"]] * strength
            frames = face[None, :] + rng.normal(0.0, 0.03, (WINDOW_FRAMES, cs.FACE_AU_COUNT))
            frames = np.where(face[None, :] > 0, frames, 0.0)       # silent muscles stay at 0
            tongue = rng.uniform(0.0, 1.0, (WINDOW_FRAMES, WINDOW_FEATURES - cs.FACE_AU_COUNT))
            window = np.clip(np.hstack([frames, tongue]), 0.0, 1.0)
            # float32, because that is what a logged action unit is.
            tick["window"] = [[float(np.float32(v)) for v in frame] for frame in window]

    return recording


def scored(trials: list[dict]) -> list[tuple[int, int, int, list[float]]]:
    """(trial_id, prompted, tick, scores) for ticks 1..9 only."""
    return [
        (t["trial_id"], t["prompted"], tick["tick"], tick["scores"])
        for t in trials
        for tick in t["ticks"]
        if tick["tick"] in SCORED_TICKS
    ]


def windows(trials: list[dict]) -> list[list[list[float]]]:
    """The action-unit windows of ticks 1..9, in the same order as `scored`."""
    return [tick["window"] for t in trials for tick in t["ticks"] if tick["tick"] in SCORED_TICKS]


def reference(recording: dict, mode: str, population: np.ndarray | None,
              dtype) -> dict:
    """Run one case through the offline functions at a given precision.

    `cs.centroids` and `cs.classify` are called exactly as `cs.evaluate` calls them, on the
    same feature transform, so this is the offline algorithm and not a description of it.
    """
    enrol = scored(recording["calibration"])
    test = scored(recording["evaluation"])

    enrol_scores = np.array([s for _, _, _, s in enrol], dtype)
    test_scores = np.array([s for _, _, _, s in test], dtype)

    enrol_y = np.array([cls for _, cls, _, _ in enrol])
    every_class = list(range(N))

    enrol_F = cs.logscore(enrol_scores, cs.DEFAULT_EPSILON)
    test_F = cs.logscore(test_scores, cs.DEFAULT_EPSILON)

    personal = (cs.centroids(enrol_F, enrol_y, every_class)
                if mode in ("personal", "hybrid", "movement") else None)
    movement = None

    if mode == "movement":
        enrol_W = np.array(windows(recording["calibration"]), dtype)
        test_W = np.array(windows(recording["evaluation"]), dtype)
        movement = cs.centroids(cs.movement_feature(enrol_W), enrol_y, every_class)
        active = personal
        predicted, top, margin = cs.classify_movement(
            personal, test_F, movement, cs.movement_feature(test_W), every_class)
    else:
        if mode == "personal":
            active = personal
        elif mode == "population":
            active = population.astype(dtype)
        elif mode == "hybrid":
            active = FROZEN_ALPHA * personal + (1.0 - FROZEN_ALPHA) * population.astype(dtype)
        else:
            raise ValueError(f"no reference for mode {mode!r}")

        predicted, top, margin = cs.classify(active, every_class, test_F, "cosine")

    # `classify` returns the winner's similarity and the margin; the runner-up is the
    # difference, and the runtime logs all three, so all three are pinned here.
    runner_up = top - margin

    truth = np.array([CLASSES[cls] for _, cls, _, _ in test])
    names = np.array([CLASSES[c] for c in predicted])
    trials = np.array([t for t, _, _, _ in test])
    ticks = np.array([tick for _, _, tick, _ in test])

    ids, trial_truth, trial_pred = cs.trial_vote(truth, names, trials, ticks, CLASSES)

    # Whether the majority was decided by the tie rule. `trial_vote` does not report it --
    # it has no need to -- so it is counted here from the same votes, which is the
    # definition rather than a second opinion about the winner.
    ties = {}
    scored_ticks = {}
    for t in ids:
        sel = trials == t
        counts = {}
        for label in names[sel]:
            counts[label] = counts.get(label, 0) + 1
        best = max(counts.values())
        ties[t] = sum(1 for n in counts.values() if n == best) > 1
        scored_ticks[t] = int(sel.sum())

    return {
        "centroids": [[float(v) for v in row] for row in active],
        "personal_centroids": ([[float(v) for v in row] for row in personal]
                               if personal is not None else None),
        "movement_centroids": ([[float(v) for v in row] for row in movement]
                               if movement is not None else None),
        "ticks": [
            {
                "trial": int(trials[i]),
                "tick": int(ticks[i]),
                "class_index": int(predicted[i]),
                "class": CLASSES[predicted[i]],
                "winner_similarity": float(top[i]),
                "runner_up_similarity": float(runner_up[i]),
                "margin": float(margin[i]),
            }
            for i in range(len(predicted))
        ],
        "trials": [
            {
                "trial": int(t),
                "prompted": str(trial_truth[i]),
                "class": str(trial_pred[i]),
                "class_index": CLASSES.index(str(trial_pred[i])),
                "tie": bool(ties[t]),
                "scored_ticks": scored_ticks[t],
            }
            for i, t in enumerate(ids)
        ],
    }


def case(name: str, description: str, recording: dict, mode: str,
         population: np.ndarray | None) -> dict:
    f64 = reference(recording, mode, population, np.float64)
    f32 = reference(recording, mode, population, np.float32)

    return {
        "name": name,
        "description": description,
        "mode": mode,
        "uses_population": population is not None,
        "recording": recording,
        "expected_f64": f64,
        # Only the decisions: the float32 magnitudes are a different arithmetic and the C#
        # is not asked to reproduce them, just to agree about who won.
        "expected_f32": {
            "ticks": [{"trial": t["trial"], "tick": t["tick"], "class_index": t["class_index"]}
                      for t in f32["ticks"]],
            "trials": [{"trial": t["trial"], "class_index": t["class_index"], "tie": t["tie"]}
                       for t in f32["trials"]],
        },
    }


# --------------------------------------------------------------------- unit vectors


def unit_cases() -> dict:
    """Small pinned values for the pieces, so a failure localises.

    A parity failure on a whole session says "something differs". These say which operation.
    """
    rng = np.random.default_rng(FIXTURE_SEED + 1)

    # Epsilon inertness and the clamp, on values that straddle it.
    #
    # Rounded through float32 first, because that is what a score *is*: the ONNX graph emits
    # float32 and the C# receives float32 before widening to double. Probing with a float64
    # literal the runtime could never see would measure the width of a float, not the
    # transform.
    probes = [float(np.float32(v)) for v in (1.0, 0.5, 7e-06, 1e-12, 1e-30, 0.0)]
    logscore = [float(v) for v in cs.logscore(np.array(probes, np.float64), cs.DEFAULT_EPSILON)]

    a = rng.normal(0.0, 1.0, N)
    b = rng.normal(0.0, 1.0, N)
    an = a / (np.linalg.norm(a) + 1e-12)
    bn = b / (np.linalg.norm(b) + 1e-12)

    # Two centroids at exactly equal similarity to the probe, to pin the tie rule. Built by
    # mirroring one vector about the probe direction so the equality is structural rather
    # than a value that happens to round the same way.
    probe = np.ones(N, np.float64)
    offset = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    tie_centroids = np.vstack([
        probe + offset, probe - offset,          # equal similarity, indices 0 and 1
        probe * 0.1 + np.arange(N) * 0.001,      # a clear loser
        probe - 3.0, probe - 4.0, probe - 5.0, probe - 6.0,
    ])
    tie_pred, tie_top, tie_margin = cs.classify(
        tie_centroids, list(range(N)), probe[None, :], "cosine")

    # One window through the movement feature: the mean over frames of the first 63 values.
    window = np.array([[float(np.float32(v)) for v in rng.uniform(0.0, 1.0, WINDOW_FEATURES)]
                       for _ in range(WINDOW_FRAMES)])
    movement = cs.movement_feature(window[None, :, :])[0]

    return {
        "logscore": {
            "epsilon": cs.DEFAULT_EPSILON,
            "input": probes,
            "expected": logscore,
            "expected_bits": [bits(v) for v in logscore],
        },
        "cosine": {
            "a": [float(v) for v in a],
            "b": [float(v) for v in b],
            "expected": float(an @ bn),
            "expected_bits": bits(float(an @ bn)),
        },
        "centroid_tie": {
            "note": "two centroids at exactly equal cosine similarity; the lower index wins",
            "centroids": [[float(v) for v in row] for row in tie_centroids],
            "feature": [float(v) for v in probe],
            "expected_class_index": int(tie_pred[0]),
            "expected_margin": float(tie_margin[0]),
        },
        "movement_feature": {
            "note": "mean over the frames of the first 63 values; the 7 tongue values are dropped",
            "window": [[float(v) for v in frame] for frame in window],
            "expected": [float(v) for v in movement],
        },
        "trial_vote": vote_cases(),
    }


def vote_cases() -> list[dict]:
    """Hand-built vote sequences covering each branch of the frozen tie rule."""
    built = [
        ("clear_majority",
         "one class wins outright",
         ["Anger"] * 5 + ["Fear"] * 4),
        ("tie_resolved_by_last_tick",
         "two classes tie and the final tick's class is among them",
         ["Anger", "Anger", "Fear", "Fear", "Anger", "Fear"]),
        ("tie_resolved_by_class_order",
         "three classes tie and the final tick's class is not among the tied set",
         ["Disgust", "Disgust", "Fear", "Fear", "Sadness"]),
    ]

    cases = []
    for name, description, labels in built:
        ticks = np.arange(1, len(labels) + 1)
        trials = np.zeros(len(labels), dtype=int)
        truth = np.array(["Anger"] * len(labels))
        _, _, winner = cs.trial_vote(truth, np.array(labels), trials, ticks, CLASSES)

        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        best = max(counts.values())

        cases.append({
            "name": name,
            "description": description,
            "ticks": [{"tick": int(t), "class_index": CLASSES.index(c)}
                      for t, c in zip(ticks, labels)],
            "expected_class_index": CLASSES.index(str(winner[0])),
            "expected_tie": sum(1 for n in counts.values() if n == best) > 1,
        })

    return cases


# --------------------------------------------------------------------------- driver


def main() -> int:
    root = sa.find_repo_root() or Path.cwd()

    artifact_path = root / UNITY_ARTIFACT
    if not artifact_path.exists():
        raise SystemExit(
            f"no Unity population artifact at\n  {artifact_path}\n"
            "Run `python Analysis/export_population_artifact_for_unity.py` first."
        )

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact["class_order"] != CLASSES:
        raise SystemExit(f"artifact class order {artifact['class_order']} != {CLASSES}")

    population = np.array([row["values"] for row in artifact["centroids"]], np.float64)

    rng = np.random.default_rng(FIXTURE_SEED)

    clean = session(rng)
    confused = session(rng, confusion={1: 6})       # Disgust reads as Surprise, consistently
    flat = session(rng, sharpness=0.4)              # a barely separable participant
    # Drawn last, so the cases above keep exactly the scores they had in v1.
    moving = add_windows(session(rng, confusion={1: 6}), np.random.default_rng(FIXTURE_SEED + 2))

    cases = [
        case("personal_clean",
             "a well-separated participant, personal centroids only",
             clean, "personal", None),
        case("personal_confused",
             "the model answers Surprise whenever the subject performs Disgust; the personal "
             "centroids should undo it",
             confused, "personal", None),
        case("population_only",
             "the frozen EmojiHero profile alone, with the K block collected but unused",
             clean, "population", population),
        case("hybrid_frozen_alpha",
             "alpha = 0.40, the preregistered blend",
             confused, "hybrid", population),
        case("hybrid_on_a_flat_participant",
             "a participant whose scores barely separate, where the blend moves the most",
             flat, "hybrid", population),
        case("movement_confused",
             "the scores answer Surprise for Disgust while the face keeps Disgust's own movement; "
             "movement calibration adds the movement cosine to the personal score cosine",
             moving, "movement", None),
    ]

    fixture = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "Analysis/make_runtime_parity_fixture.py",
        "seed": FIXTURE_SEED,
        "class_order": CLASSES,
        "epsilon": cs.DEFAULT_EPSILON,
        "frozen_alpha": FROZEN_ALPHA,
        "scored_ticks": SCORED_TICKS,
        "ticks_required_per_centroid": cs.TICKS_REQUIRED_PER_CENTROID,
        "population_semantic_sha256": artifact["semantic_sha256"],
        "population_conversion_sha256": artifact["conversion_sha256"],
        "note": (
            "expected_f64 pins the arithmetic; expected_f32 pins only the decisions the "
            "float32 offline pipeline reaches. Synthetic scores throughout -- no corpus and "
            "no participant data."
        ),
        "unit_cases": unit_cases(),
        "cases": cases,
    }

    # Each window on one line. At one number per line the 210 windows of the movement case
    # would add about 75,000 lines; on one line each they stay readable and small.
    compact = {}
    for c in cases:
        for block in ("calibration", "evaluation"):
            for trial in c["recording"][block]:
                for tick in trial["ticks"]:
                    if "window" in tick:
                        key = f"@@window{len(compact)}@@"
                        compact[key] = json.dumps(tick["window"], separators=(",", ":"))
                        tick["window"] = key

    text = json.dumps(fixture, indent=1)
    for key, value in compact.items():
        text = text.replace(f'"{key}"', value)

    dest = root / DEFAULT_DEST
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text + "\n", encoding="utf-8")

    disagreements = 0
    for c in cases:
        f64 = {(t["trial"], t["tick"]): t["class_index"] for t in c["expected_f64"]["ticks"]}
        for t in c["expected_f32"]["ticks"]:
            if f64[(t["trial"], t["tick"])] != t["class_index"]:
                disagreements += 1

    print(f"wrote {dest}")
    print(f"  cases            {len(cases)}")
    print(f"  ticks per case   {len(cases[0]['expected_f64']['ticks'])}")
    print(f"  trials per case  {len(cases[0]['expected_f64']['trials'])}")
    print(f"  population pin   {artifact['conversion_sha256']}")
    print(f"  float32 vs float64 tick disagreements: {disagreements}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
