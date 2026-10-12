#!/usr/bin/env python3
"""Tests for `runtime_calibration_report.py`.

    python Analysis/test_runtime_calibration_report.py

Synthetic sessions only: each test writes a small session folder into a temporary directory,
so nothing here needs a headset, a corpus, or ONNX.

What these cover, and what they do not. The Python-to-C# parity of the calibration arithmetic
is pinned by `Analysis/fixtures/runtime_calibration_parity.json` and checked by
`Analysis/CsharpTests`. These tests cover the layer above it: that the report reads the columns
the runtime writes, scores the evaluation block and only the evaluation block, refuses to
invent a calibrated metric where there is none, and that the parity check actually fires when
the recorded predictions and the recomputation disagree.

The action-unit columns are omitted from the synthetic files, except in movement sessions. This
tool never replays a window through ONNX -- that is `calibration_study.py`'s job and its
`replay_fidelity` block -- but movement calibration builds its second term from the logged
windows, so a movement session carries them.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibration_study as cs  # noqa: E402
import runtime_calibration_report as rcr  # noqa: E402
import session_analysis as sa  # noqa: E402

CLASSES = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
N = len(CLASSES)
SEED = 4242

BASE_COLUMNS = [
    "timestamp_utc", "session_id", "trial_id", "block_role", "repeat_index",
    "prompted_emotion", "prediction_index", "predicted_emotion", "correct", "status",
    "generation",
]

CALIBRATION_COLUMNS = [
    "calibration_mode", "personal_calibration_enabled", "population_calibration_enabled",
    "movement_calibration_enabled",
    "hybrid_alpha", "raw_predicted_emotion", "calibrated_predicted_emotion",
    "calibrated_winner_similarity", "calibrated_runner_up_similarity", "calibrated_margin",
    "calibration_valid", "calibration_error",
]

FRAMES, FEATURES = 5, 70
AU_COLUMNS = [f"frame_{f}_au_{a}" for f in range(FRAMES) for a in range(FEATURES)]


def artifact_path() -> Path:
    root = sa.find_repo_root() or Path.cwd()
    return root / rcr.UNITY_ARTIFACT


def population_matrix() -> np.ndarray:
    artifact = json.loads(artifact_path().read_text(encoding="utf-8"))
    return np.array([row["values"] for row in artifact["centroids"]], np.float64)


def scores_for(rng: np.random.Generator, peak: int, sharpness: float = 3.0,
               signature: int | None = None) -> list[float]:
    """A plausible seven-class score vector favouring `peak`.

    `signature` raises a second channel without overtaking the peak. It is what makes a
    mis-mapped class recoverable: a model that answered Surprise for both Disgust and Surprise
    with *identical* distributions has destroyed the distinction, and no centroid could bring
    it back. A real mis-mapping leaves a trace of the true class, and that trace is what the
    personal centroid picks up.
    """
    logits = rng.normal(0.0, 1.0, N)
    logits[peak] += sharpness
    if signature is not None and signature != peak:
        logits[signature] += sharpness * 0.5
    shifted = np.exp(logits - logits.max())
    return [float(np.float32(v)) for v in shifted / shifted.sum()]


class SessionBuilder:
    """Writes a session folder the way the runtime would have written it."""

    def __init__(self, directory: Path, mode: str, *, valid: bool = True,
                 error: str = "", sharpness: float = 3.0, confusion: dict | None = None):
        self.directory = directory
        self.mode = mode
        self.valid = valid
        self.error = error
        self.alpha = 0.40 if mode == rcr.MODE_HYBRID else None
        self.rng = np.random.default_rng(SEED)
        self.sharpness = sharpness
        self.confusion = confusion or {}

        # Movement sessions only: a sparse face template per class, from a separate generator
        # so the scores of every other mode stay exactly what they were.
        self.window_rng = np.random.default_rng(SEED + 1)
        self.templates = np.zeros((N, cs.FACE_AU_COUNT))
        for cls in range(N):
            active = self.window_rng.choice(cs.FACE_AU_COUNT, size=6, replace=False)
            self.templates[cls, active] = self.window_rng.uniform(0.2, 0.8, size=6)

        self.enrol: list[dict] = []
        self.test: list[dict] = []
        self._plan()

    def _plan(self) -> None:
        trial_id = 0
        for cls in range(N):
            self.enrol.append(self._trial(trial_id, cls, sa.ROLE_CALIBRATION, 1))
            trial_id += 1
        for repeat in (1, 2):
            for cls in range(N):
                self.test.append(self._trial(trial_id, cls, sa.ROLE_EVALUATION, repeat))
                trial_id += 1

    def _trial(self, trial_id: int, cls: int, role: str, repeat: int) -> dict:
        peak = self.confusion.get(cls, cls)
        signature = cls if cls in self.confusion else None
        return {
            "trial_id": trial_id,
            "prompted": cls,
            "role": role,
            "repeat": repeat,
            # Tick 0 included, as a real recording has it, so the tests exercise it being
            # dropped rather than assuming it away.
            "ticks": [
                {"tick": t,
                 "scores": scores_for(self.rng, peak, self.sharpness, signature),
                 # The face follows the *prompted* class even where the scores are confused.
                 "window": self._window(cls) if self.mode == rcr.MODE_MOVEMENT else None}
                for t in range(10)
            ],
        }

    def _window(self, cls: int) -> list[list[float]]:
        face = self.templates[cls] * self.window_rng.uniform(0.3, 1.2)
        frames = face[None, :] + self.window_rng.normal(0.0, 0.03, (FRAMES, cs.FACE_AU_COUNT))
        frames = np.where(face[None, :] > 0, frames, 0.0)
        tongue = self.window_rng.uniform(0.0, 1.0, (FRAMES, FEATURES - cs.FACE_AU_COUNT))
        window = np.clip(np.hstack([frames, tongue]), 0.0, 1.0).astype(np.float32)
        return [[float(v) for v in frame] for frame in window]

    # ------------------------------------------------------------------ calibration

    def _active_centroids(self) -> np.ndarray | None:
        if self.mode == rcr.MODE_RAW or not self.valid:
            return None

        every = list(range(N))
        personal = None
        if self.mode in (rcr.MODE_PERSONAL, rcr.MODE_HYBRID, rcr.MODE_MOVEMENT):
            rows = [tick["scores"] for trial in self.enrol
                    for tick in trial["ticks"] if tick["tick"] in rcr.SCORED_TICKS]
            labels = np.array([trial["prompted"] for trial in self.enrol
                               for tick in trial["ticks"] if tick["tick"] in rcr.SCORED_TICKS])
            personal = cs.centroids(
                cs.logscore(np.array(rows, np.float32), cs.DEFAULT_EPSILON), labels, every)

        if self.mode == rcr.MODE_MOVEMENT:
            enrol_windows = np.array([tick["window"] for trial in self.enrol
                                      for tick in trial["ticks"] if tick["tick"] in rcr.SCORED_TICKS])
            self.movement_M = cs.centroids(cs.movement_feature(enrol_windows), labels, every)
            return personal

        if self.mode == rcr.MODE_PERSONAL:
            return personal
        if self.mode == rcr.MODE_POPULATION:
            return population_matrix().astype(np.float32)
        return self.alpha * personal + (1.0 - self.alpha) * population_matrix().astype(np.float32)

    # ----------------------------------------------------------------------- writing

    def write(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        active = self._active_centroids()

        rows = []
        for trial in self.enrol + self.test:
            for tick in trial["ticks"]:
                rows.append(self._row(trial, tick, active))

        columns = (BASE_COLUMNS + [f"score_{c}" for c in CLASSES]
                   + (AU_COLUMNS if self.mode == rcr.MODE_MOVEMENT else [])
                   + CALIBRATION_COLUMNS)
        with (self.directory / "predictions.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

        self._write_metadata()
        return self.directory

    def _row(self, trial: dict, tick: dict, active: np.ndarray | None) -> dict:
        raw = CLASSES[int(np.argmax(tick["scores"]))]
        row = {
            "timestamp_utc": "2026-09-21T12:00:00.000Z",
            "session_id": self.directory.name,
            "trial_id": trial["trial_id"],
            "block_role": trial["role"],
            "repeat_index": trial["repeat"],
            "prompted_emotion": CLASSES[trial["prompted"]],
            "prediction_index": tick["tick"],
            "predicted_emotion": raw,
            "correct": 1 if raw == CLASSES[trial["prompted"]] else 0,
            "status": "ok",
            "generation": 100 + tick["tick"],
            "calibration_mode": self.mode,
            "personal_calibration_enabled":
                "true" if self.mode in (rcr.MODE_PERSONAL, rcr.MODE_HYBRID) else "false",
            "population_calibration_enabled":
                "true" if self.mode in (rcr.MODE_POPULATION, rcr.MODE_HYBRID) else "false",
            "movement_calibration_enabled": "true" if self.mode == rcr.MODE_MOVEMENT else "false",
            "hybrid_alpha": "0.40" if self.alpha is not None else "",
            "raw_predicted_emotion": raw,
            "calibrated_predicted_emotion": "",
            "calibrated_winner_similarity": "",
            "calibrated_runner_up_similarity": "",
            "calibrated_margin": "",
            "calibration_valid": "",
            "calibration_error": "",
        }
        for index, name in enumerate(CLASSES):
            row[f"score_{name}"] = repr(tick["scores"][index])

        if tick["window"] is not None:
            for f in range(FRAMES):
                for a in range(FEATURES):
                    row[f"frame_{f}_au_{a}"] = repr(tick["window"][f][a])

        if trial["role"] == sa.ROLE_EVALUATION:
            row["calibration_valid"] = "true" if self.valid else "false"
            row["calibration_error"] = self.error

            if active is not None and tick["tick"] in rcr.SCORED_TICKS:
                feature = cs.logscore(
                    np.array([tick["scores"]], np.float32), cs.DEFAULT_EPSILON)
                if self.mode == rcr.MODE_MOVEMENT:
                    predicted, top, margin = cs.classify_movement(
                        active, feature, self.movement_M,
                        cs.movement_feature(np.array([tick["window"]])), list(range(N)))
                else:
                    predicted, top, margin = cs.classify(active, list(range(N)), feature, "cosine")
                row["calibrated_predicted_emotion"] = CLASSES[int(predicted[0])]
                row["calibrated_winner_similarity"] = repr(float(top[0]))
                row["calibrated_runner_up_similarity"] = repr(float(top[0] - margin[0]))
                row["calibrated_margin"] = repr(float(margin[0]))

        return row

    def _write_metadata(self) -> None:
        artifact = json.loads(artifact_path().read_text(encoding="utf-8"))
        calibration = {
            "mode": self.mode,
            "hybrid_alpha": "0.40" if self.alpha is not None else "",
            "calibration_valid": "true" if self.valid else "false",
            "calibration_error": self.error,
            "runtime_calibration_version": "runtime-calibration-v1",
        }

        if self.mode in (rcr.MODE_POPULATION, rcr.MODE_HYBRID):
            calibration["population_artifact_semantic_sha256"] = artifact["semantic_sha256"]
            calibration["population_artifact_conversion_sha256"] = artifact["conversion_sha256"]

        (self.directory / "session_metadata.json").write_text(
            json.dumps({
                "session_id": self.directory.name,
                "protocol_version": "session-local-v1",
                "fast_rehearsal": "false",
                "class_order": CLASSES,
                "calibration": calibration,
            }, indent=2),
            encoding="utf-8")


def edit_rows(directory: Path, mutate) -> None:
    """Rewrite predictions.csv through `mutate(row) -> row`."""
    path = directory / "predictions.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames)
        rows = [mutate(row) for row in reader]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class ReportTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="runtime-calibration-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def build(self, mode: str, **kwargs) -> Path:
        name = f"session_{mode}_{len(list(self.tmp.iterdir()))}"
        return SessionBuilder(self.tmp / name, mode, **kwargs).write()

    def analyse(self, directory: Path, verify: bool = False) -> dict:
        root = sa.find_repo_root() or Path.cwd()
        return rcr.analyse(rcr.read_session(directory), root, verify)


class RawSessions(ReportTestCase):
    def test_a_raw_session_reports_raw_figures_and_no_calibrated_ones(self):
        result = self.analyse(self.build(rcr.MODE_RAW))

        self.assertEqual(rcr.MODE_RAW, result["calibration_mode"])
        self.assertIsNone(result["calibrated"],
                          "a raw session has no calibrated prediction to report")
        self.assertIsNone(result["comparison"])
        self.assertIsNotNone(result["raw"]["trial"]["accuracy"])
        self.assertIn("both calibration checkboxes off", result["note"])

    def test_a_raw_session_carries_no_alpha(self):
        self.assertIsNone(self.analyse(self.build(rcr.MODE_RAW))["hybrid_alpha"])


class EvaluationScope(ReportTestCase):
    def test_only_the_fourteen_evaluation_trials_are_scored(self):
        result = self.analyse(self.build(rcr.MODE_PERSONAL))

        self.assertEqual(14, result["raw"]["trial"]["n"],
                         "the seven calibration trials must not be scored")
        self.assertEqual(14, result["calibrated"]["trial"]["n"])

    def test_tick_zero_is_excluded_from_every_figure(self):
        result = self.analyse(self.build(rcr.MODE_PERSONAL))

        # 14 evaluation trials x 9 scored ticks. Tick 0 is a transition sample.
        self.assertEqual(14 * 9, result["raw"]["tick"]["n"])
        self.assertEqual(14 * 9, result["calibrated"]["tick"]["n"])


class CalibratedSessions(ReportTestCase):
    def test_a_personal_session_reports_both_methods(self):
        result = self.analyse(self.build(rcr.MODE_PERSONAL))

        self.assertEqual(rcr.MODE_PERSONAL, result["calibration_mode"])
        self.assertIsNotNone(result["calibrated"]["trial"]["macro_f1"])
        self.assertIsNotNone(result["calibrated_minus_raw"]["trial_macro_f1"])

    def test_a_hybrid_session_records_the_frozen_alpha(self):
        result = self.analyse(self.build(rcr.MODE_HYBRID))

        self.assertEqual(0.40, result["hybrid_alpha"])

    def test_a_population_session_reports_without_a_personal_term(self):
        result = self.analyse(self.build(rcr.MODE_POPULATION))

        self.assertEqual(rcr.MODE_POPULATION, result["calibration_mode"])
        self.assertIsNone(result["hybrid_alpha"])
        self.assertIsNotNone(result["calibrated"])

    def test_an_invalid_calibration_reports_the_reason_and_no_metrics(self):
        result = self.analyse(self.build(
            rcr.MODE_PERSONAL, valid=False, error="Fear 6/9 tracked calibration ticks"))

        self.assertFalse(result["calibration_valid"])
        self.assertIsNone(result["calibrated"],
                          "an unsealed profile must not produce a calibrated figure")
        self.assertIn("Fear 6/9", result["note"])
        self.assertIsNotNone(result["raw"]["trial"]["accuracy"],
                             "the raw recording is still worth reporting")

    def test_calibration_recovers_a_consistent_mis_mapping(self):
        # The model answers Surprise whenever the subject performs Disgust, keeping only a
        # trace of the true class -- see `scores_for`. Raw argmax loses both Disgust trials;
        # the personal centroids pick the trace up and take them back.
        session = self.build(rcr.MODE_PERSONAL, confusion={1: 6})
        result = self.analyse(session)

        self.assertGreater(result["calibrated_minus_raw"]["trial_accuracy"], 0.0)
        self.assertGreater(len(result["comparison"]["fixed_by_calibration"]), 0)
        self.assertEqual(
            result["comparison"]["net"],
            len(result["comparison"]["fixed_by_calibration"])
            - len(result["comparison"]["broken_by_calibration"]))


class Verification(ReportTestCase):
    def test_an_untouched_personal_session_verifies(self):
        result = self.analyse(self.build(rcr.MODE_PERSONAL), verify=True)

        self.assertTrue(result["verification"]["parity"])
        self.assertEqual(14 * 9, result["verification"]["n_ticks_compared"])
        self.assertEqual(14, result["verification"]["n_trials_compared"])

    def test_an_untouched_hybrid_session_verifies(self):
        result = self.analyse(self.build(rcr.MODE_HYBRID), verify=True)
        self.assertTrue(result["verification"]["parity"])

    def test_an_untouched_population_session_verifies(self):
        result = self.analyse(self.build(rcr.MODE_POPULATION), verify=True)
        self.assertTrue(result["verification"]["parity"])

    def test_one_altered_tick_prediction_fails_parity(self):
        session = self.build(rcr.MODE_PERSONAL)

        def bend(row):
            if (row["block_role"] == sa.ROLE_EVALUATION
                    and row["prediction_index"] == "4"
                    and row["trial_id"] == "9"):
                row["calibrated_predicted_emotion"] = "Fear" \
                    if row["calibrated_predicted_emotion"] != "Fear" else "Anger"
            return row

        edit_rows(session, bend)
        result = self.analyse(session, verify=True)

        self.assertFalse(result["verification"]["parity"])
        self.assertEqual(1, len(result["verification"]["tick_mismatches"]))
        self.assertEqual(9, result["verification"]["tick_mismatches"][0]["trial"])

    def test_one_altered_similarity_fails_parity(self):
        session = self.build(rcr.MODE_PERSONAL)

        def bend(row):
            if (row["block_role"] == sa.ROLE_EVALUATION
                    and row["prediction_index"] == "2"
                    and row["trial_id"] == "11"):
                row["calibrated_margin"] = "0.5"
            return row

        edit_rows(session, bend)
        result = self.analyse(session, verify=True)

        self.assertFalse(result["verification"]["parity"])
        self.assertEqual(1, len(result["verification"]["magnitude_mismatches"]))
        self.assertEqual("margin", result["verification"]["magnitude_mismatches"][0]["field"])

    def test_a_bent_trial_majority_fails_parity(self):
        # Five of the nine ticks of one trial flipped: enough to move the majority, so this
        # shows up as a trial mismatch as well as tick mismatches.
        session = self.build(rcr.MODE_PERSONAL)

        def bend(row):
            if (row["block_role"] == sa.ROLE_EVALUATION
                    and row["trial_id"] == "13"
                    and row["prediction_index"] in {"1", "2", "3", "4", "5"}):
                row["calibrated_predicted_emotion"] = "Sadness"
            return row

        edit_rows(session, bend)
        result = self.analyse(session, verify=True)

        self.assertFalse(result["verification"]["parity"])
        self.assertTrue(any(m["trial"] == 13
                            for m in result["verification"]["trial_mismatches"]))

    def test_verification_is_not_run_unless_asked(self):
        self.assertNotIn("verification", self.analyse(self.build(rcr.MODE_PERSONAL)))

    def test_a_session_recorded_against_another_artifact_is_refused(self):
        session = self.build(rcr.MODE_HYBRID)
        metadata_path = session / "session_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["calibration"]["population_artifact_semantic_sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        with self.assertRaises(SystemExit) as caught:
            self.analyse(session, verify=True)

        self.assertIn("different population artifact", str(caught.exception))


class MovementSessions(ReportTestCase):
    def test_a_movement_session_reports_both_methods(self):
        result = self.analyse(self.build(rcr.MODE_MOVEMENT))

        self.assertEqual(rcr.MODE_MOVEMENT, result["calibration_mode"])
        self.assertIsNone(result["hybrid_alpha"])
        self.assertEqual(14, result["calibrated"]["trial"]["n"])
        self.assertEqual(14 * 9, result["calibrated"]["tick"]["n"])

    def test_an_untouched_movement_session_verifies(self):
        result = self.analyse(self.build(rcr.MODE_MOVEMENT), verify=True)

        self.assertTrue(result["verification"]["parity"])
        self.assertEqual(14 * 9, result["verification"]["n_ticks_compared"])

    def test_movement_recovers_what_the_scores_lost(self):
        # Disgust is answered as Surprise by the model, but the face keeps Disgust's movement.
        result = self.analyse(self.build(rcr.MODE_MOVEMENT, confusion={1: 6}))

        self.assertGreater(result["calibrated_minus_raw"]["trial_accuracy"], 0.0)
        disgust = result["calibrated"]["trial"]["per_class"]["Disgust"]
        self.assertEqual(1.0, disgust["recall"])

    def test_an_altered_action_unit_fails_movement_parity(self):
        # The movement term is recomputed from the logged windows, so changing one logged
        # value must show up -- this is what proves the check reads them at all.
        session = self.build(rcr.MODE_MOVEMENT)

        def bend(row):
            if (row["block_role"] == sa.ROLE_EVALUATION
                    and row["prediction_index"] == "3" and row["trial_id"] == "10"):
                for a in range(cs.FACE_AU_COUNT):
                    row[f"frame_0_au_{a}"] = "1.0"
            return row

        edit_rows(session, bend)
        result = self.analyse(session, verify=True)

        self.assertFalse(result["verification"]["parity"])

    def test_a_movement_file_without_windows_cannot_be_verified(self):
        session = self.build(rcr.MODE_MOVEMENT)
        path = session / "predictions.csv"

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            keep = [c for c in reader.fieldnames if c not in AU_COLUMNS]
            rows = [{k: row[k] for k in keep} for row in reader]

        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keep)
            writer.writeheader()
            writer.writerows(rows)

        with self.assertRaises(SystemExit) as caught:
            self.analyse(session, verify=True)

        self.assertIn("frame_<f>_au_<a>", str(caught.exception))


class Refusals(ReportTestCase):
    def test_a_file_without_calibration_columns_says_which_tool_to_use(self):
        session = self.build(rcr.MODE_RAW)
        path = session / "predictions.csv"

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            keep = [c for c in reader.fieldnames if c not in CALIBRATION_COLUMNS]
            rows = [{k: row[k] for k in keep} for row in reader]

        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keep)
            writer.writeheader()
            writer.writerows(rows)

        with self.assertRaises(SystemExit) as caught:
            rcr.read_session(session)

        self.assertIn("calibration_study.py", str(caught.exception))

    def test_a_file_mixing_two_modes_is_refused(self):
        session = self.build(rcr.MODE_PERSONAL)

        def bend(row):
            if row["trial_id"] == "12":
                row["calibration_mode"] = rcr.MODE_HYBRID
            return row

        edit_rows(session, bend)

        with self.assertRaises(SystemExit) as caught:
            self.analyse(session)

        self.assertIn("fixed at session start", str(caught.exception))

    def test_a_missing_predictions_file_is_refused(self):
        with self.assertRaises(SystemExit):
            rcr.read_session(self.tmp / "nothing-here")


class ModesAreNotPooled(ReportTestCase):
    def test_sessions_of_different_modes_produce_separate_results(self):
        results = [self.analyse(self.build(mode))
                   for mode in (rcr.MODE_PERSONAL, rcr.MODE_HYBRID, rcr.MODE_POPULATION)]

        modes = [r["calibration_mode"] for r in results]
        self.assertEqual([rcr.MODE_PERSONAL, rcr.MODE_HYBRID, rcr.MODE_POPULATION], modes)
        self.assertEqual(3, len({r["session"] for r in results}))

    def test_the_command_line_reports_each_session_separately(self):
        directories = [str(self.build(mode))
                       for mode in (rcr.MODE_PERSONAL, rcr.MODE_HYBRID)]
        argv = []
        for directory in directories:
            argv += ["--session", directory]

        self.assertEqual(0, rcr.main(argv))


if __name__ == "__main__":
    unittest.main(verbosity=2)
