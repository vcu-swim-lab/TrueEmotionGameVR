#!/usr/bin/env python3
"""Unit tests for emojihero_calibration_study.py.

    python Analysis/test_emojihero_calibration_study.py

Stdlib `unittest` on synthetic fixtures. **No corpus and no ONNX file are needed**: every
score vector here is hand-built, so the arithmetic is checked against values worked out by
hand rather than against the model's own output.

That separation is deliberate. The claims that can only be made about the real corpus --
964/385/378 sequences, 20/8/8 participants, 30 frames, participant disjointness, the
Label/sequence-id agreement, the one participant whose Fear class holds a single sequence --
are checked by the study script's own integrity gate on every run against the real data,
where they belong. Asserting them here would make the unit tests unrunnable without a 35 MB
corpus and would check the fixtures rather than the data.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import au_names
import calibration_study as cs
import session_analysis as sa
import emojihero_calibration_study as eh

EMOTIONS = eh.EMOTIONS
ANGER, DISGUST, FEAR, HAPPINESS, NEUTRAL, SADNESS, SURPRISE = range(7)


def sequence(participant: int, emotion: int, *, timestamp: int = 1_700_000_000_000,
             level: int = 1, emoji: int = 0, set_id: int = 1,
             au: "np.ndarray | None" = None) -> eh.Sequence:
    """One synthetic sequence with a well-formed id and a 30x63 ramp."""
    sequence_id = f"{timestamp}-{set_id}-{participant}-{level}-{emoji}-{emotion}"
    if au is None:
        au = np.tile(
            np.linspace(0.0, 1.0, eh.FRAMES_PER_SEQUENCE)[:, None],
            (1, au_names.V1_CHANNEL_COUNT))
    return eh.Sequence(
        sequence_id=sequence_id, timestamp=timestamp, set_id=set_id,
        participant=participant, level=level, emoji=emoji, emotion=emotion,
        label=emotion, au=np.asarray(au, dtype=np.float64))


def onehot(index: int, peak: float = 0.7) -> list[float]:
    rest = (1.0 - peak) / (len(EMOTIONS) - 1)
    return [peak if i == index else rest for i in range(len(EMOTIONS))]


def fixture(sequences: list[eh.Sequence], scores_for):
    """`(index_of, logf, scores)`; `scores_for(sequence, window_index)` gives the vector."""
    index_of = {s.sequence_id: i for i, s in enumerate(sequences)}
    scores = np.array([
        [scores_for(s, w) for w in range(eh.WINDOWS_PER_SEQUENCE)]
        for s in sequences
    ], dtype=np.float64)
    return index_of, cs.logscore(scores), scores


def participant_set(participant: int, per_emotion: int = 3) -> list[eh.Sequence]:
    return [
        sequence(participant, e, timestamp=1_700_000_000_000 + e * 100 + i, emoji=i)
        for e in range(7) for i in range(per_emotion)
    ]


def full_profile(sequences: list[eh.Sequence]) -> dict[int, eh.Sequence]:
    return {e: next(s for s in sequences if s.emotion == e) for e in range(7)}


# ------------------------------------------------------------- sequence ids


class SequenceIdParsing(unittest.TestCase):
    def test_six_fields_in_order(self):
        self.assertEqual(eh.parse_sequence_id("1700485979873-1-2-3-4-5"),
                         (1700485979873, 1, 2, 3, 4, 5))

    def test_the_last_field_is_the_emotion(self):
        *_, emotion = eh.parse_sequence_id("1700485979873-1-2-1-1-6")
        self.assertEqual(EMOTIONS[emotion], "Surprise")

    def test_emotion_mapping_is_the_frozen_order(self):
        self.assertEqual(EMOTIONS, ("Anger", "Disgust", "Fear", "Happiness",
                                    "Neutral", "Sadness", "Surprise"))

    def test_wrong_field_count_raises_rather_than_guessing(self):
        with self.assertRaises(ValueError):
            eh.parse_sequence_id("1700485979873-1-2-1-1")
        with self.assertRaises(ValueError):
            eh.parse_sequence_id("1700485979873-1-2-1-1-5-9")

    def test_non_integer_field_raises(self):
        with self.assertRaises(ValueError):
            eh.parse_sequence_id("1700485979873-1-p2-1-1-5")

    def test_emotion_outside_the_class_set_raises(self):
        with self.assertRaises(ValueError):
            eh.parse_sequence_id("1700485979873-1-2-1-1-7")


# ----------------------------------------------------------- preprocessing


class Preprocessing(unittest.TestCase):
    def test_pooling_is_the_mean_of_three_consecutive_frames(self):
        au = np.arange(eh.FRAMES_PER_SEQUENCE * au_names.V1_CHANNEL_COUNT, dtype=float)
        au = au.reshape(eh.FRAMES_PER_SEQUENCE, au_names.V1_CHANNEL_COUNT)
        windows = eh.to_model_windows(au)
        self.assertAlmostEqual(float(windows[0, 0, 0]), float(au[0:3, 0].mean()), places=4)

    def test_thirty_frames_become_ten(self):
        self.assertEqual(eh.POOLED_FRAMES, 10)
        self.assertEqual(eh.FRAMES_PER_SEQUENCE // eh.POOL_FACTOR, 10)

    def test_exactly_six_windows_of_five_by_seventy(self):
        self.assertEqual(eh.to_model_windows(sequence(1, ANGER).au).shape, (6, 5, 70))
        self.assertEqual(eh.WINDOWS_PER_SEQUENCE, 6)

    def test_windows_are_stride_one_and_stay_inside_the_sequence(self):
        au = np.tile(np.arange(eh.FRAMES_PER_SEQUENCE, dtype=float)[:, None],
                     (1, au_names.V1_CHANNEL_COUNT))
        windows = eh.to_model_windows(au)
        pooled = eh.tp.downsample(np.pad(au, ((0, 0), (0, 7))), eh.POOL_FACTOR)
        for w in range(eh.WINDOWS_PER_SEQUENCE):
            np.testing.assert_allclose(windows[w], pooled[w:w + 5], rtol=1e-5)
        np.testing.assert_allclose(windows[-1][-1], pooled[eh.POOLED_FRAMES - 1], rtol=1e-5)

    def test_sixty_three_channels_widen_to_seventy(self):
        windows = eh.to_model_windows(
            np.full((eh.FRAMES_PER_SEQUENCE, au_names.V1_CHANNEL_COUNT), 0.5))
        self.assertEqual(windows.shape[2], au_names.CHANNEL_COUNT)
        np.testing.assert_allclose(windows[:, :, :63], 0.5, rtol=1e-6)

    def test_the_zero_filled_channels_are_exactly_the_tongue_channels(self):
        windows = eh.to_model_windows(
            np.full((eh.FRAMES_PER_SEQUENCE, au_names.V1_CHANNEL_COUNT), 0.5))
        for index in au_names.TONGUE_INDICES:
            self.assertTrue((windows[:, :, index] == 0.0).all(),
                            f"{au_names.NAMES[index]} should be zero-filled")
        measured = [i for i in range(au_names.CHANNEL_COUNT)
                    if i not in au_names.TONGUE_INDICES]
        self.assertTrue((windows[:, :, measured] != 0.0).all(), "a measured channel was zeroed")
        self.assertEqual(au_names.TONGUE_INDICES, tuple(range(63, 70)))

    def test_no_normalization_or_scaling_is_applied(self):
        windows = eh.to_model_windows(
            np.full((eh.FRAMES_PER_SEQUENCE, au_names.V1_CHANNEL_COUNT), 0.37))
        np.testing.assert_allclose(windows[:, :, :63], 0.37, rtol=1e-6)

    def test_wrong_frame_count_is_refused(self):
        with self.assertRaises(ValueError):
            eh.to_model_windows(np.zeros((29, au_names.V1_CHANNEL_COUNT)))

    def test_wrong_channel_count_is_refused(self):
        with self.assertRaises(ValueError):
            eh.to_model_windows(np.zeros((eh.FRAMES_PER_SEQUENCE, 70)))


# ------------------------------------------------------- profile selection


class ProfileSelection(unittest.TestCase):
    def cell(self, n: int = 5) -> dict:
        return {
            (1, emotion): [sequence(1, emotion, timestamp=1_700_000_000_000 + i, emoji=i)
                           for i in range(n)]
            for emotion in range(7)
        }

    def test_selection_is_deterministic_across_calls(self):
        by_cell = self.cell()
        a = eh.calibration_profile(by_cell, "validation", 1, 0)
        b = eh.calibration_profile(by_cell, "validation", 1, 0)
        self.assertEqual({k: v.sequence_id for k, v in a.items()},
                         {k: v.sequence_id for k, v in b.items()})

    def test_different_profile_indices_are_independent_draws(self):
        by_cell = self.cell()
        seen = {tuple(sorted(s.sequence_id for s in
                             eh.calibration_profile(by_cell, "validation", 1, i).values()))
                for i in range(40)}
        self.assertGreater(len(seen), 5, "profile index barely moved the selection")

    def test_a_profile_holds_one_sequence_per_emotion(self):
        profile = eh.calibration_profile(self.cell(), "validation", 1, 7)
        self.assertEqual(sorted(profile), list(range(7)))
        for emotion, chosen in profile.items():
            self.assertEqual(chosen.emotion, emotion)

    def test_the_split_name_changes_the_selection(self):
        by_cell = self.cell()
        a = eh.calibration_profile(by_cell, "validation", 1, 0)
        b = eh.calibration_profile(by_cell, "test", 1, 0)
        self.assertNotEqual([s.sequence_id for s in a.values()],
                            [s.sequence_id for s in b.values()])

    def test_the_index_is_always_inside_the_candidate_list(self):
        for modulus in (1, 2, 3, 7, 11):
            for profile_index in range(50):
                self.assertIn(eh._stable_index("test", 18, FEAR, profile_index, modulus),
                              range(modulus))

    def test_selection_does_not_use_the_process_salted_builtin_hash(self):
        """A `hash()`-based selection would differ between runs; this must not."""
        key = f"{eh.MASTER_SEED}|validation|2|{ANGER}|3".encode()
        self.assertEqual(
            eh._stable_index("validation", 2, ANGER, 3, 9),
            int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 9)

    def test_a_cell_with_one_sequence_still_yields_that_sequence(self):
        by_cell = {(18, FEAR): [sequence(18, FEAR)]}
        profile = eh.calibration_profile(by_cell, "test", 18, 0)
        self.assertEqual(profile[FEAR].sequence_id, by_cell[(18, FEAR)][0].sequence_id)


# ---------------------------------------------------- transform and centroids


class LogTransform(unittest.TestCase):
    def test_log_transform_matches_the_shared_implementation(self):
        scores = np.array([[0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]])
        np.testing.assert_allclose(cs.logscore(scores, cs.DEFAULT_EPSILON),
                                   np.log(np.clip(scores, cs.DEFAULT_EPSILON, None)))

    def test_epsilon_clamps_a_zero_instead_of_producing_negative_infinity(self):
        transformed = cs.logscore(np.array([[0.0, 1.0, 0, 0, 0, 0, 0]]), cs.DEFAULT_EPSILON)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertAlmostEqual(float(transformed[0, 0]), float(np.log(cs.DEFAULT_EPSILON)))

    def test_the_frozen_epsilon_is_inert_over_realistic_scores(self):
        """The smallest score observed anywhere in this project is ~7e-06, so a floor at
        1e-12 sits below all of them and cannot act as a smoother."""
        scores = np.array([[7e-06, 0.3, 0.2, 0.2, 0.1, 0.1, 0.0999]])
        np.testing.assert_allclose(cs.logscore(scores, 1e-12), cs.logscore(scores, 1e-30))
        self.assertEqual(eh.FROZEN_SETTINGS["epsilon"], 1e-12)


class PersonalCentroids(unittest.TestCase):
    def test_a_personal_centroid_is_the_mean_of_its_sequence_windows(self):
        seqs = [sequence(1, e, emoji=e) for e in range(7)]
        index_of, logf, _ = fixture(seqs, lambda s, w: onehot(s.emotion, 0.4 + 0.05 * w))
        M, enrolled = eh.personal_centroids(full_profile(seqs), index_of, logf)
        self.assertEqual(enrolled, list(range(7)))
        np.testing.assert_allclose(M[ANGER], logf[index_of[seqs[ANGER].sequence_id]].mean(0))

    def test_only_the_selected_windows_enter_the_centroid(self):
        seqs = [sequence(1, e, emoji=e) for e in range(7)]
        index_of, logf, _ = fixture(seqs, lambda s, w: onehot(s.emotion, 0.4 + 0.05 * w))
        M, _ = eh.personal_centroids(full_profile(seqs), index_of, logf,
                                     eh.NON_OVERLAPPING_WINDOWS)
        expected = logf[index_of[seqs[ANGER].sequence_id]][[0, 5]].mean(0)
        np.testing.assert_allclose(M[ANGER], expected)

    def test_an_empty_profile_returns_nothing_enrolled(self):
        M, enrolled = eh.personal_centroids({}, {}, np.zeros((0, 6, 7)))
        self.assertEqual(len(M), 0)
        self.assertEqual(enrolled, [])


# ------------------------------------------------------ population centroids


class PopulationCentroids(unittest.TestCase):
    def test_the_estimator_is_hierarchical_and_participant_balanced(self):
        """One participant with many sequences must not outvote one with few.

        Participant 1 records Anger nine times with score vector A; participant 2 records
        it once with vector B. A flat mean over sequences would land at 0.9A + 0.1B. The
        participant-balanced estimator lands exactly halfway.
        """
        seqs = ([sequence(1, ANGER, emoji=i) for i in range(9)]
                + [sequence(2, ANGER, emoji=0)])
        # every other emotion, evenly, so the build does not fail on missing support
        for emotion in range(1, 7):
            seqs += [sequence(p, emotion, emoji=0) for p in range(1, 8)]
        seqs += [sequence(p, ANGER, emoji=1) for p in range(3, 8)]

        def scores_for(s, w):
            if s.emotion != ANGER:
                return onehot(s.emotion)
            return onehot(ANGER, 0.9) if s.participant == 1 else onehot(ANGER, 0.3)

        index_of, logf, _ = fixture(seqs, scores_for)
        built = eh.build_population_centroids(seqs, logf, index_of)

        per_participant = []
        for participant in sorted({s.participant for s in seqs if s.emotion == ANGER}):
            rows = [logf[index_of[s.sequence_id]].mean(0) for s in seqs
                    if s.emotion == ANGER and s.participant == participant]
            per_participant.append(np.mean(rows, axis=0))
        np.testing.assert_allclose(built["matrix"][ANGER],
                                   np.mean(per_participant, axis=0))

        flat = np.mean([logf[index_of[s.sequence_id]].mean(0) for s in seqs
                        if s.emotion == ANGER], axis=0)
        self.assertFalse(np.allclose(built["matrix"][ANGER], flat),
                         "the estimator collapsed to a flat mean over sequences")

    def test_a_prolific_participant_does_not_dominate(self):
        seqs = ([sequence(1, ANGER, emoji=i) for i in range(20)]
                + [sequence(p, ANGER, emoji=0) for p in range(2, 8)])
        for emotion in range(1, 7):
            seqs += [sequence(p, emotion, emoji=0) for p in range(1, 8)]

        def scores_for(s, w):
            if s.emotion != ANGER:
                return onehot(s.emotion)
            return onehot(ANGER, 0.9) if s.participant == 1 else onehot(SURPRISE, 0.9)

        index_of, logf, _ = fixture(seqs, scores_for)
        matrix = eh.build_population_centroids(seqs, logf, index_of)["matrix"]
        # Six of seven participants answer Surprise for Anger, so the population centroid
        # must sit nearer the Surprise pattern than the Anger one despite 20 vs 6 sequences.
        anger_like = cs.logscore(np.array([onehot(ANGER, 0.9)]))[0]
        surprise_like = cs.logscore(np.array([onehot(SURPRISE, 0.9)]))[0]

        def cosine(a, b):
            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

        self.assertGreater(cosine(matrix[ANGER], surprise_like),
                           cosine(matrix[ANGER], anger_like))

    def test_support_counts_are_recorded_per_emotion(self):
        seqs = [sequence(p, e, emoji=i) for p in range(1, 8) for e in range(7)
                for i in range(2)]
        index_of, logf, _ = fixture(seqs, lambda s, w: onehot(s.emotion))
        built = eh.build_population_centroids(seqs, logf, index_of)
        self.assertEqual(built["support"]["Anger"]["participants"], 7)
        self.assertEqual(built["support"]["Anger"]["sequences"], 14)

    def test_a_participant_missing_an_emotion_uses_the_participants_that_have_it(self):
        seqs = [sequence(p, e, emoji=0) for p in range(1, 8) for e in range(7)]
        seqs = [s for s in seqs if not (s.participant == 3 and s.emotion == FEAR)]
        index_of, logf, _ = fixture(seqs, lambda s, w: onehot(s.emotion))
        built = eh.build_population_centroids(seqs, logf, index_of)
        self.assertEqual(built["support"]["Fear"]["participants"], 6)
        self.assertEqual(built["support"]["Anger"]["participants"], 7)
        self.assertNotIn("3", built["support"]["Fear"]["sequences_per_participant"])

    def test_an_emotion_absent_from_training_is_a_hard_error(self):
        seqs = [sequence(p, e, emoji=0) for p in range(1, 8) for e in range(6)]
        index_of, logf, _ = fixture(seqs, lambda s, w: onehot(s.emotion))
        with self.assertRaises(SystemExit) as caught:
            eh.build_population_centroids(seqs, logf, index_of)
        self.assertIn("Surprise", str(caught.exception))

    def test_too_few_participants_for_an_emotion_is_a_hard_error(self):
        seqs = [sequence(p, e, emoji=0) for p in range(1, 8) for e in range(7)]
        seqs = [s for s in seqs if not (s.emotion == FEAR and s.participant > 3)]
        index_of, logf, _ = fixture(seqs, lambda s, w: onehot(s.emotion))
        with self.assertRaises(SystemExit) as caught:
            eh.build_population_centroids(seqs, logf, index_of)
        self.assertIn("Fear", str(caught.exception))

    def test_population_centroids_come_only_from_the_training_split(self):
        self.assertEqual(eh.POPULATION_SOURCE_SPLIT, "training")
        self.assertEqual(
            eh.FROZEN_SETTINGS["population_centroid"]["source_split"], "training")


class PopulationArtifact(unittest.TestCase):
    def artifact(self, model_sha: str = "a" * 64) -> dict:
        matrix = np.arange(49, dtype=float).reshape(7, 7)
        support = {c: {"participants": 20, "sequences": 100,
                       "sequences_per_participant": {}} for c in EMOTIONS}
        return eh.population_artifact(
            matrix, support, model_sha=model_sha, model_path=Path("model.onnx"),
            provenance={"sha256": "b" * 64, "rows": 10, "split": "training"},
            sequences=[sequence(1, ANGER)])

    def roundtrip(self, artifact: dict, model_sha: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "population.json"
            path.write_text(json.dumps(eh._jsonable(artifact)), encoding="utf-8")
            return eh.load_population_centroids(path, model_sha=model_sha)

    def test_a_matching_artifact_round_trips(self):
        matrix, meta = self.roundtrip(self.artifact(), "a" * 64)
        self.assertEqual(matrix.shape, (7, 7))
        np.testing.assert_allclose(matrix, np.arange(49, dtype=float).reshape(7, 7))
        self.assertEqual(meta["source_split"], "training")

    def test_an_artifact_built_against_another_model_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            self.roundtrip(self.artifact(model_sha="a" * 64), "c" * 64)
        self.assertIn("different model", str(caught.exception))

    def test_a_wrong_schema_version_is_refused(self):
        artifact = self.artifact()
        artifact["schema_version"] = "something-else"
        with self.assertRaises(SystemExit):
            self.roundtrip(artifact, "a" * 64)

    def test_an_artifact_from_the_wrong_split_is_refused(self):
        artifact = self.artifact()
        artifact["source_split"] = "validation"
        with self.assertRaises(SystemExit) as caught:
            self.roundtrip(artifact, "a" * 64)
        self.assertIn("training split", str(caught.exception))

    def test_a_different_epsilon_is_refused(self):
        artifact = self.artifact()
        artifact["epsilon"] = 1e-6
        with self.assertRaises(SystemExit):
            self.roundtrip(artifact, "a" * 64)

    def test_a_reordered_class_list_is_refused(self):
        artifact = self.artifact()
        artifact["class_order"] = list(reversed(EMOTIONS))
        with self.assertRaises(SystemExit):
            self.roundtrip(artifact, "a" * 64)

    def test_a_missing_class_is_refused(self):
        artifact = self.artifact()
        del artifact["centroids"]["Fear"]
        with self.assertRaises(SystemExit) as caught:
            self.roundtrip(artifact, "a" * 64)
        self.assertIn("Fear", str(caught.exception))

    def test_the_result_payload_records_both_hashes_under_distinct_names(self):
        """The pin and the bytes are different things and must not share a name.

        The config enforces the semantic hash; the file hash changes on every rebuild. A
        result that recorded only `sha256` invited a reader to compare it against the
        config's pin and conclude the artifact had changed when it had not.
        """
        source = Path(eh.__file__).read_text(encoding="utf-8")
        payload = source[source.index('"population_artifact": {', source.index("def main")):]
        payload = payload[:payload.index("},")]
        self.assertIn('"semantic_sha256"', payload)
        self.assertIn('"file_sha256"', payload)
        self.assertNotIn('"sha256":', payload, "the ambiguous bare name is back")

    def test_the_artifact_records_the_model_hash_and_provenance(self):
        artifact = self.artifact()
        self.assertEqual(artifact["model"]["sha256"], "a" * 64)
        self.assertFalse(artifact["model"]["retrained"])
        self.assertIn("data_provenance", artifact)
        self.assertIn("support_per_emotion", artifact)


# ---------------------------------------------------------------- hybrid


class HybridFormula(unittest.TestCase):
    def setUp(self):
        self.personal = np.arange(49, dtype=float).reshape(7, 7)
        self.population = np.full((7, 7), 100.0)

    def test_the_formula_is_alpha_personal_plus_one_minus_alpha_population(self):
        blended = eh.hybrid_centroids(self.personal, self.population, 0.25)
        np.testing.assert_allclose(blended, 0.25 * self.personal + 0.75 * self.population)

    def test_alpha_one_is_exactly_personal(self):
        np.testing.assert_allclose(
            eh.hybrid_centroids(self.personal, self.population, 1.0), self.personal)

    def test_alpha_zero_is_exactly_population(self):
        np.testing.assert_allclose(
            eh.hybrid_centroids(self.personal, self.population, 0.0), self.population)

    def test_an_alpha_outside_the_unit_interval_raises(self):
        for alpha in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                eh.hybrid_centroids(self.personal, self.population, alpha)

    def test_a_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            eh.hybrid_centroids(self.personal, np.zeros((3, 7)), 0.5)

    def test_the_grid_is_twenty_one_steps_of_five_hundredths(self):
        self.assertEqual(len(eh.ALPHA_GRID), 21)
        self.assertEqual(eh.ALPHA_GRID[0], 0.0)
        self.assertEqual(eh.ALPHA_GRID[-1], 1.0)
        self.assertAlmostEqual(eh.ALPHA_GRID[1] - eh.ALPHA_GRID[0], 0.05)


class HybridEndpointsMatchTheStandaloneMethods(unittest.TestCase):
    """alpha=0 must reproduce population-only and alpha=1 personal-only, exactly."""

    def setUp(self):
        self.seqs = participant_set(1)
        self.index_of, self.logf, self.scores = fixture(
            self.seqs, lambda s, w: onehot(s.emotion, 0.5 + 0.02 * w))
        self.population = np.random.default_rng(0).normal(size=(7, 7))
        self.profile = full_profile(self.seqs)

    def evaluate(self, alpha: float) -> dict:
        return eh.evaluate_participant(
            self.seqs, self.index_of, self.logf, self.scores, self.profile,
            eh.ALL_WINDOWS, self.population, alpha)

    def test_alpha_one_hybrid_equals_personal(self):
        result = self.evaluate(1.0)
        self.assertEqual(result["sequence_level"]["hybrid"],
                         result["sequence_level"]["personal"])

    def test_alpha_zero_hybrid_equals_population(self):
        result = self.evaluate(0.0)
        self.assertEqual(result["sequence_level"]["hybrid"],
                         result["sequence_level"]["population"])

    def test_the_hybrid_minus_personal_delta_is_zero_at_alpha_one(self):
        delta = self.evaluate(1.0)["delta"]["hybrid_minus_personal"]
        self.assertEqual(delta["accuracy"], 0.0)
        self.assertEqual(delta["macro_f1"], 0.0)


# ------------------------------------------------------- the four methods


class FourMethodsOnOneEvaluationSet(unittest.TestCase):
    def setUp(self):
        self.seqs = participant_set(1)
        self.index_of, self.logf, self.scores = fixture(
            self.seqs, lambda s, w: onehot(s.emotion))
        self.population = cs.logscore(
            np.array([onehot(e) for e in range(7)], dtype=float))
        self.profile = full_profile(self.seqs)
        self.result = eh.evaluate_participant(
            self.seqs, self.index_of, self.logf, self.scores, self.profile,
            eh.ALL_WINDOWS, self.population, 0.5)

    def test_all_four_methods_are_present(self):
        self.assertEqual(sorted(self.result["sequence_level"]), sorted(eh.METHODS))

    def test_every_method_scores_the_same_sequences(self):
        n = self.result["n_evaluation_sequences"]
        for key in ("personal_vs_raw", "population_vs_raw", "hybrid_vs_raw",
                    "hybrid_vs_personal"):
            self.assertEqual(sum(self.result["paired_sequences"][key].values()), n,
                             f"{key} does not account for every evaluation sequence")

    def test_the_four_deltas_are_reported(self):
        self.assertEqual(sorted(self.result["delta"]), sorted(eh.DELTA_KEYS))
        for key in eh.DELTA_KEYS:
            self.assertIsNotNone(self.result["delta"][key])

    def test_hybrid_minus_personal_is_the_population_contribution(self):
        level = self.result["sequence_level"]
        self.assertAlmostEqual(
            self.result["delta"]["hybrid_minus_personal"]["accuracy"],
            level["hybrid"]["accuracy"] - level["personal"]["accuracy"])

    def test_without_a_population_matrix_only_raw_and_personal_are_scored(self):
        result = eh.evaluate_participant(
            self.seqs, self.index_of, self.logf, self.scores, self.profile)
        self.assertEqual(sorted(result["sequence_level"]), ["personal", "raw"])
        self.assertIsNone(result["delta"]["hybrid_minus_personal"])


class SequenceDisjointness(unittest.TestCase):
    def test_no_calibration_sequence_is_ever_scored(self):
        seqs = participant_set(1)
        index_of, logf, scores = fixture(seqs, lambda s, w: onehot(s.emotion))
        by_cell = eh._sequences_by_cell(seqs)
        profile = eh.calibration_profile(by_cell, "validation", 1, 0)
        result = eh.evaluate_participant(seqs, index_of, logf, scores, profile)
        calibration = set(result["calibration_sequence_ids"].values())
        evaluation = set(result["evaluation_sequence_ids"])
        self.assertEqual(calibration & evaluation, set())
        self.assertEqual(len(calibration) + len(evaluation), len(seqs))

    def test_all_six_windows_of_a_calibration_sequence_stay_on_that_side(self):
        """The leak the live study measured decaying +0.150 to -0.010. A sequence is
        enrolled or scored, never split across the two."""
        seqs = participant_set(1)
        index_of, logf, scores = fixture(seqs, lambda s, w: onehot(s.emotion))
        result = eh.evaluate_participant(seqs, index_of, logf, scores, full_profile(seqs))
        self.assertEqual(result["n_evaluation_sequences"], len(seqs) - 7)
        self.assertEqual(result["n_evaluation_windows"],
                         (len(seqs) - 7) * eh.WINDOWS_PER_SEQUENCE)

    def test_the_split_helper_asserts_disjointness(self):
        seqs = participant_set(1)
        evaluation, calibration_ids = eh.split_calibration_evaluation(
            seqs, full_profile(seqs))
        self.assertEqual({s.sequence_id for s in evaluation} & calibration_ids, set())

    def test_a_single_sequence_class_is_enrolled_and_leaves_nothing_to_score(self):
        """The structural case the corpus contains: one emotion with a single sequence.
        It must not be split to manufacture both sides; the class simply has no
        evaluation support, and the seven-class figure is invalid."""
        seqs = [s for s in participant_set(18)
                if not (s.emotion == FEAR and s.emoji > 0)]
        index_of, logf, scores = fixture(seqs, lambda s, w: onehot(s.emotion))
        result = eh.evaluate_participant(seqs, index_of, logf, scores, full_profile(seqs))
        self.assertNotIn("Fear", result["classes_with_a_scored_sequence"])
        self.assertIn("Fear", result["enrolled_classes"])
        self.assertFalse(result["primary_macro_f1_valid"])


# --------------------------------------------------------------- tie rule


class TieRule(unittest.TestCase):
    """The majority vote is `calibration_study.trial_vote`, reused rather than rewritten."""

    def vote(self, predictions: list[str], truth: str = "Anger"):
        n = len(predictions)
        _, _, p = cs.trial_vote(np.array([truth] * n), np.array(predictions),
                                np.zeros(n, dtype=int), np.arange(n), list(EMOTIONS))
        return p[0]

    def test_the_class_winning_the_most_windows(self):
        self.assertEqual(
            self.vote(["Anger", "Anger", "Anger", "Fear", "Fear", "Disgust"]), "Anger")

    def test_a_tie_goes_to_the_chronologically_last_window(self):
        self.assertEqual(
            self.vote(["Anger", "Anger", "Anger", "Fear", "Fear", "Fear"]), "Fear")

    def test_a_tie_goes_to_the_last_window_when_that_class_is_in_the_tie(self):
        """Three ways, and the final window is one of the three."""
        self.assertEqual(
            self.vote(["Disgust", "Disgust", "Fear", "Fear", "Sadness", "Sadness"]),
            "Sadness")

    def test_a_tie_the_last_window_cannot_break_falls_to_class_order(self):
        """Disgust and Fear tie at two; the final window is Anger, which is not tied,
        so the rule falls through to the first tied class in the frozen order."""
        self.assertEqual(
            self.vote(["Disgust", "Disgust", "Fear", "Fear", "Anger"]), "Disgust")


class FastPathsMatchTheSharedImplementations(unittest.TestCase):
    """The sweep helpers must never become a second definition of the frozen rules."""

    def test_fast_majority_vote_matches_trial_vote(self):
        rng = np.random.default_rng(11)
        for _ in range(200):
            n_sequences, n_windows = 5, eh.WINDOWS_PER_SEQUENCE
            predictions = rng.integers(0, 7, size=(n_sequences, n_windows))
            fast = eh.fast_majority_vote(predictions)

            names = np.array(EMOTIONS)
            flat = predictions.reshape(-1)
            trials = np.repeat(np.arange(n_sequences), n_windows)
            positions = np.tile(np.arange(n_windows), n_sequences)
            _, _, shared = cs.trial_vote(
                names[flat], names[flat], trials, positions, list(EMOTIONS))
            np.testing.assert_array_equal(names[fast], shared)

    def test_fast_macro_f1_matches_session_analysis(self):
        rng = np.random.default_rng(12)
        names = np.array(EMOTIONS)
        for _ in range(200):
            truth = rng.integers(0, 7, size=20)
            predicted = rng.integers(0, 7, size=20)
            shared = sa.macro_f1(names[truth], names[predicted], list(EMOTIONS))["macro_f1"]
            self.assertAlmostEqual(eh.fast_macro_f1(truth, predicted), shared, places=12)


# ---------------------------------------------------------- alpha selection


class AlphaSelection(unittest.TestCase):
    def build(self, population_helps: bool):
        """A participant whose model output is consistently confused.

        When `population_helps`, the population centroids carry the corrective pattern and
        the personal enrolment is uninformative noise; otherwise the personal centroids do
        all the work and the population is noise.
        """
        seqs = participant_set(1, per_emotion=3)
        rng = np.random.default_rng(5)

        def scores_for(s, w):
            return onehot(s.emotion, 0.5)

        index_of, logf, scores = fixture(seqs, scores_for)
        if population_helps:
            population = cs.logscore(np.array([onehot(e, 0.5) for e in range(7)]))
        else:
            population = rng.normal(size=(7, 7)) * 5
        return seqs, index_of, logf, scores, population

    def test_selection_prefers_the_alpha_with_the_best_objective(self):
        seqs, index_of, logf, scores, population = self.build(population_helps=False)
        selection = eh.select_alpha(seqs, "validation", index_of, logf, scores,
                                    population, list(range(3)), progress=False)
        objective = selection["objective_by_alpha"]
        best = max(objective.values())
        self.assertAlmostEqual(objective[str(selection["selected_alpha"])], best)

    def test_a_useless_population_drives_alpha_to_one(self):
        seqs, index_of, logf, scores, population = self.build(population_helps=False)
        selection = eh.select_alpha(seqs, "validation", index_of, logf, scores,
                                    population, list(range(3)), progress=False)
        self.assertEqual(selection["selected_alpha"], 1.0)
        self.assertTrue(selection["population_adds_nothing"])

    def test_a_flat_objective_ties_to_the_largest_alpha(self):
        """Every alpha scoring identically must select 1.0, the most personal option."""
        seqs, index_of, logf, scores, _ = self.build(population_helps=True)
        # Population identical to what every personal centroid will be: the blend is
        # constant in alpha, so every alpha scores the same.
        M, _ = eh.personal_centroids(full_profile(seqs), index_of, logf)
        selection = eh.select_alpha(seqs, "validation", index_of, logf, scores,
                                    M, [0], progress=False)
        self.assertEqual(len(selection["tied_alphas"]), len(eh.ALPHA_GRID))
        self.assertEqual(selection["selected_alpha"], 1.0)

    def test_the_objective_is_participant_equal(self):
        self.assertIn("participant-equal", eh.FROZEN_SETTINGS["hybrid"]["alpha_objective"])
        self.assertIn("larger alpha", eh.FROZEN_SETTINGS["hybrid"]["alpha_tie_break"])

    def test_only_eligible_participants_choose_alpha(self):
        eligible = participant_set(1, per_emotion=3)
        thin = [s for s in participant_set(2, per_emotion=3)
                if not (s.emotion == FEAR and s.emoji > 0)]
        seqs = eligible + thin
        index_of, logf, scores = fixture(seqs, lambda s, w: onehot(s.emotion))
        population = cs.logscore(np.array([onehot(e) for e in range(7)]))
        selection = eh.select_alpha(seqs, "validation", index_of, logf, scores,
                                    population, [0], progress=False)
        self.assertEqual(selection["participants_used"], [1])

    def test_alpha_is_chosen_on_validation_only(self):
        self.assertEqual(eh.FROZEN_SETTINGS["hybrid"]["alpha_selected_on"],
                         "the EmojiHero validation split only")

    def test_the_sweep_endpoints_agree_with_the_standalone_methods(self):
        seqs, index_of, logf, scores, population = self.build(population_helps=True)
        profile = full_profile(seqs)
        evaluation, _ = eh.split_calibration_evaluation(seqs, profile)
        M, enrolled = eh.personal_centroids(profile, index_of, logf)
        swept = eh.sweep_alpha(evaluation, index_of, logf, M, enrolled,
                               population, eh.ALL_WINDOWS, (0.0, 1.0))
        full = eh.evaluate_participant(seqs, index_of, logf, scores, profile,
                                       eh.ALL_WINDOWS, population, 1.0)
        self.assertAlmostEqual(swept[1.0],
                               full["sequence_level"]["personal"]["macro_f1"], places=12)


# ----------------------------------------------------------- completeness


class MissingClassValidity(unittest.TestCase):
    def test_a_missing_scored_class_invalidates_the_seven_class_figure(self):
        seqs = [s for s in participant_set(1, per_emotion=2)
                if not (s.emotion == FEAR and s.emoji > 0)]
        index_of, logf, scores = fixture(seqs, lambda s, w: onehot(s.emotion))
        result = eh.evaluate_participant(seqs, index_of, logf, scores, full_profile(seqs))
        self.assertFalse(result["primary_macro_f1_valid"])
        self.assertEqual(len(result["classes_with_a_scored_sequence"]), 6)

    def test_a_complete_seven_class_evaluation_is_valid(self):
        seqs = participant_set(1, per_emotion=2)
        index_of, logf, scores = fixture(seqs, lambda s, w: onehot(s.emotion))
        result = eh.evaluate_participant(seqs, index_of, logf, scores, full_profile(seqs))
        self.assertTrue(result["primary_macro_f1_valid"])
        self.assertEqual(len(result["classes_with_a_scored_sequence"]), 7)

    def test_macro_f1_leaves_an_unsupported_class_out_which_is_why_it_is_gated(self):
        truth = np.array(["Anger", "Disgust"])
        complete = sa.macro_f1(truth, truth, list(EMOTIONS))
        self.assertEqual(complete["macro_f1"], 1.0)
        self.assertEqual(complete["classes_scored"], 2)


class Eligibility(unittest.TestCase):
    def test_two_sequences_per_emotion_are_required(self):
        self.assertEqual(eh.SEQUENCES_REQUIRED_PER_EMOTION, 2)

    def test_a_thin_cell_is_found_by_the_general_rule(self):
        seqs = ([s for s in participant_set(18) if not (s.emotion == FEAR and s.emoji > 0)]
                + participant_set(1))
        self.assertEqual(eh.thin_cells(seqs), [(18, FEAR, 1)])

    def test_the_eligibility_table_explains_each_exclusion(self):
        seqs = ([s for s in participant_set(18) if not (s.emotion == FEAR and s.emoji > 0)]
                + participant_set(1))
        table = eh.eligibility_table(seqs)
        self.assertTrue(table[1]["eligible"])
        self.assertFalse(table[18]["eligible"])
        self.assertIn("Fear", table[18]["reason"])
        self.assertIn("must not be split", table[18]["reason"])

    def test_no_participant_id_is_hard_coded(self):
        source = Path(eh.__file__).read_text(encoding="utf-8")
        for forbidden in ("participant == 18", "participant != 18", "== 18", "p18"):
            self.assertNotIn(forbidden, source)

    def test_cell_counts_are_per_participant_and_emotion(self):
        seqs = [sequence(1, ANGER), sequence(1, ANGER, emoji=1), sequence(2, ANGER)]
        counts = eh.cell_counts(seqs)
        self.assertEqual(counts[(1, ANGER)], 2)
        self.assertEqual(counts[(2, ANGER)], 1)


# ------------------------------------------------------------ diagnostics


class Diagnostics(unittest.TestCase):
    def test_level_gap_is_reported_without_being_acted_on(self):
        seqs = ([sequence(1, e, level=4, emoji=0) for e in range(7)]
                + [sequence(1, e, level=1, emoji=1) for e in range(7)])
        index_of, logf, scores = fixture(seqs, lambda s, w: onehot(s.emotion))
        profile = {e: next(s for s in seqs if s.emotion == e and s.level == 4)
                   for e in range(7)}
        result = eh.evaluate_participant(seqs, index_of, logf, scores, profile)
        self.assertAlmostEqual(
            result["level_diagnostic"]["level_gap_calibration_minus_evaluation"], 3.0)

    def test_the_non_overlapping_pair_shares_no_frame(self):
        self.assertEqual(eh.NON_OVERLAPPING_WINDOWS, (0, 5))
        seqs = participant_set(1, per_emotion=2)
        index_of, logf, scores = fixture(seqs, lambda s, w: onehot(s.emotion))
        result = eh.evaluate_participant(seqs, index_of, logf, scores, full_profile(seqs),
                                         eh.NON_OVERLAPPING_WINDOWS)
        self.assertEqual(result["n_evaluation_windows"], 7 * 2)

    def test_separation_is_reported_for_each_centroid_family(self):
        seqs = participant_set(1)
        index_of, logf, scores = fixture(seqs, lambda s, w: onehot(s.emotion))
        population = cs.logscore(np.array([onehot(e) for e in range(7)]))
        result = eh.evaluate_participant(seqs, index_of, logf, scores, full_profile(seqs),
                                         eh.ALL_WINDOWS, population, 0.5)
        for family in ("personal", "population", "hybrid"):
            self.assertIn(family, result["centroid_separation"])

    def test_personal_to_population_similarity_is_reported(self):
        seqs = participant_set(1)
        index_of, logf, scores = fixture(seqs, lambda s, w: onehot(s.emotion))
        population = cs.logscore(np.array([onehot(e) for e in range(7)]))
        result = eh.evaluate_participant(seqs, index_of, logf, scores, full_profile(seqs),
                                         eh.ALL_WINDOWS, population, 0.5)
        self.assertEqual(sorted(result["personal_to_population_cosine"]), sorted(EMOTIONS))


class AdapterBehaviour(unittest.TestCase):
    def test_a_personal_adapter_recovers_a_consistently_mislabelled_participant(self):
        """The case calibration exists for: the model is wrong the same way every time.

        This person's Disgust always reads as Surprise, but less confidently than their
        real Surprise does. Raw argmax cannot tell the two apart and scores Disgust 0/2.
        The personal centroids keep the whole score pattern, where the two peaks differ,
        and recover Disgust without losing Surprise.
        """
        seqs = participant_set(1)

        def confused(s, w):
            if s.emotion == DISGUST:
                return onehot(SURPRISE, 0.6)
            if s.emotion == SURPRISE:
                return onehot(SURPRISE, 0.9)
            return onehot(s.emotion, 0.6)

        index_of, logf, scores = fixture(seqs, confused)
        result = eh.evaluate_participant(seqs, index_of, logf, scores, full_profile(seqs))
        raw = result["sequence_level"]["raw"]["per_class_recall"]
        personal = result["sequence_level"]["personal"]["per_class_recall"]

        self.assertEqual(raw["Disgust"], 0.0)
        self.assertEqual(personal["Disgust"], 1.0)
        self.assertEqual(personal["Surprise"], 1.0, "Surprise was traded away for Disgust")
        self.assertGreater(result["delta"]["personal_minus_raw"]["accuracy"], 0)


class SignTest(unittest.TestCase):
    def test_all_positive_over_seven_participants(self):
        result = eh._sign_test([0.1] * 7)
        self.assertEqual(result["positive"], 7)
        self.assertAlmostEqual(result["p_two_sided"], 2 / 128)

    def test_ties_are_dropped_from_the_denominator(self):
        result = eh._sign_test([0.1, 0.0, -0.1, 0.2])
        self.assertEqual((result["positive"], result["negative"], result["ties"]), (2, 1, 1))

    def test_an_even_split_is_not_significant(self):
        self.assertEqual(eh._sign_test([1.0, -1.0])["p_two_sided"], 1.0)

    def test_the_reachable_floor_is_reported(self):
        """Seven participants cannot reach p < 0.0156 however clean the result is."""
        self.assertAlmostEqual(eh._sign_test([0.1] * 7)["smallest_reachable_p"], 2 / 128)


# --------------------------------------------------------- files and config


class ResultFileRules(unittest.TestCase):
    def test_an_existing_result_file_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            eh.write_json(path, {"a": 1})
            with self.assertRaises(SystemExit):
                eh.write_json(path, {"a": 2})
            self.assertEqual(json.loads(path.read_text())["a"], 1)

    def test_a_biometric_field_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            with self.assertRaises(SystemExit):
                eh.write_json(path, {"result": {"frames": [[0.1] * 70]}})
            self.assertFalse(path.exists())

    def test_a_personal_centroid_vector_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            with self.assertRaises(SystemExit):
                eh.write_json(path, {"personal_centroid": [0.0] * 7})

    def test_the_population_artifact_is_writable_because_it_is_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "population.json"
            eh.write_json(path, {"centroids": {"Anger": [0.0] * 7}})
            self.assertTrue(path.exists())

    def test_the_whitelist_names_no_per_person_vector_field(self):
        for field in eh.PARTICIPANT_FIELDS:
            self.assertNotIn("vector", field)
            self.assertNotIn("frame", field)

    def test_numpy_scalars_survive_serialisation(self):
        payload = eh._jsonable({"a": np.float64(1.5), "b": np.int64(2),
                                "c": np.array([1.0, 2.0])})
        self.assertEqual(json.loads(json.dumps(payload)),
                         {"a": 1.5, "b": 2, "c": [1.0, 2.0]})


class SemanticHash(unittest.TestCase):
    """The preregistration pin must survive a rebuild and reject an edit."""

    def parts(self, **overrides) -> dict:
        payload = dict(
            centroids={c: [float(i)] * 7 for i, c in enumerate(EMOTIONS)},
            class_order=list(EMOTIONS),
            schema_version=eh.POPULATION_SCHEMA_VERSION,
            preprocessing=eh.FROZEN_SETTINGS["preprocessing"],
            epsilon=cs.DEFAULT_EPSILON,
            feature_space="logscore",
            transform="log(clip(score, epsilon))",
            estimator="hierarchical",
            model_sha="a" * 64,
            fingerprint={"split": "training", "workbook_sha256": "b" * 64,
                         "n_sequences": 964, "n_participants": 20},
        )
        payload.update(overrides)
        return payload

    def test_identical_inputs_give_an_identical_hash(self):
        self.assertEqual(eh.population_semantic_hash(**self.parts()),
                         eh.population_semantic_hash(**self.parts()))

    def test_key_order_does_not_change_the_hash(self):
        shuffled = {c: [float(i)] * 7
                    for i, c in enumerate(EMOTIONS)}
        reordered = dict(reversed(list(shuffled.items())))
        self.assertEqual(eh.population_semantic_hash(**self.parts(centroids=shuffled)),
                         eh.population_semantic_hash(**self.parts(centroids=reordered)))

    def test_a_changed_centroid_changes_the_hash(self):
        altered = {c: [float(i)] * 7 for i, c in enumerate(EMOTIONS)}
        altered["Anger"] = [0.5] * 7
        self.assertNotEqual(eh.population_semantic_hash(**self.parts()),
                            eh.population_semantic_hash(**self.parts(centroids=altered)))

    def test_a_different_model_changes_the_hash(self):
        self.assertNotEqual(eh.population_semantic_hash(**self.parts()),
                            eh.population_semantic_hash(**self.parts(model_sha="c" * 64)))

    def test_a_different_training_corpus_changes_the_hash(self):
        other = {"split": "training", "workbook_sha256": "d" * 64,
                 "n_sequences": 964, "n_participants": 20}
        self.assertNotEqual(eh.population_semantic_hash(**self.parts()),
                            eh.population_semantic_hash(**self.parts(fingerprint=other)))

    def test_different_preprocessing_changes_the_hash(self):
        altered = dict(eh.FROZEN_SETTINGS["preprocessing"])
        altered["window_stride"] = 2
        self.assertNotEqual(eh.population_semantic_hash(**self.parts()),
                            eh.population_semantic_hash(**self.parts(preprocessing=altered)))

    def test_a_different_epsilon_changes_the_hash(self):
        self.assertNotEqual(eh.population_semantic_hash(**self.parts()),
                            eh.population_semantic_hash(**self.parts(epsilon=1e-6)))

    def test_a_reordered_class_list_changes_the_hash(self):
        self.assertNotEqual(
            eh.population_semantic_hash(**self.parts()),
            eh.population_semantic_hash(**self.parts(class_order=list(reversed(EMOTIONS)))))

    def test_the_timestamp_is_not_in_the_hash(self):
        """Two artifacts built a moment apart from identical inputs must pin identically,
        or the frozen config would reject a correct rebuild."""
        matrix = np.arange(49, dtype=float).reshape(7, 7)
        support = {c: {"participants": 20, "sequences": 100,
                       "sequences_per_participant": {}} for c in EMOTIONS}
        kwargs = dict(model_sha="a" * 64, model_path=Path("m.onnx"),
                      provenance={"sha256": "b" * 64, "rows": 28920, "split": "training"},
                      sequences=[sequence(1, ANGER)])
        first = eh.population_artifact(matrix, support, **kwargs)
        second = eh.population_artifact(matrix, support, **kwargs)
        self.assertEqual(first["semantic_sha256"], second["semantic_sha256"])
        self.assertIn("created_utc", first)
        self.assertIn("created_utc", second["semantic_hash_excludes"] and second)

    def test_a_tampered_artifact_is_refused_on_load(self):
        matrix = np.arange(49, dtype=float).reshape(7, 7)
        support = {c: {"participants": 20, "sequences": 100,
                       "sequences_per_participant": {}} for c in EMOTIONS}
        artifact = eh.population_artifact(
            matrix, support, model_sha="a" * 64, model_path=Path("m.onnx"),
            provenance={"sha256": "b" * 64, "rows": 28920, "split": "training"},
            sequences=[sequence(1, ANGER)])
        artifact["centroids"]["Anger"] = [99.0] * 7
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "population.json"
            path.write_text(json.dumps(eh._jsonable(artifact)), encoding="utf-8")
            with self.assertRaises(SystemExit) as caught:
                eh.load_population_centroids(path, model_sha="a" * 64)
            self.assertIn("semantic hash", str(caught.exception))

    def test_an_artifact_without_a_semantic_hash_is_refused(self):
        matrix = np.arange(49, dtype=float).reshape(7, 7)
        support = {c: {"participants": 20, "sequences": 100,
                       "sequences_per_participant": {}} for c in EMOTIONS}
        artifact = eh.population_artifact(
            matrix, support, model_sha="a" * 64, model_path=Path("m.onnx"),
            provenance={"sha256": "b" * 64, "rows": 28920, "split": "training"},
            sequences=[sequence(1, ANGER)])
        del artifact["semantic_sha256"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "population.json"
            path.write_text(json.dumps(eh._jsonable(artifact)), encoding="utf-8")
            with self.assertRaises(SystemExit) as caught:
                eh.load_population_centroids(path, model_sha="a" * 64)
            self.assertIn("semantic", str(caught.exception))


class FrozenPrimaryAnalysis(unittest.TestCase):
    def test_the_primary_comparison_is_hybrid_minus_personal(self):
        primary = eh.FROZEN_SETTINGS["primary_analysis"]
        self.assertEqual(primary["comparison"], "hybrid minus personal")

    def test_the_primary_metric_is_participant_level_macro_f1(self):
        primary = eh.FROZEN_SETTINGS["primary_analysis"]
        self.assertIn("participant-level", primary["metric"])
        self.assertIn("macro-F1", primary["metric"])

    def test_the_inference_unit_is_the_participant(self):
        self.assertEqual(eh.FROZEN_SETTINGS["primary_analysis"]["inference_unit"],
                         "participant")

    def test_the_test_is_an_exact_paired_sign_test(self):
        statistic = eh.FROZEN_SETTINGS["primary_analysis"]["statistical_test"]
        self.assertIn("exact", statistic)
        self.assertIn("sign test", statistic)
        self.assertIn("participants", statistic)

    def test_profiles_are_declared_not_to_be_a_sample_size(self):
        note = eh.FROZEN_SETTINGS["primary_analysis"]["profiles_are_not_a_sample_size"]
        self.assertIn("not independent observations", note)
        self.assertIn("ONE value per participant", note)

    def test_hybrid_minus_raw_and_accuracy_are_secondary(self):
        secondary = eh.FROZEN_SETTINGS["primary_analysis"]["secondary_only"]
        self.assertIn("hybrid minus raw", secondary)
        self.assertTrue(any("accuracy" in item for item in secondary))

    def test_the_sign_test_is_computed_over_participants_not_profiles(self):
        """The aggregate feeding `_sign_test` is one value per participant."""
        source = Path(eh.__file__).read_text(encoding="utf-8")
        self.assertIn("mean_values.append(summary[\"mean\"])", source)


class FrozenConfig(unittest.TestCase):
    def config(self, **overrides) -> dict:
        payload = {
            "frozen_settings": eh._jsonable(eh.FROZEN_SETTINGS),
            "selected_alpha": 0.6,
            "primary_analysis": {**eh._jsonable(eh.FROZEN_SETTINGS["primary_analysis"]),
                                 "fixed_alpha": 0.6},
            "test_plan": {"primary_profile_index": 0, "robustness_profile_count": 1000,
                          "window_condition": "stride1"},
            "model_sha256": "abc",
        }
        payload.update(overrides)
        return payload

    def load(self, payload: dict):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return eh.load_frozen_config(path)

    def test_a_matching_config_loads(self):
        self.assertEqual(self.load(self.config())["selected_alpha"], 0.6)

    def test_a_changed_setting_refuses_the_run(self):
        altered = eh._jsonable(eh.FROZEN_SETTINGS)
        altered["metric"] = "euclidean"
        with self.assertRaises(SystemExit):
            self.load(self.config(frozen_settings=altered))

    def test_a_config_without_an_alpha_is_refused(self):
        payload = self.config()
        del payload["selected_alpha"]
        with self.assertRaises(SystemExit) as caught:
            self.load(payload)
        self.assertIn("selected_alpha", str(caught.exception))

    def test_an_alpha_outside_the_unit_interval_is_refused(self):
        with self.assertRaises(SystemExit):
            self.load(self.config(selected_alpha=1.5))

    def test_a_config_without_the_block_is_refused(self):
        with self.assertRaises(SystemExit):
            self.load({"nothing": True})

    def test_the_frozen_block_names_every_decision_that_could_move_a_number(self):
        for key in ("epsilon", "metric", "centroid", "tie_rule", "preprocessing",
                    "master_seed", "primary_test_profile_index", "eligibility_rule",
                    "missing_class_rule", "class_set", "aggregation", "hybrid",
                    "population_centroid", "methods"):
            self.assertIn(key, eh.FROZEN_SETTINGS)

    def test_a_config_without_a_fixed_alpha_is_refused(self):
        payload = self.config()
        del payload["primary_analysis"]["fixed_alpha"]
        with self.assertRaises(SystemExit) as caught:
            self.load(payload)
        self.assertIn("fixed_alpha", str(caught.exception))

    def test_a_fixed_alpha_disagreeing_with_the_selected_alpha_is_refused(self):
        payload = self.config()
        payload["primary_analysis"]["fixed_alpha"] = 0.35
        with self.assertRaises(SystemExit) as caught:
            self.load(payload)
        self.assertIn("disagrees", str(caught.exception))

    def test_a_config_missing_a_test_plan_pin_is_refused(self):
        for key in ("primary_profile_index", "robustness_profile_count",
                    "window_condition"):
            payload = self.config()
            del payload["test_plan"][key]
            with self.assertRaises(SystemExit) as caught:
                self.load(payload)
            self.assertIn(key, str(caught.exception))


class TestSplitCannotBeOverriddenFromTheCommandLine(unittest.TestCase):
    """Nothing that could move a primary number may be supplied at the prompt on test."""

    def test_the_test_split_cannot_run_without_a_frozen_config(self):
        with self.assertRaises(SystemExit) as caught:
            eh.main(["--split", "test"])
        self.assertIn("frozen-config", str(caught.exception))

    def test_alpha_cannot_be_given(self):
        with self.assertRaises(SystemExit) as caught:
            eh.main(["--split", "test", "--frozen-config", "x.json",
                     "--hybrid-alpha", "0.5"])
        self.assertIn("--hybrid-alpha", str(caught.exception))

    def test_the_profile_count_cannot_be_given(self):
        with self.assertRaises(SystemExit) as caught:
            eh.main(["--split", "test", "--frozen-config", "x.json",
                     "--profile-count", "10"])
        self.assertIn("--profile-count", str(caught.exception))

    def test_the_window_condition_cannot_be_given(self):
        with self.assertRaises(SystemExit) as caught:
            eh.main(["--split", "test", "--frozen-config", "x.json",
                     "--windows", "non-overlapping"])
        self.assertIn("--windows", str(caught.exception))

    def test_several_overrides_are_all_named(self):
        with self.assertRaises(SystemExit) as caught:
            eh.main(["--split", "test", "--frozen-config", "x.json",
                     "--hybrid-alpha", "0.5", "--profile-count", "10",
                     "--windows", "non-overlapping"])
        message = str(caught.exception)
        for flag in ("--hybrid-alpha", "--profile-count", "--windows"):
            self.assertIn(flag, message)

    def test_the_config_cannot_be_re_frozen_from_a_test_run(self):
        with self.assertRaises(SystemExit) as caught:
            eh.main(["--split", "test", "--frozen-config", "x.json",
                     "--frozen-config-output", "y.json"])
        self.assertIn("frozen before the test split", str(caught.exception))

    def test_the_same_flags_are_still_allowed_on_validation(self):
        """The guard is about the confirmatory split, not about the sensitivity work."""
        with self.assertRaises(SystemExit) as caught:
            eh.main(["--split", "validation", "--hybrid-alpha", "0.5",
                     "--profile-count", "10", "--windows", "non-overlapping",
                     "--data", "/definitely/not/here"])
        self.assertIn("--data does not exist", str(caught.exception))

    def test_there_is_no_flag_for_epsilon_metric_seed_or_primary_profile(self):
        """These are module constants precisely so they cannot be typed at a prompt."""
        import contextlib
        import io as _io
        for flag in ("--epsilon", "--metric", "--seed", "--master-seed",
                     "--primary-profile", "--tie-rule", "--centroid"):
            with contextlib.redirect_stderr(_io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    eh.main([flag, "0"])
            # argparse exits 2 on an unrecognised flag; anything else would mean the
            # parser quietly accepted it.
            self.assertEqual(caught.exception.code, 2, f"{flag} was accepted")

    def test_build_population_and_split_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            eh.main(["--build-population", "--split", "validation"])


class ProvenanceManifest(unittest.TestCase):
    def test_package_versions_are_recorded(self):
        versions = eh.package_versions()
        self.assertIn("numpy", versions)
        self.assertIn("python", versions)

    def test_the_expected_corpus_shape_is_stated_as_constants(self):
        self.assertEqual(eh.EXPECTED_SEQUENCES,
                         {"training": 964, "validation": 385, "test": 378})
        self.assertEqual(eh.EXPECTED_PARTICIPANTS,
                         {"training": 20, "validation": 8, "test": 8})

    def test_limitations_are_part_of_the_artefact_not_only_the_readme(self):
        source = Path(eh.__file__).read_text(encoding="utf-8")
        self.assertIn("not an upper or lower bound", source)
        self.assertIn("70.4%", source)
        self.assertIn("hybrid - personal", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
