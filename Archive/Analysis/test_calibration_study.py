#!/usr/bin/env python3
"""Unit tests for calibration_study.py.

    python Analysis/test_calibration_study.py

Stdlib `unittest`, so there is nothing to install beyond numpy, which the module
under test already requires. onnxruntime is deliberately not needed: everything
here works on hand-built score vectors, so the arithmetic can be checked against
values worked out by hand rather than against the model's own output.

The one test that does need the model and a recording is marked as an integration
test and skips itself when either is missing.
"""

from __future__ import annotations

import unittest

import numpy as np

import calibration_study as cs
import session_analysis as sa

CLASSES = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
ANGER, DISGUST, NEUTRAL, SURPRISE = 0, 1, 4, 6


def row(tick: int | None, trial: int, prompted: str, scores: list[float],
        block_role: str | None = None, repeat_index: int | None = None) -> sa.Row:
    """One `ok` tick carrying a flat, non-degenerate window."""
    return sa.Row(
        timestamp=None,
        trial_id=trial,
        prompted=prompted,
        prediction_index=tick,
        predicted=CLASSES[int(np.argmax(scores))],
        logged_correct=False,
        status=sa.STATUS_OK,
        generation=None,
        scores=scores,
        frames=[[0.1 * (f + 1)] * 3 for f in range(5)],
        block_role=block_role,
        repeat_index=repeat_index,
    )


def onehot(index: int, peak: float = 0.7) -> list[float]:
    rest = (1.0 - peak) / (len(CLASSES) - 1)
    return [peak if i == index else rest for i in range(len(CLASSES))]


class TickSelection(unittest.TestCase):
    def test_prefers_recorded_prediction_index(self):
        rows = [row(3, 0, "Anger", onehot(ANGER)), row(4, 0, "Anger", onehot(ANGER))]
        ticks, source = cs.tick_positions(rows)
        self.assertEqual(source, "prediction_index")
        np.testing.assert_array_equal(ticks, [3, 4])

    def test_gap_from_a_dropped_row_is_preserved(self):
        """The regression this column exists to prevent.

        Ticks 0 and 3 survive; 1 and 2 were `no_tracking` and carry no window, so
        they never reach this function. Enumerating would relabel tick 3 as tick 1
        and move it from the test split into the enrolment split.
        """
        rows = [row(0, 0, "Anger", onehot(ANGER)), row(3, 0, "Anger", onehot(ANGER))]
        ticks, _ = cs.tick_positions(rows)
        np.testing.assert_array_equal(ticks, [0, 3])

    def test_falls_back_when_the_column_is_absent(self):
        rows = [row(None, 0, "Anger", onehot(ANGER)), row(None, 0, "Anger", onehot(ANGER))]
        ticks, source = cs.tick_positions(rows)
        self.assertIn("enumerated", source)
        np.testing.assert_array_equal(ticks, [0, 1])

    def test_fallback_restarts_the_count_per_trial(self):
        rows = [
            row(None, 0, "Anger", onehot(ANGER)),
            row(None, 0, "Anger", onehot(ANGER)),
            row(None, 1, "Disgust", onehot(DISGUST)),
        ]
        ticks, _ = cs.tick_positions(rows)
        np.testing.assert_array_equal(ticks, [0, 1, 0])


class Centroids(unittest.TestCase):
    def test_is_the_per_class_mean_in_the_requested_class_order(self):
        F = np.array([[0.0, 0.0], [2.0, 4.0], [10.0, 10.0]])
        y = np.array([ANGER, ANGER, DISGUST])
        M = cs.centroids(F, y, [ANGER, DISGUST])
        np.testing.assert_allclose(M, [[1.0, 2.0], [10.0, 10.0]])

    def test_class_order_follows_the_list_not_the_data(self):
        F = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = np.array([DISGUST, ANGER])
        M = cs.centroids(F, y, [DISGUST, ANGER])
        np.testing.assert_allclose(M, [[1.0, 0.0], [0.0, 1.0]])


class Metrics(unittest.TestCase):
    def test_cosine_ignores_magnitude(self):
        M = np.array([[1.0, 0.0], [0.0, 1.0]])
        F = np.array([[50.0, 0.0]])
        pred, top, _ = cs.classify(M, [ANGER, DISGUST], F, "cosine")
        self.assertEqual(pred[0], ANGER)
        self.assertAlmostEqual(float(top[0]), 1.0, places=6)

    def test_the_two_metrics_disagree_on_the_same_data(self):
        """The case that separates them, so neither can silently stand in for the other.

        Centroids (1,0) and (4,4) against a sample at (5,1). By direction the sample
        is almost exactly on the first (cos 0.98 against 0.83); by distance it is
        nearer the second (3.16 against 4.12).
        """
        M = np.array([[1.0, 0.0], [4.0, 4.0]])
        F = np.array([[5.0, 1.0]])
        by_cosine, _, _ = cs.classify(M, [ANGER, DISGUST], F, "cosine")
        by_euclidean, _, _ = cs.classify(M, [ANGER, DISGUST], F, "euclidean")
        self.assertEqual(by_cosine[0], ANGER)
        self.assertEqual(by_euclidean[0], DISGUST)

    def test_margin_is_the_gap_to_the_runner_up(self):
        M = np.array([[1.0, 0.0], [0.0, 1.0]])
        F = np.array([[1.0, 0.0]])
        _, top, margin = cs.classify(M, [ANGER, DISGUST], F, "cosine")
        self.assertAlmostEqual(float(margin[0]), float(top[0]) - 0.0, places=6)

    def test_margin_is_zero_when_a_sample_sits_between_two_centroids(self):
        M = np.array([[1.0, 0.0], [0.0, 1.0]])
        F = np.array([[1.0, 1.0]])
        _, _, margin = cs.classify(M, [ANGER, DISGUST], F, "cosine")
        self.assertAlmostEqual(float(margin[0]), 0.0, places=6)

    def test_a_far_sample_can_still_win_by_a_wide_margin(self):
        """Why `--abstain-on distance` exists alongside `margin`.

        A point unlike anything enrolled is caught by absolute similarity and missed
        by the margin, so the two criteria are not interchangeable.
        """
        M = np.array([[1.0, 0.0], [0.9, 0.1]])
        F = np.array([[-1.0, 0.0]])
        _, top, margin = cs.classify(M, [ANGER, DISGUST], F, "cosine")
        self.assertLess(float(top[0]), 0.0)         # far from everything
        self.assertGreater(float(margin[0]), 0.0)   # yet a clear winner

    def test_unknown_metric_is_refused(self):
        with self.assertRaises(ValueError):
            cs.classify(np.zeros((1, 2)), [ANGER], np.zeros((1, 2)), "manhattan")


class Features(unittest.TestCase):
    def test_epsilon_floors_the_log_instead_of_producing_negative_infinity(self):
        data = _dataset([row(0, 0, "Anger", [0.0] + [1.0 / 6] * 6)])
        F = cs.features(data, "logscore", cs.DEFAULT_EPSILON)
        self.assertTrue(np.isfinite(F).all())
        self.assertAlmostEqual(float(F[0, 0]), float(np.log(cs.DEFAULT_EPSILON)), places=6)

    def test_log_and_raw_are_genuinely_different_spaces(self):
        data = _dataset([row(0, 0, "Anger", onehot(ANGER))])
        log = cs.features(data, "logscore", cs.DEFAULT_EPSILON)
        raw = cs.features(data, "score", cs.DEFAULT_EPSILON)
        self.assertFalse(np.allclose(log, raw))

    def test_unknown_space_is_refused(self):
        data = _dataset([row(0, 0, "Anger", onehot(ANGER))])
        with self.assertRaises(ValueError):
            cs.features(data, "pca", cs.DEFAULT_EPSILON)


class NeutralHandling(unittest.TestCase):
    def test_default_enrols_only_prompted_classes(self):
        data = _session()
        result = cs.evaluate(data, cs.Condition(), *_ticks(data, {1}, {2}))
        self.assertEqual(result["enrolled_classes"], ["Anger", "Disgust"])
        self.assertNotIn("Neutral", result["enrolled_classes"])

    def test_restricted_baseline_uses_the_same_class_set_as_the_adapter(self):
        """The confound this row exists to expose.

        Both trials here are dominated by Neutral, so a seven-way argmax scores 0
        while an argmax over the two enrolled classes can still be right. Reporting
        only the former credits the adapter for excluding Neutral.
        """
        rows = [
            row(1, 0, "Anger", _mostly(NEUTRAL, ANGER)),
            row(2, 0, "Anger", _mostly(NEUTRAL, ANGER)),
            row(1, 1, "Disgust", _mostly(NEUTRAL, DISGUST)),
            row(2, 1, "Disgust", _mostly(NEUTRAL, DISGUST)),
        ]
        data = _dataset(rows)
        result = cs.evaluate(data, cs.Condition(), *_ticks(data, {1}, {2}))
        self.assertEqual(result["accuracy"]["argmax_full"], 0.0)
        self.assertEqual(result["accuracy"]["argmax_restricted"], 1.0)

    def test_rest_adds_neutral_from_tick_zero(self):
        data = _session(with_rest=True)
        result = cs.evaluate(data, cs.Condition(neutral="rest"),
                             *_ticks(data, {1}, {2}))
        self.assertIn("Neutral", result["enrolled_classes"])

    def test_rest_without_any_tick_zero_row_is_refused(self):
        data = _session()
        with self.assertRaises(SystemExit):
            cs.evaluate(data, cs.Condition(neutral="rest"), *_ticks(data, {1}, {2}))

    def test_prompted_is_refused_until_neutral_is_actually_prompted(self):
        data = _session()
        with self.assertRaises(SystemExit):
            cs.evaluate(data, cs.Condition(neutral="prompted"), *_ticks(data, {1}, {2}))

    def test_empty_split_is_refused_rather_than_scored(self):
        data = _session()
        with self.assertRaises(SystemExit):
            cs.evaluate(data, cs.Condition(), *_ticks(data, {1}, {99}))


class Fidelity(unittest.TestCase):
    def _replayed(self, logged: np.ndarray, scores: np.ndarray) -> cs.Replayed:
        n = len(logged)
        return cs.Replayed(
            name="s", classes=CLASSES,
            windows=np.zeros((n, 5, 3), np.float32),
            prompted=np.zeros(n, int), tick=np.arange(n), trial=np.zeros(n, int),
            role=np.array([""] * n), repeat=np.full(n, -1),
            scores=scores, logged=logged, frame_reuse=0, tick_source="prediction_index",
            attempted={}, protocol_version=None, rehearsal=False,
        )

    def test_float_drift_within_tolerance_is_faithful(self):
        logged = np.array([onehot(ANGER)], np.float32)
        r = self._replayed(logged, logged + 4e-7)
        self.assertTrue(r.faithful)

    def test_a_different_model_is_not_faithful(self):
        logged = np.array([onehot(ANGER)], np.float32)
        r = self._replayed(logged, np.array([onehot(SURPRISE)], np.float32))
        self.assertFalse(r.faithful)

    def test_same_argmax_but_shifted_probabilities_still_fails(self):
        """Agreement alone is not enough -- a double softmax preserves argmax."""
        logged = np.array([onehot(ANGER, peak=0.84)], np.float32)
        shifted = np.array([onehot(ANGER, peak=0.27)], np.float32)
        r = self._replayed(logged, shifted)
        self.assertEqual(r.fidelity[0], 1.0)
        self.assertFalse(r.faithful)

    def test_mismatch_exits_nonzero_even_without_check(self):
        import contextlib, io, tempfile, os
        from pathlib import Path

        bad = Path(tempfile.gettempdir()) / "definitely-not-a-model.onnx"
        bad.write_bytes(b"not an onnx file")
        try:
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(BaseException) as caught:
                    cs.main(["--model", str(bad)])
            # Either a clean nonzero return or a SystemExit -- never a success.
            code = caught.exception.code if isinstance(caught.exception, SystemExit) else 1
            self.assertNotEqual(code, 0)
        finally:
            os.unlink(bad)


class Abstention(unittest.TestCase):
    def _result(self):
        return {
            "_arrays": {
                "y": np.array([ANGER, ANGER, DISGUST]),
                "predicted": np.array([ANGER, DISGUST, DISGUST]),
                "top": np.array([0.9, -0.5, 0.8]),
                "margin": np.array([0.5, 0.4, 0.3]),
            }
        }

    def test_zero_threshold_keeps_everything(self):
        out = cs.abstention_sweep(self._result(), [0.0], "margin")
        self.assertEqual(out[0]["coverage"], 1.0)
        self.assertAlmostEqual(out[0]["accuracy"], 2 / 3)

    def test_margin_and_distance_select_different_rows(self):
        by_margin = cs.abstention_sweep(self._result(), [0.35], "margin")[0]
        by_distance = cs.abstention_sweep(self._result(), [0.0], "distance")[0]
        self.assertAlmostEqual(by_margin["coverage"], 2 / 3)
        self.assertAlmostEqual(by_margin["accuracy"], 0.5)   # keeps the far outlier
        self.assertAlmostEqual(by_distance["coverage"], 2 / 3)
        self.assertAlmostEqual(by_distance["accuracy"], 1.0)  # drops it

    def test_accuracy_is_none_rather_than_nan_when_everything_is_declined(self):
        out = cs.abstention_sweep(self._result(), [9.0], "margin")
        self.assertEqual(out[0]["coverage"], 0.0)
        self.assertIsNone(out[0]["accuracy"])


class TickParsing(unittest.TestCase):
    def test_ranges_lists_and_mixtures(self):
        self.assertEqual(cs.parse_ticks("1-5"), {1, 2, 3, 4, 5})
        self.assertEqual(cs.parse_ticks("0,2,4"), {0, 2, 4})
        self.assertEqual(cs.parse_ticks("0,2-4"), {0, 2, 3, 4})

    def test_empty_is_refused(self):
        with self.assertRaises(Exception):
            cs.parse_ticks(" ")


class ReplayIntegration(unittest.TestCase):
    """The one test that needs the real model and a real recording."""

    def test_replay_reproduces_the_logged_scores(self):
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest("onnxruntime not installed")
        try:
            model = cs.find_model(None)
        except SystemExit:
            self.skipTest("sparseVR_teacher_model.onnx not found")
        files = sa.autodiscover()[0]
        if not files:
            self.skipTest("no session recordings on this machine")

        data = cs.replay(sa.load_session(files[0]), model)
        agree, diff = data.fidelity
        self.assertEqual(agree, 1.0)
        self.assertLess(diff, cs.FIDELITY_TOLERANCE)
        self.assertTrue(data.faithful)


# ------------------------------------------------------------------ fixtures


def _mostly(winner: int, runner_up: int) -> list[float]:
    """A score vector a seven-way argmax gets wrong and a restricted one gets right."""
    scores = [0.02] * len(CLASSES)
    scores[winner] = 0.6
    scores[runner_up] = 0.3
    return scores


def _dataset(rows: list[sa.Row], rehearsal: bool = False,
             protocol_version: str | None = None) -> cs.Replayed:
    """A `Replayed` built from rows directly, with no model involved."""
    ticks, source = cs.tick_positions(rows)
    scores = np.array([r.scores for r in rows], np.float32)

    attempted: dict[str, dict[int, str]] = {}
    for r in rows:
        attempted.setdefault(r.block_role or "", {})[r.trial_id] = r.prompted

    return cs.Replayed(
        name="fixture",
        classes=CLASSES,
        windows=np.array([r.frames for r in rows], np.float32),
        prompted=np.array([CLASSES.index(r.prompted) for r in rows]),
        tick=ticks,
        trial=np.array([r.trial_id for r in rows]),
        role=np.array([r.block_role or "" for r in rows]),
        repeat=np.array([r.repeat_index if r.repeat_index is not None else -1
                         for r in rows]),
        scores=scores,
        logged=scores,
        frame_reuse=0,
        tick_source=source,
        attempted=attempted,
        protocol_version=protocol_version,
        rehearsal=rehearsal,
    )


def _ticks(data: cs.Replayed, enrol: set[int], test: set[int]):
    """The old tick split, as masks, so `evaluate` can be called the same way either way."""
    return cs.split_masks(data, enrol, test)


def _session(with_rest: bool = False) -> cs.Replayed:
    """Two classes, one enrolment tick and one test tick each."""
    rows: list[sa.Row] = []
    for trial, (name, index) in enumerate((("Anger", ANGER), ("Disgust", DISGUST))):
        if with_rest:
            rows.append(row(0, trial, name, onehot(NEUTRAL)))
        rows.append(row(1, trial, name, onehot(index)))
        rows.append(row(2, trial, name, onehot(index)))
    return _dataset(rows)


def _study_session(k_ticks: int = 9, d_ticks: int = 9,
                   skip_d_trial: int | None = None) -> cs.Replayed:
    """A `session-local-v1`-shaped session: 7 states in K, the same 7 twice in D.

    `skip_d_trial` drops every tick of one evaluation trial while leaving it in the
    attempted set, which is what an entirely untracked trial looks like on disk.
    """
    rows: list[sa.Row] = []
    trial = 0
    for name in CLASSES:
        for t in range(1, k_ticks + 1):
            rows.append(row(t, trial, name, onehot(CLASSES.index(name)),
                            sa.ROLE_CALIBRATION, 1))
        trial += 1
    for repeat in (1, 2):
        for name in CLASSES:
            if trial != skip_d_trial:
                for t in range(1, d_ticks + 1):
                    rows.append(row(t, trial, name, onehot(CLASSES.index(name)),
                                    sa.ROLE_EVALUATION, repeat))
            trial += 1

    data = _dataset(rows, protocol_version="session-local-v1")
    if skip_d_trial is not None:
        # The trial existed; every tick of it was no_tracking, so it carries no window.
        data.attempted[sa.ROLE_EVALUATION][skip_d_trial] = "Anger"
    return data


class RoleSplit(unittest.TestCase):
    def test_enrolment_and_test_come_from_different_trials(self):
        data = _study_session()
        enrol, test = cs.role_masks(data, sa.ROLE_CALIBRATION, sa.ROLE_EVALUATION)

        self.assertFalse((enrol & test).any(), "a row cannot be in both blocks")
        self.assertTrue(
            set(data.trial[enrol].tolist()).isdisjoint(set(data.trial[test].tolist())),
            "the whole point of the split: no trial appears on both sides",
        )

    def test_tick_zero_is_in_neither_block(self):
        data = _study_session()
        rows_with_zero = _dataset(
            [row(0, 0, "Anger", onehot(ANGER), sa.ROLE_CALIBRATION, 1),
             row(1, 0, "Anger", onehot(ANGER), sa.ROLE_CALIBRATION, 1),
             row(1, 1, "Anger", onehot(ANGER), sa.ROLE_EVALUATION, 1)]
        )
        enrol, test = cs.role_masks(rows_with_zero, sa.ROLE_CALIBRATION,
                                    sa.ROLE_EVALUATION)
        self.assertNotIn(0, rows_with_zero.tick[enrol].tolist())
        self.assertNotIn(0, rows_with_zero.tick[test].tolist())
        del data

    def test_a_file_without_block_roles_is_refused_by_name(self):
        with self.assertRaises(SystemExit) as caught:
            cs.role_masks(_session(), sa.ROLE_CALIBRATION, sa.ROLE_EVALUATION)
        self.assertIn("block_role", str(caught.exception))

    def test_block_role_is_lowercased_when_read(self):
        """A hand-edited `Calibration` must not read as a different block from
        `calibration`, because the split would then silently drop those rows."""
        import csv
        import tempfile
        from pathlib import Path as _Path

        with tempfile.TemporaryDirectory() as tmp:
            path = _Path(tmp) / "predictions.csv"
            header = (["timestamp_utc", "session_id", "trial_id", "block_role",
                       "repeat_index", "prompted_emotion", "prediction_index",
                       "predicted_emotion", "correct", "status", "generation"]
                      + [f"score_{c}" for c in CLASSES]
                      + [f"frame_{f}_valid" for f in range(5)]
                      + [f"frame_{f}_au_{a}" for f in range(5) for a in range(3)])
            with path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(header)
                w.writerow(["2026-01-01T00:00:00.000Z", "s", 0, "Calibration", 1,
                            "Anger", 1, "Anger", 1, "ok", 1]
                           + onehot(ANGER) + [1] * 5 + [0.1] * 15)
            session = sa.load_session(path)

        self.assertEqual(session.rows[0].block_role, sa.ROLE_CALIBRATION)
        self.assertEqual(session.rows[0].repeat_index, 1)


class TrialVote(unittest.TestCase):
    CLASS_ORDER = CLASSES

    def _vote(self, predicted: list[str], ticks: list[int]) -> str:
        truth = np.array(["Anger"] * len(predicted))
        trials = np.zeros(len(predicted), int)
        _, _, votes = cs.trial_vote(truth, np.array(predicted), trials,
                                    np.array(ticks), self.CLASS_ORDER)
        return votes[0]

    def test_a_clear_majority_wins(self):
        self.assertEqual(
            self._vote(["Anger", "Anger", "Disgust"], [1, 2, 3]), "Anger")

    def test_a_tie_is_broken_by_the_final_tracked_tick(self):
        """Step 2 of the frozen rule. The last tick is the part of a held expression
        the participant had most settled into."""
        self.assertEqual(
            self._vote(["Anger", "Anger", "Disgust", "Disgust"], [1, 2, 3, 9]),
            "Disgust")
        self.assertEqual(
            self._vote(["Disgust", "Disgust", "Anger", "Anger"], [1, 2, 3, 9]),
            "Anger")

    def test_the_final_tick_is_by_tick_number_not_row_order(self):
        # Same votes, but the highest tick is listed first.
        self.assertEqual(
            self._vote(["Disgust", "Anger", "Anger", "Disgust"], [9, 1, 2, 3]),
            "Disgust")

    def test_a_tie_the_final_tick_cannot_break_falls_back_to_class_order(self):
        """Step 3. The final tick voted for a class that is not among the tied ones,
        so the tie needs something fixed and decided in advance."""
        self.assertEqual(
            self._vote(["Anger", "Anger", "Disgust", "Disgust", "Fear"],
                       [1, 2, 3, 4, 9]),
            "Anger")  # Anger precedes Disgust in CLASSES

    def test_the_same_rule_is_applied_to_every_method(self):
        data = _study_session()
        enrol, test = cs.role_masks(data, sa.ROLE_CALIBRATION, sa.ROLE_EVALUATION)
        result = cs.evaluate(data, cs.Condition(), enrol, test,
                             required_ticks=9, test_role=sa.ROLE_EVALUATION)
        # 14 evaluation trials collapse to 14 votes for all three methods alike.
        self.assertEqual(result["n_test_trials"], 14)
        for key in ("argmax_full", "argmax_restricted", "adapter"):
            self.assertIn("accuracy", result["trial_majority"][key])


class MissingTracking(unittest.TestCase):
    def test_a_state_short_of_nine_calibration_ticks_gets_no_centroid(self):
        data = _study_session(k_ticks=9)
        # Drop one Anger calibration tick, leaving eight.
        keep = ~((data.role == sa.ROLE_CALIBRATION)
                 & (data.prompted == ANGER)
                 & (data.tick == 9))
        trimmed = cs.Replayed(
            name=data.name, classes=data.classes, windows=data.windows[keep],
            prompted=data.prompted[keep], tick=data.tick[keep],
            trial=data.trial[keep], role=data.role[keep], repeat=data.repeat[keep],
            scores=data.scores[keep], logged=data.logged[keep],
            frame_reuse=0, tick_source=data.tick_source,
            attempted=data.attempted, protocol_version=data.protocol_version,
            rehearsal=False,
        )
        enrol, test = cs.role_masks(trimmed, sa.ROLE_CALIBRATION, sa.ROLE_EVALUATION)
        result = cs.evaluate(trimmed, cs.Condition(), enrol, test,
                             required_ticks=9, test_role=sa.ROLE_EVALUATION)

        self.assertEqual(result["ticks_per_centroid"]["Anger"], 8)
        self.assertIn("Anger", result["incomplete_centroids"])
        self.assertNotIn("Anger", result["enrolled_classes"])
        self.assertFalse(result["calibration_valid"])
        self.assertFalse(result["primary_macro_f1_valid"])

    def test_a_partial_hold_is_accepted_when_no_requirement_is_given(self):
        """The tick-split reference condition enrols five ticks by design, so the
        requirement has to be opt-in rather than always on."""
        data = _session()
        result = cs.evaluate(data, cs.Condition(), *_ticks(data, {1}, {2}))
        self.assertTrue(result["calibration_valid"])
        self.assertEqual(result["incomplete_centroids"], [])

    def test_an_entirely_untracked_trial_is_reported_not_dropped(self):
        data = _study_session(skip_d_trial=8)
        enrol, test = cs.role_masks(data, sa.ROLE_CALIBRATION, sa.ROLE_EVALUATION)
        result = cs.evaluate(data, cs.Condition(), enrol, test,
                             required_ticks=9, test_role=sa.ROLE_EVALUATION)

        self.assertEqual(result["n_test_trials"], 13)
        self.assertIn("8", result["unscored_trials"])
        self.assertFalse(result["primary_macro_f1_valid"],
                         "an unscored trial must not leave the primary figure valid")


class PrimaryMacroF1(unittest.TestCase):
    def test_a_complete_seven_class_session_is_valid(self):
        data = _study_session()
        enrol, test = cs.role_masks(data, sa.ROLE_CALIBRATION, sa.ROLE_EVALUATION)
        result = cs.evaluate(data, cs.Condition(), enrol, test,
                             required_ticks=9, test_role=sa.ROLE_EVALUATION)
        self.assertTrue(result["primary_macro_f1_valid"])
        self.assertEqual(result["trial_majority"]["adapter"]["classes_scored"],
                         len(CLASSES))

    def test_a_six_class_session_does_not_yield_a_primary_figure(self):
        """`macro_f1` leaves an unsupported class out of the mean, which raises it. A
        six-class recording therefore cannot produce a seven-class primary number."""
        data = _session()
        result = cs.evaluate(data, cs.Condition(), *_ticks(data, {1}, {2}))
        self.assertFalse(result["primary_macro_f1_valid"])
        self.assertLess(len(result["classes_with_a_scored_trial"]), len(CLASSES))


class ResultFiles(unittest.TestCase):
    def _result(self) -> dict:
        data = _study_session()
        enrol, test = cs.role_masks(data, sa.ROLE_CALIBRATION, sa.ROLE_EVALUATION)
        return cs.evaluate(data, cs.Condition(), enrol, test,
                           required_ticks=9, test_role=sa.ROLE_EVALUATION)

    def test_one_file_per_session_and_never_an_overwrite(self):
        import tempfile

        result = self._result()
        with tempfile.TemporaryDirectory() as tmp:
            written = cs.write_results([result], {"model": "m"}, tmp)
            self.assertEqual(len(written), 1)
            self.assertIn(result["session"], written[0].name)

            with self.assertRaises(SystemExit) as caught:
                cs.write_results([result], {"model": "m"}, tmp)
            self.assertIn("refusing to overwrite", str(caught.exception))

    def test_no_centroids_scores_or_windows_reach_the_file(self):
        """The research CSV is the consented data record. The analysis output must not
        become a second copy of the face signal."""
        import json as _json
        import tempfile

        result = self._result()
        with tempfile.TemporaryDirectory() as tmp:
            written = cs.write_results([result], {"model": "m"}, tmp)
            body = _json.loads(written[0].read_text(encoding="utf-8"))

        for field in ("_arrays", "_abstention"):
            self.assertNotIn(field, body)
        self.assertTrue(set(body) <= set(cs.RESULT_FIELDS) | {"model"},
                        f"unexpected fields: {sorted(set(body) - set(cs.RESULT_FIELDS))}")

    def test_the_frozen_settings_are_recorded_alongside_the_numbers(self):
        import json as _json
        import tempfile

        result = self._result()
        settings = {"tie_rule": "most_ticks -> final_tracked_tick -> class_order",
                    "epsilon": 1e-12}
        with tempfile.TemporaryDirectory() as tmp:
            written = cs.write_results([result], settings, tmp)
            body = _json.loads(written[0].read_text(encoding="utf-8"))

        self.assertEqual(body["tie_rule"], settings["tie_rule"])
        self.assertEqual(body["epsilon"], 1e-12)
        self.assertIn("replay_fidelity", body)


class EpsilonIsAGuardNotAKnob(unittest.TestCase):
    def test_it_is_inert_over_the_observed_score_range(self):
        """The smallest score seen in any recording is ~7.2e-06. A floor at 1e-12 is
        below every one of them, so it cannot act as a smoother -- which the old 1e-6
        default could, and did: sweeping it moved one session's accuracy by 0.09."""
        data = _dataset([row(1, 0, "Anger", onehot(ANGER))])
        a = cs.features(data, "logscore", 1e-12)
        b = cs.features(data, "logscore", 1e-30)
        np.testing.assert_allclose(a, b)

    def test_it_still_prevents_negative_infinity(self):
        data = _dataset([row(1, 0, "Anger", [0.0] + [1.0 / 6] * 6)])
        F = cs.features(data, "logscore", 1e-12)
        self.assertTrue(np.isfinite(F).all())


# --------------------------------------------------- VR shadow evaluation


def _shadow_session(confuse: bool = False) -> cs.Replayed:
    """A session-local-v1 fixture: 7 calibration trials, 14 evaluation trials.

    With `confuse`, the model answers Surprise whenever the subject performs Disgust --
    consistently, in both blocks -- which is the mis-mapping calibration exists for.
    """
    rows, trial = [], 0
    for role, repeats in ((sa.ROLE_CALIBRATION, 1), (sa.ROLE_EVALUATION, 2)):
        for repeat in range(1, repeats + 1):
            for index, name in enumerate(CLASSES):
                emitted = SURPRISE if (confuse and index == DISGUST) else index
                for tick in range(10):
                    rows.append(row(tick, trial, name, onehot(emitted, 0.6),
                                    block_role=role, repeat_index=repeat))
                trial += 1
    return _dataset(rows, protocol_version="session-local-v1")


def _population(peak: float = 0.6) -> "np.ndarray":
    """A seven-row population profile in the same log-score space."""
    return np.log(np.clip(np.array([onehot(i, peak) for i in range(len(CLASSES))]),
                          cs.DEFAULT_EPSILON, None))


def _shadow_masks(data):
    return cs.role_masks(data, sa.ROLE_CALIBRATION, sa.ROLE_EVALUATION)


class ShadowEvaluationLeavesPersonalOnlyAlone(unittest.TestCase):
    """The published personal-only numbers must not move because a new option exists."""

    def test_the_result_is_identical_without_the_new_options(self):
        data = _shadow_session()
        enrol, test = _shadow_masks(data)
        cond = cs.Condition()

        before = cs.evaluate(data, cond, enrol, test,
                             required_ticks=cs.TICKS_REQUIRED_PER_CENTROID,
                             test_role=sa.ROLE_EVALUATION)
        after = cs.evaluate(data, cond, enrol, test,
                            required_ticks=cs.TICKS_REQUIRED_PER_CENTROID,
                            test_role=sa.ROLE_EVALUATION,
                            population_M=None, hybrid_alpha=None)

        self.assertEqual(sorted(before), sorted(after))
        self.assertEqual(before["accuracy"], after["accuracy"])
        self.assertEqual(before["trial_majority"], after["trial_majority"])

    def test_no_shadow_key_reaches_a_personal_only_result(self):
        data = _shadow_session()
        enrol, test = _shadow_masks(data)
        result = cs.evaluate(data, cs.Condition(), enrol, test,
                             required_ticks=cs.TICKS_REQUIRED_PER_CENTROID,
                             test_role=sa.ROLE_EVALUATION)
        self.assertNotIn("shadow_deltas", result)
        self.assertNotIn("shadow_alpha", result)
        self.assertNotIn("population", result["accuracy"])
        self.assertNotIn("hybrid", result["trial_majority"])

    def test_the_personal_number_is_unchanged_by_the_population_profile(self):
        """Adding methods must not alter the adapter score itself."""
        data = _shadow_session(confuse=True)
        enrol, test = _shadow_masks(data)
        cond = cs.Condition()
        plain = cs.evaluate(data, cond, enrol, test,
                            required_ticks=cs.TICKS_REQUIRED_PER_CENTROID,
                            test_role=sa.ROLE_EVALUATION)
        shadow = cs.evaluate(data, cond, enrol, test,
                             required_ticks=cs.TICKS_REQUIRED_PER_CENTROID,
                             test_role=sa.ROLE_EVALUATION,
                             population_M=_population(), hybrid_alpha=0.4)
        self.assertEqual(plain["accuracy"]["adapter"], shadow["accuracy"]["adapter"])
        self.assertEqual(plain["trial_majority"]["adapter"],
                         shadow["trial_majority"]["adapter"])


class ShadowEvaluationMethods(unittest.TestCase):
    def shadow(self, alpha: float, confuse: bool = True) -> dict:
        data = _shadow_session(confuse=confuse)
        enrol, test = _shadow_masks(data)
        return cs.evaluate(data, cs.Condition(), enrol, test,
                           required_ticks=cs.TICKS_REQUIRED_PER_CENTROID,
                           test_role=sa.ROLE_EVALUATION,
                           population_M=_population(), hybrid_alpha=alpha)

    def test_all_four_methods_are_scored(self):
        result = self.shadow(0.4)
        for method in ("argmax_full", "adapter", "population", "hybrid"):
            self.assertIn(method, result["trial_majority"])
        for method in ("adapter", "population", "hybrid"):
            self.assertIn(method, result["accuracy"])

    def test_every_method_sees_the_same_number_of_trials(self):
        result = self.shadow(0.4)
        n = result["n_test_trials"]
        for method in ("adapter", "population", "hybrid"):
            scored = sum(sum(r.values()) for r in result["confusion"][method].values())
            self.assertEqual(scored, n, f"{method} scored a different trial count")

    def test_alpha_one_makes_hybrid_the_personal_adapter(self):
        result = self.shadow(1.0)
        self.assertEqual(result["trial_majority"]["hybrid"],
                         result["trial_majority"]["adapter"])
        self.assertEqual(result["shadow_deltas"]["hybrid_minus_personal"]["macro_f1"], 0.0)

    def test_alpha_zero_makes_hybrid_the_population_method(self):
        result = self.shadow(0.0)
        self.assertEqual(result["trial_majority"]["hybrid"],
                         result["trial_majority"]["population"])

    def test_the_two_deltas_answer_different_questions(self):
        result = self.shadow(0.4)
        deltas = result["shadow_deltas"]
        for key in ("hybrid_minus_personal", "hybrid_minus_raw",
                    "population_minus_raw", "personal_minus_raw"):
            self.assertIn(key, deltas)
        tm = result["trial_majority"]
        self.assertAlmostEqual(deltas["hybrid_minus_personal"]["macro_f1"],
                               tm["hybrid"]["macro_f1"] - tm["adapter"]["macro_f1"])
        self.assertAlmostEqual(deltas["hybrid_minus_raw"]["macro_f1"],
                               tm["hybrid"]["macro_f1"] - tm["argmax_full"]["macro_f1"])

    def test_the_alpha_is_recorded_with_the_result(self):
        self.assertEqual(self.shadow(0.4)["shadow_alpha"], 0.4)

    def test_a_personal_adapter_recovers_a_consistent_mis_mapping(self):
        """Sanity check on the fixture: Disgust reads as Surprise, enrolment fixes it."""
        result = self.shadow(1.0)
        self.assertEqual(result["per_class"]["Disgust"]["argmax_full"], 0)
        self.assertEqual(result["per_class"]["Disgust"]["adapter"],
                         result["per_class"]["Disgust"]["n"])

    def test_a_wrong_shaped_population_matrix_is_refused(self):
        data = _shadow_session()
        enrol, test = _shadow_masks(data)
        with self.assertRaises(SystemExit):
            cs.evaluate(data, cs.Condition(), enrol, test,
                        required_ticks=cs.TICKS_REQUIRED_PER_CENTROID,
                        test_role=sa.ROLE_EVALUATION,
                        population_M=np.zeros((3, 7)), hybrid_alpha=0.4)

    def test_a_wrong_dimensioned_population_matrix_is_refused(self):
        data = _shadow_session()
        enrol, test = _shadow_masks(data)
        with self.assertRaises(SystemExit):
            cs.evaluate(data, cs.Condition(), enrol, test,
                        required_ticks=cs.TICKS_REQUIRED_PER_CENTROID,
                        test_role=sa.ROLE_EVALUATION,
                        population_M=np.zeros((7, 5)), hybrid_alpha=0.4)

    def test_an_alpha_outside_the_unit_interval_is_refused(self):
        data = _shadow_session()
        enrol, test = _shadow_masks(data)
        for alpha in (-0.1, 1.5):
            with self.assertRaises(SystemExit):
                cs.evaluate(data, cs.Condition(), enrol, test,
                            required_ticks=cs.TICKS_REQUIRED_PER_CENTROID,
                            test_role=sa.ROLE_EVALUATION,
                            population_M=_population(), hybrid_alpha=alpha)


class ShadowCommandLine(unittest.TestCase):
    def test_the_two_options_must_be_given_together(self):
        with self.assertRaises(SystemExit) as caught:
            cs.main(["--population-centroids", "x.json"])
        self.assertIn("must be given together", str(caught.exception))
        with self.assertRaises(SystemExit) as caught:
            cs.main(["--hybrid-alpha", "0.4"])
        self.assertIn("must be given together", str(caught.exception))

    def test_compare_all_needs_a_population_profile(self):
        with self.assertRaises(SystemExit) as caught:
            cs.main(["--compare-all"])
        self.assertIn("--compare-all", str(caught.exception))

    def test_the_result_whitelist_names_the_shadow_fields(self):
        self.assertIn("shadow_deltas", cs.RESULT_FIELDS)
        self.assertIn("shadow_alpha", cs.RESULT_FIELDS)

    def test_no_centroid_or_score_vector_can_reach_a_result_file(self):
        for field in cs.RESULT_FIELDS:
            self.assertNotIn("centroid_vector", field)
            self.assertNotIn("frames", field)

if __name__ == "__main__":
    unittest.main(verbosity=2)
