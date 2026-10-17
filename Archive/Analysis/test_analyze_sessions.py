#!/usr/bin/env python3
"""Tests for the four-conditions additions to `session_analysis.py` and `analyze_sessions.py`.

    python Analysis/test_analyze_sessions.py

Synthetic sessions only (the same builders as `test_score_conditions.py`). What is covered:

* the study's calibration block is left out of accuracy, F1, the confusion matrix and the trial table,
  because the calibration is fitted on it, and the report says so;
* a session with no calibration block (the game, an uncalibrated study session, an older recording) is
  scored whole, with the same numbers as before;
* the new `backend_error` status is recognised and counted, and does not raise an "unrecognised status"
  warning;
* macro-F1, the ticks-1-to-9 view and the calibrated-vs-model comparison are in the report;
* pooling keeps the block role, so a pooled report leaves the calibration block out too.

The expected numbers are worked out by hand in the comments.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_sessions as an  # noqa: E402
import session_analysis as sa  # noqa: E402
from test_score_conditions import tick, trial, write_session  # noqa: E402


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="analyze-sessions-"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def analyse(self, folder: Path) -> dict:
        return sa.analyse(sa.load_session(folder))


def study_with_calibration_block(root: Path, name="s1", **kwargs) -> Path:
    # Calibration block: Anger prompted, all ten ticks wrong. It must not count.
    # Evaluation block: Anger prompted, Anger on ticks 1-9 (tick 0 wrong), so 9 of 10 right.
    k = trial("Anger", ["Fear"] * 10, role="calibration")
    d = trial("Anger", ["Fear"] + ["Anger"] * 9, role="evaluation")
    return write_session(root, name, k + d, condition="sparsevr_cal", study=True, **kwargs)


class CalibrationBlock(Base):
    def test_accuracy_is_computed_on_the_test_trials_only(self):
        report = self.analyse(study_with_calibration_block(self.root))

        acc = report["accuracy"]
        self.assertEqual(10, acc["ticks_scored"])                       # not 20
        self.assertAlmostEqual(0.9, acc["tick_accuracy"])
        self.assertEqual(1, len(report["trials"]))                      # the one test trial
        self.assertEqual({"calibration_block_ticks_excluded": 10, "calibration_block_trials_excluded": 1},
                         report["scope"])

    def test_the_confusion_matrix_and_per_emotion_figures_leave_the_block_out(self):
        report = self.analyse(study_with_calibration_block(self.root))

        # Anger prompted on 10 test ticks: 9 Anger, 1 Fear. The calibration block's ten Fears are not here.
        self.assertEqual(9, report["confusion"]["Anger"]["Anger"])
        self.assertEqual(1, report["confusion"]["Anger"]["Fear"])
        self.assertEqual(10, report["per_prompted"]["Anger"]["ticks"])

    def test_tracking_and_timing_still_describe_the_whole_recording(self):
        report = self.analyse(study_with_calibration_block(self.root))
        self.assertEqual(20, report["tracking"]["ticks"])

    def test_a_pooled_report_leaves_the_block_out_too(self):
        folders = [study_with_calibration_block(self.root, "a"), study_with_calibration_block(self.root, "b")]
        pooled = sa.pool([sa.load_session(f) for f in folders])
        report = sa.analyse(pooled)

        self.assertEqual(20, report["accuracy"]["ticks_scored"])        # 2 x 10 test ticks
        self.assertEqual(20, report["scope"]["calibration_block_ticks_excluded"])
        self.assertAlmostEqual(0.9, report["accuracy"]["tick_accuracy"])

    def test_a_session_with_no_calibration_block_is_scored_whole(self):
        # The game: no block_role. Tick 0 wrong, ticks 1-9 right -> 9 of 10.
        folder = write_session(self.root, "g", trial("Anger", ["Fear"] + ["Anger"] * 9), condition="jev_nocal")
        report = self.analyse(folder)

        self.assertEqual(10, report["accuracy"]["ticks_scored"])
        self.assertAlmostEqual(0.9, report["accuracy"]["tick_accuracy"])
        self.assertEqual(0, report["scope"]["calibration_block_ticks_excluded"])

    def test_the_report_says_the_block_was_left_out_and_still_prints_plain_ascii(self):
        folder = study_with_calibration_block(self.root)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            an.print_report(self.analyse(folder), top_aus=3, show_trials=True)
        text = out.getvalue()

        self.assertIn("the calibration block (1 trials, 10 ticks) is left out", text)
        self.assertIn("2  (1 scored, 1 in the calibration block)", text)       # 1 test trial + 1 calibration trial
        text.encode("ascii")


class MacroF1(Base):
    def two_trials(self):
        # Prompted Anger: tick 0 Happiness, then Anger x6, Fear x3.   Prompted Fear: tick 0 Happiness, then Fear x9.
        return (trial("Anger", ["Happiness"] + ["Anger"] * 6 + ["Fear"] * 3)
                + trial("Fear", ["Happiness"] + ["Fear"] * 9))

    def test_macro_f1_over_all_ticks(self):
        report = self.analyse(write_session(self.root, "g", self.two_trials(), condition="sparsevr_nocal"))
        acc = report["accuracy"]

        # truth: Anger x10, Fear x10.  predicted: Happiness x2, Anger x6, Fear x12.  15 of 20 right.
        # Anger: tp 6, fp 0, fn 4 -> P 1, R 0.6, F1 0.75.   Fear: tp 9, fp 3, fn 1 -> P 0.75, R 0.9, F1 0.8182.
        # Happiness: predicted twice, never true -> F1 0, and it counts (a class that is predicted is scored).
        self.assertAlmostEqual(15 / 20, acc["tick_accuracy"])
        self.assertAlmostEqual((0.75 + 2 * 0.75 * 0.9 / 1.65 + 0.0) / 3, acc["macro_f1"])
        self.assertEqual(3, acc["classes_scored"])

    def test_the_ticks_1_to_9_view_matches_score_conditions(self):
        report = self.analyse(write_session(self.root, "g", self.two_trials(), condition="sparsevr_nocal"))
        window = report["accuracy"]["ticks_1_to_9"]

        # Same hand-worked numbers as test_score_conditions: 15 of 18; F1 (0.8 + 6/7) / 2 over two classes.
        self.assertEqual(18, window["ticks_scored"])
        self.assertAlmostEqual(15 / 18, window["tick_accuracy"])
        self.assertAlmostEqual((0.8 + 6 / 7) / 2, window["macro_f1"])
        self.assertEqual(2, window["classes_scored"])


class BackendErrors(Base):
    def rows(self):
        # Prompted Happiness: 6 right, 2 Jev failures, 1 lost tracking (ticks 1-9) plus a right tick 0.
        return ([tick("Happiness", "Happiness", 0)]
                + [tick("Happiness", "Happiness", i) for i in range(1, 7)]
                + [tick("Happiness", "", i, status="backend_error", error="Unreachable") for i in (7, 8)]
                + [tick("Happiness", "", 9, status="no_tracking", error="NoTracking")])

    def test_backend_error_is_a_known_status_and_is_counted(self):
        session = sa.load_session(write_session(self.root, "j", self.rows(), condition="jev_nocal"))
        report = sa.analyse(session)

        self.assertEqual([], [w for w in report["warnings"] if "unrecognised status" in w])
        self.assertEqual(2, report["tracking"]["backend_error"])
        self.assertEqual(1, report["tracking"]["no_tracking"])
        self.assertEqual(7, report["tracking"]["ok"])
        self.assertEqual("Unreachable", [r for r in session.rows if r.status == sa.STATUS_BACKEND_ERROR][0].error_code)

    def test_unanswered_ticks_are_not_scored_as_wrong_predictions(self):
        report = self.analyse(write_session(self.root, "j", self.rows(), condition="jev_nocal"))

        self.assertEqual(7, report["accuracy"]["ticks_scored"])
        self.assertAlmostEqual(1.0, report["accuracy"]["tick_accuracy"])

    def test_the_overview_shows_backend_errors_only_when_there_are_some(self):
        with_errors = write_session(self.root, "j", self.rows(), condition="jev_nocal")
        clean = write_session(self.root, "c", trial("Anger", ["Anger"] * 10), condition="jev_nocal")

        def overview(folder):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                an._print_overview(self.analyse(folder))
            return out.getvalue()

        self.assertIn("backend_error 2", overview(with_errors))
        self.assertNotIn("backend_error", overview(clean))


class CalibrationComparison(Base):
    def test_the_calibrated_decision_is_compared_with_the_model_on_the_same_ticks(self):
        # Prompted Happiness, ticks 0-9. Model right on 0-3. Calibrated right on 1-6, wrong on 7-8, none on 0 and 9.
        predicted = ["Happiness"] * 4 + ["Sadness"] * 6
        calibrated = ["", "Happiness", "Happiness", "Happiness", "Happiness", "Happiness", "Happiness",
                      "Neutral", "Neutral", ""]
        rows = trial("Happiness", predicted, role="evaluation", calibrated=calibrated)
        folder = write_session(self.root, "s", rows, condition="sparsevr_cal", study=True, calibrated_column=True)

        c = self.analyse(folder)["calibration"]

        # 8 ticks have a decision (1-8). The model is right on 1-3 -> 3/8; the calibration on 1-6 -> 6/8.
        self.assertEqual(8, c["ticks"])
        self.assertAlmostEqual(3 / 8, c["model_same_ticks"]["accuracy"])
        self.assertAlmostEqual(6 / 8, c["calibrated"]["accuracy"])

    def test_no_calibrated_decision_means_no_comparison(self):
        folder = write_session(self.root, "s", trial("Anger", ["Anger"] * 10), condition="jev_nocal", study=True,
                               calibrated_column=True)
        self.assertIsNone(self.analyse(folder)["calibration"])

    def test_the_comparison_is_printed(self):
        rows = trial("Happiness", ["Sadness"] * 10, role="evaluation", calibrated=["Happiness"] * 10)
        folder = write_session(self.root, "s", rows, condition="sparsevr_cal", study=True, calibrated_column=True)
        report = self.analyse(folder)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            an._print_accuracy(report, report["schema"]["classes"])

        self.assertIn("calibration (the headset's decision vs the model", out.getvalue())
        self.assertIn("calibrated  accuracy 100.0%", out.getvalue())


class CommandLine(Base):
    def test_the_whole_report_runs_on_a_calibrated_session(self):
        study_with_calibration_block(self.root)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = an.main([str(self.root)])

        self.assertEqual(0, code)
        self.assertIn("macro-F1", out.getvalue())
        out.getvalue().encode("ascii")


if __name__ == "__main__":
    unittest.main(verbosity=2)
