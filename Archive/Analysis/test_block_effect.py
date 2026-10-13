#!/usr/bin/env python3
"""Tests for `block_effect.py`.

Stdlib `unittest`, no third-party dependency at all -- the script under test has none.
The integration tests skip themselves when the pulled sessions are not on the machine.

What is worth guarding here is the splitting, not the arithmetic. A block comparison that
puts a trial in the wrong block, or that averages the resting prompt into the activity
meter, produces a number that looks perfectly reasonable and is wrong in the direction
that matters -- it would make a collapsed session look usable. So: trials land in the
block their rows claim, repeats merge without double counting, Neutral stays out of the
activity meter, and a recording with no blocks is refused rather than reported as one.

    python Analysis/test_block_effect.py
"""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import block_effect as be    # noqa: E402
import session_analysis as sa    # noqa: E402

CLASSES = ("Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise")
FRAMES = 5
FEATURES = 70

SESSIONS = Path(__file__).resolve().parents[1] / "Logs" / "quest-sessions" / "sessions"


def row(trial_id: int, role: str, repeat: int, prompted: str, index: int,
        predicted: str, activity: float = 1.0) -> sa.Row:
    """One tick. `activity` is spread over the newest frame, so a row's contribution to
    the activity meter is exactly that number."""
    scores = [1.0 if c == predicted else 0.0 for c in CLASSES]
    blank = [0.0] * FEATURES
    newest = [activity / FEATURES] * FEATURES
    return sa.Row(
        timestamp=None, trial_id=trial_id, prompted=prompted, prediction_index=index,
        predicted=predicted, logged_correct=predicted == prompted, status=sa.STATUS_OK,
        generation=index, scores=scores,
        frames=[blank] * (FRAMES - 1) + [newest],
        block_role=role, repeat_index=repeat,
    )


def session(rows: list[sa.Row], metadata: dict | None = None) -> sa.Session:
    return sa.Session(
        path=Path("synthetic") / "predictions.csv",
        session_id="synthetic",
        schema=sa.Schema(classes=CLASSES, window_frames=FRAMES, features_per_frame=FEATURES),
        rows=rows,
        metadata=metadata,
    )


def trial(trial_id: int, role: str, repeat: int, prompted: str, predicted: str,
          ticks: int = 10, activity: float = 1.0) -> list[sa.Row]:
    return [row(trial_id, role, repeat, prompted, i, predicted, activity)
            for i in range(ticks)]


class TestSplitting(unittest.TestCase):
    """The part that would be silently wrong: which rows land where."""

    def test_trials_land_in_the_block_their_rows_name(self):
        rows = (trial(0, sa.ROLE_CALIBRATION, 1, "Anger", "Anger")
                + trial(1, sa.ROLE_EVALUATION, 1, "Anger", "Neutral"))
        blocks = be.collect(session(rows))
        by_label = {b.label: b for b in blocks}
        self.assertEqual(by_label["calibration r1"].tick_accuracy, 1.0)
        self.assertEqual(by_label["evaluation r1"].tick_accuracy, 0.0)

    def test_repeats_merge_without_double_counting(self):
        rows = (trial(0, sa.ROLE_CALIBRATION, 1, "Anger", "Anger")
                + trial(1, sa.ROLE_EVALUATION, 1, "Anger", "Anger")
                + trial(2, sa.ROLE_EVALUATION, 2, "Anger", "Neutral"))
        blocks = be.collect(session(rows))
        merged = be.pick(blocks, sa.ROLE_EVALUATION)
        self.assertEqual(merged.label, "evaluation (all)")
        self.assertEqual(len(merged.trials), 2)
        self.assertEqual(merged.ticks, 20)
        self.assertEqual(merged.tick_accuracy, 0.5)
        # The per-repeat rows are still there, and the merge is not one of them.
        self.assertEqual(sum(len(b.trials) for b in blocks if b.repeat is not None), 3)

    def test_single_repeat_role_has_no_merged_row(self):
        rows = (trial(0, sa.ROLE_CALIBRATION, 1, "Anger", "Anger")
                + trial(1, sa.ROLE_EVALUATION, 1, "Anger", "Anger"))
        labels = [b.label for b in be.collect(session(rows))]
        self.assertEqual(labels, ["calibration r1", "evaluation r1"])

    def test_calibration_is_ordered_before_evaluation(self):
        rows = (trial(0, sa.ROLE_EVALUATION, 2, "Anger", "Anger")
                + trial(1, sa.ROLE_EVALUATION, 1, "Anger", "Anger")
                + trial(2, sa.ROLE_CALIBRATION, 1, "Anger", "Anger"))
        roles = [b.role for b in be.collect(session(rows))]
        self.assertEqual(roles[0], sa.ROLE_CALIBRATION)

    def test_a_recording_without_blocks_is_refused(self):
        plain = sa.Row(
            timestamp=None, trial_id=0, prompted="Anger", prediction_index=0,
            predicted="Anger", logged_correct=True, status=sa.STATUS_OK, generation=0,
            scores=[1.0] + [0.0] * 6, frames=[[0.0] * FEATURES] * FRAMES,
        )
        with self.assertRaises(be.BlockError):
            be.collect(session([plain]))

    def test_untracked_ticks_are_excluded(self):
        rows = trial(0, sa.ROLE_CALIBRATION, 1, "Anger", "Anger")
        rows[0].status = sa.STATUS_NO_TRACKING
        block = be.collect(session(rows))[0]
        self.assertEqual(block.ticks, 9)


class TestTickFilter(unittest.TestCase):

    def test_ticks_filter_selects_the_named_positions(self):
        rows = trial(0, sa.ROLE_CALIBRATION, 1, "Anger", "Anger")
        block = be.collect(session(rows), be.parse_ticks("1-9"))[0]
        self.assertEqual(block.ticks, 9)

    def test_parse_ticks_forms(self):
        self.assertIsNone(be.parse_ticks(""))
        self.assertEqual(be.parse_ticks("1-3"), {1, 2, 3})
        self.assertEqual(be.parse_ticks("0,4"), {0, 4})
        self.assertEqual(be.parse_ticks("1-2,5"), {1, 2, 5})

    def test_filtering_everything_away_is_an_error(self):
        rows = trial(0, sa.ROLE_CALIBRATION, 1, "Anger", "Anger", ticks=2)
        with self.assertRaises(be.BlockError):
            be.collect(session(rows), {50})


class TestActivityMeter(unittest.TestCase):
    """Sitting still is correct for Neutral, so it must not count as not performing."""

    def test_resting_prompt_is_left_out(self):
        rows = (trial(0, sa.ROLE_CALIBRATION, 1, "Anger", "Anger", activity=4.0)
                + trial(1, sa.ROLE_CALIBRATION, 1, "Neutral", "Neutral", activity=0.0))
        block = be.collect(session(rows))[0]
        self.assertAlmostEqual(block.activity, 4.0)
        # ... but its ticks still count towards accuracy.
        self.assertEqual(block.ticks, 20)

    def test_activity_is_the_newest_frame_only(self):
        rows = trial(0, sa.ROLE_CALIBRATION, 1, "Anger", "Anger", activity=2.5)
        block = be.collect(session(rows))[0]
        self.assertAlmostEqual(block.activity, 2.5)

    def test_ratio_reflects_a_weaker_second_block(self):
        rows = (trial(0, sa.ROLE_CALIBRATION, 1, "Anger", "Anger", activity=4.0)
                + trial(1, sa.ROLE_EVALUATION, 1, "Anger", "Neutral", activity=1.0))
        blocks = be.collect(session(rows))
        ratio = be._ratio(be.pick(blocks, sa.ROLE_EVALUATION).activity,
                          be.pick(blocks, sa.ROLE_CALIBRATION).activity)
        self.assertAlmostEqual(ratio, 0.25)


class TestTrialRollup(unittest.TestCase):

    def test_majority_decides_a_trial(self):
        rows = [row(0, sa.ROLE_CALIBRATION, 1, "Anger", i,
                    "Anger" if i < 6 else "Neutral") for i in range(10)]
        block = be.collect(session(rows))[0]
        self.assertEqual(block.trial_hits, 1)
        self.assertEqual(block.tick_accuracy, 0.6)

    def test_a_minority_of_correct_ticks_is_not_a_hit(self):
        rows = [row(0, sa.ROLE_CALIBRATION, 1, "Anger", i,
                    "Anger" if i < 4 else "Neutral") for i in range(10)]
        block = be.collect(session(rows))[0]
        self.assertEqual(block.trial_hits, 0)
        self.assertEqual(block.tick_accuracy, 0.4)


class TestFisher(unittest.TestCase):
    """Checked against values that can be worked out by hand, because a wrong p here
    would be quoted in a report."""

    def test_a_perfect_separation_is_small(self):
        # 7/7 against 0/14 cannot happen by relabelling; p is the two-tail extreme.
        p = be.fisher_exact(7, 0, 0, 14)
        self.assertLess(p, 0.001)

    def test_no_difference_is_one(self):
        self.assertAlmostEqual(be.fisher_exact(5, 5, 5, 5), 1.0)

    def test_symmetric_in_the_rows(self):
        self.assertAlmostEqual(be.fisher_exact(5, 2, 3, 11), be.fisher_exact(3, 11, 5, 2))

    def test_degenerate_tables_give_none(self):
        self.assertIsNone(be.fisher_exact(0, 0, 0, 0))
        self.assertIsNone(be.fisher_exact(7, 0, 14, 0))   # every trial a hit

    def test_known_value(self):
        # 2x2 [[1,9],[11,3]], worked out from the hypergeometric by hand rather than
        # copied from a run of this code: n=24, row total 10, column total 12, so
        # C(24,12) = 2704156 tables. The observed cell (1) and the three others at or
        # below its probability are k = 0, 1, 9, 10, with numerators 91, 3640, 3640, 91.
        self.assertAlmostEqual(be.fisher_exact(1, 9, 11, 3), 7462 / 2704156, places=12)


class TestReading(unittest.TestCase):
    """The verdict paragraph is the output anyone will actually act on."""

    def _reading(self, calibration_activity: float, evaluation_activity: float,
                 evaluation_predicted: str) -> str:
        rows = []
        for i in range(7):
            rows += trial(i, sa.ROLE_CALIBRATION, 1, "Anger", "Anger",
                          activity=calibration_activity)
        for i in range(7):
            rows += trial(7 + i, sa.ROLE_EVALUATION, 1, "Anger", evaluation_predicted,
                          activity=evaluation_activity)
        blocks = be.collect(session(rows))
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            be._print_reading(be.pick(blocks, sa.ROLE_CALIBRATION),
                              be.pick(blocks, sa.ROLE_EVALUATION))
        return buffer.getvalue()

    def test_a_quiet_second_block_is_blamed_on_the_participant(self):
        text = self._reading(4.0, 1.0, "Neutral")
        self.assertIn("the face moved", text)
        self.assertIn("cannot separate", text)

    def test_an_equally_active_second_block_is_blamed_on_the_model(self):
        text = self._reading(4.0, 4.0, "Neutral")
        self.assertIn("kept moving about as much", text)
        self.assertNotIn("cannot separate", text)

    def test_blocks_that_agree_are_reported_as_comparable(self):
        text = self._reading(4.0, 4.0, "Anger")
        self.assertIn("Accuracy holds", text)


class TestPulledSessions(unittest.TestCase):
    """Integration: run against whatever is actually on this machine."""

    @staticmethod
    def sessions() -> list[Path]:
        if not SESSIONS.is_dir():
            return []
        return sorted(p / "predictions.csv" for p in SESSIONS.iterdir()
                      if (p / "predictions.csv").is_file())

    def test_every_session_local_recording_splits(self):
        found = 0
        for path in self.sessions():
            loaded = sa.load_session(path)
            if (loaded.metadata or {}).get("protocol_version") != "session-local-v1":
                continue
            found += 1
            blocks = be.collect(loaded)
            calibration = be.pick(blocks, sa.ROLE_CALIBRATION)
            evaluation = be.pick(blocks, sa.ROLE_EVALUATION)
            self.assertIsNotNone(calibration, path)
            self.assertIsNotNone(evaluation, path)
            # The protocol is one calibration pass and two evaluation passes.
            self.assertEqual(len(calibration.trials), 7, path)
            self.assertEqual(len(evaluation.trials), 14, path)
            # Every trial is accounted for exactly once across the per-repeat blocks.
            per_repeat = [b for b in blocks if b.repeat is not None]
            self.assertEqual(sum(len(b.trials) for b in per_repeat), 21, path)
        if not found:
            self.skipTest("no session-local-v1 recordings pulled to this machine")

    def test_evaluation_block_matches_the_calibration_study_baseline(self):
        """Both tools score the deployed argmax on the same held-out ticks, so their
        numbers have to agree. They are computed by different code from the same file,
        which is the only reason this is worth asserting."""
        path = SESSIONS / "session_20260917_133343_759c71" / "predictions.csv"
        if not path.is_file():
            self.skipTest(f"{path.name} not pulled to this machine")
        blocks = be.collect(sa.load_session(path), be.parse_ticks("1-9"))
        evaluation = be.pick(blocks, sa.ROLE_EVALUATION)
        # calibration_study.py reports 0.222 for `argmax, all classes (deployed)`.
        self.assertAlmostEqual(evaluation.tick_accuracy, 0.222, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
