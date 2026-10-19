#!/usr/bin/env python3
"""Tests for `score_conditions.py`.

    python Analysis/test_score_conditions.py

Synthetic sessions only: each test writes small session folders into a temporary directory, so nothing
here needs a headset, a corpus, numpy or ONNX.

The expected numbers are worked out by hand in the comments, not computed with the code under test.
What is covered: which ticks count (not the calibration block, not tick 0 unless asked), how unanswered
ticks are counted, that the calibrated decision is compared with the model on the very same ticks, and
that sessions are pooled only within a condition, a program and a protocol version.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_conditions as sc  # noqa: E402

CLASSES = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]


def tick(prompted, predicted, index, *, role="", calibrated="", status="ok", error=""):
    return {"role": role, "prompted": prompted, "predicted": predicted, "tick": index,
            "calibrated": calibrated, "status": status, "error": error}


def write_session(root: Path, name: str, rows: list[dict], *, condition: str | None = "sparsevr_nocal",
                  protocol: str = "", study: bool = False, calibrated_column: bool = False,
                  block_column: bool = True) -> Path:
    """One session folder in the layout SessionLogger writes (action-unit columns reduced to one)."""
    folder = root / name
    folder.mkdir(parents=True)

    header = ["timestamp_utc", "session_id", "trial_id"]
    if block_column:
        header += ["block_role", "repeat_index"]
    header += ["prompted_emotion", "prediction_index", "predicted_emotion", "correct", "status", "generation"]
    header += [f"score_{c}" for c in CLASSES]
    header += ["frame_0_valid", "frame_0_au_0"]
    if calibrated_column:
        header += ["calibrated_predicted_emotion"]
    header += ["backend_id", "error_code"] + [f"raw_{c}" for c in CLASSES]

    with (folder / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for i, row in enumerate(rows):
            ok = row["status"] == "ok"
            line = {"timestamp_utc": "2026-10-17T05:00:00.000Z", "session_id": name, "trial_id": i // 10,
                    "prompted_emotion": row["prompted"], "prediction_index": row["tick"],
                    "predicted_emotion": row["predicted"] if ok else "",
                    "correct": "1" if ok and row["prompted"] == row["predicted"] else "0",
                    "status": row["status"], "generation": i,
                    "frame_0_valid": "1" if ok else "", "frame_0_au_0": "0.1" if ok else "",
                    "backend_id": "x", "error_code": row["error"]}
            if block_column:
                line["block_role"] = row["role"]
                line["repeat_index"] = "1" if row["role"] else ""
            if calibrated_column:
                line["calibrated_predicted_emotion"] = row["calibrated"]
            for c in CLASSES:
                line[f"score_{c}"] = "0.142857" if ok else ""
            writer.writerow(line)

    if condition is not None:
        meta = {"session_id": name, "models": {"condition": condition, "backend_id": "x"}}
        if study:
            meta["protocol_version"] = protocol or "session-local-v2"
        (folder / "session_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return folder


def trial(prompted, predictions, *, role="", calibrated=None, start=0):
    """One trial: `predictions` is the list of predicted emotions for ticks start, start+1, ..."""
    rows = []
    for i, predicted in enumerate(predictions):
        cal = calibrated[i] if calibrated else ""
        rows.append(tick(prompted, predicted, start + i, role=role, calibrated=cal))
    return rows


def run_main(*argv) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        code = sc.main([str(a) for a in argv])
    return code, out.getvalue()


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="score-conditions-"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def groups(self, ticks=sc.DEFAULT_TICKS):
        files = [p / "predictions.csv" for p in sorted(self.root.iterdir()) if p.is_dir()]
        return sc.build_groups([sc.read_session(f) for f in files], ticks)


class WhichTicksCount(Base):
    def two_trials(self):
        # Trial A, prompted Anger: tick 0 wrong, then ticks 1-9: Anger x6, Fear x3.
        # Trial B, prompted Fear:  tick 0 wrong, then ticks 1-9: Fear x9.
        a = trial("Anger", ["Happiness"] + ["Anger"] * 6 + ["Fear"] * 3)
        b = trial("Fear", ["Happiness"] + ["Fear"] * 9, start=0)
        return a + b

    def test_accuracy_and_macro_f1_over_ticks_1_to_9(self):
        write_session(self.root, "s1", self.two_trials())
        (g,) = self.groups()

        # truth: Anger x9, Fear x9.  predicted: Anger x6, Fear x12.
        # accuracy = (6 + 9) / 18.  Anger: P 6/6, R 6/9 -> F1 0.8.  Fear: P 9/12, R 9/9 -> F1 6/7.
        m = g["model"]
        self.assertEqual(18, m["n"])
        self.assertAlmostEqual(15 / 18, m["accuracy"])
        self.assertAlmostEqual((0.8 + 6 / 7) / 2, m["macro_f1"])
        # Only Anger and Fear appear as truth or prediction, so the mean covers two classes.
        self.assertEqual(2, m["classes_scored"])

    def test_tick_zero_is_counted_only_when_asked_for(self):
        write_session(self.root, "s1", self.two_trials())
        (g,) = self.groups(ticks=None)

        # Two more ticks, both wrong (Happiness): 15 of 20.
        self.assertEqual(20, g["model"]["n"])
        self.assertAlmostEqual(15 / 20, g["model"]["accuracy"])

    def test_a_single_tick_can_be_selected(self):
        write_session(self.root, "s1", self.two_trials())
        (g,) = self.groups(ticks=(0, 0))
        self.assertEqual(2, g["model"]["n"])
        self.assertEqual(0, g["model"]["correct"])

    def test_the_calibration_block_is_never_scored(self):
        k = trial("Anger", ["Fear"] * 10, role="calibration")                 # all wrong, must not count
        d = trial("Anger", ["Anger"] * 10, role="evaluation")                 # all right
        write_session(self.root, "s1", k + d, study=True)
        (g,) = self.groups()

        self.assertEqual(9, g["ticks"])                                       # ticks 1-9 of the evaluation trial
        self.assertAlmostEqual(1.0, g["model"]["accuracy"])

    def test_the_game_has_no_block_role_and_all_its_ticks_are_test_ticks(self):
        write_session(self.root, "g1", trial("Anger", ["Anger"] * 10), study=False)
        (g,) = self.groups()
        self.assertEqual("game", g["program"])
        self.assertEqual(9, g["ticks"])

    def test_a_file_without_a_block_role_column_still_scores(self):
        write_session(self.root, "old", trial("Anger", ["Anger"] * 10), condition=None, block_column=False)
        (g,) = self.groups()
        self.assertEqual(9, g["ticks"])
        self.assertEqual(sc.NO_CONDITION, g["condition"])


class UnansweredTicks(Base):
    def test_failures_are_counted_but_not_scored(self):
        # Ticks 1-9, prompted Happiness: 6 right, 2 backend_error (Unreachable), 1 no_tracking.
        rows = ([tick("Happiness", "Happiness", i) for i in range(1, 7)]
                + [tick("Happiness", "", i, status="backend_error", error="Unreachable") for i in (7, 8)]
                + [tick("Happiness", "", 9, status="no_tracking", error="NoTracking")])
        write_session(self.root, "s1", rows, condition="jev_nocal")
        (g,) = self.groups()

        self.assertEqual(9, g["ticks"])
        self.assertEqual(6, g["answered"])
        self.assertAlmostEqual(6 / 9, g["coverage"])
        self.assertAlmostEqual(1.0, g["model"]["accuracy"])                    # on what was answered
        self.assertAlmostEqual(6 / 9, g["accuracy_failures_wrong"])            # unanswered counted wrong
        self.assertEqual({"backend_error": 2, "no_tracking": 1}, g["failures"])
        self.assertEqual({"Unreachable": 2}, g["backend_errors"])

    def test_a_group_with_no_answered_tick_has_no_accuracy_instead_of_a_zero(self):
        rows = [tick("Happiness", "", i, status="backend_error", error="Unauthorized") for i in range(1, 10)]
        write_session(self.root, "s1", rows, condition="jev_nocal")
        (g,) = self.groups()

        self.assertEqual(0, g["answered"])
        self.assertIsNone(g["model"]["accuracy"])
        self.assertIsNone(g["model"]["macro_f1"])
        self.assertEqual(0.0, g["coverage"])


class TheCalibratedDecision(Base):
    def rows(self):
        # Prompted Happiness, ticks 0-9. Model: Happiness on ticks 0-3, Sadness after.
        # Calibrated: Happiness on 1-6, Neutral on 7-8, blank on 9 (no decision), nothing on tick 0.
        predicted = ["Happiness"] * 4 + ["Sadness"] * 6
        calibrated = ["", "Happiness", "Happiness", "Happiness", "Happiness", "Happiness", "Happiness",
                      "Neutral", "Neutral", ""]
        return trial("Happiness", predicted, role="evaluation", calibrated=calibrated)

    def test_calibration_is_compared_with_the_model_on_the_same_ticks(self):
        write_session(self.root, "s1", self.rows(), condition="sparsevr_cal", study=True, calibrated_column=True)
        (g,) = self.groups()

        # The model on all nine ticks: Happiness on 1-3 -> 3/9.
        self.assertAlmostEqual(3 / 9, g["model"]["accuracy"])

        # Calibrated decisions exist on ticks 1-8 only: model right on 1-3 (3/8), calibrated right on 1-6 (6/8).
        c = g["calibration"]
        self.assertEqual(8, c["ticks"])
        self.assertAlmostEqual(3 / 8, c["model_same_ticks"]["accuracy"])
        self.assertAlmostEqual(6 / 8, c["calibrated"]["accuracy"])
        self.assertAlmostEqual(0.375, sc.delta(c["model_same_ticks"]["accuracy"], c["calibrated"]["accuracy"]))

    def test_no_calibrated_decision_means_no_calibration_block_in_the_result(self):
        write_session(self.root, "s1", trial("Anger", ["Anger"] * 10), condition="jev_nocal", study=True,
                      calibrated_column=True)
        (g,) = self.groups()
        self.assertIsNone(g["calibration"])

    def test_the_game_is_compared_too_when_its_file_has_the_calibrated_decision(self):
        # Game rows (no block_role), calibrated condition: the model is wrong, the calibration right.
        rows = trial("Anger", ["Fear"] * 10, calibrated=[""] + ["Anger"] * 9)
        write_session(self.root, "g1", rows, condition="jev_cal", study=False, calibrated_column=True)
        (g,) = self.groups()

        self.assertEqual("game", g["program"])
        self.assertEqual(9, g["calibration"]["ticks"])
        self.assertAlmostEqual(0.0, g["calibration"]["model_same_ticks"]["accuracy"])
        self.assertAlmostEqual(1.0, g["calibration"]["calibrated"]["accuracy"])

    def test_an_old_game_file_without_the_column_says_why_it_cannot_be_compared(self):
        write_session(self.root, "g1", trial("Anger", ["Anger"] * 10), condition="jev_cal", study=False)
        (g,) = self.groups()
        self.assertIsNone(g["calibration"])
        self.assertFalse(g["has_calibrated_column"])

        code, text = run_main(self.root)
        self.assertEqual(0, code)
        self.assertIn("jev_cal / game: no calibrated decisions to compare", text)
        self.assertIn("recorded before the game wrote it", text)

    def test_a_skipped_calibration_is_reported_as_such(self):
        # The column exists but every decision is blank: the player pressed B.
        write_session(self.root, "g1", trial("Anger", ["Anger"] * 10), condition="sparsevr_cal", study=False,
                      calibrated_column=True)
        code, text = run_main(self.root)
        self.assertEqual(0, code)
        self.assertIn("every calibrated decision is blank", text)


class Pooling(Base):
    def test_sessions_of_one_condition_are_pooled_tick_by_tick(self):
        write_session(self.root, "a", trial("Anger", ["Anger"] * 10), condition="jev_nocal")
        write_session(self.root, "b", trial("Anger", ["Anger"] * 5 + ["Fear"] * 5), condition="jev_nocal")
        (g,) = self.groups()

        # 9 + 9 ticks; session a is all right (9), session b right on ticks 1-4 only (4).
        self.assertEqual(2, g["sessions"])
        self.assertEqual(18, g["ticks"])
        self.assertAlmostEqual(13 / 18, g["model"]["accuracy"])

    def test_conditions_and_programs_are_kept_apart(self):
        write_session(self.root, "a", trial("Anger", ["Anger"] * 10), condition="jev_nocal")
        write_session(self.root, "b", trial("Anger", ["Fear"] * 10), condition="sparsevr_nocal")
        write_session(self.root, "c", trial("Anger", ["Anger"] * 10, role="evaluation"),
                      condition="jev_nocal", study=True)
        groups = self.groups()

        self.assertEqual([("sparsevr_nocal", "game"), ("jev_nocal", "game"), ("jev_nocal", "study")],
                         [(g["condition"], g["program"]) for g in groups])
        self.assertEqual([0.0, 1.0, 1.0], [g["model"]["accuracy"] for g in groups])

    def test_protocol_versions_are_never_pooled(self):
        rows = trial("Anger", ["Anger"] * 10, role="evaluation")
        write_session(self.root, "v1", rows, condition="sparsevr_nocal", study=True, protocol="session-local-v1")
        write_session(self.root, "v2", rows, condition="sparsevr_nocal", study=True, protocol="session-local-v2")
        groups = self.groups()

        self.assertEqual(2, len(groups))
        self.assertEqual({"session-local-v1", "session-local-v2"}, {g["protocol_version"] for g in groups})

    def test_a_session_without_metadata_is_scored_under_no_condition(self):
        write_session(self.root, "old", trial("Anger", ["Anger"] * 10), condition=None)
        (g,) = self.groups()
        self.assertEqual(sc.NO_CONDITION, g["condition"])
        self.assertAlmostEqual(1.0, g["model"]["accuracy"])


class CommandLine(Base):
    def test_a_report_is_printed_and_the_json_and_csv_are_written(self):
        write_session(self.root, "a", trial("Anger", ["Anger"] * 10), condition="jev_nocal")
        out_json, out_csv = self.root / "out.json", self.root / "out.csv"

        code, text = run_main(self.root, "--per-class", "--per-session", "--json", out_json, "--csv", out_csv)

        self.assertEqual(0, code)
        self.assertIn("jev_nocal", text)
        self.assertIn("per class: jev_nocal / game", text)
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertEqual(sc.REPORT_SCHEMA_VERSION, payload["schema"])
        self.assertEqual(1.0, payload["groups"][0]["model"]["accuracy"])
        with out_csv.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(["model"], [r["variant"] for r in rows])
        self.assertEqual("jev_nocal", rows[0]["condition"])

    def test_the_output_is_plain_ascii(self):
        write_session(self.root, "a", trial("Anger", ["Anger"] * 10), condition="jev_nocal")
        _, text = run_main(self.root, "--per-class", "--per-session")
        text.encode("ascii")                                                  # raises if anything else sneaks in

    def test_a_missing_path_is_an_error(self):
        code, _ = run_main(self.root / "nothing-here")
        self.assertEqual(2, code)

    def test_an_unreadable_file_is_skipped_not_fatal(self):
        write_session(self.root, "good", trial("Anger", ["Anger"] * 10), condition="jev_nocal")
        bad = self.root / "bad"
        bad.mkdir()
        (bad / "predictions.csv").write_text("not,a,session\n1,2,3\n", encoding="utf-8")

        code, text = run_main(self.root)

        self.assertEqual(0, code)
        self.assertIn("1 file(s) could not be read", text)

    def test_tick_ranges_are_parsed(self):
        self.assertEqual((1, 9), sc.parse_ticks("1-9"))
        self.assertEqual((3, 3), sc.parse_ticks("3"))
        self.assertIsNone(sc.parse_ticks("all"))
        for bad in ("9-1", "x", "-2"):
            with self.assertRaises(Exception):
                sc.parse_ticks(bad)

    def test_the_real_header_width_is_accepted(self):
        # The study's file has 394 columns: 5 frames x 70 action units. Build that header and read it.
        header = ["timestamp_utc", "session_id", "trial_id", "block_role", "repeat_index", "prompted_emotion",
                  "prediction_index", "predicted_emotion", "correct", "status", "generation"]
        header += [f"score_{c}" for c in CLASSES]
        header += [f"frame_{f}_valid" for f in range(5)]
        header += [f"frame_{f}_au_{a}" for f in range(5) for a in range(70)]
        header += ["calibrated_predicted_emotion", "backend_id", "error_code"] + [f"raw_{c}" for c in CLASSES]
        folder = self.root / "wide"
        folder.mkdir()
        with (folder / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            row = {h: "" for h in header}
            row.update({"block_role": "evaluation", "prompted_emotion": "Anger", "prediction_index": "3",
                        "predicted_emotion": "Anger", "status": "ok"})
            writer.writerow([row[h] for h in header])

        session = sc.read_session(folder / "predictions.csv")
        self.assertEqual(CLASSES, session["classes"])
        self.assertEqual(1, len(session["rows"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
