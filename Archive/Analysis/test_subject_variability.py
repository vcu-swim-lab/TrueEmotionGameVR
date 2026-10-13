#!/usr/bin/env python3
"""Unit tests for au_names.py, acting_dataset.py and subject_variability_study.py.

    python Analysis/test_subject_variability.py

Stdlib `unittest`, numpy only. Tests that need the corpus or the Meta SDK on disk skip
themselves when those are absent, so this runs on a machine that has neither.

Most of these guard a specific way the analysis could be silently wrong rather than merely
broken: a channel mapping that is off by one, a variance decomposition that resamples frames
instead of participants, an estimand that fills a missing class with a zero. Each of those
produces plausible numbers, which is what makes them worth a test.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

import acting_dataset as ad
import au_names
import subject_variability_study as svs
import model_performance_study as mps
import teacher_preprocessing as tp

SDK_ENUM = Path(r'C:\Users\alime\vcu-swim-lab\TrueEmotionGameVR\Library\PackageCache'
                r'\com.meta.xr.sdk.core\Scripts\OVRPlugin.cs')


class ChannelNames(unittest.TestCase):
    def test_seventy_unique_names(self):
        self.assertEqual(au_names.CHANNEL_COUNT, 70)
        self.assertEqual(len(set(au_names.NAMES)), 70)

    def test_index_round_trips(self):
        for i, name in enumerate(au_names.NAMES):
            self.assertEqual(au_names.INDEX[name], i)
            self.assertEqual(au_names.name(i), name)

    def test_regions_partition_every_channel_exactly_once(self):
        seen = [i for group in au_names.REGIONS.values() for i in group]
        self.assertEqual(sorted(seen), list(range(70)),
                         'regions must cover all 70 channels without overlap')

    def test_expression_channels_exclude_gaze_and_tongue(self):
        expression = set(au_names.EXPRESSION_INDICES)
        self.assertFalse(expression & set(au_names.REGIONS['eye_gaze']))
        self.assertFalse(expression & set(au_names.REGIONS['tongue']))
        self.assertEqual(len(expression), 70 - 8 - 7)

    def test_bilateral_pairs_are_left_right_and_disjoint(self):
        used = []
        for left, right in au_names.BILATERAL_PAIRS:
            self.assertNotEqual(left, right)
            used += [left, right]
        self.assertEqual(len(used), len(set(used)), 'a channel is in two pairs')

    def test_tongue_indices_are_the_v1_gap(self):
        """EmojiHeroVR's 63 channels are 0-62; 63-69 are unobserved there, not zero."""
        self.assertEqual(au_names.TONGUE_INDICES, tuple(range(63, 70)))
        self.assertEqual(au_names.V1_CHANNEL_COUNT, 63)
        for i in au_names.TONGUE_INDICES:
            self.assertIn('Tongue', au_names.NAMES[i])

    def test_names_match_the_sdk_enum(self):
        """The check that licenses naming the live logs' numeric AU indices."""
        if not SDK_ENUM.is_file():
            self.skipTest('Meta XR SDK not on this machine')
        self.assertEqual(au_names.names_from_sdk_enum(SDK_ENUM), au_names.NAMES)

    def test_names_match_the_corpus_header(self):
        try:
            root = ad.find_corpus_root()
        except ad.CorpusError:
            self.skipTest('ActingRecordings not on this machine')
        header = next(iter(sorted(root.glob('*/*/acting.csv'))), None)
        if header is None:
            self.skipTest('no acting.csv found')
        self.assertEqual(au_names.names_from_csv_header(header), au_names.NAMES)


class MetadataFiles(unittest.TestCase):
    def test_exclusions_parse_and_are_unique(self):
        exclusions = ad.load_exclusions()
        self.assertEqual(len(exclusions), 29)
        for (participant, emotion) in exclusions:
            self.assertIn(emotion, ad.EMOTIONS)
            self.assertRegex(participant, r'^\d{8}-\d{4}$')

    def test_one_participant_is_excluded_on_every_emotion(self):
        """P41 loses all seven, which is why E2 has no full-coverage subject."""
        exclusions = ad.load_exclusions()
        lost = {e for (p, e) in exclusions if p == 'P41'}
        self.assertEqual(lost, set(ad.EMOTIONS))

    def test_exclusion_reasons_are_categories_not_verbatim_reports(self):
        """The participant's own words must not reach version control.

        `acted_exclusions_derived.csv` replaces each verbatim self-report with one of two
        categories. If a future edit reintroduced the raw text this would fail, which is
        the point: the raw file is deliberately not tracked.
        """
        reasons = set(ad.load_exclusions().values())
        self.assertEqual(reasons, {'reported_different_prompted_class',
                                   'reported_affect_outside_class_set'})

    def test_a_missing_exclusions_file_is_not_an_error(self):
        """The primary unfiltered analysis must run without the optional file.

        The exclusions feed one clearly labelled sensitivity analysis. Raising here would
        make an optional input mandatory and stop the headline analysis.
        """
        self.assertEqual(ad.load_exclusions(Path('does-not-exist.csv')), {})

    def test_splits_cover_the_corpus_and_use_known_labels(self):
        splits = ad.load_splits()
        self.assertEqual(len(splits), 15)
        for participant, (split, source) in splits.items():
            self.assertIn(split, ('train', 'validation', 'test'))
            self.assertIn(source, ('named', 'assumed'))

    def test_held_out_participants_are_the_named_ones(self):
        splits = ad.load_splits()
        held = {p for p, (s, _) in splits.items() if s in ('validation', 'test')}
        self.assertEqual(held, {'P41', 'P46',
                                'P40', 'P48', 'P49'})
        for participant in held:
            self.assertEqual(splits[participant][1], 'named',
                             'a held-out participant must not be an assumption')


class SyntheticCorpus(unittest.TestCase):
    """A corpus built on disk, so the loader is exercised rather than bypassed."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'ActingRecordings'
        self.data = Path(self.tmp.name) / 'data'
        self.data.mkdir(parents=True)

        for participant in ('P1', 'P2'):
            for emotion in ('Anger', 'Neutral'):
                self._write(participant, emotion)

        (self.data / 'exclusions.csv').write_text(
            '# comment line that must be skipped\n'
            'participant_id,emotion,excluded,reason_category\n'
            'P2,Anger,1,reported_different_prompted_class\n', encoding='utf-8')
        (self.data / 'splits.csv').write_text(
            '# comment\nparticipant,split,source\n'
            'P1,train,assumed\nP2,test,named\n', encoding='utf-8')

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, participant: str, emotion: str, frames: int = 20):
        folder = self.root / participant / emotion
        folder.mkdir(parents=True)
        header = (['Timestamp', 'RealtimeSeconds', 'Emotion']
                  + list(ad.QUALITY_COLUMNS) + list(au_names.NAMES))
        with (folder / 'acting.csv').open('w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for i in range(frames):
                quality = [1] * len(ad.QUALITY_COLUMNS)
                au = [0.0] * 70
                # A single ramping channel, so peak and timing are known exactly.
                au[0] = i / (frames - 1)
                writer.writerow([f't{i}', i * 0.1, emotion] + quality + au)

    def _load(self):
        return ad.load_corpus(self.root,
                              exclusions_file=self.data / 'exclusions.csv',
                              splits_file=self.data / 'splits.csv')

    def test_loads_every_recording_with_split_and_exclusion(self):
        corpus = self._load()
        self.assertEqual(len(corpus.recordings), 4)
        self.assertEqual(corpus.participants, ['P1', 'P2'])
        excluded = [r for r in corpus.recordings if r.excluded]
        self.assertEqual([(r.participant, r.emotion) for r in excluded], [('P2', 'Anger')])
        self.assertEqual({r.split for r in corpus.recordings}, {'train', 'test'})

    def test_estimands_differ_by_exactly_the_exclusion(self):
        corpus = self._load()
        self.assertEqual(len(corpus.subset()), 4)
        self.assertEqual(len(corpus.subset(self_report_consistent=True)), 3)

    def test_e2_omits_a_class_rather_than_zero_filling_it(self):
        """The forbidden move: a missing class must be absent, never scored as zero."""
        corpus = self._load()
        coverage = corpus.coverage(self_report_consistent=True)
        self.assertEqual(coverage['P2'], {'Neutral'})
        self.assertNotIn('Anger', coverage['P2'])

        profiles = svs.build_profiles(corpus, self_report_consistent=True)
        self.assertNotIn(('P2', 'Anger'), profiles)
        self.assertIn(('P1', 'Anger'), profiles)

    def test_sample_rate_is_frames_over_elapsed_time(self):
        """The two rate definitions disagree, and only one of them is the rate.

        Twenty frames spaced 0.1 s apart span 1.9 s, because there are nineteen gaps. So
        frames/elapsed is 10.53 while 1/median-gap is 10.0. On the real corpus that gap is
        much wider -- 60.2 against 70.9 Hz, because the capture is bursty -- and reporting
        the median made the data look inconsistent with the pipeline that reads it.
        """
        corpus = self._load()
        recording = corpus.recordings[0]
        self.assertAlmostEqual(recording.duration, 1.9, places=6)
        self.assertAlmostEqual(recording.sample_rate, 20 / 1.9, places=6)
        self.assertAlmostEqual(recording.median_interval_rate, 10.0, places=6)

    def test_profile_peak_and_timing_on_a_known_ramp(self):
        corpus = self._load()
        recording = next(r for r in corpus.recordings
                         if r.participant == 'P1' and r.emotion == 'Anger')
        prof = svs.profile(recording, None)
        self.assertTrue(prof['usable'])
        self.assertAlmostEqual(float(prof['peak'][0]), 1.0, places=6)
        # A ramp peaks on its last frame: 19 steps of 0.1 s.
        self.assertAlmostEqual(float(prof['time_to_peak'][0]), 1.9, places=6)
        # Rising 1.0 over 1.9 s in equal steps at 10 Hz.
        self.assertAlmostEqual(float(prof['max_velocity'][0]), 1 / 19 / 0.1, places=5)
        self.assertEqual(int(prof['flat_channels']), 69)

    def test_mismatched_emotion_column_is_refused(self):
        corpus_dir = self.root / 'P1' / 'Anger'
        with (corpus_dir / 'acting.csv').open(encoding='utf-8') as fh:
            rows = list(csv.reader(fh))
        rows[1][2] = 'Disgust'
        with (corpus_dir / 'acting.csv').open('w', newline='', encoding='utf-8') as fh:
            csv.writer(fh).writerows(rows)
        with self.assertRaises(ad.CorpusError):
            self._load()

    def test_face_invalid_frames_are_dropped(self):
        corpus_dir = self.root / 'P1' / 'Neutral'
        with (corpus_dir / 'acting.csv').open(encoding='utf-8') as fh:
            rows = list(csv.reader(fh))
        valid_col = rows[0].index('FaceValid')
        for row in rows[1:6]:
            row[valid_col] = '0'
        with (corpus_dir / 'acting.csv').open('w', newline='', encoding='utf-8') as fh:
            csv.writer(fh).writerows(rows)

        recording = next(r for r in self._load().recordings
                         if r.participant == 'P1' and r.emotion == 'Neutral')
        self.assertEqual(recording.frames, 20)
        self.assertEqual(len(recording.valid_au), 15)
        self.assertAlmostEqual(recording.tracking_rate, 0.75)


class VarianceDecomposition(unittest.TestCase):
    def _profiles(self, grid: np.ndarray, participants, emotions) -> dict:
        out = {}
        for i, p in enumerate(participants):
            for j, e in enumerate(emotions):
                if not np.isnan(grid[i, j]).all():
                    vector = np.zeros(au_names.CHANNEL_COUNT)
                    vector[:grid.shape[2]] = grid[i, j]
                    out[(p, e)] = {'peak': vector}
        return out

    def test_identity_dominated_channel_scores_below_one(self):
        """Channel varies by person, not by emotion -> ratio well under 1."""
        grid = np.array([[[0.1], [0.1]], [[0.9], [0.9]]])
        var = svs.variance_comparison(
            self._profiles(grid, ['P1', 'P2'], ['Anger', 'Fear']),
            rng=np.random.default_rng(0), bootstrap=50)
        self.assertLess(var['ratio'][0], 0.01)

    def test_emotion_dominated_channel_scores_above_one(self):
        grid = np.array([[[0.1], [0.9]], [[0.1], [0.9]]])
        var = svs.variance_comparison(
            self._profiles(grid, ['P1', 'P2'], ['Anger', 'Fear']),
            rng=np.random.default_rng(0), bootstrap=50)
        self.assertGreater(var['ratio'][0], 100)

    def test_bootstrap_resamples_participants(self):
        """Resampling frames instead would give an interval the data has not earned."""
        grid = np.array([[[0.1], [0.9]], [[0.2], [0.8]], [[0.15], [0.85]]])
        var = svs.variance_comparison(
            self._profiles(grid, ['P1', 'P2', 'P3'], ['Anger', 'Fear']),
            rng=np.random.default_rng(0), bootstrap=200)
        self.assertEqual(var['resampled'], 'participants')
        lo, hi = var['ratio_ci']
        self.assertLessEqual(lo[0], var['ratio'][0])
        self.assertGreaterEqual(hi[0], var['ratio'][0])

    def test_missing_recordings_do_not_become_zeros(self):
        """A participant missing an emotion must shrink the sample, not add a 0."""
        full = np.array([[[0.5], [0.5]], [[0.5], [0.5]]])
        holed = full.copy()
        holed[1, 1] = np.nan
        a = svs.variance_comparison(self._profiles(full, ['P1', 'P2'], ['A', 'F']),
                                    rng=np.random.default_rng(0), bootstrap=20)
        b = svs.variance_comparison(self._profiles(holed, ['P1', 'P2'], ['A', 'F']),
                                    rng=np.random.default_rng(0), bootstrap=20)
        self.assertEqual(int(a['coverage'][0]), 4)
        self.assertEqual(int(b['coverage'][0]), 3)
        # Every remaining value is identical, so no variance appears from the hole.
        self.assertAlmostEqual(float(b['between_participant'][0]), 0.0, places=9)


class Similarity(unittest.TestCase):
    def test_identical_profiles_score_one(self):
        vector = np.zeros(au_names.CHANNEL_COUNT)
        vector[au_names.INDEX['JawDrop']] = 0.8
        profiles = {('P1', 'Fear'): {'peak': vector}, ('P2', 'Fear'): {'peak': vector}}
        self.assertAlmostEqual(
            svs.profile_similarity(profiles)['Fear']['mean_cosine'], 1.0, places=6)

    def test_disjoint_profiles_score_zero(self):
        a = np.zeros(au_names.CHANNEL_COUNT)
        b = np.zeros(au_names.CHANNEL_COUNT)
        a[au_names.INDEX['JawDrop']] = 0.8
        b[au_names.INDEX['BrowLowererL']] = 0.8
        profiles = {('P1', 'Fear'): {'peak': a}, ('P2', 'Fear'): {'peak': b}}
        self.assertAlmostEqual(
            svs.profile_similarity(profiles)['Fear']['mean_cosine'], 0.0, places=6)

    def test_gaze_channels_are_not_counted(self):
        """Looking around a scene is not expressiveness and must not create similarity."""
        a = np.zeros(au_names.CHANNEL_COUNT)
        b = np.zeros(au_names.CHANNEL_COUNT)
        a[au_names.INDEX['JawDrop']] = 0.8
        b[au_names.INDEX['JawDrop']] = 0.8
        b[au_names.REGIONS['eye_gaze'][0]] = 5.0
        profiles = {('P1', 'Fear'): {'peak': a}, ('P2', 'Fear'): {'peak': b}}
        self.assertAlmostEqual(
            svs.profile_similarity(profiles)['Fear']['mean_cosine'], 1.0, places=6)


class TeacherPreprocessing(unittest.TestCase):
    """The reproduction of the training pipeline, checked against hand-worked examples."""

    def test_downsample_averages_every_six_rows_by_index(self):
        au = np.arange(12, dtype=np.float64).reshape(12, 1)
        out = tp.downsample(au)
        # rows 0-5 mean 2.5, rows 6-11 mean 8.5
        np.testing.assert_allclose(out, [[2.5], [8.5]])

    def test_downsample_drops_the_trailing_remainder(self):
        au = np.arange(14, dtype=np.float64).reshape(14, 1)
        self.assertEqual(len(tp.downsample(au)), 2)

    def test_downsample_ignores_timestamps(self):
        """Index-based, as the notebook is. A different capture rate silently shifts the
        output frequency, and reproducing the model's input means reproducing that."""
        au = np.ones((12, 1))
        self.assertEqual(len(tp.downsample(au)), 2)

    def test_clean_forward_then_back_fills(self):
        au = np.array([[np.nan], [1.0], [np.nan], [np.inf], [2.0]])
        np.testing.assert_allclose(tp.clean(au), [[1.0], [1.0], [1.0], [1.0], [2.0]])

    def test_clean_zeroes_an_all_missing_channel(self):
        np.testing.assert_allclose(tp.clean(np.full((3, 1), np.nan)), [[0.0], [0.0], [0.0]])

    def test_activity_is_any_channel_over_threshold(self):
        au = np.array([[0.1, 0.3], [0.1, 0.1]])
        np.testing.assert_array_equal(tp.activity_labels(au, 'Anger'), [True, False])

    def test_neutral_activity_is_inverted(self):
        """Neutral is the absence of movement, so a fidgeting Neutral trial yields nothing."""
        au = np.array([[0.1, 0.3], [0.1, 0.1]])
        np.testing.assert_array_equal(tp.activity_labels(au, 'Neutral'), [False, True])

    def test_gated_windows_need_three_consecutive_active_frames(self):
        # Ten frames, active at 0-2 only -> one window at i=0, then i jumps to 5 and finds
        # nothing, so exactly one window.
        au = np.zeros((10, 1))
        au[0:3] = 0.9
        windows = tp.make_windows(au, 'Anger')
        self.assertEqual(windows.shape, (1, 5, 1))
        np.testing.assert_allclose(windows[0, :3, 0], 0.9)

    def test_two_isolated_active_frames_yield_nothing(self):
        au = np.zeros((10, 1))
        au[0] = au[2] = 0.9      # active but not three in a row
        self.assertEqual(len(tp.make_windows(au, 'Anger')), 0)

    def test_accepted_windows_do_not_overlap(self):
        au = np.full((20, 1), 0.9)
        windows = tp.make_windows(au, 'Anger')
        # i advances by 5 on every accept, and the loop stops at len-5.
        self.assertEqual(len(windows), 3)

    def test_ungated_windows_are_every_non_overlapping_block(self):
        au = np.zeros((20, 1))
        self.assertEqual(len(tp.make_windows(au, 'Anger', gate=False)), 4)
        self.assertEqual(len(tp.make_windows(au, 'Anger', gate=True)), 0)

    def test_short_recording_yields_no_windows(self):
        self.assertEqual(len(tp.make_windows(np.ones((4, 1)), 'Anger')), 0)

    def test_teacher_windows_use_every_frame_not_just_valid_ones(self):
        """The pipeline never read FaceValid, so neither may the reproduction."""
        au = np.full((60, au_names.CHANNEL_COUNT), 0.9, np.float32)
        recording = ad.Recording(
            domain='t', participant='P', emotion='Anger', path=Path('x'),
            split='train', split_source='assumed', excluded=False, exclusion_reason=None,
            times=np.arange(60) / 60, au=au,
            face_valid=np.zeros(60, bool),        # every frame flagged invalid
        )
        self.assertEqual(len(recording.valid_au), 0)
        self.assertGreater(len(tp.teacher_windows(recording)), 0)

    def test_decimate_takes_samples_not_averages(self):
        """The runtime never averages, so its proxy must not either."""
        au = np.arange(12, dtype=np.float64).reshape(12, 1)
        np.testing.assert_allclose(tp.decimate(au), [[0.0], [6.0]])
        np.testing.assert_allclose(tp.downsample(au), [[2.5], [8.5]])

    def test_decimate_preserves_spikes_that_averaging_removes(self):
        """The whole reason the two conditions are reported separately.

        A single-frame spike survives sampling and is diluted sixfold by averaging, which is
        why an averaged window must never be labelled a runtime measurement.
        """
        au = np.zeros((12, 1))
        au[0] = 1.2
        self.assertAlmostEqual(float(tp.decimate(au)[0, 0]), 1.2)
        self.assertAlmostEqual(float(tp.downsample(au)[0, 0]), 0.2)

    def test_the_three_conditions_are_distinct(self):
        au = np.zeros((120, au_names.CHANNEL_COUNT), np.float32)
        au[:, 0] = np.linspace(0, 1, 120)     # ramps past the 0.25 gate partway through
        recording = ad.Recording(
            domain='t', participant='P', emotion='Anger', path=Path('x'),
            split='train', split_source='assumed', excluded=False, exclusion_reason=None,
            times=np.arange(120) / 60, au=au, face_valid=np.ones(120, bool),
        )
        exact = tp.windows_for_condition(recording, 'notebook_exact')
        ungated = tp.windows_for_condition(recording, 'notebook_ungated')
        proxy = tp.windows_for_condition(recording, 'runtime_proxy')

        # The gate throws windows away; ungated and proxy keep every block.
        self.assertLess(len(exact), len(ungated))
        self.assertEqual(len(ungated), len(proxy))
        # Same count, different content: one is averaged, the other sampled.
        self.assertFalse(np.allclose(ungated, proxy))

    def test_unknown_condition_is_refused(self):
        recording = ad.Recording(
            domain='t', participant='P', emotion='Anger', path=Path('x'),
            split='train', split_source='assumed', excluded=False, exclusion_reason=None,
            times=np.arange(60) / 60,
            au=np.zeros((60, au_names.CHANNEL_COUNT), np.float32),
            face_valid=np.ones(60, bool),
        )
        with self.assertRaises(ValueError):
            tp.windows_for_condition(recording, 'runtime')

    def test_channel_count_is_checked(self):
        recording = ad.Recording(
            domain='t', participant='P', emotion='Anger', path=Path('x'),
            split='train', split_source='assumed', excluded=False, exclusion_reason=None,
            times=np.arange(60) / 60, au=np.zeros((60, 12), np.float32),
            face_valid=np.ones(60, bool),
        )
        with self.assertRaises(ValueError):
            tp.teacher_windows(recording)


class NotebookAgreement(unittest.TestCase):
    """Golden values taken from the teacher notebook's own saved outputs.

    These are the facts that let an acted recording be scored at all. If any of them drifts,
    every label in Phase B is silently attached to the wrong output channel, and nothing in
    the live-session fidelity check would catch it -- that check proves the graph is right,
    not that this corpus is being labelled the way the notebook labelled it.
    """

    def test_class_order_matches_the_notebooks_label_encoder(self):
        """`final_teacher_notebook (4).ipynb` cell 8 output:

            le.classes_ = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral',
                           'Sadness', 'Surprise']
        """
        self.assertEqual(mps.CLASS_ORDER,
                         ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral',
                          'Sadness', 'Surprise'])

    def test_class_order_is_the_sorted_order_the_encoder_produces(self):
        """sklearn's LabelEncoder sorts, so the order is alphabetical by construction."""
        self.assertEqual(mps.CLASS_ORDER, sorted(mps.CLASS_ORDER))

    def test_au_order_matches_the_notebooks_feature_list(self):
        """Cell 7 defines AU_NAMES starting BrowLowererL, BrowLowererR, CheekPuffL, and
        cell 12's saved output asserts there are 70 of them, eye channels included."""
        self.assertEqual(au_names.NAMES[:3],
                         ('BrowLowererL', 'BrowLowererR', 'CheekPuffL'))
        self.assertEqual(au_names.CHANNEL_COUNT, 70)

    def test_pipeline_constants_match_the_notebook(self):
        self.assertEqual(tp.SEQ_LENGTH, 5)
        self.assertEqual(tp.DOWNSAMPLE_FACTOR, 6)
        self.assertEqual(tp.CONSECUTIVE, 3)
        self.assertEqual(tp.ACTIVITY_THRESHOLD, 0.25)


class FixedWindowGate(unittest.TestCase):
    """The control that separates the gate's selection from its window placement."""

    def _recording(self, au):
        return ad.Recording(
            domain='t', participant='P', emotion='Anger', path=Path('x'),
            split='test', split_source='named', excluded=False, exclusion_reason=None,
            times=np.arange(len(au)) / 60, au=au.astype(np.float32),
            face_valid=np.ones(len(au), bool))

    def test_it_offers_the_same_windows_as_ungated(self):
        au = np.zeros((120, au_names.CHANNEL_COUNT))
        au[:, 0] = np.linspace(0, 1, 120)
        recording = self._recording(au)
        np.testing.assert_allclose(
            tp.windows_for_condition(recording, 'fixed_windows_gated'),
            tp.windows_for_condition(recording, 'notebook_ungated'))

    def test_the_mask_refuses_quiet_windows_and_keeps_active_ones(self):
        quiet = np.zeros((5, au_names.CHANNEL_COUNT))
        active = np.zeros((5, au_names.CHANNEL_COUNT))
        active[:, 0] = 0.9
        windows = np.stack([quiet, active])
        np.testing.assert_array_equal(tp.gate_mask(windows, 'Anger'), [False, True])

    def test_neutral_is_inverted_here_too(self):
        quiet = np.zeros((5, au_names.CHANNEL_COUNT))
        active = np.zeros((5, au_names.CHANNEL_COUNT))
        active[:, 0] = 0.9
        np.testing.assert_array_equal(
            tp.gate_mask(np.stack([quiet, active]), 'Neutral'), [True, False])

    def test_two_active_frames_are_not_enough(self):
        window = np.zeros((5, au_names.CHANNEL_COUNT))
        window[0:2, 0] = 0.9
        self.assertFalse(tp.window_passes_gate(window, 'Anger'))
        window[2, 0] = 0.9
        self.assertTrue(tp.window_passes_gate(window, 'Anger'))

    def test_empty_input_gives_an_empty_mask(self):
        self.assertEqual(
            len(tp.gate_mask(np.empty((0, 5, au_names.CHANNEL_COUNT)), 'Anger')), 0)


class PhaseBMetrics(unittest.TestCase):
    CLASSES = list(ad.EMOTIONS)

    def test_perfect_prediction_scores_one(self):
        m = mps.macro_f1(self.CLASSES, self.CLASSES, self.CLASSES)
        self.assertAlmostEqual(m['macro_f1'], 1.0)
        self.assertEqual(m['classes_scored'], 7)

    def test_absent_class_is_skipped_not_scored_zero(self):
        """The zero-filling rule, at the metric level.

        A class with no true instance and no prediction must leave the mean alone. Scoring
        it 0 would drag macro-F1 down for a class the participant's data never contained.
        """
        truth = ['Anger', 'Disgust']
        m = mps.macro_f1(truth, truth, self.CLASSES)
        self.assertEqual(m['classes_scored'], 2)
        self.assertAlmostEqual(m['macro_f1'], 1.0)
        self.assertIsNone(m['per_class']['Fear']['f1'])
        self.assertEqual(m['per_class']['Fear']['support'], 0)

    def test_a_predicted_but_never_true_class_is_scored(self):
        """Over-prediction must cost something, so a class with predictions but no support
        is scored rather than skipped."""
        m = mps.macro_f1(['Anger', 'Anger'], ['Anger', 'Neutral'], self.CLASSES)
        self.assertEqual(m['per_class']['Neutral']['support'], 0)
        self.assertEqual(m['per_class']['Neutral']['precision'], 0.0)
        self.assertIn('Neutral', [c for c in self.CLASSES
                                  if m['per_class'][c]['f1'] is not None])

    def test_recall_and_precision_on_a_worked_example(self):
        truth = ['Anger', 'Anger', 'Fear', 'Fear', 'Fear']
        pred = ['Anger', 'Fear', 'Fear', 'Fear', 'Anger']
        m = mps.macro_f1(truth, pred, self.CLASSES)['per_class']
        self.assertAlmostEqual(m['Anger']['recall'], 0.5)       # 1 of 2
        self.assertAlmostEqual(m['Anger']['precision'], 0.5)    # 1 of 2 predicted
        self.assertAlmostEqual(m['Fear']['recall'], 2 / 3)
        self.assertAlmostEqual(m['Fear']['precision'], 2 / 3)

    def test_unscored_recordings_are_excluded_not_counted_wrong(self):
        """A recording the gate produced no window for is not a model error."""
        rows = [
            {'participant': 'P', 'emotion': 'Anger', 'split': 'test', 'excluded': False,
             'windows': 3, 'predicted': 'Anger', 'correct': True, 'window_correct': 3,
             'mean_confidence': 0.8, 'mean_prompted_score': 0.8},
            {'participant': 'P', 'emotion': 'Fear', 'split': 'test', 'excluded': False,
             'windows': 0, 'predicted': None, 'correct': None, 'window_correct': 0,
             'mean_confidence': None, 'mean_prompted_score': None},
        ]
        stats = mps.per_participant(rows, self.CLASSES)[0]
        self.assertEqual(stats['recordings'], 2)
        self.assertEqual(stats['unscored'], 1)
        self.assertAlmostEqual(stats['trial_accuracy'], 1.0)   # 1 of 1 scored, not 1 of 2

    def test_confusion_counts_only_scored_recordings(self):
        rows = [
            {'emotion': 'Anger', 'predicted': 'Neutral'},
            {'emotion': 'Anger', 'predicted': None},
        ]
        matrix = mps.pooled_confusion(rows, self.CLASSES)
        self.assertEqual(matrix['Anger']['Neutral'], 1)
        self.assertEqual(sum(matrix['Anger'].values()), 1)


class RealCorpus(unittest.TestCase):
    """Integration checks that skip when the corpus is not on this machine."""

    def setUp(self):
        try:
            self.corpus = ad.load_corpus(ad.find_corpus_root())
        except ad.CorpusError as e:
            self.skipTest(str(e))

    def test_shape_is_what_the_report_claims(self):
        self.assertEqual(len(self.corpus.participants), 15)
        self.assertEqual(len(self.corpus.recordings), 105)
        self.assertEqual(len(self.corpus.subset(self_report_consistent=True)), 76)
        self.assertEqual(self.corpus.warnings, [])

    def test_every_participant_has_all_seven_emotions_before_filtering(self):
        for participant, emotions in self.corpus.coverage().items():
            self.assertEqual(emotions, set(ad.EMOTIONS), participant)

    def test_no_held_out_participant_has_all_seven_after_filtering(self):
        """The reason E2 carries no seven-class subject-level score."""
        held = self.corpus.coverage(self_report_consistent=True,
                                    splits=('validation', 'test'))
        for participant, emotions in held.items():
            self.assertLess(len(emotions), 7, participant)

    def test_held_out_counts(self):
        e1 = self.corpus.subset(splits=('validation', 'test'))
        e2 = self.corpus.subset(self_report_consistent=True, splits=('validation', 'test'))
        self.assertEqual((len(e1), len({r.participant for r in e1})), (35, 5))
        self.assertEqual((len(e2), len({r.participant for r in e2})), (19, 4))


if __name__ == '__main__':
    unittest.main(verbosity=2)
