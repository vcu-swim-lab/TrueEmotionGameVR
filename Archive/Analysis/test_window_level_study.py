#!/usr/bin/env python3
"""Tests for `window_level_study.py`.

Stdlib `unittest`, needing only numpy. The integration tests skip themselves when the acted
corpus, the live sessions or onnxruntime are not on the machine.

The tests that matter most are the ones guarding a silent wrong answer. Flattening a window
in the wrong order, letting a validation participant into the training reference, pooling two
sessions instead of weighting them, or reporting a 55-channel number as the 70-channel result
all produce a table that prints, reads plausibly, and is about something else.

    python Analysis/test_window_level_study.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import acting_dataset as ad
import au_names
import live_vs_acted_study as lva
import model_performance_study as mp
import session_analysis as sa
import teacher_preprocessing as tp
import window_level_study as wl

N = au_names.CHANNEL_COUNT
F = tp.SEQ_LENGTH


def row(prompted: str, index: int, frame_values: list[float], *,
        predicted: str | None = None, status: str = sa.STATUS_OK,
        scores: list[float] | None = None) -> sa.Row:
    """One tick whose frame `f` is `frame_values[f]` on every channel."""
    ok = status == sa.STATUS_OK
    return sa.Row(
        timestamp=None,
        trial_id=0,
        prompted=prompted,
        prediction_index=index,
        predicted=predicted if predicted is not None else prompted,
        logged_correct=predicted in (None, prompted),
        status=status,
        generation=index,
        scores=(scores or [1.0] + [0.0] * 6) if ok else None,
        frames=[[v] * N for v in frame_values] if ok else None,
    )


def session(rows: list[sa.Row], name: str = 'synthetic') -> sa.Session:
    return sa.Session(
        path=Path(name) / 'predictions.csv',
        session_id=name,
        schema=sa.Schema(classes=tuple(wl.CLASS_ORDER), window_frames=F,
                         features_per_frame=N),
        rows=rows,
    )


def simple_session(per_emotion: dict[str, list[list[float]]]) -> sa.Session:
    rows = [row(emotion, i, frames)
            for emotion, ticks in per_emotion.items()
            for i, frames in enumerate(ticks)]
    return session(rows)


def uniform(value: float) -> list[float]:
    return [value] * F


def recording(emotion: str, au: np.ndarray, participant: str = 'p1',
              split: str = 'train') -> ad.Recording:
    n = len(au)
    return ad.Recording(
        domain=ad.DOMAIN, participant=participant, emotion=emotion, path=Path('x'),
        split=split, split_source='assumed', excluded=False, exclusion_reason=None,
        times=np.arange(n) / 60.0,
        au=au.astype(np.float32),
        face_valid=np.ones(n, bool),
    )


class Flattening(unittest.TestCase):

    def test_flatten_preserves_time_order(self):
        """Position f*70 + c must be channel c at frame f."""
        window = np.zeros((1, F, N), np.float32)
        for f in range(F):
            window[0, f, :] = f                     # frame f is all f
        flat = wl.flatten(window)

        self.assertEqual(flat.shape, (1, F * N))
        for f in range(F):
            block = flat[0, f * N:(f + 1) * N]
            self.assertTrue(np.all(block == f), f'frame {f} landed in the wrong block')

    def test_flatten_distinguishes_a_reversed_window(self):
        """A rising expression and a falling one must not flatten to the same vector."""
        rising = np.zeros((1, F, N), np.float32)
        for f in range(F):
            rising[0, f, :] = f
        falling = rising[:, ::-1, :].copy()

        self.assertFalse(np.array_equal(wl.flatten(rising), wl.flatten(falling)))
        # And the cosine must see the difference, which a peak profile could not.
        self.assertLess(float(wl.cosine_matrix(wl.flatten(rising), wl.flatten(falling))[0, 0]),
                        0.999)

    def test_flatten_rejects_the_wrong_shape(self):
        with self.assertRaises(wl.StudyError):
            wl.flatten(np.zeros((3, 4, N), np.float32))
        with self.assertRaises(wl.StudyError):
            wl.flatten(np.zeros((3, F, 63), np.float32))
        with self.assertRaises(wl.StudyError):
            wl.flatten(np.zeros((F, N), np.float32))        # missing the batch axis


class LiveWindows(unittest.TestCase):

    def test_window_shape_is_exactly_five_by_seventy(self):
        s = simple_session({e: [uniform(0.1), uniform(0.2)] for e in wl.COMMON_EMOTIONS})
        windows = wl.live_windows(s)
        for emotion in wl.COMMON_EMOTIONS:
            self.assertEqual(windows[emotion]['windows'].shape[1:], (F, N))

    def test_transition_tick_excluded(self):
        # Tick 0 is a distinctive 9.0; every later tick is 0.5.
        s = simple_session({e: [uniform(9.0)] + [uniform(0.5)] * 4
                            for e in wl.COMMON_EMOTIONS})
        windows = wl.live_windows(s)
        for emotion in wl.COMMON_EMOTIONS:
            data = windows[emotion]
            self.assertEqual(len(data['windows']), 4)
            self.assertEqual(data['n_prompted_ticks'], 5)
            self.assertFalse((data['windows'] == 9.0).any(), 'tick 0 leaked in')
            self.assertEqual(data['tick'], [1, 2, 3, 4])

    def test_no_tracking_row_produces_no_window(self):
        rows = []
        for emotion in wl.COMMON_EMOTIONS:
            rows.append(row(emotion, 0, uniform(9.0)))
            rows.append(row(emotion, 1, uniform(0.5)))
            rows.append(row(emotion, 2, uniform(0.0), status=sa.STATUS_NO_TRACKING))
            rows.append(row(emotion, 3, uniform(0.5)))
        windows = wl.live_windows(session(rows))
        for emotion in wl.COMMON_EMOTIONS:
            self.assertEqual(len(windows[emotion]['windows']), 2)
            self.assertEqual(windows[emotion]['tick'], [1, 3], 'ticks renumbered')

    def test_only_the_six_common_emotions(self):
        self.assertEqual(set(wl.COMMON_EMOTIONS), set(ad.EMOTIONS) - {'Neutral'})
        s = simple_session({e: [uniform(0.1), uniform(0.2)] for e in wl.COMMON_EMOTIONS})
        s.rows.extend(row('Neutral', i, uniform(5.0)) for i in range(3))
        windows = wl.live_windows(s)
        self.assertNotIn('Neutral', windows)
        self.assertEqual(sorted(windows), sorted(wl.COMMON_EMOTIONS))

    def test_class_order_must_match_the_model(self):
        s = simple_session({e: [uniform(0.1), uniform(0.2)] for e in wl.COMMON_EMOTIONS})
        s.schema = sa.Schema(classes=tuple(reversed(wl.CLASS_ORDER)),
                             window_frames=F, features_per_frame=N)
        with self.assertRaises(wl.StudyError):
            wl.live_windows(s)

    def test_wrong_window_size_is_refused(self):
        s = simple_session({e: [uniform(0.1), uniform(0.2)] for e in wl.COMMON_EMOTIONS})
        s.schema = sa.Schema(classes=tuple(wl.CLASS_ORDER), window_frames=3,
                             features_per_frame=N)
        with self.assertRaises(wl.StudyError):
            wl.live_windows(s)

    def test_missing_emotion_raises(self):
        values = {e: [uniform(0.1), uniform(0.2)]
                  for e in wl.COMMON_EMOTIONS if e != 'Sadness'}
        with self.assertRaises(wl.StudyError) as caught:
            wl.live_windows(simple_session(values))
        self.assertIn('Sadness', str(caught.exception))


class RuntimeProxyWindows(unittest.TestCase):
    """The acted side is built by the shared pipeline, and it is built right."""

    def test_runtime_proxy_takes_every_sixth_frame(self):
        au = np.tile(np.arange(60, dtype=np.float32)[:, None], (1, N))   # frame i == i
        windows = tp.windows_for_condition(recording('Anger', au), 'runtime_proxy')

        self.assertEqual(windows.shape, (2, F, N))
        # decimate -> frames 0, 6, 12, ... 54; first window is the first five of those.
        np.testing.assert_allclose(windows[0, :, 0], [0, 6, 12, 18, 24])
        np.testing.assert_allclose(windows[1, :, 0], [30, 36, 42, 48, 54])

    def test_runtime_proxy_does_not_average(self):
        """The distinguishing property against notebook_exact: no mean of six."""
        au = np.tile(np.arange(60, dtype=np.float32)[:, None], (1, N))
        proxy = tp.windows_for_condition(recording('Anger', au), 'runtime_proxy')
        exact = tp.windows_for_condition(recording('Anger', au), 'notebook_exact')
        self.assertEqual(proxy[0, 0, 0], 0.0)
        # notebook_exact averages frames 0-5 -> 2.5
        self.assertAlmostEqual(float(exact[0, 0, 0]), 2.5, places=5)

    def test_acted_windows_only_take_named_participants(self):
        corpus = ad.Corpus(domain=ad.DOMAIN, root=Path('x'), recordings=[
            recording('Anger', np.ones((60, N), np.float32), 'train1', 'train'),
            recording('Anger', np.ones((60, N), np.float32), 'val1', 'validation'),
            recording('Anger', np.ones((60, N), np.float32), 'test1', 'test'),
        ])
        acted = wl.acted_windows(corpus, 'runtime_proxy', ['train1'])
        self.assertEqual(list(acted), [('train1', 'Anger')])

    def test_acted_windows_exclude_neutral(self):
        corpus = ad.Corpus(domain=ad.DOMAIN, root=Path('x'), recordings=[
            recording('Neutral', np.ones((60, N), np.float32), 'train1', 'train'),
            recording('Anger', np.ones((60, N), np.float32), 'train1', 'train'),
        ])
        acted = wl.acted_windows(corpus, 'runtime_proxy', ['train1'])
        self.assertEqual(list(acted), [('train1', 'Anger')])


class ChannelSets(unittest.TestCase):
    """70 and 55 are separate results and never substituted for one another."""

    def test_channel_sets_are_distinct_and_correctly_sized(self):
        self.assertEqual(len(wl.CHANNEL_SETS['all70']), 70)
        self.assertEqual(len(wl.CHANNEL_SETS['expression55']), 55)
        self.assertEqual(wl.PRIMARY_CHANNELS, 'all70',
                         'the primary must be what the model actually sees')

    def test_channel_columns_span_every_frame(self):
        columns = wl._channel_columns([0, 69])
        self.assertEqual(len(columns), 2 * F)
        np.testing.assert_array_equal(
            columns, [0, 69, 70, 139, 140, 209, 210, 279, 280, 349])

    def test_channel_subset_actually_drops_those_channels(self):
        """A difference confined to gaze must vanish under expression55 and not under all70."""
        gaze = au_names.REGIONS['eye_gaze'][0]
        a = np.full((1, F, N), 0.5, np.float32)
        b = a.copy()
        b[0, :, gaze] = 9.0

        flat_a, flat_b = wl.flatten(a), wl.flatten(b)
        all70 = wl._channel_columns(wl.CHANNEL_SETS['all70'])
        expr55 = wl._channel_columns(wl.CHANNEL_SETS['expression55'])

        self.assertLess(float(wl.cosine_matrix(flat_a[:, all70], flat_b[:, all70])[0, 0]), 0.99)
        self.assertAlmostEqual(
            float(wl.cosine_matrix(flat_a[:, expr55], flat_b[:, expr55])[0, 0]), 1.0, places=6)

    def test_primary_is_not_silently_the_subset(self):
        """The 55-channel set must be a strict subset, so the two can never coincide."""
        self.assertLess(set(wl.CHANNEL_SETS['expression55']),
                        set(wl.CHANNEL_SETS['all70']))
        dropped = set(wl.CHANNEL_SETS['all70']) - set(wl.CHANNEL_SETS['expression55'])
        self.assertEqual(dropped,
                         set(au_names.REGIONS['eye_gaze']) | set(au_names.REGIONS['tongue']))


class Cosine(unittest.TestCase):

    def test_cosine_matrix_on_known_vectors(self):
        a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        b = np.array([[2.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        sims = wl.cosine_matrix(a, b)
        self.assertAlmostEqual(sims[0, 0], 1.0)
        self.assertAlmostEqual(sims[0, 1], 1 / np.sqrt(2))
        self.assertAlmostEqual(sims[1, 0], 0.0)
        self.assertAlmostEqual(sims[1, 1], 1 / np.sqrt(2))

    def test_zero_window_is_refused_rather_than_scored(self):
        with self.assertRaises(wl.StudyError):
            wl.cosine_matrix(np.zeros((1, 3)), np.ones((1, 3)))


class ParticipantBalance(unittest.TestCase):
    """One prolific participant must not decide the answer."""

    def setUp(self):
        self.n_live = 1
        # p1: a single, near-perfect match. p2: a hundred mediocre ones.
        self.per_participant = {
            ('p1', 'Anger'): np.array([[1.0]]),
            ('p2', 'Anger'): np.full((1, 100), 0.2),
        }
        for emotion in wl.COMMON_EMOTIONS:
            if emotion != 'Anger':
                self.per_participant[('p1', emotion)] = np.array([[0.1]])
                self.per_participant[('p2', emotion)] = np.full((1, 100), 0.1)

    def test_each_participant_contributes_their_best_window_once(self):
        balanced = wl.balanced_by_emotion(self.per_participant, ['p1', 'p2'])
        # (1.0 + 0.2) / 2 = 0.6. A pooled maximum would give 1.0 and a pooled mean ~0.208,
        # both of which are decided by how many windows p2 happened to contribute.
        self.assertAlmostEqual(float(balanced['Anger'][0]), 0.6)

    def test_window_count_does_not_change_the_result(self):
        fewer = dict(self.per_participant)
        fewer[('p2', 'Anger')] = np.full((1, 3), 0.2)      # 3 windows instead of 100
        self.assertAlmostEqual(
            float(wl.balanced_by_emotion(self.per_participant, ['p1', 'p2'])['Anger'][0]),
            float(wl.balanced_by_emotion(fewer, ['p1', 'p2'])['Anger'][0]))

    def test_participant_absent_for_an_emotion_is_skipped_not_zeroed(self):
        partial = {k: v for k, v in self.per_participant.items()
                   if k != ('p2', 'Anger')}
        balanced = wl.balanced_by_emotion(partial, ['p1', 'p2'])
        self.assertAlmostEqual(float(balanced['Anger'][0]), 1.0)   # not (1.0 + 0) / 2

    def test_no_window_for_an_emotion_raises(self):
        empty = {k: v for k, v in self.per_participant.items() if k[1] != 'Anger'}
        with self.assertRaises(wl.StudyError):
            wl.balanced_by_emotion(empty, ['p1', 'p2'])


class NearestNeighbour(unittest.TestCase):
    """The per-window description, on an acted set built by hand."""

    def setUp(self):
        def window(scale: float, channel: int) -> np.ndarray:
            w = np.zeros((1, F, N), np.float32)
            w[0, :, channel] = scale
            return w

        # Each emotion gets its own channel, so which emotion a live window resembles is
        # decided entirely by which channel it uses.
        self.acted = {}
        for i, emotion in enumerate(wl.COMMON_EMOTIONS):
            for p in ('p1', 'p2'):
                self.acted[(p, emotion)] = window(1.0, i)
        self.participants = ['p1', 'p2']

    def live_on(self, channel: int) -> np.ndarray:
        w = np.zeros((1, F, N), np.float32)
        w[0, :, channel] = 1.0
        return w

    def test_nearest_emotion_is_the_matching_channel(self):
        prompted = wl.COMMON_EMOTIONS[3]
        described = wl.describe_windows(self.live_on(3), self.acted, self.participants,
                                        prompted, wl.CHANNEL_SETS['all70'])
        self.assertEqual(described['nearest_emotion'], [prompted])
        self.assertEqual(described['prompted_rank'].tolist(), [1])
        self.assertAlmostEqual(float(described['balanced_to_prompted'][0]), 1.0, places=6)

    def test_a_window_on_the_wrong_channel_ranks_the_prompt_last(self):
        prompted = wl.COMMON_EMOTIONS[0]
        other = wl.COMMON_EMOTIONS[5]
        described = wl.describe_windows(self.live_on(5), self.acted, self.participants,
                                        prompted, wl.CHANNEL_SETS['all70'])
        self.assertEqual(described['nearest_emotion'], [other])
        self.assertAlmostEqual(float(described['balanced_to_prompted'][0]), 0.0, places=6)
        self.assertEqual(described['balanced_best_emotion'], [other])

    def test_top_k_and_nearest_participant_are_reported(self):
        prompted = wl.COMMON_EMOTIONS[2]
        described = wl.describe_windows(self.live_on(2), self.acted, self.participants,
                                        prompted, wl.CHANNEL_SETS['all70'])
        self.assertEqual(len(described['top_k_emotions'][0]), wl.TOP_K)
        self.assertIn(described['nearest_participant'][0], self.participants)

    def test_only_named_participants_contribute(self):
        polluted = dict(self.acted)
        w = np.zeros((1, F, N), np.float32)
        w[0, :, 3] = 1.0
        polluted[('val1', wl.COMMON_EMOTIONS[0])] = w      # a perfect decoy, wrong split

        prompted = wl.COMMON_EMOTIONS[3]
        described = wl.describe_windows(self.live_on(3), polluted, self.participants,
                                        prompted, wl.CHANNEL_SETS['all70'])
        self.assertNotIn('val1', described['nearest_participant'])
        self.assertEqual(described['nearest_emotion'], [prompted])


class ActedSelfSimilarity(unittest.TestCase):
    """The control that bounds every nearest-neighbour claim."""

    def acted_on_channels(self, assignment: dict[str, int],
                          participants: list[str]) -> dict:
        acted = {}
        for emotion, channel in assignment.items():
            for p in participants:
                w = np.zeros((1, F, N), np.float32)
                w[0, :, channel] = 1.0
                acted[(p, emotion)] = w
        return acted

    def test_perfectly_separated_emotions_are_self_nearest(self):
        participants = ['p1', 'p2', 'p3']
        acted = self.acted_on_channels(
            {e: i for i, e in enumerate(wl.COMMON_EMOTIONS)}, participants)
        result = wl.acted_self_similarity(acted, participants, wl.CHANNEL_SETS['all70'])
        for emotion in wl.COMMON_EMOTIONS:
            self.assertTrue(result[emotion]['self_is_nearest'], emotion)
            self.assertAlmostEqual(result[emotion]['cosine'][emotion], 1.0, places=6)
            self.assertAlmostEqual(result[emotion]['margin'], 1.0, places=6)

    def test_two_emotions_sharing_a_channel_are_not_separated(self):
        participants = ['p1', 'p2', 'p3']
        assignment = {e: i for i, e in enumerate(wl.COMMON_EMOTIONS)}
        assignment['Fear'] = assignment['Surprise']        # indistinguishable
        acted = self.acted_on_channels(assignment, participants)
        result = wl.acted_self_similarity(acted, participants, wl.CHANNEL_SETS['all70'])
        self.assertAlmostEqual(result['Fear']['cosine']['Surprise'], 1.0, places=6)
        self.assertAlmostEqual(result['Fear']['margin'], 0.0, places=6)

    def test_a_participant_is_never_compared_against_themselves(self):
        """One participant with a unique profile must not score 1.0 off their own windows."""
        participants = ['p1', 'p2']
        acted = self.acted_on_channels(
            {e: i for i, e in enumerate(wl.COMMON_EMOTIONS)}, participants)
        odd = np.zeros((1, F, N), np.float32)
        odd[0, :, 69] = 1.0                       # p1's Anger uses a channel nobody else does
        acted[('p1', 'Anger')] = odd

        result = wl.acted_self_similarity(acted, participants, wl.CHANNEL_SETS['all70'])
        # p1 scores 0 against p2's Anger; p2 scores 0 against p1's. Mean is 0, not 0.5 and
        # certainly not 1.0, which is what including one's own windows would give.
        self.assertAlmostEqual(result['Anger']['cosine']['Anger'], 0.0, places=6)


class SessionWeighting(unittest.TestCase):
    """A's two evenings weigh the same, whatever their length."""

    def test_floats_are_averaged_not_pooled(self):
        a = {'model_accuracy': 1.0, 'n_windows': 2, 'prediction_counts': {'Anger': 2}}
        b = {'model_accuracy': 0.0, 'n_windows': 40, 'prediction_counts': {'Fear': 40}}
        combined = wl.mean_of_summaries([a, b])
        # Equal weight per session: 0.5. Pooling windows would give 2/42 = 0.048.
        self.assertAlmostEqual(combined['model_accuracy'], 0.5)
        self.assertEqual(combined['n_windows'], 42)        # counts still total
        self.assertEqual(combined['prediction_counts'], {'Fear': 40, 'Anger': 2})
        self.assertEqual(combined['n_sessions'], 2)

    def test_nested_float_dicts_are_averaged(self):
        a = {'input_cos_by_emotion': {'Anger': 1.0, 'Fear': 0.0}}
        b = {'input_cos_by_emotion': {'Anger': 0.0, 'Fear': 1.0}}
        combined = wl.mean_of_summaries([a, b])
        self.assertAlmostEqual(combined['input_cos_by_emotion']['Anger'], 0.5)
        self.assertAlmostEqual(combined['input_cos_by_emotion']['Fear'], 0.5)

    def test_single_session_is_unchanged(self):
        one = {'model_accuracy': 0.7, 'n_windows': 9}
        self.assertEqual(wl.mean_of_summaries([one]), one)

    def test_subject_registry_is_two_people(self):
        self.assertEqual(sorted(wl.LIVE_SUBJECTS), ['A', 'B'])
        self.assertEqual(len(wl.LIVE_SUBJECTS['A']), 2)
        self.assertEqual(len(wl.LIVE_SUBJECTS['B']), 1)
        self.assertIs(wl.LIVE_SUBJECTS, lva.LIVE_SUBJECTS, 'must reuse, not restate')


class Fidelity(unittest.TestCase):
    """No ONNX number is reported without a passing replay check."""

    def setUp(self):
        self.original = mp.fidelity_check
        self.addCleanup(lambda: setattr(mp, 'fidelity_check', self.original))

    def test_failed_fidelity_stops_the_study(self):
        mp.fidelity_check = lambda path: {
            'ran': True, 'passed': False, 'tolerance': 1e-4,
            'sessions': [{'session': 's', 'rows': 1, 'agreement': 0.5, 'max_diff': 0.9}]}
        with self.assertRaises(wl.StudyError) as caught:
            wl.run(None, None, None, 'all70', ('runtime_proxy',))
        self.assertIn('FAILED', str(caught.exception))

    def test_unrun_fidelity_stops_the_study(self):
        mp.fidelity_check = lambda path: {'ran': False, 'reason': 'no live session logs'}
        with self.assertRaises(wl.StudyError) as caught:
            wl.run(None, None, None, 'all70', ('runtime_proxy',))
        self.assertIn('fidelity', str(caught.exception).lower())


# --------------------------------------------------------------- integration


def requirements_or_skip():
    try:
        import onnxruntime           # noqa: F401
    except ImportError:
        raise unittest.SkipTest('onnxruntime not installed')
    try:
        ad.find_corpus_root(None)
    except ad.CorpusError as exc:
        raise unittest.SkipTest(f'acted corpus unavailable: {exc}')
    try:
        lva.resolve_live_sessions(None)
    except lva.StudyError as exc:
        raise unittest.SkipTest(f'live sessions unavailable: {exc}')


class RealData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        requirements_or_skip()
        cls.result = wl.run(None, None, None, 'all70',
                            (wl.PRIMARY_CONDITION, wl.CONTROL_CONDITION))

    def test_fidelity_passed(self):
        self.assertTrue(self.result['fidelity']['passed'])

    def test_conditions_are_kept_under_separate_names(self):
        self.assertEqual(sorted(self.result['conditions']),
                         sorted({wl.PRIMARY_CONDITION, wl.CONTROL_CONDITION}))
        self.assertNotEqual(wl.PRIMARY_CONDITION, wl.CONTROL_CONDITION)

        primary = self.result['conditions'][wl.PRIMARY_CONDITION]
        control = self.result['conditions'][wl.CONTROL_CONDITION]
        self.assertNotEqual(primary['acted_windows'], control['acted_windows'],
                            'the two conditions produced identical window sets')

    def test_channel_set_is_recorded_in_the_result(self):
        self.assertEqual(self.result['channel_set'], 'all70')
        self.assertEqual(self.result['channel_count'], 70)

    def test_reference_is_the_ten_training_participants(self):
        self.assertEqual(len(self.result['training_participants']), 10)

    def test_subject_a_has_two_sessions_and_twice_the_windows(self):
        primary = self.result['conditions'][wl.PRIMARY_CONDITION]['subjects']
        self.assertEqual(primary['A']['n_sessions'], 2)
        self.assertEqual(primary['B']['n_sessions'], 1)
        for emotion in wl.COMMON_EMOTIONS:
            self.assertEqual(primary['A']['emotions'][emotion]['n_windows'],
                             2 * primary['B']['emotions'][emotion]['n_windows'])

    def test_channel_set_changes_the_numbers(self):
        """A 55-channel run must not silently return the 70-channel result."""
        other = wl.run(None, None, None, 'expression55', (wl.PRIMARY_CONDITION,))
        self.assertEqual(other['channel_count'], 55)

        a70 = self.result['conditions'][wl.PRIMARY_CONDITION]['subjects']['A']
        a55 = other['conditions'][wl.PRIMARY_CONDITION]['subjects']['A']
        self.assertNotEqual(
            a70['emotions']['Anger']['input_cos_to_prompted'],
            a55['emotions']['Anger']['input_cos_to_prompted'])

    def test_no_live_subject_appears_among_acted_participants(self):
        live_ids = {i for ids in wl.LIVE_SUBJECTS.values() for i in ids}
        for participant in self.result['training_participants']:
            self.assertNotIn(participant, live_ids)


if __name__ == '__main__':
    unittest.main(verbosity=2)
