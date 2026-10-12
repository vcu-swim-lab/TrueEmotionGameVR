#!/usr/bin/env python3
"""Tests for `live_pairwise_study.py`.

Stdlib `unittest`, needing only numpy. The integration tests skip themselves when the pulled
live sessions are not on the machine.

The comparison this file guards is small enough to eyeball and easy to get backwards: the
within-person number and the between-person numbers must come from the sets they claim to,
the two representations must not be conflated, and `compare` must be symmetric -- otherwise
"A1 against B" and "B against A1" would differ and the table would depend on column order.

    python Analysis/test_live_pairwise_study.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import au_names
import live_pairwise_study as lp
import live_vs_acted_study as lva
import session_analysis as sa
import teacher_preprocessing as tp
import window_level_study as wl

N = au_names.CHANNEL_COUNT
F = tp.SEQ_LENGTH


def row(prompted: str, index: int, frame_values: list[float]) -> sa.Row:
    return sa.Row(
        timestamp=None, trial_id=0, prompted=prompted, prediction_index=index,
        predicted=prompted, logged_correct=True, status=sa.STATUS_OK, generation=index,
        scores=[1.0] + [0.0] * 6,
        frames=[[v] * N for v in frame_values],
    )


def session(per_emotion: dict[str, list[list[float]]]) -> sa.Session:
    rows = [row(e, i, frames) for e, ticks in per_emotion.items()
            for i, frames in enumerate(ticks)]
    return sa.Session(
        path=Path('synthetic') / 'predictions.csv', session_id='synthetic',
        schema=sa.Schema(classes=tuple(wl.CLASS_ORDER), window_frames=F,
                         features_per_frame=N),
        rows=rows,
    )


def uniform(value: float) -> list[float]:
    return [value] * F


def simple(value: float, ticks: int = 4) -> sa.Session:
    """A session where every emotion is the same flat window, repeated."""
    return session({e: [uniform(9.0)] + [uniform(value)] * ticks
                    for e in lp.COMMON_EMOTIONS})


class SessionVectors(unittest.TestCase):

    def test_peak_gives_one_row_per_emotion(self):
        vectors = lp.session_vectors(simple(0.5), 'peak', lp.CHANNEL_SETS['all70'])
        self.assertEqual(sorted(vectors), sorted(lp.COMMON_EMOTIONS))
        for emotion in lp.COMMON_EMOTIONS:
            self.assertEqual(vectors[emotion].shape, (1, 70))
            self.assertAlmostEqual(float(vectors[emotion].max()), 0.5)

    def test_window_gives_one_row_per_scored_window(self):
        vectors = lp.session_vectors(simple(0.5, ticks=4), 'window',
                                     lp.CHANNEL_SETS['all70'])
        for emotion in lp.COMMON_EMOTIONS:
            self.assertEqual(vectors[emotion].shape, (4, F * 70))

    def test_both_representations_exclude_the_transition_tick(self):
        # Tick 0 is 9.0; every later tick is 0.5. Neither representation may see the 9.0.
        for representation in lp.REPRESENTATIONS:
            vectors = lp.session_vectors(simple(0.5), representation,
                                         lp.CHANNEL_SETS['all70'])
            for emotion in lp.COMMON_EMOTIONS:
                self.assertLess(float(vectors[emotion].max()), 1.0,
                                f'{representation}/{emotion}: tick 0 leaked in')

    def test_channel_subset_narrows_the_vector(self):
        peak = lp.session_vectors(simple(0.5), 'peak', lp.CHANNEL_SETS['expression55'])
        window = lp.session_vectors(simple(0.5), 'window', lp.CHANNEL_SETS['expression55'])
        self.assertEqual(peak['Anger'].shape, (1, 55))
        self.assertEqual(window['Anger'].shape, (4, F * 55))

    def test_unknown_representation_raises(self):
        with self.assertRaises(lp.StudyError):
            lp.session_vectors(simple(0.5), 'peek', lp.CHANNEL_SETS['all70'])

    def test_representations_reuse_the_other_modules(self):
        """These must not be reimplementations that can drift."""
        self.assertIs(lp.COMMON_EMOTIONS, lva.COMMON_EMOTIONS)
        self.assertIs(lp.LIVE_SUBJECTS, lva.LIVE_SUBJECTS)
        self.assertIs(lp.CHANNEL_SETS, wl.CHANNEL_SETS)


class Compare(unittest.TestCase):

    def test_a_set_compared_against_itself_matches_perfectly(self):
        """`best` must be 1.0; `mean` must not, since it includes unlike cross pairs."""
        x = np.random.default_rng(0).random((4, 12)) + 0.1
        result = lp.compare(x, x)
        self.assertAlmostEqual(result['best'], 1.0, places=9)
        self.assertAlmostEqual(result['max'], 1.0, places=9)
        self.assertLess(result['mean'], 1.0)

    def test_one_vector_against_itself_scores_one_on_both(self):
        x = np.array([[0.3, 0.7, 0.1]])
        result = lp.compare(x, x)
        self.assertAlmostEqual(result['mean'], 1.0, places=9)
        self.assertAlmostEqual(result['best'], 1.0, places=9)

    def test_orthogonal_sets_score_zero(self):
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[0.0, 1.0, 0.0]])
        result = lp.compare(a, b)
        self.assertAlmostEqual(result['mean'], 0.0, places=9)
        self.assertAlmostEqual(result['best'], 0.0, places=9)

    def test_mean_is_over_every_cross_pair(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = lp.compare(a, b)
        # cosines are 1, 0, 0, 1 -> mean 0.5
        self.assertAlmostEqual(result['mean'], 0.5, places=9)
        self.assertEqual(result['n_pairs'], 4)
        # ...but each vector has a perfect match on the other side.
        self.assertAlmostEqual(result['best'], 1.0, places=9)

    def test_best_is_symmetric(self):
        """Swapping the arguments must not change the answer."""
        rng = np.random.default_rng(1)
        a, b = rng.random((3, 8)) + 0.1, rng.random((7, 8)) + 0.1
        forward, backward = lp.compare(a, b), lp.compare(b, a)
        self.assertAlmostEqual(forward['best'], backward['best'], places=12)
        self.assertAlmostEqual(forward['mean'], backward['mean'], places=12)

    def test_single_vectors_make_mean_and_best_coincide(self):
        a = np.array([[1.0, 1.0, 0.0]])
        b = np.array([[1.0, 0.0, 0.0]])
        result = lp.compare(a, b)
        self.assertAlmostEqual(result['mean'], result['best'], places=12)
        self.assertAlmostEqual(result['mean'], 1 / np.sqrt(2), places=9)


class Ranking(unittest.TestCase):

    def test_rank_shares_ties(self):
        np.testing.assert_allclose(lp._rank(np.array([3.0, 1.0, 2.0])), [2.0, 0.0, 1.0])
        np.testing.assert_allclose(lp._rank(np.array([1.0, 1.0, 2.0])), [0.5, 0.5, 2.0])

    def test_rank_of_a_constant_array_is_flat(self):
        np.testing.assert_allclose(lp._rank(np.array([5.0, 5.0, 5.0])), [1.0, 1.0, 1.0])


# --------------------------------------------------------------- integration


def live_or_skip():
    try:
        return lva.resolve_live_sessions(None)
    except lva.StudyError as exc:
        raise unittest.SkipTest(f'live sessions unavailable: {exc}')


class RealData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        live_or_skip()
        cls.results = {r: lp.run(None, r, 'all70') for r in lp.REPRESENTATIONS}

    def test_the_repeated_subject_is_the_one_with_two_sessions(self):
        for representation, result in self.results.items():
            self.assertEqual(result['repeated_subject'], 'A', representation)
            self.assertEqual(result['single_subject'], 'B', representation)
            self.assertEqual(len(result['sessions']['A']), 2)
            self.assertEqual(len(result['sessions']['B']), 1)

    def test_the_two_representations_give_different_numbers(self):
        """A `peak` result must never be reported as a `window` result."""
        peak = self.results['peak']['emotions']
        window = self.results['window']['emotions']
        self.assertEqual(peak.keys(), window.keys())
        differing = sum(1 for e in peak
                        if abs(peak[e]['within_person']['mean']
                               - window[e]['within_person']['mean']) > 1e-6)
        self.assertEqual(differing, len(peak),
                         'the two representations produced identical numbers')
        for representation, result in self.results.items():
            self.assertEqual(result['representation'], representation)

    def test_peak_has_one_pair_per_emotion_and_window_has_many(self):
        for emotion in lp.COMMON_EMOTIONS:
            self.assertEqual(
                self.results['peak']['emotions'][emotion]['within_person']['n_pairs'], 1)
            self.assertGreater(
                self.results['window']['emotions'][emotion]['within_person']['n_pairs'], 1)

    def test_verdict_follows_the_numbers(self):
        for result in self.results.values():
            for emotion, data in result['emotions'].items():
                expected = data['within_person']['mean'] > max(
                    data['cross_1']['mean'], data['cross_2']['mean'])
                self.assertEqual(data['identity_dominates'], expected, emotion)
                self.assertAlmostEqual(
                    data['margin'],
                    data['within_person']['mean'] - max(data['cross_1']['mean'],
                                                        data['cross_2']['mean']))

    def test_within_person_compares_As_two_sessions_not_A_against_B(self):
        """Guards the one mistake that would invert the whole table.

        Both cross columns use B. If the within-person column were accidentally an A-B
        comparison too, it would equal one of them.
        """
        for result in self.results.values():
            for emotion, data in result['emotions'].items():
                self.assertNotAlmostEqual(data['within_person']['mean'],
                                          data['cross_1']['mean'], places=9, msg=emotion)
                self.assertNotAlmostEqual(data['within_person']['mean'],
                                          data['cross_2']['mean'], places=9, msg=emotion)

    def test_channel_set_changes_the_result(self):
        other = lp.run(None, 'window', 'expression55')
        self.assertEqual(other['channel_count'], 55)
        self.assertEqual(self.results['window']['channel_count'], 70)
        self.assertNotAlmostEqual(
            other['emotions']['Anger']['within_person']['mean'],
            self.results['window']['emotions']['Anger']['within_person']['mean'], places=9)

    def test_association_covers_the_repeated_subject_only(self):
        assoc = self.results['window']['association']
        self.assertEqual(assoc['subject'], 'A')
        self.assertEqual(assoc['n_emotions'], len(lp.COMMON_EMOTIONS))
        self.assertEqual({r['emotion'] for r in assoc['rows']}, set(lp.COMMON_EMOTIONS))


if __name__ == '__main__':
    unittest.main(verbosity=2)
