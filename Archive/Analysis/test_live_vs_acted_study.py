#!/usr/bin/env python3
"""Tests for `live_vs_acted_study.py`.

Stdlib `unittest`, needing only numpy. The integration tests skip themselves when the acted
corpus or the pulled live sessions are not on the machine, so this file runs anywhere.

Each test here exists because getting the thing wrong would change the report's conclusion
without changing its shape -- the numbers would still print, still look plausible, and be
about something else. In particular: counting A's two evenings as two people, letting a live
subject into the centroid it is scored against, and returning a quiet 0.0 where an emotion is
missing all produce a readable table and a false finding.

    python Analysis/test_live_vs_acted_study.py
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
import session_analysis as sa

N = au_names.CHANNEL_COUNT


def row(prompted: str, index: int, value: float, *,
        predicted: str | None = None, status: str = sa.STATUS_OK) -> sa.Row:
    """One synthetic tick whose five frames are all `value` on every channel."""
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
        scores=[1.0] + [0.0] * 6 if ok else None,
        frames=[[value] * N for _ in range(5)] if ok else None,
    )


def session(rows: list[sa.Row], name: str = 'synthetic') -> sa.Session:
    return sa.Session(
        path=Path(name) / 'predictions.csv',
        session_id=name,
        schema=sa.Schema(classes=tuple(ad.EMOTIONS), window_frames=5, features_per_frame=N),
        rows=rows,
    )


def full_session(values: dict[str, list[float]], name: str = 'synthetic') -> sa.Session:
    """A session with one trial per common emotion, one tick per value given."""
    rows = [row(emotion, i, v)
            for emotion, vals in values.items()
            for i, v in enumerate(vals)]
    return session(rows, name)


def flat(value: float) -> np.ndarray:
    return np.full(N, value, dtype=float)


class SubjectIdentity(unittest.TestCase):
    """A's two evenings are one person."""

    def test_registry_has_two_subjects_and_three_sessions(self):
        self.assertEqual(sorted(lva.LIVE_SUBJECTS), ['A', 'B'])
        self.assertEqual(len(lva.LIVE_SUBJECTS['A']), 2)
        self.assertEqual(len(lva.LIVE_SUBJECTS['B']), 1)

        all_ids = [i for ids in lva.LIVE_SUBJECTS.values() for i in ids]
        self.assertEqual(len(all_ids), 3, 'three sessions')
        self.assertEqual(len(set(all_ids)), 3, 'no session claimed by two subjects')

    def test_the_two_a_sessions_are_the_expected_ones(self):
        # Named explicitly: if the mapping is ever edited, this fails rather than the report
        # silently becoming a three-person comparison.
        self.assertIn('20260912_213907_eec228', lva.LIVE_SUBJECTS['A'])
        self.assertIn('20260913_005336_07c050', lva.LIVE_SUBJECTS['A'])
        self.assertEqual(lva.LIVE_SUBJECTS['B'], ('20260914_175512_8ccb69',))


class TickSelection(unittest.TestCase):
    """Tick 0 out, and selection by column rather than by row order."""

    def test_transition_tick_excluded_from_the_profile(self):
        # Tick 0 is a huge outlier; every later tick is 0.5. If tick 0 leaked in, the peak
        # would be 9.0.
        values = {e: [9.0] + [0.5] * 9 for e in lva.COMMON_EMOTIONS}
        profiles = lva.live_profiles(full_session(values))
        for emotion in lva.COMMON_EMOTIONS:
            self.assertAlmostEqual(profiles[emotion]['peak'].max(), 0.5)
            self.assertEqual(profiles[emotion]['ticks'], 9)
            self.assertEqual(profiles[emotion]['ticks_prompted'], 10)

    def test_dropped_tick_does_not_shift_the_boundary(self):
        """A no_tracking tick mid-trial must not renumber the ones after it.

        Selection is on `prediction_index`. Were it done by enumerating rows, dropping the
        unscored tick 3 would pull tick 1 into position 0 and exclude the wrong sample.
        """
        rows = []
        for emotion in lva.COMMON_EMOTIONS:
            rows.append(row(emotion, 0, 9.0))                              # transition
            rows.append(row(emotion, 1, 0.5))
            rows.append(row(emotion, 2, 0.5))
            rows.append(row(emotion, 3, 0.0, status=sa.STATUS_NO_TRACKING))  # no window
            rows.append(row(emotion, 4, 0.5))
        profiles = lva.live_profiles(session(rows))
        for emotion in lva.COMMON_EMOTIONS:
            self.assertEqual(profiles[emotion]['ticks'], 3, 'three scored ticks after tick 0')
            self.assertAlmostEqual(profiles[emotion]['peak'].max(), 0.5)

    def test_missing_prediction_index_is_refused(self):
        rows = []
        for emotion in lva.COMMON_EMOTIONS:
            r = row(emotion, 1, 0.5)
            r.prediction_index = None
            rows.append(r)
        with self.assertRaises(lva.StudyError):
            lva.live_profiles(session(rows))

    def test_missing_emotion_raises_rather_than_returning_zero(self):
        values = {e: [0.0, 0.5] for e in lva.COMMON_EMOTIONS if e != 'Fear'}
        with self.assertRaises(lva.StudyError) as caught:
            lva.live_profiles(full_session(values))
        self.assertIn('Fear', str(caught.exception))

    def test_trial_with_only_a_transition_tick_raises(self):
        values = {e: ([0.5] if e != 'Sadness' else [0.5] * 3) for e in lva.COMMON_EMOTIONS}
        with self.assertRaises(lva.StudyError):
            lva.live_profiles(full_session(values))


class Scope(unittest.TestCase):
    """Six emotions, expression channels only."""

    def test_only_the_six_common_emotions(self):
        self.assertEqual(set(lva.COMMON_EMOTIONS), set(ad.EMOTIONS) - {'Neutral'})
        self.assertNotIn('Neutral', lva.COMMON_EMOTIONS)
        self.assertEqual(len(lva.COMMON_EMOTIONS), 6)

    def test_neutral_rows_are_ignored_not_profiled(self):
        values = {e: [0.0, 0.5] for e in lva.COMMON_EMOTIONS}
        s = full_session(values)
        s.rows.extend(row('Neutral', i, 9.0) for i in range(3))
        profiles = lva.live_profiles(s)
        self.assertNotIn('Neutral', profiles)
        self.assertEqual(sorted(profiles), sorted(lva.COMMON_EMOTIONS))

    def test_channel_order_matches_au_names(self):
        self.assertEqual(N, 70)
        self.assertEqual(lva.EXPRESSION, list(au_names.EXPRESSION_INDICES))
        # Gaze and tongue are the channels the comparison deliberately drops.
        for index in au_names.REGIONS['eye_gaze'] + au_names.REGIONS['tongue']:
            self.assertNotIn(index, lva.EXPRESSION)
        self.assertEqual(len(lva.EXPRESSION), 70 - 8 - 7)

    def test_wrong_channel_count_is_refused(self):
        s = full_session({e: [0.0, 0.5] for e in lva.COMMON_EMOTIONS})
        s.schema = sa.Schema(classes=s.schema.classes, window_frames=5, features_per_frame=63)
        with self.assertRaises(lva.StudyError):
            lva.live_profiles(s)


class SessionCombination(unittest.TestCase):
    """Equal weight per session, not per frame."""

    def test_equal_weight_regardless_of_session_length(self):
        short = lva.live_profiles(full_session({e: [0.0, 1.0] for e in lva.COMMON_EMOTIONS}))
        long = lva.live_profiles(full_session({e: [0.0] + [0.0] * 40 + [0.0]
                                               for e in lva.COMMON_EMOTIONS}))
        combined = lva.combine_sessions([short, long])
        for emotion in lva.COMMON_EMOTIONS:
            # Equal weight: (1.0 + 0.0) / 2. Pooling frames would give the 41-tick session
            # ~95% of the say and land near 0.0; pooling peaks would give 1.0.
            self.assertAlmostEqual(combined[emotion]['peak'].max(), 0.5)
            self.assertEqual(combined[emotion]['sessions'], 2)

    def test_single_session_passes_through(self):
        one = lva.live_profiles(full_session({e: [0.0, 0.7] for e in lva.COMMON_EMOTIONS}))
        combined = lva.combine_sessions([one])
        for emotion in lva.COMMON_EMOTIONS:
            self.assertAlmostEqual(combined[emotion]['peak'].max(), 0.7)
            self.assertEqual(combined[emotion]['sessions'], 1)

    def test_hole_in_one_session_is_refused(self):
        full = lva.live_profiles(full_session({e: [0.0, 0.5] for e in lva.COMMON_EMOTIONS}))
        partial = dict(full)
        del partial['Fear']
        with self.assertRaises(lva.StudyError):
            lva.combine_sessions([full, partial])


class Similarity(unittest.TestCase):
    """Cosine and nearest-emotion, on vectors worked out by hand."""

    def test_cosine_on_known_vectors(self):
        a, b = np.zeros(N), np.zeros(N)
        x, y = lva.EXPRESSION[0], lva.EXPRESSION[1]

        a[x] = 1.0
        b[x] = 5.0
        self.assertAlmostEqual(lva.cosine(a, b), 1.0, places=9)   # same direction

        b[:] = 0.0
        b[y] = 1.0
        self.assertAlmostEqual(lva.cosine(a, b), 0.0, places=9)   # orthogonal

        b[y] = 1.0
        b[x] = 1.0
        self.assertAlmostEqual(lva.cosine(a, b), 1 / np.sqrt(2), places=9)   # 45 degrees

    def test_cosine_ignores_gaze_and_tongue(self):
        gaze = au_names.REGIONS['eye_gaze'][0]
        a, b = np.zeros(N), np.zeros(N)
        a[lva.EXPRESSION[0]] = 1.0
        b[lva.EXPRESSION[0]] = 1.0
        b[gaze] = 100.0     # a player who looked around the scene
        self.assertAlmostEqual(lva.cosine(a, b), 1.0, places=9)

    def test_cosine_against_zero_vector_raises(self):
        with self.assertRaises(lva.StudyError):
            lva.cosine(flat(1.0), np.zeros(N))

    def test_nearest_emotion_picks_the_closest_centroid(self):
        centroids = {}
        for i, emotion in enumerate(lva.COMMON_EMOTIONS):
            v = np.zeros(N)
            v[lva.EXPRESSION[i]] = 1.0
            centroids[emotion] = v

        live = np.zeros(N)
        live[lva.EXPRESSION[2]] = 1.0     # exactly the third emotion's centroid
        third = lva.COMMON_EMOTIONS[2]

        result = lva.nearest_emotion(live, centroids, prompted=third)
        self.assertEqual(result['nearest'], third)
        self.assertTrue(result['nearest_is_prompted'])
        self.assertEqual(result['prompted_rank'], 1)
        self.assertAlmostEqual(result['prompted_cosine'], 1.0, places=9)
        self.assertAlmostEqual(result['margin_over_nearest_wrong'], 1.0, places=9)

    def test_nearest_emotion_reports_a_collision(self):
        """Asked for one emotion, nearest to another, and last of the six.

        The live vector is a graded mixture of all six one-hot centroids, so every cosine is
        distinct and the ranking is fully determined -- no tie for stable-sort order to
        decide. The prompted emotion gets the smallest weight, so it must come last.
        """
        centroids = {}
        for i, emotion in enumerate(lva.COMMON_EMOTIONS):
            v = np.zeros(N)
            v[lva.EXPRESSION[i]] = 1.0
            centroids[emotion] = v

        prompted = lva.COMMON_EMOTIONS[0]
        nearest = lva.COMMON_EMOTIONS[5]
        live = np.zeros(N)
        for i in range(len(lva.COMMON_EMOTIONS)):
            live[lva.EXPRESSION[i]] = float(i + 1)     # 1..6, prompted weakest

        result = lva.nearest_emotion(live, centroids, prompted=prompted)
        self.assertEqual(result['nearest'], nearest)
        self.assertFalse(result['nearest_is_prompted'])
        self.assertEqual(result['nearest_wrong'], nearest)
        self.assertLess(result['margin_over_nearest_wrong'], 0)
        self.assertEqual(result['prompted_rank'], len(lva.COMMON_EMOTIONS))
        self.assertEqual([r['emotion'] for r in result['ranking']],
                         list(reversed(lva.COMMON_EMOTIONS)))

    def test_nearest_emotion_without_a_centroid_for_the_prompt_raises(self):
        centroids = {'Anger': flat(1.0)}
        with self.assertRaises(lva.StudyError):
            lva.nearest_emotion(flat(1.0), centroids, prompted='Fear')


class Centroids(unittest.TestCase):
    """Equal weight per participant, and nothing built from a hole."""

    def test_equal_weight_per_participant(self):
        profiles = {}
        for i, p in enumerate(['p1', 'p2', 'p3']):
            v = np.zeros(N)
            v[lva.EXPRESSION[0]] = float(i)      # 0, 1, 2
            profiles[(p, 'Anger')] = v
        c = lva.centroid(profiles, ['p1', 'p2', 'p3'], 'Anger')
        self.assertAlmostEqual(c[lva.EXPRESSION[0]], 1.0)

    def test_missing_participant_raises(self):
        profiles = {('p1', 'Anger'): flat(1.0)}
        with self.assertRaises(lva.StudyError) as caught:
            lva.centroid(profiles, ['p1', 'p2'], 'Anger')
        self.assertIn('p2', str(caught.exception))

    def test_empty_participant_list_raises(self):
        with self.assertRaises(lva.StudyError):
            lva.centroid({}, [], 'Anger')


class LeaveOneOut(unittest.TestCase):
    """The reference distribution, and the live subject's position in it."""

    def setUp(self):
        # Four training participants on a circle in a two-channel plane, so every
        # leave-one-out cosine is computable by hand rather than by running the code.
        self.x, self.y = lva.EXPRESSION[0], lva.EXPRESSION[1]
        self.train = ['p1', 'p2', 'p3', 'p4']
        self.profiles = {}
        for p, angle in zip(self.train, [0.0, 0.1, -0.1, 0.0]):
            v = np.zeros(N)
            v[self.x], v[self.y] = np.cos(angle), np.sin(angle)
            self.profiles[(p, 'Anger')] = v

    def vec(self, angle: float) -> np.ndarray:
        v = np.zeros(N)
        v[self.x], v[self.y] = np.cos(angle), np.sin(angle)
        return v

    def test_reference_excludes_the_participant_being_scored(self):
        ref = lva.loo_reference(self.profiles, self.train, 'Anger')
        self.assertEqual(len(ref), len(self.train))
        self.assertEqual(sorted(r['participant'] for r in ref), sorted(self.train))

        # p2 sits at +0.1 rad; its leave-one-out centroid is the mean of 0, -0.1 and 0, which
        # points at a negative angle, so p2 must score *below* p1 (which sits at 0 and whose
        # own centroid also points near 0).
        by_name = {r['participant']: r['cosine'] for r in ref}
        self.assertLess(by_name['p2'], by_name['p1'])
        self.assertAlmostEqual(by_name['p2'], by_name['p3'], places=9)  # mirrored

    def test_self_inclusion_would_inflate_the_reference(self):
        """A participant scored against a centroid containing them looks closer."""
        loo = {r['participant']: r['cosine']
               for r in lva.loo_reference(self.profiles, self.train, 'Anger')}
        for p in self.train:
            inclusive = lva.cosine(self.profiles[(p, 'Anger')],
                                   lva.centroid(self.profiles, self.train, 'Anger'))
            self.assertGreaterEqual(round(inclusive, 9), round(loo[p], 9))

    def test_live_subject_never_enters_the_centroid(self):
        """Adding the live profile to the profile dict must not change its own position."""
        live = self.vec(1.2)                       # far from every training participant
        before = lva.position_in_training(live, self.profiles, self.train, 'Anger')

        polluted = dict(self.profiles)
        polluted[('live_A', 'Anger')] = live       # present in the dict, absent from `train`
        after = lva.position_in_training(live, polluted, self.train, 'Anger')

        self.assertAlmostEqual(before['cosine_matched_loo'], after['cosine_matched_loo'])
        self.assertEqual(before['n_reference_below'], after['n_reference_below'])
        self.assertEqual(before['cosine_to_full_centroid'], after['cosine_to_full_centroid'])
        self.assertNotIn('live_A', [r['participant'] for r in after['reference']])

    def test_matched_comparison_uses_nine_member_centroids(self):
        """Both sides of the percentile are scored against same-sized centroids."""
        live = self.vec(0.0)
        pos = lva.position_in_training(live, self.profiles, self.train, 'Anger')
        # Computed by hand: mean over the four leave-one-out centroids.
        expected = float(np.mean([
            lva.cosine(live, lva.centroid(self.profiles,
                                          [p for p in self.train if p != held], 'Anger'))
            for held in self.train]))
        self.assertAlmostEqual(pos['cosine_matched_loo'], expected, places=12)

    def test_an_outlier_lands_below_everyone(self):
        live = self.vec(1.4)                       # ~80 degrees off the training direction
        pos = lva.position_in_training(live, self.profiles, self.train, 'Anger')
        self.assertEqual(pos['n_reference_below'], 0)
        self.assertEqual(pos['percentile'], 0.0)
        self.assertFalse(pos['inside_reference_range'])

    def test_a_typical_profile_lands_inside_the_range(self):
        live = self.vec(0.0)
        pos = lva.position_in_training(live, self.profiles, self.train, 'Anger')
        self.assertTrue(pos['inside_reference_range'])
        self.assertGreater(pos['n_reference_below'], 0)

    def test_percentile_counts_strictly_below(self):
        live = self.vec(0.0)
        pos = lva.position_in_training(live, self.profiles, self.train, 'Anger')
        values = [r['cosine'] for r in pos['reference']]
        self.assertEqual(pos['n_reference_below'], sum(1 for v in values
                                                       if v < pos['cosine_matched_loo']))
        self.assertAlmostEqual(pos['percentile'],
                               100.0 * pos['n_reference_below'] / len(values))


class ModelResults(unittest.TestCase):

    def test_accuracy_over_scored_ticks_and_over_all_ticks(self):
        rows = []
        for emotion in lva.COMMON_EMOTIONS:
            rows.append(row(emotion, 0, 0.5))                              # tick 0, a hit
            rows.append(row(emotion, 1, 0.5))                              # hit
            rows.extend(row(emotion, i, 0.5, predicted='Neutral')          # three misses
                        for i in (2, 3, 4))
        model = lva.live_model_results([session(rows)])
        for emotion in lva.COMMON_EMOTIONS:
            self.assertAlmostEqual(model[emotion]['acc_scored'], 1 / 4)    # ticks 1-4
            self.assertAlmostEqual(model[emotion]['acc_all'], 2 / 5)       # ticks 0-4
            self.assertEqual(model[emotion]['dominant_wrong'], 'Neutral')
            self.assertEqual(model[emotion]['dominant_wrong_ticks'], 3)

    def test_no_scored_tick_raises(self):
        rows = [row(e, 0, 0.5) for e in lva.COMMON_EMOTIONS]
        with self.assertRaises(lva.StudyError):
            lva.live_model_results([session(rows)])


class Diagnostics(unittest.TestCase):

    def test_dead_and_active_channels_ignore_gaze(self):
        peak = np.zeros(N)
        for i in lva.EXPRESSION[:5]:
            peak[i] = 1.0
        for i in au_names.REGIONS['eye_gaze']:
            peak[i] = 1.0           # gaze must not count as expressive
        d = lva.diagnostics_from_peaks([peak], tracking_rate=1.0)
        self.assertEqual(d['active_channels'], 5)
        self.assertEqual(d['dead_channels'], len(lva.EXPRESSION) - 5)

    def test_peaks_combine_by_maximum_across_emotions(self):
        a, b = np.zeros(N), np.zeros(N)
        a[lva.EXPRESSION[0]] = 1.0
        b[lva.EXPRESSION[1]] = 1.0
        d = lva.diagnostics_from_peaks([a, b], tracking_rate=1.0)
        self.assertEqual(d['active_channels'], 2)

    def test_symmetric_face_has_zero_asymmetry(self):
        peak = np.zeros(N)
        for left, right in au_names.BILATERAL_PAIRS:
            peak[left] = peak[right] = 1.0
        d = lva.diagnostics_from_peaks([peak], tracking_rate=1.0)
        self.assertAlmostEqual(d['mean_bilateral_asymmetry'], 0.0)
        self.assertEqual(d['n_asymmetry_pairs'], len(au_names.BILATERAL_PAIRS))

    def test_one_sided_pair_reports_full_asymmetry(self):
        peak = np.zeros(N)
        left, right = au_names.BILATERAL_PAIRS[0]
        peak[left], peak[right] = 1.0, 0.0
        d = lva.diagnostics_from_peaks([peak], tracking_rate=1.0)
        self.assertAlmostEqual(d['mean_bilateral_asymmetry'], 1.0)
        self.assertEqual(d['n_asymmetry_pairs'], 1)     # the rest are below threshold


# --------------------------------------------------------------- integration


def corpus_or_skip() -> ad.Corpus:
    try:
        return ad.load_corpus(ad.find_corpus_root(None))
    except ad.CorpusError as exc:
        raise unittest.SkipTest(f'acted corpus unavailable: {exc}')


def live_or_skip() -> dict[str, list[Path]]:
    try:
        return lva.resolve_live_sessions(None)
    except lva.StudyError as exc:
        raise unittest.SkipTest(f'live sessions unavailable: {exc}')


class RealData(unittest.TestCase):

    def test_training_split_has_ten_participants(self):
        corpus = corpus_or_skip()
        self.assertEqual(len(lva.participants_in_split(corpus, 'train')), 10)
        self.assertEqual(len(corpus.participants), 15)

    def test_acted_profiles_cover_every_participant_and_common_emotion(self):
        corpus = corpus_or_skip()
        for sampling in ('full', 'decimated'):
            profiles = lva.acted_profiles(corpus, sampling)
            self.assertEqual(len(profiles), 15 * 6, sampling)
            self.assertFalse(any(e == 'Neutral' for _, e in profiles), sampling)

    def test_live_sessions_resolve_to_two_subjects(self):
        files = live_or_skip()
        self.assertEqual(len(files['A']), 2)
        self.assertEqual(len(files['B']), 1)
        self.assertEqual(len({str(p) for ps in files.values() for p in ps}), 3)

    def test_end_to_end_conclusions_survive_both_samplings(self):
        """The sampling choice must not decide which subject is closer."""
        corpus_or_skip()
        live_or_skip()
        results = {s: lva.run(None, None, s, lva.DEFAULT_CONDITION)
                   for s in ('full', 'decimated')}

        for emotion in lva.COMMON_EMOTIONS:
            signs = set()
            for result in results.values():
                a = result['subjects']['A']['emotions'][emotion]['position']
                b = result['subjects']['B']['emotions'][emotion]['position']
                signs.add(a['cosine_matched_loo'] > b['cosine_matched_loo'])
            self.assertEqual(len(signs), 1,
                             f'{emotion}: which subject is closer depends on the sampling')

    def test_live_subjects_are_absent_from_the_acted_profiles(self):
        corpus = corpus_or_skip()
        profiles = lva.acted_profiles(corpus, 'full')
        live_ids = {i for ids in lva.LIVE_SUBJECTS.values() for i in ids}
        for participant, _ in profiles:
            self.assertNotIn(participant, live_ids)
            self.assertNotIn(participant, ('A', 'B'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
