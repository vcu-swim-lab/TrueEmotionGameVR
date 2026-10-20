using System.Collections.Generic;

using NUnit.Framework;

namespace VcuSeCenter.Calibration.Tests
{
    public class TrialMajorityTests
    {
        private const int Anger = 0;
        private const int Disgust = 1;
        private const int Fear = 2;
        private const int Sadness = 5;
        private const int Classes = 7;

        private static List<TickVote> Ticks(params int[] classes)
        {
            var votes = new List<TickVote>();
            for (int i = 0; i < classes.Length; ++i)
            {
                votes.Add(new TickVote(i + 1, classes[i]));
            }

            return votes;
        }

        [Test]
        public void An_outright_majority_wins_without_a_tie()
        {
            var vote = TrialMajority.Vote(
                Ticks(Anger, Anger, Anger, Anger, Anger, Fear, Fear, Fear, Fear), Classes);

            Assert.IsTrue(vote.Scorable);
            Assert.AreEqual(Anger, vote.ClassIndex);
            Assert.IsFalse(vote.TieOccurred);
            Assert.AreEqual(9, vote.ScoredTicks);
        }

        [Test]
        public void A_tie_is_broken_by_the_final_tick_when_that_class_is_tied()
        {
            // Anger 3, Fear 3; the last tick is Fear, which is in the tied set.
            var vote = TrialMajority.Vote(
                Ticks(Anger, Anger, Fear, Fear, Anger, Fear), Classes);

            Assert.AreEqual(Fear, vote.ClassIndex);
            Assert.IsTrue(vote.TieOccurred);
        }

        [Test]
        public void A_tie_falls_through_to_class_order_when_the_final_tick_is_not_tied()
        {
            // Disgust 2, Fear 2, Sadness 1. The last tick is Sadness, which did not tie, so
            // step 2 cannot decide it and the first tied class in class order takes it.
            var vote = TrialMajority.Vote(
                Ticks(Disgust, Disgust, Fear, Fear, Sadness), Classes);

            Assert.AreEqual(Disgust, vote.ClassIndex);
            Assert.IsTrue(vote.TieOccurred);
        }

        [Test]
        public void The_final_tick_is_the_highest_tick_index_not_the_last_row()
        {
            // Rows deliberately out of order. `np.argmax(ticks)` picks the highest tick
            // index, so tick 6 decides this even though tick 2 is written last.
            var vote = TrialMajority.Vote(new List<TickVote>
            {
                new TickVote(1, Anger),
                new TickVote(6, Fear),
                new TickVote(3, Anger),
                new TickVote(2, Fear),
            }, Classes);

            Assert.AreEqual(Fear, vote.ClassIndex);
            Assert.IsTrue(vote.TieOccurred);
        }

        [Test]
        public void A_unanimous_trial_is_not_a_tie()
        {
            var vote = TrialMajority.Vote(Ticks(Fear, Fear, Fear), Classes);

            Assert.AreEqual(Fear, vote.ClassIndex);
            Assert.IsFalse(vote.TieOccurred);
        }

        [Test]
        public void A_single_tick_decides_the_trial_on_its_own()
        {
            var vote = TrialMajority.Vote(Ticks(Sadness), Classes);

            Assert.IsTrue(vote.Scorable);
            Assert.AreEqual(Sadness, vote.ClassIndex);
            Assert.IsFalse(vote.TieOccurred);
            Assert.AreEqual(1, vote.ScoredTicks);
        }

        [Test]
        public void A_trial_with_no_scored_tick_is_not_scorable()
        {
            var vote = TrialMajority.Vote(new List<TickVote>(), Classes);

            Assert.IsFalse(vote.Scorable);
            Assert.AreEqual(0, vote.ScoredTicks);
        }

        [Test]
        public void A_seven_way_tie_resolves_to_the_first_class_in_order()
        {
            // Every class once, and the final tick is Surprise -- which is tied, so step 2
            // takes it. This pins that step 2 outranks step 3 rather than the other way round.
            var vote = TrialMajority.Vote(Ticks(0, 1, 2, 3, 4, 5, 6), Classes);

            Assert.AreEqual(6, vote.ClassIndex);
            Assert.IsTrue(vote.TieOccurred);
        }

        [Test]
        public void A_vote_for_a_class_outside_the_schema_is_refused()
        {
            Assert.Throws<System.ArgumentOutOfRangeException>(
                () => TrialMajority.Vote(Ticks(9), Classes));
            Assert.Throws<System.ArgumentOutOfRangeException>(
                () => TrialMajority.Vote(Ticks(-1), Classes));
        }
    }
}
