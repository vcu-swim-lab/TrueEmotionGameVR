using System;
using System.Collections.Generic;

namespace VcuSeCenter.Calibration
{
    /// <summary>One tick's contribution to a trial vote.</summary>
    public readonly struct TickVote
    {
        public readonly int TickIndex;
        public readonly int ClassIndex;

        public TickVote(int tickIndex, int classIndex)
        {
            TickIndex = tickIndex;
            ClassIndex = classIndex;
        }
    }

    /// <summary>The outcome of collapsing a trial's ticks to one prediction.</summary>
    public readonly struct TrialVote
    {
        /// <summary>False when the trial contributed no scored tick at all.</summary>
        public readonly bool Scorable;

        public readonly int ClassIndex;

        /// <summary>
        /// True when more than one class shared the highest vote count, so the outcome was
        /// decided by the tie rule rather than by the count. Recorded per trial: a session
        /// whose trials are routinely decided by a tie-break is a session whose ticks did not
        /// agree, and that is worth seeing without recounting the votes by hand.
        /// </summary>
        public readonly bool TieOccurred;

        public readonly int ScoredTicks;

        public TrialVote(bool scorable, int classIndex, bool tieOccurred, int scoredTicks)
        {
            Scorable = scorable;
            ClassIndex = classIndex;
            TieOccurred = tieOccurred;
            ScoredTicks = scoredTicks;
        }

        public static TrialVote Unscorable => new TrialVote(false, -1, false, 0);
    }

    /// <summary>
    /// `calibration_study.trial_vote`: ticks to one prediction per trial, by majority.
    /// </summary>
    public static class TrialMajority
    {
        /// <summary>
        /// The frozen tie rule, applied identically to every method being compared:
        ///
        ///   1. the class winning the most ticks in the trial;
        ///   2. on a tie, the prediction of the trial's final tracked tick;
        ///   3. if that still does not decide it, the first tied class in class order.
        /// </summary>
        /// <remarks>
        /// Frozen before collection and reused verbatim here rather than re-derived, because
        /// a rule chosen per method is a way of choosing the winner. Step 2 is not arbitrary:
        /// a tie is usually a trial that changed its mind partway through, and the end of a
        /// held expression is the part the participant had most settled into. Step 3 is
        /// arbitrary on purpose -- it has to be fixed, decided in advance, and not read off
        /// the scores.
        ///
        /// Step 1 counts into a per-class array rather than a dictionary. The Python keys a
        /// dictionary by label, whose insertion order never reaches the result: a single
        /// winner is the same class either way, and a tie is resolved by the final tick and
        /// then by class order, neither of which depends on how the counts were stored.
        /// </remarks>
        public static TrialVote Vote(IReadOnlyList<TickVote> ticks, int classCount)
        {
            if (ticks == null) throw new ArgumentNullException(nameof(ticks));
            if (classCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(classCount), classCount, "need at least one class");
            }

            if (ticks.Count == 0)
            {
                return TrialVote.Unscorable;
            }

            var counts = new int[classCount];
            int lastTickIndex = int.MinValue;
            int lastTickClass = -1;

            for (int i = 0; i < ticks.Count; ++i)
            {
                int cls = ticks[i].ClassIndex;
                if (cls < 0 || cls >= classCount)
                {
                    throw new ArgumentOutOfRangeException(
                        nameof(ticks), cls, $"tick {i} voted for class {cls}, outside 0..{classCount - 1}");
                }

                counts[cls]++;

                // Strictly greater, so the first row carrying the highest tick index wins --
                // matching `np.argmax`, which returns the first occurrence of the maximum.
                if (ticks[i].TickIndex > lastTickIndex)
                {
                    lastTickIndex = ticks[i].TickIndex;
                    lastTickClass = cls;
                }
            }

            // Seeded from the first class rather than from zero: seeding from zero would let
            // a class that no tick voted for set the bar, and the highest count would then be
            // whichever class merely beat nothing.
            int best = counts[0];
            for (int c = 1; c < classCount; ++c)
            {
                if (counts[c] > best) best = counts[c];
            }

            int tiedCount = 0;
            int firstTied = -1;
            bool lastTickIsTied = false;
            for (int c = 0; c < classCount; ++c)
            {
                if (counts[c] != best) continue;

                tiedCount++;
                if (firstTied < 0) firstTied = c;
                if (c == lastTickClass) lastTickIsTied = true;
            }

            if (tiedCount == 1)
            {
                return new TrialVote(true, firstTied, false, ticks.Count);
            }

            int winner = lastTickIsTied ? lastTickClass : firstTied;
            return new TrialVote(true, winner, true, ticks.Count);
        }
    }
}
