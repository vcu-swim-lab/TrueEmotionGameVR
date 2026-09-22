using System;
using System.Collections.Generic;
using System.Globalization;

namespace VcuSwimLab.Calibration
{
    /// <summary>
    /// Collects the calibration block's log-score vectors and fits one centroid per class.
    /// </summary>
    /// <remarks>
    /// Mirrors what `calibration_study.evaluate` does with `role_masks(..., SCORED_TICKS)`:
    /// take the calibration-block rows whose tick index is 1..9, transform them to log-scores,
    /// and average them per class. The `required_ticks` gate is here too, and it refuses
    /// rather than averages -- a centroid built from two of nine ticks is not a weaker version
    /// of the same estimate, it is a different one, and producing it quietly is how a bad
    /// recording turns into a result.
    /// </remarks>
    public sealed class PersonalCentroidBuilder
    {
        /// <summary>
        /// First scored tick. Tick 0 is a transition sample, caught as the hold screen appears
        /// and before the participant has begun; every block drops it.
        /// </summary>
        public const int FirstScoredTick = 1;

        /// <summary>Last scored tick, so a complete hold contributes nine.</summary>
        public const int LastScoredTick = 9;

        /// <summary>
        /// Ticks a class must contribute before a centroid is fitted for it.
        /// </summary>
        /// <remarks>
        /// `calibration_study.TICKS_REQUIRED_PER_CENTROID`. The Python asks for *at least*
        /// nine; with ticks 1..9 the most a trial can offer is nine, so the two conditions
        /// coincide. This side checks for exactly nine, which additionally catches a protocol
        /// that started emitting more ticks per trial without the analysis being told.
        /// </remarks>
        public const int RequiredTicksPerCentroid = 9;

        private readonly int classCount;
        private readonly List<double[]>[] observations;
        private readonly int[] offered;
        private string fault;

        public PersonalCentroidBuilder(int classCount)
        {
            if (classCount <= 1)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(classCount), classCount, "need at least two classes");
            }

            this.classCount = classCount;
            observations = new List<double[]>[classCount];
            offered = new int[classCount];
            for (int c = 0; c < classCount; ++c)
            {
                observations[c] = new List<double[]>(RequiredTicksPerCentroid);
            }
        }

        /// <summary>Whether a tick index is inside the scored window.</summary>
        public static bool IsScoredTick(int tickIndex)
            => tickIndex >= FirstScoredTick && tickIndex <= LastScoredTick;

        /// <summary>Scored ticks accepted so far for a class.</summary>
        public int TickCount(int classIndex) => observations[classIndex].Count;

        /// <summary>
        /// Offer one calibration tick. Ticks outside 1..9 are ignored, which is how tick 0
        /// stays out of the centroid.
        /// </summary>
        /// <returns>True when the tick was taken into the centroid.</returns>
        public bool Observe(int classIndex, int tickIndex, IReadOnlyList<float> scores)
        {
            if (classIndex < 0 || classIndex >= classCount)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(classIndex), classIndex, $"outside 0..{classCount - 1}");
            }

            if (!IsScoredTick(tickIndex))
            {
                return false;
            }

            offered[classIndex]++;

            // A tick with no scores is an untracked tick; it is logged as such and simply does
            // not reach the centroid, which is what makes the count below fall short and the
            // whole calibration fail rather than quietly average over fewer samples.
            if (scores == null)
            {
                return false;
            }

            if (scores.Count != classCount)
            {
                Fault($"a calibration tick carried {scores.Count} scores but the schema has "
                      + $"{classCount} classes");
                return false;
            }

            try
            {
                observations[classIndex].Add(CalibrationMath.LogScore(scores));
                return true;
            }
            catch (ArgumentException e)
            {
                Fault($"a calibration tick for class {classIndex} held a value the log-score "
                      + $"transform refused: {e.Message}");
                return false;
            }
        }

        /// <summary>
        /// Fit the centroids, or explain why not.
        /// </summary>
        /// <remarks>
        /// Never returns a partial set. The caller has exactly two outcomes to handle --
        /// a complete profile, or a session marked invalid with a reason -- and no third one
        /// where some classes calibrated and the rest fell back to something else.
        /// </remarks>
        public bool TryBuild(IReadOnlyList<string> classNames, out double[][] centroids, out string error)
        {
            centroids = null;

            if (fault != null)
            {
                error = fault;
                return false;
            }

            var short_ = new List<string>();
            for (int c = 0; c < classCount; ++c)
            {
                int have = observations[c].Count;
                if (have != RequiredTicksPerCentroid)
                {
                    string name = classNames != null && c < classNames.Count
                        ? classNames[c]
                        : c.ToString(CultureInfo.InvariantCulture);
                    short_.Add($"{name} {have}/{RequiredTicksPerCentroid}");
                }
            }

            if (short_.Count > 0)
            {
                error = "the calibration block did not contribute "
                        + $"{RequiredTicksPerCentroid} tracked ticks for every class: "
                        + string.Join(", ", short_)
                        + ". Ticks 0 and any untracked tick are excluded by design, so this "
                        + "normally means face tracking dropped during the calibration block.";
                return false;
            }

            var fitted = new double[classCount][];
            for (int c = 0; c < classCount; ++c)
            {
                var centroid = CalibrationMath.Mean(observations[c]);

                for (int i = 0; i < centroid.Length; ++i)
                {
                    if (double.IsNaN(centroid[i]) || double.IsInfinity(centroid[i]))
                    {
                        string name = classNames != null && c < classNames.Count
                            ? classNames[c]
                            : c.ToString(CultureInfo.InvariantCulture);
                        error = $"the centroid for {name} came out non-finite at dimension {i}";
                        return false;
                    }
                }

                fitted[c] = centroid;
            }

            centroids = fitted;
            error = null;
            return true;
        }

        /// <summary>Per-class scored-tick counts, for the session metadata.</summary>
        public int[] TickCounts()
        {
            var counts = new int[classCount];
            for (int c = 0; c < classCount; ++c)
            {
                counts[c] = observations[c].Count;
            }

            return counts;
        }

        private void Fault(string message)
        {
            // First fault wins: the later ones are usually the same cause reported again, and
            // the operator needs the thing that went wrong first.
            if (fault == null) fault = message;
        }
    }
}
