using System;
using System.Collections.Generic;
using System.Globalization;

namespace VcuSwimLab.Calibration
{
    /// <summary>
    /// Collects the calibration block's movement features and fits one movement centroid per
    /// class: the player's typical face for each emotion.
    /// </summary>
    /// <remarks>
    /// The movement counterpart of `PersonalCentroidBuilder`, with the same rules: ticks 1..9
    /// only, exactly nine per class, and refuse rather than average over fewer. The feature is
    /// `CalibrationMath.MovementFeature` -- the mean over the window's frames of the 63 face
    /// action units -- instead of the log-scores.
    /// </remarks>
    public sealed class MovementCentroidBuilder
    {
        private readonly int classCount;
        private readonly List<double[]>[] observations;
        private string fault;

        public MovementCentroidBuilder(int classCount)
        {
            if (classCount <= 1)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(classCount), classCount, "need at least two classes");
            }

            this.classCount = classCount;
            observations = new List<double[]>[classCount];
            for (int c = 0; c < classCount; ++c)
            {
                observations[c] = new List<double[]>(PersonalCentroidBuilder.RequiredTicksPerCentroid);
            }
        }

        /// <summary>Scored windows accepted so far for a class.</summary>
        public int TickCount(int classIndex) => observations[classIndex].Count;

        /// <summary>
        /// Offer one calibration tick's window. Ticks outside 1..9 are ignored, and a null
        /// window is an untracked tick that does not reach the centroid.
        /// </summary>
        /// <returns>True when the window was taken into the centroid.</returns>
        public bool Observe(int classIndex, int tickIndex, IReadOnlyList<float[]> window)
        {
            if (classIndex < 0 || classIndex >= classCount)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(classIndex), classIndex, $"outside 0..{classCount - 1}");
            }

            if (!PersonalCentroidBuilder.IsScoredTick(tickIndex) || window == null)
            {
                return false;
            }

            try
            {
                observations[classIndex].Add(CalibrationMath.MovementFeature(window));
                return true;
            }
            catch (ArgumentException e)
            {
                Fault($"a calibration window for class {classIndex} was refused by the movement "
                      + $"feature: {e.Message}");
                return false;
            }
        }

        /// <summary>Fit the movement centroids, or explain why not. Never a partial set.</summary>
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
                if (have != PersonalCentroidBuilder.RequiredTicksPerCentroid)
                {
                    short_.Add($"{Name(classNames, c)} {have}/{PersonalCentroidBuilder.RequiredTicksPerCentroid}");
                }
            }

            if (short_.Count > 0)
            {
                error = "the calibration block did not contribute "
                        + $"{PersonalCentroidBuilder.RequiredTicksPerCentroid} tracked windows for "
                        + "every class: " + string.Join(", ", short_)
                        + ". Movement calibration needs the action-unit window of every scored "
                        + "calibration tick.";
                return false;
            }

            var fitted = new double[classCount][];
            for (int c = 0; c < classCount; ++c)
            {
                fitted[c] = CalibrationMath.Mean(observations[c]);
            }

            centroids = fitted;
            error = null;
            return true;
        }

        private static string Name(IReadOnlyList<string> classNames, int c)
            => classNames != null && c < classNames.Count
                ? classNames[c]
                : c.ToString(CultureInfo.InvariantCulture);

        private void Fault(string message)
        {
            if (fault == null) fault = message;
        }
    }
}
