using System;
using System.Collections.Generic;

namespace VcuSwimLab.Calibration
{
    /// <summary>
    /// Thrown when a session is configured for something it cannot do. Always fatal, and
    /// always before the first trial.
    /// </summary>
    public sealed class CalibrationConfigurationException : Exception
    {
        public CalibrationConfigurationException(string message) : base(message) { }
    }

    /// <summary>One evaluation tick's calibrated decision.</summary>
    public readonly struct TickCalibration
    {
        /// <summary>False in raw mode, and when the calibration profile is invalid.</summary>
        public readonly bool Scored;

        public readonly int ClassIndex;
        public readonly double WinnerSimilarity;
        public readonly double RunnerUpSimilarity;
        public readonly double Margin;

        public TickCalibration(bool scored, int classIndex, double winnerSimilarity,
                               double runnerUpSimilarity, double margin)
        {
            Scored = scored;
            ClassIndex = classIndex;
            WinnerSimilarity = winnerSimilarity;
            RunnerUpSimilarity = runnerUpSimilarity;
            Margin = margin;
        }

        public static TickCalibration NotScored => new TickCalibration(false, -1, 0.0, 0.0, 0.0);
    }

    /// <summary>
    /// Runtime calibration for one session: collect the K block, seal a centroid set, then
    /// classify every D tick against it.
    /// </summary>
    /// <remarks>
    /// The centroid set is resolved once, at `Seal`, and never rebuilt. Everything that
    /// decides what this object does -- the mode, the alpha, the population profile -- is
    /// fixed at construction. An operator toggling a checkbox mid-session cannot reach it,
    /// because it holds values rather than a reference to the component the checkboxes live
    /// on.
    ///
    /// This class does no logging and knows nothing about Unity. It answers questions and
    /// reports faults; deciding what a fault means for the session belongs to the caller.
    /// </remarks>
    public sealed class RuntimeCalibrator
    {
        /// <summary>
        /// Bumped when anything that changes a calibrated prediction changes. Written into
        /// the session metadata so a pooled analysis can refuse to mix implementations.
        /// </summary>
        public const string ImplementationVersion = "runtime-calibration-v1";

        private readonly string[] classNames;
        private readonly PersonalCentroidBuilder builder;

        private double[][] activeCentroids;

        public CalibrationMode Mode { get; }

        /// <summary>The blend weight in hybrid mode; `double.NaN` in every other mode.</summary>
        public double Alpha { get; }

        /// <summary>Null unless the mode uses it, so personal-only cannot be influenced by it.</summary>
        public PopulationProfile Population { get; }

        public bool IsSealed { get; private set; }

        /// <summary>
        /// True once a usable centroid set exists. Always true in raw mode, where there is
        /// nothing to invalidate.
        /// </summary>
        public bool CalibrationValid { get; private set; }

        /// <summary>Empty when nothing went wrong.</summary>
        public string CalibrationError { get; private set; } = string.Empty;

        /// <summary>
        /// True when personal centroids were fitted. False in raw and -- by design -- in
        /// population-only, where the K block is still recorded for a comparable protocol but
        /// contributes nothing to classification.
        /// </summary>
        public bool PersonalCentroidsBuilt { get; private set; }

        public IReadOnlyList<string> ClassNames => classNames;

        private RuntimeCalibrator(CalibrationMode mode, string[] classNames,
                                  PopulationProfile population, double alpha)
        {
            Mode = mode;
            this.classNames = classNames;
            Population = population;
            Alpha = alpha;
            builder = new PersonalCentroidBuilder(classNames.Length);
        }

        /// <summary>
        /// Build a calibrator for a session, refusing a configuration it cannot honour.
        /// </summary>
        /// <param name="population">
        /// Required exactly when the mode uses it, and refused when it does not. Passing a
        /// profile into a personal-only session is an error rather than a no-op: it is the
        /// only way to guarantee at the type level that personal-only results cannot depend on
        /// the artifact, and a caller that hands one over has misunderstood the mode.
        /// </param>
        public static RuntimeCalibrator Create(CalibrationMode mode, IReadOnlyList<string> classNames,
                                               PopulationProfile population)
        {
            if (classNames == null) throw new ArgumentNullException(nameof(classNames));
            if (classNames.Count < 2)
            {
                throw new CalibrationConfigurationException(
                    $"the schema has {classNames.Count} classes; calibration needs at least two");
            }

            bool needsPopulation = CalibrationModes.UsesPopulation(mode);

            if (needsPopulation && population == null)
            {
                throw new CalibrationConfigurationException(
                    $"{CalibrationModes.Canonical(mode)} mode needs the EmojiHero population "
                    + "profile, and it did not load. Fix the artifact rather than running: a "
                    + "session that silently scored raw would be recorded as a calibrated one.");
            }

            if (!needsPopulation && population != null)
            {
                throw new CalibrationConfigurationException(
                    $"{CalibrationModes.Canonical(mode)} mode does not use a population profile, "
                    + "so one must not be supplied. Personal-only results have to be independent "
                    + "of the artifact by construction, not by coincidence.");
            }

            if (population != null && population.Centroids().Length != classNames.Count)
            {
                throw new CalibrationConfigurationException(
                    $"the population profile has {population.Centroids().Length} centroids but the "
                    + $"schema has {classNames.Count} classes");
            }

            double alpha = mode == CalibrationMode.Hybrid
                ? CalibrationMath.FrozenHybridAlpha
                : double.NaN;

            var names = new string[classNames.Count];
            for (int i = 0; i < classNames.Count; ++i) names[i] = classNames[i];

            return new RuntimeCalibrator(mode, names, population, alpha);
        }

        /// <summary>
        /// Offer a calibration-block tick.
        /// </summary>
        /// <remarks>
        /// Called in every mode, including raw and population-only. The K block runs and is
        /// recorded identically whatever the mode, so recordings stay comparable across modes;
        /// what changes is whether the collected ticks are ever turned into centroids.
        /// </remarks>
        public void ObserveCalibrationTick(int classIndex, int tickIndex, IReadOnlyList<float> scores)
        {
            if (IsSealed)
            {
                throw new InvalidOperationException(
                    "a calibration tick arrived after the profile was sealed; the calibration "
                    + "block cannot reopen once the evaluation block has started");
            }

            builder.Observe(classIndex, tickIndex, scores);
        }

        /// <summary>
        /// Close the calibration block and resolve the centroid set the evaluation block will
        /// be scored against.
        /// </summary>
        public void Seal()
        {
            if (IsSealed)
            {
                throw new InvalidOperationException("the calibration profile is already sealed");
            }

            IsSealed = true;

            if (Mode == CalibrationMode.Raw)
            {
                CalibrationValid = true;
                return;
            }

            double[][] personal = null;
            if (CalibrationModes.UsesPersonal(Mode))
            {
                if (!builder.TryBuild(classNames, out personal, out string error))
                {
                    CalibrationValid = false;
                    CalibrationError = error;
                    return;
                }

                PersonalCentroidsBuilt = true;
            }

            switch (Mode)
            {
                case CalibrationMode.Personal:
                    activeCentroids = personal;
                    break;

                case CalibrationMode.Population:
                    activeCentroids = Population.Centroids();
                    break;

                case CalibrationMode.Hybrid:
                    activeCentroids = CalibrationMath.Blend(personal, Population.Centroids(), Alpha);
                    break;

                default:
                    throw new InvalidOperationException($"unmapped mode {Mode}");
            }

            CalibrationValid = true;
        }

        /// <summary>
        /// Classify one evaluation tick against the sealed centroid set.
        /// </summary>
        /// <remarks>
        /// Returns `NotScored` in raw mode, when the profile is invalid, and when the tick
        /// itself carried no scores. Each of those is a real state of the recording and none
        /// of them is a fallback to a different method: an unscored tick is logged blank and
        /// contributes no vote, exactly as an untracked tick does in the offline analysis.
        /// </remarks>
        public TickCalibration Classify(int tickIndex, IReadOnlyList<float> scores)
        {
            if (!IsSealed)
            {
                throw new InvalidOperationException(
                    "an evaluation tick arrived before the calibration profile was sealed");
            }

            if (activeCentroids == null || !CalibrationValid || scores == null)
            {
                return TickCalibration.NotScored;
            }

            if (scores.Count != classNames.Length)
            {
                return TickCalibration.NotScored;
            }

            double[] feature;
            try
            {
                feature = CalibrationMath.LogScore(scores);
            }
            catch (ArgumentException)
            {
                return TickCalibration.NotScored;
            }

            var nearest = CalibrationMath.Nearest(activeCentroids, feature);
            return new TickCalibration(true, nearest.ClassIndex, nearest.WinnerSimilarity,
                                       nearest.RunnerUpSimilarity, nearest.Margin);
        }

        /// <summary>Per-class scored calibration-tick counts, for the session metadata.</summary>
        public int[] CalibrationTickCounts() => builder.TickCounts();

        /// <summary>
        /// The sealed centroid set, copied.
        /// </summary>
        /// <remarks>
        /// Exposed for tests and for nothing else. These vectors are a biometric description
        /// of one participant and are deliberately absent from every file this project
        /// writes -- see `SessionLogger`, which records similarities and margins but no
        /// centroid.
        /// </remarks>
        public double[][] ActiveCentroidsForTesting()
        {
            if (activeCentroids == null) return null;

            var copy = new double[activeCentroids.Length][];
            for (int e = 0; e < activeCentroids.Length; ++e)
            {
                copy[e] = (double[])activeCentroids[e].Clone();
            }

            return copy;
        }
    }
}
