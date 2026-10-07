using System;

namespace VcuSwimLab.Calibration
{
    /// <summary>
    /// The calibration settings a session actually ran under, captured once and immutable
    /// thereafter.
    /// </summary>
    /// <remarks>
    /// The two operator toggles live on an inspector component, and inspector fields can be
    /// changed while the game is playing. A session that started as personal-only and finished
    /// as hybrid would be a recording of neither, and nothing in the file would say so.
    ///
    /// So the toggles are read exactly once, at session start, into this object. Everything
    /// downstream -- the calibrator, the CSV, the metadata -- reads from here. There is no
    /// path back to the component, which is what makes "locked at session start" a property of
    /// the design rather than a discipline someone has to remember.
    /// </remarks>
    public sealed class CalibrationSessionConfig
    {
        public bool PersonalEnabled { get; }
        public bool PopulationEnabled { get; }
        public bool MovementEnabled { get; }
        public CalibrationMode Mode { get; }

        /// <summary>
        /// Whether the per-tick calibration decision was written to the Unity console.
        /// </summary>
        /// <remarks>
        /// Console only, over `adb logcat` on the headset. Nothing is drawn on the headset
        /// panel in any mode: showing a participant what the model thinks of their face --
        /// right or wrong -- turns the evaluation block into a feedback task, and the
        /// protocol's promise that nothing is scored stops being true.
        ///
        /// Named for what it does. It was briefly called `DebugFeedbackShown`, which promised
        /// an overlay that never existed; a flag whose name overstates its effect is worse
        /// than no flag, because a reader of the metadata would conclude the participant had
        /// seen feedback when they had not.
        ///
        /// Recorded either way, so a session run with logging on is identifiable afterwards.
        /// </remarks>
        public bool DebugLoggingEnabled { get; }

        /// <summary>The blend weight, or `double.NaN` outside hybrid mode.</summary>
        public double Alpha => Mode == CalibrationMode.Hybrid
            ? CalibrationMath.FrozenHybridAlpha
            : double.NaN;

        public bool NeedsPopulationArtifact => CalibrationModes.UsesPopulation(Mode);

        private CalibrationSessionConfig(bool personal, bool population, bool movement, bool debugLogging)
        {
            // First, so an impossible combination never produces a half-built object.
            Mode = CalibrationModes.FromToggles(personal, population, movement);
            PersonalEnabled = personal;
            PopulationEnabled = population;
            MovementEnabled = movement;
            DebugLoggingEnabled = debugLogging;
        }

        /// <summary>Capture the two original toggles, with movement off. Call once, at session start.</summary>
        public static CalibrationSessionConfig Lock(bool personalEnabled, bool populationEnabled,
                                                    bool debugLoggingEnabled)
            => new CalibrationSessionConfig(personalEnabled, populationEnabled, false, debugLoggingEnabled);

        /// <summary>Capture all three toggles. Call once, at session start.</summary>
        /// <exception cref="CalibrationConfigurationException">
        /// Movement ticked together with personal or population.
        /// </exception>
        public static CalibrationSessionConfig Lock(bool personalEnabled, bool populationEnabled,
                                                    bool movementEnabled, bool debugLoggingEnabled)
            => new CalibrationSessionConfig(personalEnabled, populationEnabled, movementEnabled,
                                            debugLoggingEnabled);

        public string ModeName => CalibrationModes.Canonical(Mode);

        /// <summary>The alpha as it is written to the CSV and metadata; empty outside hybrid.</summary>
        public string AlphaText => Mode == CalibrationMode.Hybrid
            ? CalibrationMath.FrozenHybridAlpha.ToString(
                "0.00", System.Globalization.CultureInfo.InvariantCulture)
            : string.Empty;

        public override string ToString()
            => Mode == CalibrationMode.Hybrid
                ? $"{ModeName} (alpha {AlphaText}: {AlphaText} personal, "
                  + $"{(1.0 - CalibrationMath.FrozenHybridAlpha).ToString("0.00", System.Globalization.CultureInfo.InvariantCulture)} population)"
                : ModeName;
    }
}
