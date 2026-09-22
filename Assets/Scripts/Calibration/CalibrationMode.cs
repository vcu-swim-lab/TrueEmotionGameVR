using System;

namespace VcuSwimLab.Calibration
{
    /// <summary>
    /// Which centroid set the evaluation block is scored against.
    /// </summary>
    /// <remarks>
    /// The mode is derived from two independent operator toggles rather than picked from a
    /// dropdown, because the two terms of the hybrid are separately meaningful: "personal"
    /// and "population" are the things being switched on, and `Hybrid` is what having both
    /// on means. A dropdown would let an operator select `Hybrid` while believing only one
    /// term was active.
    ///
    /// `Raw` is not a calibration. It is the untouched argmax the protocol has always
    /// recorded, kept as a named mode so a recording says explicitly that calibration was
    /// off rather than leaving a reader to infer it from absent columns.
    /// </remarks>
    public enum CalibrationMode
    {
        Raw = 0,
        Personal = 1,
        Population = 2,
        Hybrid = 3,
    }

    public static class CalibrationModes
    {
        // On-disk spellings. Written instead of `mode.ToString()` for the same reason
        // `SessionLogger.Canonical` exists: renaming an enum member must not silently
        // change the file vocabulary the Python side matches on.
        public const string RawName = "raw";
        public const string PersonalName = "personal";
        public const string PopulationName = "population";
        public const string HybridName = "hybrid";

        /// <summary>The two-checkbox truth table, in one place.</summary>
        public static CalibrationMode FromToggles(bool personalEnabled, bool populationEnabled)
        {
            if (personalEnabled && populationEnabled) return CalibrationMode.Hybrid;
            if (personalEnabled) return CalibrationMode.Personal;
            if (populationEnabled) return CalibrationMode.Population;
            return CalibrationMode.Raw;
        }

        public static string Canonical(CalibrationMode mode)
        {
            switch (mode)
            {
                case CalibrationMode.Raw: return RawName;
                case CalibrationMode.Personal: return PersonalName;
                case CalibrationMode.Population: return PopulationName;
                case CalibrationMode.Hybrid: return HybridName;
                default:
                    throw new ArgumentOutOfRangeException(nameof(mode), mode, "unmapped calibration mode");
            }
        }

        /// <summary>True when the mode needs personal centroids fitted from the K block.</summary>
        public static bool UsesPersonal(CalibrationMode mode)
            => mode == CalibrationMode.Personal || mode == CalibrationMode.Hybrid;

        /// <summary>True when the mode needs the frozen EmojiHero population artifact.</summary>
        public static bool UsesPopulation(CalibrationMode mode)
            => mode == CalibrationMode.Population || mode == CalibrationMode.Hybrid;

        /// <summary>True when any centroid classification happens at all.</summary>
        public static bool IsCalibrated(CalibrationMode mode) => mode != CalibrationMode.Raw;
    }
}
