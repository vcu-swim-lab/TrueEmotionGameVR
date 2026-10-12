namespace VcuSwimLab.Calibration
{
    /// <summary>
    /// The calibration settings a session actually ran under, captured once and immutable
    /// thereafter.
    /// </summary>
    /// <remarks>
    /// The toggles live on an inspector component, and inspector fields can be changed while
    /// the game is playing. A session that started raw and finished calibrated would be a
    /// recording of neither, and nothing in the file would say so.
    ///
    /// So the toggles are read exactly once, at session start, into this object. Everything
    /// downstream -- the calibrator, the CSV, the metadata -- reads from here. There is no
    /// path back to the component, which is what makes "locked at session start" a property of
    /// the design rather than a discipline someone has to remember.
    ///
    /// The only calibration is movement calibration, from the VREmotionDetection library. The
    /// personal, population and hybrid modes were removed; recordings still carry their
    /// columns, always "false", so older and newer files read the same way.
    /// </remarks>
    public sealed class CalibrationSessionConfig
    {
        // On-disk spellings, matched by the Python analysis.
        public const string RawName = "raw";
        public const string MovementName = "movement";

        public bool MovementEnabled { get; }

        /// <summary>
        /// Whether the per-tick calibration decision was written to the Unity console.
        /// </summary>
        /// <remarks>
        /// Console only, over `adb logcat` on the headset. Nothing is drawn on the headset
        /// panel: showing a participant what the model thinks of their face -- right or wrong
        /// -- turns the evaluation block into a feedback task, and the protocol's promise that
        /// nothing is scored stops being true. Recorded either way, so a session run with
        /// logging on is identifiable afterwards.
        /// </remarks>
        public bool DebugLoggingEnabled { get; }

        private CalibrationSessionConfig(bool movement, bool debugLogging)
        {
            MovementEnabled = movement;
            DebugLoggingEnabled = debugLogging;
        }

        /// <summary>Capture the toggles. Call once, at session start.</summary>
        public static CalibrationSessionConfig Lock(bool movementEnabled, bool debugLoggingEnabled)
            => new CalibrationSessionConfig(movementEnabled, debugLoggingEnabled);

        public string ModeName => MovementEnabled ? MovementName : RawName;

        public override string ToString() => ModeName;
    }
}
