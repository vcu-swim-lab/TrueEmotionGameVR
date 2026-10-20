namespace VcuSeCenter.Calibration
{
    /// <summary>
    /// The names of the four experiment conditions: which backend predicts (SparseVR or Jev)
    /// crossed with whether the player's calibration is used.
    /// </summary>
    /// <remarks>
    /// Pure text rules, no Unity types, so they can be unit-tested. Nothing here selects a
    /// condition: that is done once, in <c>ConditionSetup.Configure</c>, by calling the
    /// library. These helpers only turn what the library reports (is the active backend Jev?
    /// is calibration on?) into the labels and file ids the game writes down.
    ///
    ///   1  SparseVR, no calibration     id "sparsevr_nocal"
    ///   2  SparseVR, calibration        id "sparsevr_cal"
    ///   3  Jev, no calibration          id "jev_nocal"
    ///   4  Jev, calibration             id "jev_cal"
    /// </remarks>
    public static class ExperimentCondition
    {
        /// <summary>The library gives every Jev backend an id that starts with this.</summary>
        public const string JevBackendIdPrefix = "jev:";

        /// <summary>True when the library's active backend id is a Jev backend's.</summary>
        public static bool IsJevBackend(string backendId)
            => backendId != null
               && backendId.StartsWith(JevBackendIdPrefix, System.StringComparison.Ordinal);

        /// <summary>1 to 4, in the order of the table above.</summary>
        public static int Number(bool usesJev, bool usesCalibration)
            => 1 + (usesJev ? 2 : 0) + (usesCalibration ? 1 : 0);

        /// <summary>The on-disk spelling written into session metadata.</summary>
        public static string Id(bool usesJev, bool usesCalibration)
            => (usesJev ? "jev" : "sparsevr") + (usesCalibration ? "_cal" : "_nocal");

        /// <summary>"SparseVr" or "Jev", as written to <c>face_backend</c>.</summary>
        public static string BackendName(bool usesJev) => usesJev ? "Jev" : "SparseVr";

        /// <summary>For screens and logs, e.g. "Condition 3: Jev, no calibration".</summary>
        public static string Label(bool usesJev, bool usesCalibration)
            => $"Condition {Number(usesJev, usesCalibration)}: "
               + (usesJev ? "Jev" : "SparseVR")
               + (usesCalibration ? ", calibration" : ", no calibration");

        /// <summary>
        /// The id a player's calibration is saved under, which keeps one file per backend.
        /// </summary>
        /// <remarks>
        /// The library stores one calibration file per id and refuses to load a file made by a
        /// different backend. If SparseVR and Jev shared a player id, testing one would
        /// overwrite the other's calibration every time. Letters, digits, '-' and '_' only, as
        /// the library requires; a valid <paramref name="playerId"/> stays valid.
        /// </remarks>
        public static string CalibrationFileId(string playerId, bool usesJev)
            => $"{playerId}-{(usesJev ? "jev" : "sparsevr")}";

        /// <summary>The text Assets/Resources/jev_api_key.txt ships with, until the key is pasted over it.</summary>
        public const string KeyPlaceholder = "PASTE_YOUR_TYPESAFE_API_KEY_HERE";

        /// <summary>
        /// The key in the text of the key file: the first line that is not blank and does not
        /// start with '#'. Null when there is none, or when it is still the placeholder, so a
        /// forgotten paste is reported as "no key" and not sent to Jev as a key.
        /// </summary>
        public static string KeyFromFileText(string text)
        {
            if (text == null)
            {
                return null;
            }

            foreach (string raw in text.Split('\n'))
            {
                string line = raw.Trim();
                if (line.Length == 0 || line[0] == '#')
                {
                    continue;
                }

                return line == KeyPlaceholder ? null : line;
            }

            return null;
        }
    }
}
