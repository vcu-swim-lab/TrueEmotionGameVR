using System;
using System.Collections.Generic;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace VcuSwimLab.Calibration
{
    /// <summary>
    /// Thrown when the population artifact is absent, malformed, or built against something
    /// other than the model and corpus this build expects.
    /// </summary>
    /// <remarks>
    /// Always fatal to the session. There is no fallback to raw: a run configured for
    /// population or hybrid that quietly scored raw would be recorded as a calibrated
    /// session and pooled with real ones.
    /// </remarks>
    public sealed class PopulationProfileException : Exception
    {
        public PopulationProfileException(string message) : base(message) { }
        public PopulationProfileException(string message, Exception inner) : base(message, inner) { }
    }

    /// <summary>
    /// The frozen EmojiHero population centroids, validated on load.
    /// </summary>
    /// <remarks>
    /// Seven seven-dimensional vectors in the model's own log-score output space, averaged
    /// over the twenty EmojiHero training participants. This is aggregate model *output*: it
    /// carries no action units, no face windows and no individual's centroid, which is why it
    /// is the one centroid set in this project that ships inside a build.
    ///
    /// The caveat the reports carry, restated here because this is where the value enters the
    /// headset: these centroids describe how the teacher responds to *EmojiHero* input --
    /// 30-frame captures mean-pooled three at a time, windows sharing four frames of five. A
    /// live window is five instantaneous samples sharing none. The blend is well defined
    /// because both live in the model's seven-dimensional output space, but the population
    /// term is being applied off the distribution it was estimated on, and the one headset
    /// session measured so far had it moving the result the wrong way.
    /// </remarks>
    public sealed class PopulationProfile
    {
        /// <summary>Schema of the Unity-side artifact, written by the export script.</summary>
        public const string UnitySchemaVersion = "emojihero-population-centroids-unity-v1";

        /// <summary>Schema of the analysis artifact the export script reads.</summary>
        public const string SourceSchemaVersion = "emojihero-population-centroids-v1";

        /// <summary>
        /// The preregistration pin, frozen in commit `cc8ff2d` and recorded in
        /// `Analysis/EMOJIHERO_HYBRID_CALIBRATION_VALIDATION_REPORT.md`.
        /// </summary>
        /// <remarks>
        /// A content hash over the centroids, class order, epsilon, feature space, transform,
        /// estimator, preprocessing, model hash and corpus fingerprint -- and deliberately not
        /// over timestamps, paths or the git commit, so a correct rebuild from identical
        /// inputs reproduces it inside a byte-different file. Compiled in as a constant so a
        /// build carries the identity of the artifact it expects, rather than trusting
        /// whatever file happens to sit in StreamingAssets.
        /// </remarks>
        public const string ExpectedSemanticSha256 =
            "ab1f0a1de18229374c104434562892ac3fc2b0993d4d925538152ffb045b5ffd";

        /// <summary>SHA-256 of `Runtime/Models/sparseVR_teacher_model.onnx`, never retrained.</summary>
        /// <remarks>
        /// The centroids live in this graph's output space, so a profile built from a
        /// different graph describes a different space entirely and blending it in would
        /// produce numbers that look ordinary and mean nothing.
        ///
        /// Known limitation: this checks the hash the *artifact* records against the hash
        /// this build expects. It cannot re-hash the model at runtime, because Sentis
        /// re-serialises the ONNX into its own asset format at import and the original bytes
        /// are not on the device. The end-to-end pin is therefore the export script, which
        /// hashes the real file, plus this constant.
        /// </remarks>
        public const string ExpectedModelSha256 =
            "ef6cd4134f4d64ab5a7233ac719cc05de88963b33757eca4d92f1b625413343c";

        /// <summary>
        /// Population centroids must come from the EmojiHero *training* split.
        /// </summary>
        /// <remarks>
        /// Validation is where alpha was selected and test was the confirmatory split; a
        /// population term estimated from either would partly describe the very people being
        /// scored. Training is the one split never used as an evaluation set, so it is the
        /// only one that can supply classifier parameters without contaminating a
        /// measurement.
        /// </remarks>
        public const string ExpectedSourceSplit = "training";

        public const string ExpectedFeatureSpace = CalibrationMath.FeatureSpaceName;
        public const string ExpectedTransform = CalibrationMath.TransformName;
        public const string ExpectedMetric = CalibrationMath.MetricName;

        public static readonly string[] ExpectedClassOrder =
        {
            "Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise",
        };

        private readonly double[][] centroids;

        public string SemanticSha256 { get; }
        public string ConversionSha256 { get; }
        public string ModelSha256 { get; }
        public string SourceSplit { get; }
        public string SchemaVersion { get; }
        public string Estimator { get; }
        public double Epsilon { get; }
        public int ParticipantCount { get; }
        public int SequenceCount { get; }

        private PopulationProfile(double[][] centroids, string semantic, string conversion,
                                  string modelSha, string sourceSplit, string schemaVersion,
                                  string estimator, double epsilon,
                                  int participantCount, int sequenceCount)
        {
            this.centroids = centroids;
            SemanticSha256 = semantic;
            ConversionSha256 = conversion;
            ModelSha256 = modelSha;
            SourceSplit = sourceSplit;
            SchemaVersion = schemaVersion;
            Estimator = estimator;
            Epsilon = epsilon;
            ParticipantCount = participantCount;
            SequenceCount = sequenceCount;
        }

        /// <summary>
        /// The seven centroid rows, in class order. Copied on the way out: the caller blends
        /// these with a personal set, and a blend that wrote through to the profile would
        /// corrupt every later trial in the session.
        /// </summary>
        public double[][] Centroids()
        {
            var copy = new double[centroids.Length][];
            for (int e = 0; e < centroids.Length; ++e)
            {
                copy[e] = (double[])centroids[e].Clone();
            }

            return copy;
        }

        // ---------------------------------------------------------------------- loading

        /// <summary>
        /// Parse and fully validate an artifact. Every failure throws; nothing is repaired.
        /// </summary>
        public static PopulationProfile Parse(string json)
        {
            if (string.IsNullOrEmpty(json))
            {
                throw new PopulationProfileException(
                    "the population artifact is empty. Export it with "
                    + "`python Analysis/export_population_artifact_for_unity.py`.");
            }

            // Every structural complaint the reader can make -- unreadable JSON, a field of
            // the wrong type, a field that is simply absent -- is rethrown as one exception
            // type. A caller configuring a session has exactly one thing to do about any of
            // them, and should not have to know which layer noticed.
            try
            {
                return Validate(MiniJson.AsObject(MiniJson.Parse(json), "population artifact"));
            }
            catch (JsonParseException e)
            {
                throw new PopulationProfileException(
                    $"the population artifact is not usable: {e.Message}", e);
            }
        }

        private static PopulationProfile Validate(Dictionary<string, object> map)
        {
            string schemaVersion = RequireString(map, "schema_version");
            if (schemaVersion != UnitySchemaVersion)
            {
                throw new PopulationProfileException(
                    $"schema_version is '{schemaVersion}', expected '{UnitySchemaVersion}'");
            }

            string sourceSchema = RequireString(map, "source_schema_version");
            if (sourceSchema != SourceSchemaVersion)
            {
                throw new PopulationProfileException(
                    $"source_schema_version is '{sourceSchema}', expected '{SourceSchemaVersion}'");
            }

            string sourceSplit = RequireString(map, "source_split");
            if (sourceSplit != ExpectedSourceSplit)
            {
                throw new PopulationProfileException(
                    $"source_split is '{sourceSplit}', expected '{ExpectedSourceSplit}'. Population "
                    + "centroids must come from the training split; anything else leaks the data "
                    + "being scored.");
            }

            var classOrder = RequireStringArray(map, "class_order");
            if (classOrder.Length != ExpectedClassOrder.Length)
            {
                throw new PopulationProfileException(
                    $"class_order has {classOrder.Length} entries, expected {ExpectedClassOrder.Length}");
            }
            for (int i = 0; i < classOrder.Length; ++i)
            {
                if (classOrder[i] != ExpectedClassOrder[i])
                {
                    throw new PopulationProfileException(
                        $"class_order[{i}] is '{classOrder[i]}', expected '{ExpectedClassOrder[i]}'. "
                        + "A permuted class order silently relabels every centroid.");
                }
            }

            double epsilon = RequireDouble(map, "epsilon");
            if (epsilon != CalibrationMath.Epsilon)
            {
                throw new PopulationProfileException(
                    $"epsilon is {epsilon.ToString("R", CultureInfo.InvariantCulture)}, this build uses "
                    + $"{CalibrationMath.Epsilon.ToString("R", CultureInfo.InvariantCulture)}. "
                    + "The feature spaces would not match.");
            }

            RequireExactString(map, "feature_space", ExpectedFeatureSpace);
            RequireExactString(map, "transform", ExpectedTransform);
            RequireExactString(map, "metric", ExpectedMetric);

            string estimator = RequireString(map, "estimator");
            if (estimator.Length == 0)
            {
                throw new PopulationProfileException("estimator is empty");
            }

            string modelSha = RequireString(map, "model_sha256");
            if (modelSha != ExpectedModelSha256)
            {
                throw new PopulationProfileException(
                    "the artifact was built against a different model.\n"
                    + $"  artifact: {modelSha}\n"
                    + $"  expected: {ExpectedModelSha256}\n"
                    + "Population centroids live in that model's output space and cannot be reused "
                    + "across models.");
            }

            string semantic = RequireString(map, "semantic_sha256");
            if (semantic != ExpectedSemanticSha256)
            {
                throw new PopulationProfileException(
                    "the artifact's semantic hash is not the preregistered one.\n"
                    + $"  artifact: {semantic}\n"
                    + $"  expected: {ExpectedSemanticSha256}\n"
                    + "This build is pinned to the population profile frozen before the EmojiHero "
                    + "test split was opened.");
            }

            var centroids = ReadCentroids(map, classOrder);

            string recordedConversion = RequireString(map, "conversion_sha256");
            string recomputed = ConversionHash(
                schemaVersion: schemaVersion,
                sourceSchemaVersion: sourceSchema,
                semanticSha256: semantic,
                sourceSplit: sourceSplit,
                classOrder: classOrder,
                epsilon: epsilon,
                featureSpace: ExpectedFeatureSpace,
                transform: ExpectedTransform,
                estimator: estimator,
                metric: ExpectedMetric,
                modelSha256: modelSha,
                centroids: centroids);

            if (recordedConversion != recomputed)
            {
                throw new PopulationProfileException(
                    "the artifact's conversion hash does not match its contents.\n"
                    + $"  recorded:   {recordedConversion}\n"
                    + $"  recomputed: {recomputed}\n"
                    + "A value was edited after the file was exported. Re-export it rather than "
                    + "patching it by hand.");
            }

            return new PopulationProfile(
                centroids, semantic, recordedConversion, modelSha, sourceSplit, schemaVersion,
                estimator, epsilon,
                (int)OptionalDouble(map, "n_participants", 0),
                (int)OptionalDouble(map, "n_sequences", 0));
        }

        private static double[][] ReadCentroids(Dictionary<string, object> map, string[] classOrder)
        {
            var rows = MiniJson.AsArray(MiniJson.Field(map, "centroids", "population artifact"),
                                        "centroids");

            if (rows.Count != classOrder.Length)
            {
                throw new PopulationProfileException(
                    $"the centroid matrix has {rows.Count} rows, expected {classOrder.Length}");
            }

            var centroids = new double[rows.Count][];

            for (int e = 0; e < rows.Count; ++e)
            {
                var row = MiniJson.AsObject(rows[e], $"centroids[{e}]");
                string emotion = MiniJson.AsString(
                    MiniJson.Field(row, "emotion", $"centroids[{e}]"), $"centroids[{e}].emotion");

                if (emotion != classOrder[e])
                {
                    throw new PopulationProfileException(
                        $"centroids[{e}] is '{emotion}' but class_order[{e}] is '{classOrder[e]}'. "
                        + "The rows must be in class order.");
                }

                var values = MiniJson.AsArray(
                    MiniJson.Field(row, "values", $"centroids[{e}]"), $"centroids[{e}].values");

                if (values.Count != classOrder.Length)
                {
                    throw new PopulationProfileException(
                        $"centroids[{e}] ('{emotion}') is {values.Count}-dimensional, expected "
                        + $"{classOrder.Length}. The profile must be square: seven centroids in the "
                        + "model's seven-dimensional output space.");
                }

                var vector = new double[values.Count];
                for (int i = 0; i < values.Count; ++i)
                {
                    double value = MiniJson.AsDouble(values[i], $"centroids[{e}].values[{i}]");
                    if (double.IsNaN(value) || double.IsInfinity(value))
                    {
                        throw new PopulationProfileException(
                            $"centroids[{e}] ('{emotion}') value {i} is {value}; every centroid "
                            + "value must be finite");
                    }

                    vector[i] = value;
                }

                centroids[e] = vector;
            }

            return centroids;
        }

        // ------------------------------------------------------------- conversion hash

        /// <summary>
        /// A hash over the converted artifact's meaning, recomputable in both C# and Python.
        /// </summary>
        /// <remarks>
        /// The analysis-side `semantic_sha256` cannot be recomputed here: it hashes a
        /// canonical `json.dumps`, and Python's shortest round-trip float repr and .NET's
        /// "R" format agree on digits but not on exponent spelling ("1e-12" against "1E-12"),
        /// so a C# reimplementation would be a source of false failures.
        ///
        /// This hash sidesteps formatting entirely by serialising every double as its
        /// IEEE-754 bit pattern in hex. `struct.pack('>d', v).hex()` in Python and
        /// `BitConverter.DoubleToInt64Bits(v).ToString("x16")` in C# are exactly equal for
        /// every finite double, with no rounding, locale or notation in the path.
        ///
        /// The two hashes do different jobs and both are checked: `semantic_sha256` pins
        /// *which* profile this is, back to the preregistered one; `conversion_sha256` proves
        /// the converted file still holds what the exporter put in it.
        ///
        /// `Analysis/export_population_artifact_for_unity.py` builds the identical string; a
        /// golden test in both languages holds them together.
        /// </remarks>
        public static string ConversionHash(
            string schemaVersion, string sourceSchemaVersion, string semanticSha256,
            string sourceSplit, IReadOnlyList<string> classOrder, double epsilon,
            string featureSpace, string transform, string estimator, string metric,
            string modelSha256, IReadOnlyList<double[]> centroids)
        {
            var sb = new StringBuilder();
            sb.Append("emojihero-population-centroids-unity-conversion-v1\n");
            sb.Append("schema_version=").Append(schemaVersion).Append('\n');
            sb.Append("source_schema_version=").Append(sourceSchemaVersion).Append('\n');
            sb.Append("semantic_sha256=").Append(semanticSha256).Append('\n');
            sb.Append("source_split=").Append(sourceSplit).Append('\n');
            sb.Append("class_order=").Append(string.Join(",", classOrder)).Append('\n');
            sb.Append("epsilon=").Append(Bits(epsilon)).Append('\n');
            sb.Append("feature_space=").Append(featureSpace).Append('\n');
            sb.Append("transform=").Append(transform).Append('\n');
            sb.Append("estimator=").Append(estimator).Append('\n');
            sb.Append("metric=").Append(metric).Append('\n');
            sb.Append("model_sha256=").Append(modelSha256).Append('\n');

            for (int e = 0; e < centroids.Count; ++e)
            {
                sb.Append("centroid[").Append(classOrder[e]).Append("]=");
                for (int i = 0; i < centroids[e].Length; ++i)
                {
                    if (i > 0) sb.Append(',');
                    sb.Append(Bits(centroids[e][i]));
                }
                sb.Append('\n');
            }

            using (var sha = SHA256.Create())
            {
                var digest = sha.ComputeHash(Encoding.UTF8.GetBytes(sb.ToString()));
                var hex = new StringBuilder(digest.Length * 2);
                foreach (byte b in digest)
                {
                    hex.Append(b.ToString("x2", CultureInfo.InvariantCulture));
                }

                return hex.ToString();
            }
        }

        /// <summary>The IEEE-754 bit pattern of a double, as sixteen lowercase hex digits.</summary>
        public static string Bits(double value)
            => BitConverter.DoubleToInt64Bits(value).ToString("x16", CultureInfo.InvariantCulture);

        // -------------------------------------------------------------------- helpers

        private static string RequireString(Dictionary<string, object> map, string key)
            => MiniJson.AsString(MiniJson.Field(map, key, "population artifact"), key);

        private static double RequireDouble(Dictionary<string, object> map, string key)
            => MiniJson.AsDouble(MiniJson.Field(map, key, "population artifact"), key);

        private static double OptionalDouble(Dictionary<string, object> map, string key, double fallback)
            => map.TryGetValue(key, out object value) && value is double number ? number : fallback;

        private static void RequireExactString(Dictionary<string, object> map, string key, string expected)
        {
            string actual = RequireString(map, key);
            if (actual != expected)
            {
                throw new PopulationProfileException($"{key} is '{actual}', expected '{expected}'");
            }
        }

        private static string[] RequireStringArray(Dictionary<string, object> map, string key)
        {
            var list = MiniJson.AsArray(MiniJson.Field(map, key, "population artifact"), key);
            var result = new string[list.Count];
            for (int i = 0; i < list.Count; ++i)
            {
                result[i] = MiniJson.AsString(list[i], $"{key}[{i}]");
            }

            return result;
        }
    }
}
