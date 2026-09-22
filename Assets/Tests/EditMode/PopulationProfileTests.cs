using System;

using NUnit.Framework;

namespace VcuSwimLab.Calibration.Tests
{
    public class PopulationProfileTests
    {
        private static string Artifact => TestPaths.ReadPopulationArtifact();

        private static string Tamper(string original, string find, string replace)
        {
            StringAssert.Contains(find, original,
                                  "the fixture no longer contains the text this test edits");
            return ReplaceFirst(original, find, replace);
        }

        private static string ReplaceFirst(string text, string find, string replace)
        {
            int at = text.IndexOf(find, StringComparison.Ordinal);
            return text.Substring(0, at) + replace + text.Substring(at + find.Length);
        }

        [Test]
        public void The_exported_artifact_loads_and_is_seven_by_seven()
        {
            var profile = PopulationProfile.Parse(Artifact);
            var centroids = profile.Centroids();

            Assert.AreEqual(7, centroids.Length);
            foreach (var row in centroids)
            {
                Assert.AreEqual(7, row.Length);
                foreach (double value in row)
                {
                    Assert.IsFalse(double.IsNaN(value));
                    Assert.IsFalse(double.IsInfinity(value));
                }
            }
        }

        [Test]
        public void The_artifact_carries_the_preregistered_semantic_hash()
        {
            var profile = PopulationProfile.Parse(Artifact);

            Assert.AreEqual(
                "ab1f0a1de18229374c104434562892ac3fc2b0993d4d925538152ffb045b5ffd",
                profile.SemanticSha256);
            Assert.AreEqual(PopulationProfile.ExpectedSemanticSha256, profile.SemanticSha256);
        }

        [Test]
        public void The_artifact_was_built_against_the_shipped_teacher()
        {
            var profile = PopulationProfile.Parse(Artifact);

            Assert.AreEqual(PopulationProfile.ExpectedModelSha256, profile.ModelSha256);
            Assert.AreEqual("training", profile.SourceSplit);
        }

        [Test]
        public void The_centroids_are_copied_on_the_way_out()
        {
            var profile = PopulationProfile.Parse(Artifact);

            var first = profile.Centroids();
            double original = first[0][0];
            first[0][0] = 12345.0;

            Assert.AreEqual(original, profile.Centroids()[0][0],
                            "a caller blending these must not be able to corrupt the profile");
        }

        // ------------------------------------------------------------------ refusals

        [Test]
        public void A_missing_artifact_is_a_hard_failure()
        {
            Assert.Throws<PopulationProfileException>(() => PopulationProfile.Parse(null));
            Assert.Throws<PopulationProfileException>(() => PopulationProfile.Parse(""));
        }

        [Test]
        public void Unreadable_json_is_a_hard_failure()
        {
            Assert.Throws<PopulationProfileException>(() => PopulationProfile.Parse("{ not json"));
            Assert.Throws<PopulationProfileException>(() => PopulationProfile.Parse("[1, 2, 3]"));
            Assert.Throws<PopulationProfileException>(
                () => PopulationProfile.Parse(Artifact + "trailing"));
        }

        [Test]
        public void A_wrong_schema_version_is_refused()
        {
            string tampered = Tamper(Artifact,
                "\"schema_version\": \"emojihero-population-centroids-unity-v1\"",
                "\"schema_version\": \"emojihero-population-centroids-unity-v2\"");

            var e = Assert.Throws<PopulationProfileException>(() => PopulationProfile.Parse(tampered));
            StringAssert.Contains("schema_version", e.Message);
        }

        [Test]
        public void A_wrong_source_schema_version_is_refused()
        {
            string tampered = Tamper(Artifact,
                "\"source_schema_version\": \"emojihero-population-centroids-v1\"",
                "\"source_schema_version\": \"something-else\"");

            Assert.Throws<PopulationProfileException>(() => PopulationProfile.Parse(tampered));
        }

        [Test]
        public void A_profile_built_from_a_split_other_than_training_is_refused()
        {
            string tampered = Tamper(Artifact,
                "\"source_split\": \"training\"",
                "\"source_split\": \"validation\"");

            var e = Assert.Throws<PopulationProfileException>(() => PopulationProfile.Parse(tampered));
            StringAssert.Contains("leaks the data being scored", e.Message);
        }

        [Test]
        public void A_permuted_class_order_is_refused()
        {
            string tampered = Tamper(Artifact, "\"Anger\",\n    \"Disgust\"",
                                               "\"Disgust\",\n    \"Anger\"");

            var e = Assert.Throws<PopulationProfileException>(() => PopulationProfile.Parse(tampered));
            StringAssert.Contains("class_order", e.Message);
        }

        [Test]
        public void A_different_model_hash_is_refused()
        {
            string tampered = Tamper(Artifact, PopulationProfile.ExpectedModelSha256,
                                               new string('0', 64));

            var e = Assert.Throws<PopulationProfileException>(() => PopulationProfile.Parse(tampered));
            StringAssert.Contains("different model", e.Message);
        }

        [Test]
        public void A_different_semantic_hash_is_refused()
        {
            string tampered = Tamper(Artifact, PopulationProfile.ExpectedSemanticSha256,
                                               new string('a', 64));

            var e = Assert.Throws<PopulationProfileException>(() => PopulationProfile.Parse(tampered));
            StringAssert.Contains("semantic hash", e.Message);
        }

        [Test]
        public void A_different_epsilon_is_refused()
        {
            string tampered = Tamper(Artifact, "\"epsilon\": 1e-12", "\"epsilon\": 1e-06");

            var e = Assert.Throws<PopulationProfileException>(() => PopulationProfile.Parse(tampered));
            StringAssert.Contains("epsilon", e.Message);
        }

        [Test]
        public void A_different_feature_space_is_refused()
        {
            string tampered = Tamper(Artifact, "\"feature_space\": \"logscore\"",
                                               "\"feature_space\": \"score\"");

            Assert.Throws<PopulationProfileException>(() => PopulationProfile.Parse(tampered));
        }

        [Test]
        public void A_different_metric_is_refused()
        {
            string tampered = Tamper(Artifact, "\"metric\": \"cosine\"",
                                               "\"metric\": \"euclidean\"");

            Assert.Throws<PopulationProfileException>(() => PopulationProfile.Parse(tampered));
        }

        [Test]
        public void An_edited_centroid_value_breaks_the_conversion_hash()
        {
            var profile = PopulationProfile.Parse(Artifact);
            double first = profile.Centroids()[0][0];

            // Every centroid value is a log of a probability, so they are all negative and
            // this substring is unambiguous in the file.
            string original = first.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
            StringAssert.Contains(original, Artifact,
                                  "the exporter should write round-trippable literals");

            string tampered = ReplaceFirst(Artifact, original, "-1.5");

            var e = Assert.Throws<PopulationProfileException>(() => PopulationProfile.Parse(tampered));
            StringAssert.Contains("conversion hash", e.Message);
        }

        [Test]
        public void A_non_finite_centroid_value_is_refused()
        {
            var profile = PopulationProfile.Parse(Artifact);
            string original = profile.Centroids()[0][0]
                .ToString("R", System.Globalization.CultureInfo.InvariantCulture);

            // A magnitude no double can hold. .NET Core parses it to negative infinity and
            // Mono refuses the parse; the reader reports both the same way, so this assertion
            // holds under `dotnet test` and under Unity's EditMode runner alike.
            string tampered = ReplaceFirst(Artifact, original, "-1e400");

            var e = Assert.Throws<PopulationProfileException>(() => PopulationProfile.Parse(tampered));
            StringAssert.Contains("finite", e.Message);
        }

        [Test]
        public void A_number_the_reader_cannot_represent_is_refused_the_same_way_on_every_runtime()
        {
            foreach (string literal in new[] { "1e400", "-1e400", "1e-400" })
            {
                // 1e-400 underflows to zero rather than overflowing, and zero is a perfectly
                // representable double, so it is included to pin that underflow is *not*
                // treated as a failure.
                object value = null;
                try
                {
                    value = MiniJson.Parse("{\"v\": " + literal + "}");
                }
                catch (JsonParseException e)
                {
                    StringAssert.Contains("finite", e.Message, literal);
                    continue;
                }

                var map = MiniJson.AsObject(value, "probe");
                Assert.AreEqual(0.0, MiniJson.AsDouble(map["v"], "v"), literal);
            }
        }

        [Test]
        public void A_missing_field_is_refused_rather_than_defaulted()
        {
            string tampered = Tamper(Artifact, "\"conversion_sha256\"", "\"conversion_sha256_typo\"");

            var e = Assert.Throws<PopulationProfileException>(() => PopulationProfile.Parse(tampered));
            StringAssert.Contains("conversion_sha256", e.Message);
        }

        // -------------------------------------------------------------- conversion hash

        [Test]
        public void The_bit_pattern_encoding_matches_the_python_side()
        {
            // `struct.pack('>d', v).hex()` in Python. Negative values are the normal case
            // here -- every centroid value is a log probability -- so the sign bit is the
            // thing most worth pinning.
            Assert.AreEqual("0000000000000000", PopulationProfile.Bits(0.0));
            Assert.AreEqual("3ff0000000000000", PopulationProfile.Bits(1.0));
            Assert.AreEqual("bff0000000000000", PopulationProfile.Bits(-1.0));
            Assert.AreEqual("3d719799812dea11", PopulationProfile.Bits(1e-12));
            Assert.AreEqual("3fd999999999999a", PopulationProfile.Bits(0.4));
            Assert.AreEqual("c008000000000000", PopulationProfile.Bits(-3.0));
        }

        [Test]
        public void The_conversion_hash_recomputes_to_the_recorded_value()
        {
            // Parse succeeding at all already proves this, because the parser recomputes the
            // hash and refuses a mismatch. Asserting it here as well says what the file
            // guarantees, rather than leaving it as a side effect of the loader.
            var profile = PopulationProfile.Parse(Artifact);

            Assert.AreEqual(64, profile.ConversionSha256.Length);
            StringAssert.IsMatch("^[0-9a-f]{64}$", profile.ConversionSha256);
        }
    }
}
