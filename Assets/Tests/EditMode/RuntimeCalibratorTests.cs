using System;

using NUnit.Framework;

namespace VcuSwimLab.Calibration.Tests
{
    public class RuntimeCalibratorTests
    {
        private static readonly string[] Names =
        {
            "Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise",
        };

        private const int Classes = 7;

        private static PopulationProfile Population => PopulationProfile.Parse(
            TestPaths.ReadPopulationArtifact());

        private static float[] Scores(int peak, float mass = 0.6f)
        {
            var scores = new float[Classes];
            float rest = (1.0f - mass) / (Classes - 1);
            for (int i = 0; i < Classes; ++i) scores[i] = rest;
            scores[peak] = mass;
            return scores;
        }

        /// <summary>A complete calibration block, with tick 0 offered and correctly ignored.</summary>
        private static void RunCalibrationBlock(RuntimeCalibrator calibrator, Func<int, float[]> scores)
        {
            for (int cls = 0; cls < Classes; ++cls)
            {
                for (int tick = 0; tick < 10; ++tick)
                {
                    calibrator.ObserveCalibrationTick(cls, tick, scores(cls));
                }
            }
        }

        private static RuntimeCalibrator Sealed(CalibrationMode mode, PopulationProfile population = null)
        {
            var calibrator = RuntimeCalibrator.Create(mode, Names, population);
            RunCalibrationBlock(calibrator, cls => Scores(cls));
            calibrator.Seal();
            return calibrator;
        }

        // ------------------------------------------------------------------- wiring

        [Test]
        public void Raw_mode_scores_nothing_and_is_always_valid()
        {
            var calibrator = Sealed(CalibrationMode.Raw);

            Assert.IsTrue(calibrator.CalibrationValid);
            Assert.AreEqual(string.Empty, calibrator.CalibrationError);
            Assert.IsFalse(calibrator.PersonalCentroidsBuilt);
            Assert.IsNull(calibrator.ActiveCentroidsForTesting());

            Assert.IsFalse(calibrator.Classify(1, Scores(0)).Scored,
                           "raw mode produces no calibrated prediction at all");
        }

        [Test]
        public void Personal_mode_fits_centroids_and_scores_ticks()
        {
            var calibrator = Sealed(CalibrationMode.Personal);

            Assert.IsTrue(calibrator.CalibrationValid);
            Assert.IsTrue(calibrator.PersonalCentroidsBuilt);
            Assert.IsTrue(double.IsNaN(calibrator.Alpha), "alpha only exists in hybrid mode");

            var tick = calibrator.Classify(1, Scores(4));
            Assert.IsTrue(tick.Scored);
            Assert.AreEqual(4, tick.ClassIndex);
        }

        [Test]
        public void Population_mode_does_not_build_personal_centroids()
        {
            var calibrator = Sealed(CalibrationMode.Population, Population);

            Assert.IsTrue(calibrator.CalibrationValid);
            Assert.IsFalse(calibrator.PersonalCentroidsBuilt,
                           "the K block is recorded for a comparable protocol, not used");

            // The active set is the artifact, unmodified.
            var active = calibrator.ActiveCentroidsForTesting();
            var expected = Population.Centroids();
            for (int e = 0; e < Classes; ++e)
            {
                CollectionAssert.AreEqual(expected[e], active[e]);
            }
        }

        [Test]
        public void Population_mode_still_collects_the_calibration_block()
        {
            var calibrator = Sealed(CalibrationMode.Population, Population);

            foreach (int count in calibrator.CalibrationTickCounts())
            {
                Assert.AreEqual(9, count, "the K block must be recorded identically in every mode");
            }
        }

        [Test]
        public void Population_mode_survives_a_calibration_block_it_never_uses()
        {
            // Not a single calibration tick offered. Population-only does not depend on the
            // K block, so it must still seal valid.
            var calibrator = RuntimeCalibrator.Create(CalibrationMode.Population, Names, Population);
            calibrator.Seal();

            Assert.IsTrue(calibrator.CalibrationValid);
            Assert.IsTrue(calibrator.Classify(1, Scores(0)).Scored);
        }

        [Test]
        public void Hybrid_mode_blends_at_the_frozen_alpha()
        {
            var population = Population;
            var personalOnly = Sealed(CalibrationMode.Personal);
            var hybrid = Sealed(CalibrationMode.Hybrid, population);

            Assert.AreEqual(0.40, hybrid.Alpha);

            var expected = CalibrationMath.Blend(
                personalOnly.ActiveCentroidsForTesting(), population.Centroids(), 0.40);
            var actual = hybrid.ActiveCentroidsForTesting();

            for (int e = 0; e < Classes; ++e)
            {
                for (int i = 0; i < Classes; ++i)
                {
                    Assert.AreEqual(expected[e][i], actual[e][i], 1e-15);
                }
            }
        }

        // -------------------------------------------------------------- configuration

        [Test]
        public void A_population_mode_without_a_profile_is_refused_before_the_session_runs()
        {
            var e = Assert.Throws<CalibrationConfigurationException>(
                () => RuntimeCalibrator.Create(CalibrationMode.Population, Names, null));
            StringAssert.Contains("did not load", e.Message);

            Assert.Throws<CalibrationConfigurationException>(
                () => RuntimeCalibrator.Create(CalibrationMode.Hybrid, Names, null));
        }

        [Test]
        public void Personal_only_refuses_a_population_profile_outright()
        {
            // Test 16 stated as a construction rather than an observation: personal-only
            // cannot be influenced by the artifact because it cannot hold one.
            var e = Assert.Throws<CalibrationConfigurationException>(
                () => RuntimeCalibrator.Create(CalibrationMode.Personal, Names, Population));
            StringAssert.Contains("by construction", e.Message);

            Assert.Throws<CalibrationConfigurationException>(
                () => RuntimeCalibrator.Create(CalibrationMode.Raw, Names, Population));
        }

        [Test]
        public void Personal_only_predictions_do_not_move_when_a_profile_exists_on_disk()
        {
            // The complement of the test above: having exported an artifact must not change
            // what a personal-only session predicts.
            var calibrator = Sealed(CalibrationMode.Personal);
            Assert.IsNull(calibrator.Population);

            for (int cls = 0; cls < Classes; ++cls)
            {
                var tick = calibrator.Classify(3, Scores(cls));
                Assert.IsTrue(tick.Scored);
                Assert.AreEqual(cls, tick.ClassIndex);
            }
        }

        [Test]
        public void A_schema_with_fewer_than_two_classes_is_refused()
        {
            Assert.Throws<CalibrationConfigurationException>(
                () => RuntimeCalibrator.Create(CalibrationMode.Personal, new[] { "Anger" }, null));
        }

        // ---------------------------------------------------------------- sealing

        [Test]
        public void An_incomplete_calibration_block_invalidates_a_personal_session()
        {
            var calibrator = RuntimeCalibrator.Create(CalibrationMode.Personal, Names, null);
            for (int cls = 0; cls < Classes; ++cls)
            {
                int ticks = cls == 2 ? 5 : 9;
                for (int tick = 1; tick <= ticks; ++tick)
                {
                    calibrator.ObserveCalibrationTick(cls, tick, Scores(cls));
                }
            }

            calibrator.Seal();

            Assert.IsFalse(calibrator.CalibrationValid);
            StringAssert.Contains("Fear 5/9", calibrator.CalibrationError);
            Assert.IsFalse(calibrator.Classify(1, Scores(0)).Scored,
                           "an invalid profile must not silently fall back to raw or population");
        }

        [Test]
        public void An_incomplete_calibration_block_invalidates_a_hybrid_session_too()
        {
            var calibrator = RuntimeCalibrator.Create(CalibrationMode.Hybrid, Names, Population);
            for (int cls = 0; cls < Classes; ++cls)
            {
                for (int tick = 1; tick <= 9; ++tick)
                {
                    if (cls == 0 && tick == 3) continue;
                    calibrator.ObserveCalibrationTick(cls, tick, Scores(cls));
                }
            }

            calibrator.Seal();

            Assert.IsFalse(calibrator.CalibrationValid);
            Assert.IsFalse(calibrator.Classify(1, Scores(0)).Scored,
                           "hybrid must not quietly degrade to population-only");
        }

        [Test]
        public void Classifying_before_the_seal_is_a_programming_error()
        {
            var calibrator = RuntimeCalibrator.Create(CalibrationMode.Personal, Names, null);

            Assert.Throws<InvalidOperationException>(() => calibrator.Classify(1, Scores(0)));
        }

        [Test]
        public void The_calibration_block_cannot_reopen_after_the_seal()
        {
            var calibrator = Sealed(CalibrationMode.Personal);

            Assert.Throws<InvalidOperationException>(
                () => calibrator.ObserveCalibrationTick(0, 1, Scores(0)));
            Assert.Throws<InvalidOperationException>(() => calibrator.Seal());
        }

        [Test]
        public void An_untracked_evaluation_tick_is_not_scored()
        {
            var calibrator = Sealed(CalibrationMode.Personal);

            Assert.IsFalse(calibrator.Classify(1, null).Scored);
            Assert.IsFalse(calibrator.Classify(1, new float[3]).Scored);
        }

        [Test]
        public void A_non_finite_evaluation_tick_is_not_scored_rather_than_throwing_mid_session()
        {
            var calibrator = Sealed(CalibrationMode.Personal);
            var poisoned = Scores(0);
            poisoned[2] = float.NaN;

            Assert.IsFalse(calibrator.Classify(1, poisoned).Scored);
        }

        [Test]
        public void Calibration_recovers_a_consistent_mis_mapping()
        {
            // The case calibration exists for: the model answers Surprise whenever the
            // subject performs Disgust. Raw argmax gets Disgust wrong every time; the
            // personal centroid learns that this face's Disgust looks like that.
            //
            // The mis-mapping is deliberately asymmetric. Were Disgust and Surprise to
            // produce the identical vector, their centroids would coincide and every tick
            // would resolve to the lower class index -- which would pass this test while
            // demonstrating the tie rule rather than the calibration.
            float[] Confused(int cls)
            {
                if (cls == 1) return Scores(6, 0.6f);   // Disgust reads as a mild Surprise
                if (cls == 6) return Scores(6, 0.9f);   // a real Surprise reads stronger
                return Scores(cls);
            }

            var calibrator = RuntimeCalibrator.Create(CalibrationMode.Personal, Names, null);
            RunCalibrationBlock(calibrator, Confused);
            calibrator.Seal();

            var disgust = Confused(1);
            Assert.AreEqual(6, ArgMax(disgust), "raw argmax mistakes this face's Disgust");
            Assert.AreEqual(1, calibrator.Classify(1, disgust).ClassIndex,
                            "the personal centroid should undo it");
        }

        private static int ArgMax(float[] scores)
        {
            int best = 0;
            for (int i = 1; i < scores.Length; ++i)
            {
                if (scores[i] > scores[best]) best = i;
            }

            return best;
        }

        [Test]
        public void The_implementation_version_is_recorded()
        {
            Assert.AreEqual("runtime-calibration-v1", RuntimeCalibrator.ImplementationVersion);
        }

        // ------------------------------------------------------------------ movement

        /// <summary>A 5 x 70 window whose face part moves the given action units.</summary>
        private static float[][] Window(params int[] moving)
        {
            var window = new float[5][];
            for (int f = 0; f < 5; ++f)
            {
                window[f] = new float[70];
                foreach (int a in moving) window[f][a] = 0.5f;
                for (int t = 63; t < 70; ++t) window[f][t] = 0.9f;   // tongue: must be ignored
            }

            return window;
        }

        /// <summary>Each class moves its own two face action units.</summary>
        private static float[][] FaceOf(int cls) => Window(cls * 2, cls * 2 + 1);

        private static RuntimeCalibrator SealedMovement(Func<int, float[]> scores)
        {
            var calibrator = RuntimeCalibrator.Create(CalibrationMode.Movement, Names, null);
            for (int cls = 0; cls < Classes; ++cls)
            {
                for (int tick = 0; tick < 10; ++tick)
                {
                    calibrator.ObserveCalibrationTick(cls, tick, scores(cls), FaceOf(cls));
                }
            }

            calibrator.Seal();
            return calibrator;
        }

        [Test]
        public void Movement_mode_builds_score_and_movement_centroids()
        {
            var calibrator = SealedMovement(cls => Scores(cls));

            Assert.IsTrue(calibrator.CalibrationValid, calibrator.CalibrationError);
            Assert.IsTrue(calibrator.PersonalCentroidsBuilt);
            Assert.IsTrue(calibrator.MovementCentroidsBuilt);

            var movement = calibrator.MovementCentroidsForTesting();
            Assert.AreEqual(Classes, movement.Length);
            Assert.AreEqual(CalibrationMath.FaceActionUnitCount, movement[0].Length,
                            "the tongue values are not part of the movement centroid");
            Assert.AreEqual(0.5, movement[3][6], 1e-7);
            Assert.AreEqual(0.0, movement[3][0], 1e-12);

            var tick = calibrator.Classify(1, Scores(5), FaceOf(5));
            Assert.IsTrue(tick.Scored);
            Assert.AreEqual(5, tick.ClassIndex);
            Assert.AreEqual(2.0, tick.WinnerSimilarity, 1e-9, "both cosines are 1 for an exact match");
        }

        [Test]
        public void Other_modes_never_build_movement_centroids()
        {
            Assert.IsFalse(Sealed(CalibrationMode.Personal).MovementCentroidsBuilt);
            Assert.IsNull(Sealed(CalibrationMode.Personal).MovementCentroidsForTesting());
            Assert.IsFalse(Sealed(CalibrationMode.Hybrid, Population).MovementCentroidsBuilt);
        }

        [Test]
        public void Movement_mode_refuses_a_population_profile()
        {
            Assert.Throws<CalibrationConfigurationException>(
                () => RuntimeCalibrator.Create(CalibrationMode.Movement, Names, Population));
        }

        [Test]
        public void Movement_mode_without_calibration_windows_is_invalid()
        {
            // Scores arrive but no windows: the movement term cannot be fitted, and the
            // session must not quietly fall back to personal-only.
            var calibrator = RuntimeCalibrator.Create(CalibrationMode.Movement, Names, null);
            RunCalibrationBlock(calibrator, cls => Scores(cls));
            calibrator.Seal();

            Assert.IsFalse(calibrator.CalibrationValid);
            StringAssert.Contains("tracked windows", calibrator.CalibrationError);
            Assert.IsFalse(calibrator.Classify(1, Scores(0), FaceOf(0)).Scored);
        }

        [Test]
        public void A_movement_tick_without_a_window_is_not_scored()
        {
            var calibrator = SealedMovement(cls => Scores(cls));

            Assert.IsFalse(calibrator.Classify(1, Scores(2), null).Scored);
            Assert.IsFalse(calibrator.Classify(1, Scores(2), new[] { new float[10] }).Scored,
                           "a frame narrower than 63 action units is refused, not padded");
        }

        [Test]
        public void Movement_separates_two_emotions_the_scores_cannot()
        {
            // The model gives Disgust and Surprise the identical score vector, so their score
            // centroids coincide and personal calibration has to pick one of them for both.
            // The faces differ, and the movement term tells them apart.
            float[] Collapsed(int cls) => cls == 1 ? Scores(6) : Scores(cls);

            var personal = RuntimeCalibrator.Create(CalibrationMode.Personal, Names, null);
            RunCalibrationBlock(personal, Collapsed);
            personal.Seal();

            var movement = SealedMovement(Collapsed);

            Assert.AreEqual(personal.Classify(1, Collapsed(1)).ClassIndex,
                            personal.Classify(1, Collapsed(6)).ClassIndex,
                            "personal calibration cannot separate identical score vectors");
            Assert.AreEqual(1, movement.Classify(1, Collapsed(1), FaceOf(1)).ClassIndex);
            Assert.AreEqual(6, movement.Classify(1, Collapsed(6), FaceOf(6)).ClassIndex);
        }
    }
}
