using System;

using NUnit.Framework;

namespace VcuSwimLab.Calibration.Tests
{
    public class CalibrationModeTests
    {
        [Test]
        public void The_two_checkboxes_map_to_four_modes()
        {
            Assert.AreEqual(CalibrationMode.Raw,
                            CalibrationModes.FromToggles(personalEnabled: false, populationEnabled: false));
            Assert.AreEqual(CalibrationMode.Personal,
                            CalibrationModes.FromToggles(personalEnabled: true, populationEnabled: false));
            Assert.AreEqual(CalibrationMode.Population,
                            CalibrationModes.FromToggles(personalEnabled: false, populationEnabled: true));
            Assert.AreEqual(CalibrationMode.Hybrid,
                            CalibrationModes.FromToggles(personalEnabled: true, populationEnabled: true));
        }

        [Test]
        public void Each_mode_has_one_canonical_spelling()
        {
            Assert.AreEqual("raw", CalibrationModes.Canonical(CalibrationMode.Raw));
            Assert.AreEqual("personal", CalibrationModes.Canonical(CalibrationMode.Personal));
            Assert.AreEqual("population", CalibrationModes.Canonical(CalibrationMode.Population));
            Assert.AreEqual("hybrid", CalibrationModes.Canonical(CalibrationMode.Hybrid));
        }

        [Test]
        public void Only_the_modes_that_need_a_term_report_needing_it()
        {
            Assert.IsFalse(CalibrationModes.UsesPersonal(CalibrationMode.Raw));
            Assert.IsTrue(CalibrationModes.UsesPersonal(CalibrationMode.Personal));
            Assert.IsFalse(CalibrationModes.UsesPersonal(CalibrationMode.Population));
            Assert.IsTrue(CalibrationModes.UsesPersonal(CalibrationMode.Hybrid));

            Assert.IsFalse(CalibrationModes.UsesPopulation(CalibrationMode.Raw));
            Assert.IsFalse(CalibrationModes.UsesPopulation(CalibrationMode.Personal));
            Assert.IsTrue(CalibrationModes.UsesPopulation(CalibrationMode.Population));
            Assert.IsTrue(CalibrationModes.UsesPopulation(CalibrationMode.Hybrid));
        }

        [Test]
        public void Raw_is_the_only_uncalibrated_mode()
        {
            Assert.IsFalse(CalibrationModes.IsCalibrated(CalibrationMode.Raw));
            Assert.IsTrue(CalibrationModes.IsCalibrated(CalibrationMode.Personal));
            Assert.IsTrue(CalibrationModes.IsCalibrated(CalibrationMode.Population));
            Assert.IsTrue(CalibrationModes.IsCalibrated(CalibrationMode.Hybrid));
        }
    }

    public class LogScoreTests
    {
        [Test]
        public void The_transform_is_the_natural_log()
        {
            var result = CalibrationMath.LogScore(new[] { 1.0f, 0.5f, 0.25f });

            Assert.AreEqual(Math.Log(1.0), result[0], 1e-15);
            Assert.AreEqual(Math.Log(0.5f), result[1], 1e-15);
            Assert.AreEqual(Math.Log(0.25f), result[2], 1e-15);
        }

        [Test]
        public void The_epsilon_is_the_frozen_one()
        {
            Assert.AreEqual(1e-12, CalibrationMath.Epsilon);
        }

        [Test]
        public void Zero_is_clamped_to_the_epsilon_rather_than_sent_to_negative_infinity()
        {
            var result = CalibrationMath.LogScore(new[] { 0.0f });

            Assert.IsFalse(double.IsNegativeInfinity(result[0]));
            Assert.AreEqual(Math.Log(CalibrationMath.Epsilon), result[0], 1e-15);
        }

        [Test]
        public void The_epsilon_is_inert_on_every_score_the_corpus_actually_holds()
        {
            // The smallest score observed anywhere in the EmojiHero corpus is about 7e-06,
            // seven orders of magnitude above the clamp. A clamp that never fires cannot be
            // shaping a result, which is the property the reports claim.
            float smallestObserved = 7e-06f;
            var result = CalibrationMath.LogScore(new[] { smallestObserved });

            Assert.AreEqual(Math.Log(smallestObserved), result[0], 1e-15);
            Assert.AreNotEqual(Math.Log(CalibrationMath.Epsilon), result[0]);
        }

        [Test]
        public void A_value_below_the_epsilon_is_raised_to_it()
        {
            var result = CalibrationMath.LogScore(new[] { 1e-30f });
            Assert.AreEqual(Math.Log(CalibrationMath.Epsilon), result[0], 1e-15);
        }

        [Test]
        public void A_non_finite_score_is_refused_rather_than_clamped()
        {
            Assert.Throws<ArgumentException>(() => CalibrationMath.LogScore(new[] { float.NaN }));
            Assert.Throws<ArgumentException>(
                () => CalibrationMath.LogScore(new[] { float.PositiveInfinity }));
            Assert.Throws<ArgumentException>(
                () => CalibrationMath.LogScore(new[] { float.NegativeInfinity }));
        }
    }

    public class BlendTests
    {
        private static double[][] Personal => new[]
        {
            new[] { 1.0, 2.0, 3.0 },
            new[] { -1.0, -2.0, -3.0 },
        };

        private static double[][] Population => new[]
        {
            new[] { 10.0, 20.0, 30.0 },
            new[] { 0.0, 0.0, 0.0 },
        };

        [Test]
        public void Alpha_one_is_the_personal_centroid_exactly()
        {
            var blended = CalibrationMath.Blend(Personal, Population, 1.0);

            CollectionAssert.AreEqual(Personal[0], blended[0]);
            CollectionAssert.AreEqual(Personal[1], blended[1]);
        }

        [Test]
        public void Alpha_zero_is_the_population_centroid_exactly()
        {
            var blended = CalibrationMath.Blend(Personal, Population, 0.0);

            CollectionAssert.AreEqual(Population[0], blended[0]);
            CollectionAssert.AreEqual(Population[1], blended[1]);
        }

        [Test]
        public void The_frozen_alpha_weights_the_personal_term_at_forty_percent()
        {
            Assert.AreEqual(0.40, CalibrationMath.FrozenHybridAlpha);

            var blended = CalibrationMath.Blend(Personal, Population, CalibrationMath.FrozenHybridAlpha);

            // H_e = 0.40 * P_e + 0.60 * G_e
            Assert.AreEqual(0.40 * 1.0 + 0.60 * 10.0, blended[0][0], 1e-15);
            Assert.AreEqual(0.40 * 2.0 + 0.60 * 20.0, blended[0][1], 1e-15);
            Assert.AreEqual(0.40 * 3.0 + 0.60 * 30.0, blended[0][2], 1e-15);
            Assert.AreEqual(0.40 * -1.0, blended[1][0], 1e-15);
        }

        [Test]
        public void An_alpha_outside_the_unit_interval_is_refused()
        {
            Assert.Throws<ArgumentOutOfRangeException>(
                () => CalibrationMath.Blend(Personal, Population, -0.1));
            Assert.Throws<ArgumentOutOfRangeException>(
                () => CalibrationMath.Blend(Personal, Population, 1.5));
            Assert.Throws<ArgumentOutOfRangeException>(
                () => CalibrationMath.Blend(Personal, Population, double.NaN));
        }

        [Test]
        public void Mismatched_shapes_are_refused()
        {
            Assert.Throws<ArgumentException>(
                () => CalibrationMath.Blend(Personal, new[] { new[] { 1.0, 2.0, 3.0 } }, 0.4));
            Assert.Throws<ArgumentException>(
                () => CalibrationMath.Blend(Personal,
                                            new[] { new[] { 1.0, 2.0 }, new[] { 1.0, 2.0 } }, 0.4));
        }

        [Test]
        public void Blending_does_not_write_through_to_either_input()
        {
            var personal = Personal;
            var population = Population;
            CalibrationMath.Blend(personal, population, 0.4);

            Assert.AreEqual(1.0, personal[0][0]);
            Assert.AreEqual(10.0, population[0][0]);
        }
    }

    public class NearestCentroidTests
    {
        [Test]
        public void The_nearest_centroid_by_cosine_wins()
        {
            var centroids = new[]
            {
                new[] { 1.0, 0.0, 0.0 },
                new[] { 0.0, 1.0, 0.0 },
                new[] { 0.0, 0.0, 1.0 },
            };

            Assert.AreEqual(1, CalibrationMath.Nearest(centroids, new[] { 0.1, 5.0, 0.1 }).ClassIndex);
            Assert.AreEqual(2, CalibrationMath.Nearest(centroids, new[] { 0.1, 0.1, 5.0 }).ClassIndex);
        }

        [Test]
        public void Cosine_ignores_magnitude()
        {
            var centroids = new[] { new[] { 1.0, 0.0 }, new[] { 0.0, 1.0 } };

            var small = CalibrationMath.Nearest(centroids, new[] { 3.0, 1.0 });
            var large = CalibrationMath.Nearest(centroids, new[] { 3000.0, 1000.0 });

            Assert.AreEqual(small.ClassIndex, large.ClassIndex);
            Assert.AreEqual(small.WinnerSimilarity, large.WinnerSimilarity, 1e-12);
        }

        [Test]
        public void An_exact_tie_goes_to_the_lower_class_index()
        {
            // Mirror images about the probe direction: structurally equal similarity, not a
            // pair of values that happen to round the same way.
            var centroids = new[]
            {
                new[] { 2.0, 0.0, 1.0 },
                new[] { 0.0, 2.0, 1.0 },
                new[] { -1.0, -1.0, -1.0 },
            };

            var result = CalibrationMath.Nearest(centroids, new[] { 1.0, 1.0, 1.0 });

            Assert.AreEqual(0, result.ClassIndex);
            Assert.AreEqual(0.0, result.Margin, 1e-15,
                            "a genuine tie should have a zero margin");
        }

        [Test]
        public void The_margin_is_the_gap_to_the_runner_up()
        {
            var centroids = new[]
            {
                new[] { 1.0, 0.0, 0.0 },
                new[] { 0.9, 0.4, 0.0 },
                new[] { 0.0, 0.0, 1.0 },
            };

            var result = CalibrationMath.Nearest(centroids, new[] { 1.0, 0.0, 0.0 });

            Assert.AreEqual(0, result.ClassIndex);
            Assert.AreEqual(result.WinnerSimilarity - result.RunnerUpSimilarity, result.Margin, 1e-15);
            Assert.Greater(result.Margin, 0.0);
        }

        [Test]
        public void A_zero_vector_produces_a_number_rather_than_a_NaN()
        {
            // The `+ 1e-12` norm floor exists for exactly this: without it a zero feature
            // vector divides by zero and every similarity becomes NaN, which propagates into
            // the vote as a silently missing prediction.
            var centroids = new[] { new[] { 1.0, 0.0 }, new[] { 0.0, 1.0 } };

            var result = CalibrationMath.Nearest(centroids, new[] { 0.0, 0.0 });

            Assert.IsFalse(double.IsNaN(result.WinnerSimilarity));
            Assert.IsFalse(double.IsNaN(result.Margin));
        }

        [Test]
        public void A_single_centroid_is_refused_because_there_is_no_choice_to_make()
        {
            Assert.Throws<ArgumentException>(
                () => CalibrationMath.Nearest(new[] { new[] { 1.0, 0.0 } }, new[] { 1.0, 0.0 }));
        }
    }

    public class MeanTests
    {
        [Test]
        public void The_centroid_is_the_arithmetic_mean()
        {
            var mean = CalibrationMath.Mean(new[]
            {
                new[] { 0.0, 10.0 },
                new[] { 2.0, 20.0 },
                new[] { 4.0, 30.0 },
            });

            Assert.AreEqual(2.0, mean[0], 1e-15);
            Assert.AreEqual(20.0, mean[1], 1e-15);
        }

        [Test]
        public void Averaging_nothing_is_refused()
        {
            Assert.Throws<ArgumentException>(() => CalibrationMath.Mean(new double[0][]));
        }

        [Test]
        public void A_ragged_set_is_refused()
        {
            Assert.Throws<ArgumentException>(() => CalibrationMath.Mean(new[]
            {
                new[] { 1.0, 2.0 },
                new[] { 1.0 },
            }));
        }
    }
}
