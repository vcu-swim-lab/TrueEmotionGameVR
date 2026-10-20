using System;

using NUnit.Framework;

namespace VcuSeCenter.Calibration.Tests
{
    public class PersonalCentroidBuilderTests
    {
        private const int Classes = 7;

        private static readonly string[] Names =
        {
            "Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise",
        };

        private static float[] Scores(int peak, float mass = 0.6f)
        {
            var scores = new float[Classes];
            float rest = (1.0f - mass) / (Classes - 1);
            for (int i = 0; i < Classes; ++i) scores[i] = rest;
            scores[peak] = mass;
            return scores;
        }

        private static PersonalCentroidBuilder Complete()
        {
            var builder = new PersonalCentroidBuilder(Classes);
            for (int cls = 0; cls < Classes; ++cls)
            {
                for (int tick = 0; tick < 10; ++tick)
                {
                    builder.Observe(cls, tick, Scores(cls));
                }
            }

            return builder;
        }

        [Test]
        public void Tick_zero_is_not_taken_into_the_centroid()
        {
            var builder = new PersonalCentroidBuilder(Classes);

            Assert.IsFalse(builder.Observe(0, 0, Scores(0)),
                           "tick 0 is a transition sample and every block drops it");
            Assert.AreEqual(0, builder.TickCount(0));

            Assert.IsTrue(builder.Observe(0, 1, Scores(0)));
            Assert.AreEqual(1, builder.TickCount(0));
        }

        [Test]
        public void Only_ticks_one_through_nine_are_scored()
        {
            Assert.IsFalse(PersonalCentroidBuilder.IsScoredTick(0));
            for (int tick = 1; tick <= 9; ++tick)
            {
                Assert.IsTrue(PersonalCentroidBuilder.IsScoredTick(tick), $"tick {tick}");
            }
            Assert.IsFalse(PersonalCentroidBuilder.IsScoredTick(10));
        }

        [Test]
        public void A_complete_block_contributes_exactly_nine_ticks_per_class()
        {
            var builder = Complete();

            for (int cls = 0; cls < Classes; ++cls)
            {
                Assert.AreEqual(PersonalCentroidBuilder.RequiredTicksPerCentroid,
                                builder.TickCount(cls), Names[cls]);
            }

            Assert.IsTrue(builder.TryBuild(Names, out var centroids, out string error), error);
            Assert.IsNull(error);
            Assert.AreEqual(Classes, centroids.Length);
            Assert.AreEqual(Classes, centroids[0].Length);
        }

        [Test]
        public void The_required_tick_count_matches_the_offline_constant()
        {
            Assert.AreEqual(9, PersonalCentroidBuilder.RequiredTicksPerCentroid,
                            "calibration_study.TICKS_REQUIRED_PER_CENTROID");
        }

        [Test]
        public void One_short_class_fails_the_whole_calibration()
        {
            var builder = new PersonalCentroidBuilder(Classes);
            for (int cls = 0; cls < Classes; ++cls)
            {
                // Fear loses one tick to a tracking dropout.
                int ticks = cls == 2 ? 8 : 9;
                for (int tick = 1; tick <= ticks; ++tick)
                {
                    builder.Observe(cls, tick, Scores(cls));
                }
            }

            Assert.IsFalse(builder.TryBuild(Names, out var centroids, out string error));
            Assert.IsNull(centroids, "a partial profile must not be handed back");
            StringAssert.Contains("Fear 8/9", error);
        }

        [Test]
        public void An_untracked_tick_contributes_nothing_and_leaves_the_class_short()
        {
            var builder = new PersonalCentroidBuilder(Classes);
            for (int cls = 0; cls < Classes; ++cls)
            {
                for (int tick = 1; tick <= 9; ++tick)
                {
                    bool untracked = cls == 0 && tick == 5;
                    builder.Observe(cls, tick, untracked ? null : Scores(cls));
                }
            }

            Assert.AreEqual(8, builder.TickCount(0));
            Assert.IsFalse(builder.TryBuild(Names, out _, out string error));
            StringAssert.Contains("Anger 8/9", error);
        }

        [Test]
        public void A_missing_class_fails_rather_than_enrolling_six()
        {
            var builder = new PersonalCentroidBuilder(Classes);
            for (int cls = 0; cls < Classes - 1; ++cls)
            {
                for (int tick = 1; tick <= 9; ++tick) builder.Observe(cls, tick, Scores(cls));
            }

            Assert.IsFalse(builder.TryBuild(Names, out _, out string error));
            StringAssert.Contains("Surprise 0/9", error);
        }

        [Test]
        public void A_wrong_width_score_vector_fails_the_calibration()
        {
            var builder = Complete();
            builder.Observe(0, 1, new float[3]);

            Assert.IsFalse(builder.TryBuild(Names, out _, out string error));
            StringAssert.Contains("3 scores", error);
        }

        [Test]
        public void A_non_finite_score_fails_the_calibration_rather_than_poisoning_a_centroid()
        {
            var builder = new PersonalCentroidBuilder(Classes);
            var poisoned = Scores(0);
            poisoned[3] = float.NaN;

            for (int cls = 0; cls < Classes; ++cls)
            {
                for (int tick = 1; tick <= 9; ++tick)
                {
                    builder.Observe(cls, tick, cls == 0 && tick == 4 ? poisoned : Scores(cls));
                }
            }

            Assert.IsFalse(builder.TryBuild(Names, out _, out string error));
            StringAssert.Contains("log-score transform refused", error);
        }

        [Test]
        public void The_centroid_is_the_mean_of_the_nine_log_score_vectors()
        {
            var builder = new PersonalCentroidBuilder(Classes);

            // Nine identical ticks, so the mean is that vector and the arithmetic is checkable
            // by hand rather than only against another implementation.
            var scores = Scores(3, 0.7f);
            for (int cls = 0; cls < Classes; ++cls)
            {
                for (int tick = 1; tick <= 9; ++tick)
                {
                    builder.Observe(cls, tick, cls == 3 ? scores : Scores(cls));
                }
            }

            Assert.IsTrue(builder.TryBuild(Names, out var centroids, out _));

            var expected = CalibrationMath.LogScore(scores);
            for (int i = 0; i < Classes; ++i)
            {
                Assert.AreEqual(expected[i], centroids[3][i], 1e-12);
            }
        }

        [Test]
        public void A_class_outside_the_schema_is_refused_outright()
        {
            var builder = new PersonalCentroidBuilder(Classes);

            Assert.Throws<ArgumentOutOfRangeException>(() => builder.Observe(7, 1, Scores(0)));
            Assert.Throws<ArgumentOutOfRangeException>(() => builder.Observe(-1, 1, Scores(0)));
        }

        [Test]
        public void Tick_counts_are_reported_for_the_session_metadata()
        {
            var builder = Complete();
            var counts = builder.TickCounts();

            Assert.AreEqual(Classes, counts.Length);
            foreach (int count in counts) Assert.AreEqual(9, count);
        }
    }
}
