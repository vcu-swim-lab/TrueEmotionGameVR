using System;

using NUnit.Framework;

namespace VcuSwimLab.Calibration.Tests
{
    /// <summary>
    /// The library's `MovementCalibrator` and the protocol's `RuntimeCalibrator` in movement
    /// mode must make the same decision from the same calibration samples, so a game built on
    /// the library runs the method this project measured.
    /// </summary>
    public class MovementCalibratorParityTests
    {
        private static readonly string[] Names =
        {
            "Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise",
        };

        private const int Classes = 7;

        private static float[] RandomScores(Random rng)
        {
            var scores = new float[Classes];
            float sum = 0f;
            for (int i = 0; i < Classes; ++i) sum += scores[i] = (float)rng.NextDouble() + 0.01f;
            for (int i = 0; i < Classes; ++i) scores[i] /= sum;
            return scores;
        }

        private static float[][] RandomWindow(Random rng, int cls)
        {
            var window = new float[5][];
            for (int f = 0; f < 5; ++f)
            {
                window[f] = new float[70];
                for (int a = 0; a < 70; ++a)
                {
                    window[f][a] = (float)(rng.NextDouble() * 0.3 + (a % Classes == cls ? 0.4 : 0.0));
                }
            }

            return window;
        }

        [Test]
        public void Library_and_protocol_make_identical_movement_decisions()
        {
            var rng = new Random(20261012);
            var library = new MovementCalibrator(Names);
            var protocol = RuntimeCalibrator.Create(CalibrationMode.Movement, Names, null);

            for (int cls = 0; cls < Classes; ++cls)
            {
                library.BeginEmotion(cls);
                for (int tick = 0; tick < 10; ++tick)
                {
                    var scores = RandomScores(rng);
                    var window = RandomWindow(rng, cls);
                    Assert.IsTrue(library.Observe(scores, window));
                    protocol.ObserveCalibrationTick(cls, tick, scores, window);
                }
            }

            Assert.IsTrue(library.Finish(), library.Error);
            protocol.Seal();
            Assert.IsTrue(protocol.CalibrationValid, protocol.CalibrationError);

            for (int i = 0; i < 500; ++i)
            {
                var scores = RandomScores(rng);
                var window = RandomWindow(rng, rng.Next(Classes));

                var mine = library.Classify(scores, window);
                var theirs = protocol.Classify(1, scores, window);

                Assert.IsTrue(mine.HasValue && theirs.Scored);
                Assert.AreEqual(theirs.ClassIndex, mine.Value.ClassIndex);
                Assert.AreEqual(theirs.Margin, mine.Value.Margin, 1e-12);
                Assert.AreEqual(theirs.WinnerSimilarity, mine.Value.WinnerSimilarity, 1e-12);
            }
        }
    }
}
