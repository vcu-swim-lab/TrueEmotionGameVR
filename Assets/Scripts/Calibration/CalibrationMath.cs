using System;
using System.Collections.Generic;

namespace VcuSwimLab.Calibration
{
    /// <summary>
    /// The calibration arithmetic, ported one operation at a time from
    /// `Analysis/calibration_study.py`.
    /// </summary>
    /// <remarks>
    /// Nothing here is a re-derivation. Every method below names the Python function it
    /// mirrors, and `Analysis/make_runtime_parity_fixture.py` generates a golden fixture from
    /// those same Python functions which the tests replay through this class. If a change
    /// here alters a single prediction the parity test fails, which is the point: the offline
    /// result and the headset result have to be the same number computed twice, not two
    /// numbers that happen to agree.
    ///
    /// Everything is `double`. The Python pipeline carries `float32` through `np.log` and the
    /// matmul, so the two are not bit-identical; the fixture pins the `float64` reference
    /// tightly and separately asserts that the `float32` pipeline reaches the same argmax on
    /// every tick. Where that could stop being true -- a near-tie between two centroids -- the
    /// margin is logged, so it is visible in the data rather than assumed away.
    /// </remarks>
    public static class CalibrationMath
    {
        /// <summary>
        /// The log clamp. Frozen at 1e-12 and matched to `calibration_study.DEFAULT_EPSILON`.
        /// </summary>
        /// <remarks>
        /// The exported graph ends in Softmax and can emit values small enough to underflow;
        /// without a floor a single 0.0 sends a whole feature vector to -inf and every
        /// similarity becomes NaN. 1e-12 sits provably below every score observed in the
        /// corpus (smallest ~7e-06), so it is inert on real data and only catches underflow.
        /// </remarks>
        public const double Epsilon = 1e-12;

        /// <summary>
        /// The weight on the personal term, frozen before the EmojiHero test split was
        /// opened and never re-selected since.
        /// </summary>
        /// <remarks>
        /// Selected on the EmojiHero validation split (n=8) as the argmax of
        /// participant-equal mean sequence macro-F1 over a 21-point grid. It is `const`, and
        /// no inspector field, command line or config file can move it: the value is an
        /// experimental constant of a preregistered study, and a session recorded at some
        /// other alpha is not a session of that study.
        ///
        /// Read it as "40% personal, 60% population". The selection curve is very flat
        /// (0.40 vs 0.45 differ by 0.00053), so the plateau is the robust finding and the
        /// exact argmax is not -- which is a reason to keep the number fixed, not a reason
        /// to let it drift.
        /// </remarks>
        public const double FrozenHybridAlpha = 0.40;

        /// <summary>
        /// Added to every vector norm before dividing, mirroring the `+ 1e-12` in
        /// `calibration_study.classify`. It keeps a zero vector from producing NaN.
        /// </summary>
        public const double NormFloor = 1e-12;

        public const string FeatureSpaceName = "logscore";
        public const string TransformName = "log(clip(score, epsilon))";
        public const string MetricName = "cosine";

        /// <summary>
        /// `calibration_study.logscore`: log(max(score, epsilon)), elementwise.
        /// </summary>
        /// <exception cref="ArgumentException">
        /// A non-finite score. Refused rather than clamped: NaN out of the graph means the
        /// inference went wrong, and a centroid averaged over it is silently meaningless.
        /// </exception>
        public static double[] LogScore(IReadOnlyList<float> scores)
        {
            if (scores == null) throw new ArgumentNullException(nameof(scores));

            var result = new double[scores.Count];
            for (int i = 0; i < scores.Count; ++i)
            {
                double value = scores[i];
                if (double.IsNaN(value) || double.IsInfinity(value))
                {
                    throw new ArgumentException(
                        $"score {i} is {value}; the log-score transform needs finite model output",
                        nameof(scores));
                }

                result[i] = Math.Log(value < Epsilon ? Epsilon : value);
            }

            return result;
        }

        /// <summary>
        /// `calibration_study.centroids`, for one class: the arithmetic mean of that class's
        /// feature vectors.
        /// </summary>
        public static double[] Mean(IReadOnlyList<double[]> vectors)
        {
            if (vectors == null) throw new ArgumentNullException(nameof(vectors));
            if (vectors.Count == 0)
            {
                throw new ArgumentException("cannot average an empty set of vectors", nameof(vectors));
            }

            int width = vectors[0].Length;
            var sum = new double[width];

            for (int v = 0; v < vectors.Count; ++v)
            {
                if (vectors[v].Length != width)
                {
                    throw new ArgumentException(
                        $"vector {v} has length {vectors[v].Length}, expected {width}",
                        nameof(vectors));
                }

                for (int i = 0; i < width; ++i)
                {
                    sum[i] += vectors[v][i];
                }
            }

            for (int i = 0; i < width; ++i)
            {
                sum[i] /= vectors.Count;
            }

            return sum;
        }

        /// <summary>
        /// The hybrid blend: <c>H_e = alpha * P_e + (1 - alpha) * G_e</c>, row by row.
        /// </summary>
        /// <remarks>
        /// Both matrices are indexed by the frozen class order, so row e of each is the same
        /// emotion. Mirrors `hybrid_alpha * M + (1.0 - hybrid_alpha) * population_rows` in
        /// `calibration_study.evaluate`.
        /// </remarks>
        public static double[][] Blend(double[][] personal, double[][] population, double alpha)
        {
            if (personal == null) throw new ArgumentNullException(nameof(personal));
            if (population == null) throw new ArgumentNullException(nameof(population));
            if (personal.Length != population.Length)
            {
                throw new ArgumentException(
                    $"personal has {personal.Length} centroids, population has {population.Length}");
            }
            if (!(alpha >= 0.0 && alpha <= 1.0))
            {
                throw new ArgumentOutOfRangeException(
                    nameof(alpha), alpha, "the hybrid weight must lie in [0, 1]");
            }

            var blended = new double[personal.Length][];
            for (int e = 0; e < personal.Length; ++e)
            {
                if (personal[e].Length != population[e].Length)
                {
                    throw new ArgumentException(
                        $"centroid {e}: personal is {personal[e].Length}-dimensional, "
                        + $"population is {population[e].Length}-dimensional");
                }

                var row = new double[personal[e].Length];
                for (int i = 0; i < row.Length; ++i)
                {
                    row[i] = alpha * personal[e][i] + (1.0 - alpha) * population[e][i];
                }

                blended[e] = row;
            }

            return blended;
        }

        /// <summary>
        /// <c>v / (||v|| + 1e-12)</c>, the normalisation `calibration_study.classify` applies
        /// to both the centroids and the features before taking their dot product.
        /// </summary>
        /// <remarks>
        /// Normalising first and then taking the dot product -- rather than the algebraically
        /// equal `dot / (norm * norm)` -- keeps the order of floating-point operations the
        /// same as the Python, so the two cannot disagree in the last bits for a reason that
        /// has nothing to do with the calibration.
        /// </remarks>
        public static double[] Normalized(double[] vector)
        {
            if (vector == null) throw new ArgumentNullException(nameof(vector));

            double sumSquares = 0.0;
            for (int i = 0; i < vector.Length; ++i)
            {
                sumSquares += vector[i] * vector[i];
            }

            double scale = Math.Sqrt(sumSquares) + NormFloor;

            var unit = new double[vector.Length];
            for (int i = 0; i < vector.Length; ++i)
            {
                unit[i] = vector[i] / scale;
            }

            return unit;
        }

        public static double Dot(double[] a, double[] b)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (a.Length != b.Length)
            {
                throw new ArgumentException($"length {a.Length} against length {b.Length}");
            }

            double total = 0.0;
            for (int i = 0; i < a.Length; ++i)
            {
                total += a[i] * b[i];
            }

            return total;
        }

        /// <summary>Cosine similarity as `classify` computes it.</summary>
        public static double CosineSimilarity(double[] a, double[] b)
            => Dot(Normalized(a), Normalized(b));

        /// <summary>
        /// Nearest centroid by cosine similarity: the winner, its similarity, the runner-up's
        /// similarity, and the margin between them.
        /// </summary>
        /// <remarks>
        /// `calibration_study.classify` reads the winner off `np.argsort(-sim)`. For a
        /// seven-element row NumPy sorts with insertion sort, which is stable, so an exact
        /// tie resolves to the lower class index. The strictly-greater comparison below is
        /// that same rule stated outright rather than inherited from a sort implementation --
        /// and `make_runtime_parity_fixture.py` includes a hand-built exact tie so the two
        /// are checked against each other rather than assumed equal.
        /// </remarks>
        public static NearestCentroid Nearest(double[][] centroids, double[] feature)
        {
            if (centroids == null) throw new ArgumentNullException(nameof(centroids));
            if (feature == null) throw new ArgumentNullException(nameof(feature));
            if (centroids.Length < 2)
            {
                throw new ArgumentException(
                    $"nearest-centroid classification needs at least two centroids, got {centroids.Length}",
                    nameof(centroids));
            }

            var unitFeature = Normalized(feature);

            int best = -1;
            double bestSimilarity = double.NegativeInfinity;
            double runnerUp = double.NegativeInfinity;

            for (int e = 0; e < centroids.Length; ++e)
            {
                double similarity = Dot(unitFeature, Normalized(centroids[e]));

                // Strictly greater: on an exact tie the earlier class index keeps the win.
                if (similarity > bestSimilarity)
                {
                    runnerUp = bestSimilarity;
                    bestSimilarity = similarity;
                    best = e;
                }
                else if (similarity > runnerUp)
                {
                    runnerUp = similarity;
                }
            }

            return new NearestCentroid(best, bestSimilarity, runnerUp, bestSimilarity - runnerUp);
        }
    }

    /// <summary>One tick's nearest-centroid decision.</summary>
    public readonly struct NearestCentroid
    {
        public readonly int ClassIndex;
        public readonly double WinnerSimilarity;
        public readonly double RunnerUpSimilarity;
        public readonly double Margin;

        public NearestCentroid(int classIndex, double winnerSimilarity,
                               double runnerUpSimilarity, double margin)
        {
            ClassIndex = classIndex;
            WinnerSimilarity = winnerSimilarity;
            RunnerUpSimilarity = runnerUpSimilarity;
            Margin = margin;
        }
    }
}
