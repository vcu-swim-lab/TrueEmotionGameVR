using System.Collections.Generic;
using System.Linq;

namespace VcuSwimLab.Jev
{
    public static class JevFaceDescriber
    {
        public const double StrongFrom = 0.5;
        public const double ModerateFrom = 0.2;
        public const double SlightFrom = 0.05;
        public const double ModerateActivityFrom = 0.02;
        public const double HighActivityFrom = 0.06;
        public const double ActiveNowShare = 0.5;
        public const double NarrowGapBelow = 0.02;

        public static string Strength(double value)
        {
            if (value >= StrongFrom) return "strong";
            if (value >= ModerateFrom) return "moderate";
            return value >= SlightFrom ? "slight" : null;
        }

        public static int[] TopIndices(IReadOnlyList<double> values, int count)
        {
            return Enumerable.Range(0, values.Count)
                .Where(i => values[i] >= SlightFrom)
                .OrderByDescending(i => values[i])
                .Take(count)
                .ToArray();
        }

        public static string[] Label(IReadOnlyList<double> values, IReadOnlyList<string> names, IEnumerable<int> indices)
        {
            return indices.Select(i => $"{names[i]} ({Strength(values[i])})").ToArray();
        }

        public static string ActivityLevel(IReadOnlyList<double> values)
        {
            double mean = values.Count == 0 ? 0.0 : values.Average();
            if (mean >= HighActivityFrom) return "high";
            return mean >= ModerateActivityFrom ? "moderate" : "low";
        }

        public static string ActiveNow(IReadOnlyList<double> current, IReadOnlyList<double> reference,
                                       IReadOnlyCollection<int> signature)
        {
            int active = signature.Count(i => current[i] >= ActiveNowShare * reference[i]);
            return $"{active} of {signature.Count}";
        }

        public static int[] Ranks(IReadOnlyList<double> similarities)
        {
            var order = Enumerable.Range(0, similarities.Count).OrderByDescending(i => similarities[i]).ToArray();
            var ranks = new int[order.Length];
            for (int position = 0; position < order.Length; ++position)
            {
                ranks[order[position]] = position + 1;
            }

            return ranks;
        }

        public static string GapWord(IReadOnlyList<double> similarities)
        {
            var best = similarities.OrderByDescending(s => s).Take(2).ToArray();
            return best.Length == 2 && best[0] - best[1] < NarrowGapBelow ? "narrow" : "clear";
        }
    }
}
