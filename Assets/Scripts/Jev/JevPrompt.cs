using System;
using System.Collections.Generic;

namespace VcuSwimLab.Jev
{
    public static class JevPrompt
    {
        public const string QuestionType = "noul";

        private const string Relaxed = "These same movements are weak, absent, or relaxed.";

        private static readonly Dictionary<string, (string yes, string no)> Criteria = new()
        {
            ["Anger"] = Active("brow lowering and drawing together, lid tightening, and lip pressing or tightening"),
            ["Disgust"] = Active("nose wrinkling, upper lip raising, and cheek raising"),
            ["Fear"] = Active("brow raising and drawing together, upper lid raising, and lip stretching"),
            ["Happiness"] = Active("cheek raising, lip corners pulling up, and eye narrowing"),
            ["Neutral"] = (
                "Little or no activity overall, with no coordinated activation of any single expression's movements.",
                "A clear coordinated activation of the movements of any single expression."),
            ["Sadness"] = Active("inner brow raising, brow lowering, and lip corners pulling down"),
            ["Surprise"] = Active("inner and outer brow raising, upper lid raising, and jaw dropping"),
        };

        public static string Question(string emotion) => $"Does `current_face` show {emotion}?";

        public static string HowToJudge(string emotion)
        {
            return $"If `players_own_faces.{emotion}` exists, judge by how closely current_face matches how this " +
                   $"player makes {emotion}: its similarity, its rank, and whether this player's signature movements " +
                   $"for {emotion} are active now. Otherwise judge by general knowledge of {emotion} expressions. " +
                   "Several strong, coordinated movements outweigh one weak movement. Use only the data given.";
        }

        public static (string yes, string no) CriteriaFor(string emotion)
        {
            if (!Criteria.TryGetValue(emotion, out var criteria))
            {
                throw new ArgumentException($"no Jev criteria for '{emotion}'", nameof(emotion));
            }

            return criteria;
        }

        private static (string yes, string no) Active(string movements)
        {
            return ($"A coordinated activation of {movements}.", Relaxed);
        }
    }
}
