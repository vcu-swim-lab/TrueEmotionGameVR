using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using VcuSwimLab.Calibration;

namespace VcuSwimLab.Jev
{
    public static class JevRequestBuilder
    {
        public const int ActiveAusShown = 8;
        public const int SignatureAusShown = 5;
        public const int SimilarityDecimals = 3;

        public static string Build(string model, IReadOnlyList<float[]> window, IReadOnlyList<string> auNames,
                                   IReadOnlyList<string> classNames, CalibrationProfile profile)
        {
            var feature = CalibrationMath.MovementFeature(window);

            var state = new JObject { ["current_face"] = CurrentFace(feature, auNames) };
            if (profile != null)
            {
                state["players_own_faces"] = OwnFaces(feature, auNames, classNames, profile);
            }

            var questions = new JObject();
            foreach (var emotion in classNames)
            {
                questions[emotion] = Question(emotion);
            }

            var request = new JObject { ["state"] = state, ["model"] = model, ["questions"] = questions };
            return request.ToString(Formatting.None);
        }

        private static JObject CurrentFace(double[] feature, IReadOnlyList<string> auNames)
        {
            var active = JevFaceDescriber.TopIndices(feature, ActiveAusShown);
            return new JObject
            {
                ["overall_activity"] = JevFaceDescriber.ActivityLevel(feature),
                ["active_aus"] = JArray.FromObject(JevFaceDescriber.Label(feature, auNames, active)),
            };
        }

        private static JObject OwnFaces(double[] feature, IReadOnlyList<string> auNames,
                                        IReadOnlyList<string> classNames, CalibrationProfile profile)
        {
            var similarities = profile.MovementCentroids
                .Select(centroid => CalibrationMath.CosineSimilarity(feature, centroid))
                .ToArray();
            var ranks = JevFaceDescriber.Ranks(similarities);

            var faces = new JObject { ["closest_match_gap"] = JevFaceDescriber.GapWord(similarities) };
            for (int e = 0; e < classNames.Count; ++e)
            {
                var centroid = profile.MovementCentroids[e];
                var signature = JevFaceDescriber.TopIndices(centroid, SignatureAusShown);
                faces[classNames[e]] = new JObject
                {
                    ["similarity"] = System.Math.Round(similarities[e], SimilarityDecimals),
                    ["rank"] = ranks[e],
                    ["signature_aus"] = JArray.FromObject(JevFaceDescriber.Label(centroid, auNames, signature)),
                    ["signature_aus_active_now"] = JevFaceDescriber.ActiveNow(feature, centroid, signature),
                };
            }

            return faces;
        }

        private static JObject Question(string emotion)
        {
            var (yes, no) = JevPrompt.CriteriaFor(emotion);
            return new JObject
            {
                ["type"] = JevPrompt.QuestionType,
                ["instructions"] = new JObject
                {
                    ["question"] = JevPrompt.Question(emotion),
                    ["how_to_judge"] = JevPrompt.HowToJudge(emotion),
                },
                ["criteria"] = new JObject { ["true"] = yes, ["false"] = no },
            };
        }
    }
}
