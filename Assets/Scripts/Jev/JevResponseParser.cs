using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace VcuSwimLab.Jev
{
    public static class JevResponseParser
    {
        private const int MaxDetailLength = 200;

        private static readonly Dictionary<long, string> StatusReasons = new()
        {
            [0] = "no response from Typesafe",
            [401] = "the Typesafe API key is missing or invalid",
            [422] = "Typesafe rejected the request",
            [429] = "the Typesafe rate limit was exceeded",
            [529] = "Typesafe is overloaded",
        };

        public static bool TryParse(string json, IReadOnlyList<string> classNames, out float[] scores, out string error)
        {
            scores = null;

            if (!TryParseObject(json, out var root) || !(root["answers"] is JObject answers))
            {
                error = "Typesafe returned a response without answers";
                return false;
            }

            var values = new double[classNames.Count];
            for (int e = 0; e < values.Length; ++e)
            {
                var token = (answers[classNames[e]] as JObject)?["noul"];
                if (token == null || (token.Type != JTokenType.Float && token.Type != JTokenType.Integer))
                {
                    error = $"Typesafe gave no answer for {classNames[e]}";
                    return false;
                }

                values[e] = token.Value<double>();
                if (double.IsNaN(values[e]) || double.IsInfinity(values[e]) || values[e] < 0.0)
                {
                    error = $"Typesafe gave an invalid answer for {classNames[e]}";
                    return false;
                }
            }

            double sum = values.Sum();
            if (sum <= 0.0)
            {
                error = "Typesafe found no emotion in the face";
                return false;
            }

            scores = values.Select(v => (float)(v / sum)).ToArray();
            error = null;
            return true;
        }

        public static string FailureMessage(long statusCode, string body)
        {
            string reason = StatusReasons.TryGetValue(statusCode, out var known) ? known : $"Typesafe answered HTTP {statusCode}";
            string detail = Detail(body);
            return string.IsNullOrEmpty(detail) ? reason : $"{reason}: {detail}";
        }

        private static string Detail(string body)
        {
            if (string.IsNullOrWhiteSpace(body))
            {
                return null;
            }

            if (!TryParseObject(body, out var root))
            {
                string text = body.Trim();
                return text.Length > MaxDetailLength ? text.Substring(0, MaxDetailLength) : text;
            }

            return Text(root["error"]) ?? Text(root["message"]) ?? Text(root["detail"]);
        }

        private static string Text(JToken token)
        {
            switch (token)
            {
                case JValue value when value.Type == JTokenType.String:
                    return (string)value;
                case JArray array:
                    string joined = string.Join("; ", array.Select(Text).Where(t => !string.IsNullOrEmpty(t)));
                    return joined.Length == 0 ? null : joined;
                case JObject obj:
                    return Text(obj["message"]) ?? Text(obj["msg"]);
                default:
                    return null;
            }
        }

        private static bool TryParseObject(string json, out JObject root)
        {
            root = null;
            if (string.IsNullOrWhiteSpace(json))
            {
                return false;
            }

            try
            {
                root = JObject.Parse(json);
                return true;
            }
            catch (JsonException)
            {
                return false;
            }
        }
    }
}
