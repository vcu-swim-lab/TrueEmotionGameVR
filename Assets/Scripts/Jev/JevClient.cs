using System;
using System.Linq;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
using VcuSwimLab.Calibration;

namespace VcuSwimLab.Jev
{
    public readonly struct JevResult
    {
        public readonly float[] Scores;
        public readonly string Error;

        private JevResult(float[] scores, string error)
        {
            Scores = scores;
            Error = error;
        }

        public bool Ok => Scores != null;

        public static JevResult Success(float[] scores) => new JevResult(scores, null);

        public static JevResult Failure(string error) => new JevResult(null, error);
    }

    public static class JevClient
    {
        public const string SystemOnePath = "/v1/systemone";
        public const float InitialBackoffSeconds = 0.5f;
        public const float MaxBackoffSeconds = 5f;

        private static readonly string[] ClassNames = Enum.GetNames(typeof(Emotion));

        private static readonly string[] AuNames = Enum.GetNames(typeof(OVRFaceExpressions.FaceExpression))
            .Take(CalibrationMath.FaceActionUnitCount)
            .ToArray();

        public static bool ShouldRetry(long statusCode) => statusCode == 0 || statusCode == 408 || statusCode == 429 || statusCode >= 500;

        public static float BackoffSeconds(int attempt) => Mathf.Min(MaxBackoffSeconds, InitialBackoffSeconds * Mathf.Pow(2f, attempt));

        public static async Awaitable<JevResult> Classify(JevSettings settings, float[][] window, CalibrationProfile profile)
        {
            if (settings == null)
            {
                return JevResult.Failure("no Jev Settings asset is assigned");
            }

            string apiKey = settings.ResolveApiKey();
            if (apiKey == null)
            {
                return JevResult.Failure($"no Typesafe API key: set it in the Jev Settings asset or in {JevSettings.ApiKeyVariable}");
            }

            string body;
            try
            {
                body = JevRequestBuilder.Build(settings.Model, window, AuNames, ClassNames, profile);
            }
            catch (ArgumentException e)
            {
                return JevResult.Failure(e.Message);
            }

            string url = settings.BaseUrl.TrimEnd('/') + SystemOnePath;

            for (int attempt = 0; ; ++attempt)
            {
                var (status, text) = await Post(url, apiKey, body, settings.TimeoutSeconds);

                if (status >= 200 && status < 300)
                {
                    return JevResponseParser.TryParse(text, ClassNames, out var scores, out string error)
                        ? JevResult.Success(scores)
                        : JevResult.Failure(error);
                }

                if (attempt >= settings.MaxRetries || !ShouldRetry(status))
                {
                    return JevResult.Failure(JevResponseParser.FailureMessage(status, text));
                }

                await Awaitable.WaitForSecondsAsync(BackoffSeconds(attempt));
            }
        }

        private static async Awaitable<(long status, string text)> Post(string url, string apiKey, string body, float timeoutSeconds)
        {
            using var request = new UnityWebRequest(url, UnityWebRequest.kHttpVerbPOST);
            request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body));
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            request.SetRequestHeader("Authorization", $"Bearer {apiKey}");
            request.timeout = Mathf.CeilToInt(timeoutSeconds);

            await request.SendWebRequest();

            string text = request.responseCode == 0 ? request.error : request.downloadHandler.text;
            return (request.responseCode, text);
        }
    }
}
