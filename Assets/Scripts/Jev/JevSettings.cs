using System;
using UnityEngine;

namespace VcuSwimLab.Jev
{
    [CreateAssetMenu(fileName = "JevSettings", menuName = "VR Emotion/Jev Settings")]
    public class JevSettings : ScriptableObject
    {
        public const string ApiKeyVariable = "TYPESAFE_API_KEY";
        public const string DefaultBaseUrl = "https://api.typesafe.ai";
        public const string DefaultModel = "jev-1.13.0";
        public const float DefaultTimeoutSeconds = 10f;
        public const int DefaultMaxRetries = 2;

        [Tooltip("Your own Typesafe API key (console.typesafe.ai/keys). The library provides none. " +
                 "Leave empty to use the TYPESAFE_API_KEY environment variable. Do not commit this asset.")]
        [SerializeField]
        private string apiKey = "";

        [Tooltip("Typesafe endpoint, or your own proxy that forwards to it.")]
        [SerializeField]
        private string baseUrl = DefaultBaseUrl;

        [Tooltip("Pin a versioned model for stable results.")]
        [SerializeField]
        private string model = DefaultModel;

        [SerializeField]
        private float timeoutSeconds = DefaultTimeoutSeconds;

        [SerializeField]
        private int maxRetries = DefaultMaxRetries;

        public string ApiKey
        {
            get => apiKey;
            set => apiKey = value;
        }

        public string BaseUrl => baseUrl;
        public string Model => model;
        public float TimeoutSeconds => timeoutSeconds;
        public int MaxRetries => maxRetries;

        public string ResolveApiKey() => Resolve(apiKey, Environment.GetEnvironmentVariable(ApiKeyVariable));

        internal static string Resolve(string assetKey, string environmentKey)
        {
            if (!string.IsNullOrWhiteSpace(assetKey)) return assetKey.Trim();
            return string.IsNullOrWhiteSpace(environmentKey) ? null : environmentKey.Trim();
        }
    }
}
