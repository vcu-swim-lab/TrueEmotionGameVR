using System.Collections.Generic;

using UnityEngine;
using Random = UnityEngine.Random;

using TMPro;
using Unity.InferenceEngine;


[RequireComponent(typeof(EmotionPredictor))]
[RequireComponent(typeof(OVRFaceExpressions))]
public class ProgressGame : MonoBehaviour
{
    private TextMeshProUGUI text;
    private OVRFaceExpressions faceExpressions;
    private EmotionPredictor predictor;
    [SerializeField] private ModelAsset auModel;

    // Map emotion name to emoji for display
    private static readonly Dictionary<Emotion, string> emotionToEmoji = new()
    {
        { Emotion.Anger, "🤬" },
        { Emotion.Disgust, "🤢" },
        { Emotion.Fear, "😱" },
        { Emotion.Happiness, "😀" },
        { Emotion.Sadness, "☹️" },
        { Emotion.Surprise, "😲" },
    };

    private readonly Emotion[] emotionList = System.Enum.GetValues(typeof(Emotion)) as Emotion[];

    // Store per-emotion confidence history
    private readonly Dictionary<Emotion, List<float>> emotionConfidenceLogs = new();

    private Awaitable gameTask;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        text = GameObject.Find("Instruction").GetComponent<TextMeshProUGUI>();
        faceExpressions = GetComponent<OVRFaceExpressions>();

        predictor = GetComponent<EmotionPredictor>();
        predictor.Setup(
            new DeviceReader[] { new AUDevice(faceExpressions) },
            new ModelAsset[][] { new[] { auModel } }
        );


        gameTask = RunGame();
    }

    private async Awaitable RunGame()
    {
        // Shuffle emotion list
        for (int i = emotionList.Length - 1; i > 0; --i)
        {
            int j = Random.Range(0, i + 1);
            (emotionList[i], emotionList[j]) = (emotionList[j], emotionList[i]);
        }

        text.text = "You got 10s to act each emotion shown to you. Good luck.";
        await Awaitable.WaitForSecondsAsync(2f);

        foreach (var emotion in emotionList)
        {
            // Countdown before showing emotion
            for (int j = 3; j >= 0; --j)
            {
                text.text = $"{j}";
                await Awaitable.WaitForSecondsAsync(1f);
            }

            // Show emoji for current emotion
            string emoji = emotionToEmoji[emotion];
            text.text = emoji;

            // Run prediction loop for this emotion
            await RunPredictionCoroutine(emotion);
        }

        text.text = "Thanks for playing!";
        Debug.Log("All emotion predictions collected.");
    }

    private async Awaitable RunPredictionCoroutine(Emotion emotion)
    {
        print("Prediction started for: " + emotion);
        await RunPredictionLoop(emotion);
    }

    private async Awaitable RunPredictionLoop(Emotion emotion)
    {
        emotionConfidenceLogs[emotion] = new List<float>();

        const int intervalMs = 1000;
        var startTime = Time.realtimeSinceStartup;

        for (int i = 0; i < 10; ++i)
        {
            // Calculate next target time
            float nextTick = startTime + (i + 1) * (intervalMs / 1000f);

            // Await prediction
            var allConfidences = await PredictEmotionAsync();

            if (allConfidences.TryGetValue(emotion, out float confidence))
            {
                emotionConfidenceLogs[emotion].Add(confidence);
                Debug.Log($"[{emotion}] Prediction {i + 1}: {confidence:F2}");
            }

            // Wait for the remaining time until the next 1-second mark
            float remaining = nextTick - Time.realtimeSinceStartup;
            if (remaining > 0)
            {
                await Awaitable.WaitForSecondsAsync(remaining);
            }
        }
    }

    // Simulated async emotion predictor returning full confidence dictionary
    private async Awaitable<Dictionary<Emotion, float>> PredictEmotionAsync()
    {
        // await Awaitable.NextFrameAsync(); // simulate async

        // var emotions = new Dictionary<Emotion, float>();
        // foreach (var emo in emotionList)
        // {
        //     emotions[emo] = Random.Range(0f, 1f);
        // }

        // // Normalize
        // float total = 0f;
        // foreach (var val in emotions.Values) total += val;
        // foreach (var key in new List<Emotion>(emotions.Keys)) emotions[key] /= total;

        // return emotions;

        var emo = await predictor.Predict();
        return new Dictionary<Emotion, float>
        {
            { emo, 1f } // Simulate full confidence for the predicted emotion
        };
    }
}