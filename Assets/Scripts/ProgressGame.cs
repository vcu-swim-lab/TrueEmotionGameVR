using System.Collections.Generic;

using UnityEngine;
using Random = UnityEngine.Random;

using TMPro;
using Unity.InferenceEngine;
using System.Linq;
using UnityEngine.UI;


[RequireComponent(typeof(EmotionPredictor))]
[RequireComponent(typeof(OVRFaceExpressions))]
public class ProgressGame : MonoBehaviour
{
    private TextMeshProUGUI text;
    // private TextMeshProUGUI debug;
    private Button restart;

    // private OVRFaceExpressions faceExpressions;
    private EmotionPredictor predictor;
    // [SerializeField] private ModelAsset auModel;

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

    private readonly Emotion[] emotionList = emotionToEmoji.Keys.ToArray();

    // Store per-emotion confidence history
    private readonly Dictionary<Emotion, List<float>> emotionConfidenceLogs = new();


    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        text = GameObject.Find("Instruction").GetComponent<TextMeshProUGUI>();
        // debug = GameObject.Find("Debug").GetComponent<TextMeshProUGUI>();
        restart = GameObject.Find("Restart").GetComponent<Button>();

        // faceExpressions = GetComponent<OVRFaceExpressions>();

        predictor = GetComponent<EmotionPredictor>();
        // predictor.Setup(
        //     new DeviceReader[] { new AUDevice(faceExpressions) },
        //     new ModelAsset[][] { new[] { auModel } }
        // );
        predictor.Polling = false;


        RunGame();
    }

    private async void RunGame()
    {
        restart.gameObject.SetActive(false);
        int score = 0;

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
            score += await RunPredictionCoroutine(emotion);

            // debug.text = "Prediction complete for " + emotion;
        }

        text.text = $"Thanks for playing! Score: {score}";
        Debug.Log("All emotion predictions collected.");

        restart.gameObject.SetActive(true);

        restart.onClick.AddListener(() =>
        {
            // Reset the game
            emotionConfidenceLogs.Clear();
            RunGame();
        });
    }

    private async Awaitable<int> RunPredictionCoroutine(Emotion emotion)
    {
        predictor.Flush(); // Ensure we start with fresh data
        predictor.Polling = true;

        int score = 0;

        emotionConfidenceLogs[emotion] = new List<float>();

        const int intervalMs = 1000;

        int times = 10;
        for (int i = 0; i < times; ++i)
        {
            // Calculate next target time
            float nextTick = Time.time + (intervalMs / 1000f);

            // Await prediction
            var allConfidences = await PredictEmotionAsync();

            if (allConfidences.TryGetValue(emotion, out float confidence))
            {
                emotionConfidenceLogs[emotion].Add(confidence);
                // debug.text = $"[{emotion}] Prediction {i + 1}: {confidence:F2}";

                score += (int)(confidence * 10);
            }


            // Wait for the remaining time until the next 1-second mark
            float remaining = nextTick - Time.time;
            if (remaining > 0)
            {
                await Awaitable.WaitForSecondsAsync(remaining);
            }
        }

        print("Done with predicting " + emotion);

        predictor.Polling = false;

        return score;
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
            { emo.Item1, emo.Item2 } // Simulate full confidence for the predicted emotion
        };
    }
}