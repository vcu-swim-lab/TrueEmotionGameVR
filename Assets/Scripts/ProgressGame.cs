using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using TMPro;


public class ProgressGame : MonoBehaviour
{
    private TextMeshProUGUI text;

    // Map emotion name to emoji for display
    private static readonly Dictionary<string, string> emotionToEmoji = new()
    {
        { "angry", "🤬" },
        { "disgust", "🤢" },
        { "fear", "😱" },
        { "happy", "😀" },
        { "sad", "☹️" },
        { "surprised", "😲" }
    };

    // Emotion name list (shuffled later)
    private List<string> emotionList = new()
    {
        "angry", "disgust", "fear", "happy", "sad", "surprised"
    };


    // Store per-emotion confidence history
    private Dictionary<string, List<float>> emotionConfidenceLogs = new();

    void Start()
    {
        text = GameObject.Find("Instruction").GetComponent<TextMeshProUGUI>();

        StartCoroutine(RunGame());
    }

    private IEnumerator RunGame()
    {
        // Shuffle emotion list
        for (int i = emotionList.Count - 1; i > 0; --i)
        {
            int j = Random.Range(0, i + 1);
            (emotionList[i], emotionList[j]) = (emotionList[j], emotionList[i]);
        }

        text.text = "You got 10s to act each emotion shown to you. Good luck.";
        yield return new WaitForSeconds(2.0f);

        foreach (string emotion in emotionList)
        {
            // Countdown before showing emotion
            for (int j = 3; j >= 0; --j)
            {
                text.text = $"{j}";
                yield return new WaitForSeconds(1.0f);
            }

            // Show emoji for current emotion
            string emoji = emotionToEmoji[emotion];
            text.text = emoji;

            // Run prediction loop for this emotion
            yield return RunPredictionCoroutine(emotion);
        }

        text.text = "Thanks for playing!";
        Debug.Log("All emotion predictions collected.");
    }

    private IEnumerator RunPredictionCoroutine(string emotion)
    {
        print("Prediction started for: " + emotion);
        Task task = RunPredictionLoop(emotion);
        while (!task.IsCompleted)
            yield return null;

        if (task.Exception != null)
            Debug.LogError(task.Exception);
    }

    private async Task RunPredictionLoop(string emotion)
    {
        emotionConfidenceLogs[emotion] = new List<float>();

        const int intervalMs = 1000;
        var startTime = Time.realtimeSinceStartup;

        for (int i = 0; i < 10; ++i)
        {
            // Fire off prediction but don't await it yet
            var predictionTask = PredictEmotionAsync();

            // Calculate next target time
            float nextTick = startTime + (i + 1) * (intervalMs / 1000f);

            // Await prediction
            var allConfidences = await predictionTask;

            if (allConfidences.TryGetValue(emotion, out float confidence))
            {
                emotionConfidenceLogs[emotion].Add(confidence);
                Debug.Log($"[{emotion}] Prediction {i + 1}: {confidence:F2}");
            }

            // Wait for the remaining time until the next 1-second mark
            float remaining = nextTick - Time.realtimeSinceStartup;
            if (remaining > 0)
            {
                int delayMs = Mathf.RoundToInt(remaining * 1000f);
                await Task.Delay(delayMs);
            }
        }
    }

    // Simulated async emotion predictor returning full confidence dictionary
    private async Task<Dictionary<string, float>> PredictEmotionAsync()
    {
        await Task.Yield(); // simulate async

        var emotions = new Dictionary<string, float>
        {
            { "angry", Random.Range(0f, 1f) },
            { "disgust", Random.Range(0f, 1f) },
            { "fear", Random.Range(0f, 1f) },
            { "happy", Random.Range(0f, 1f) },
            { "sad", Random.Range(0f, 1f) },
            { "surprised", Random.Range(0f, 1f) },
        };

        // Normalize
        float total = 0f;
        foreach (var val in emotions.Values) total += val;
        foreach (var key in new List<string>(emotions.Keys)) emotions[key] /= total;

        return emotions;
    }
}