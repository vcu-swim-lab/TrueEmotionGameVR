using System.Collections.Generic;

using UnityEngine;
using Random = UnityEngine.Random;

using TMPro;
using Unity.InferenceEngine;
using System.Linq;
using UnityEngine.UI;
using Oculus.Interaction;


[RequireComponent(typeof(EmotionPredictor))]
[RequireComponent(typeof(OVRFaceExpressions))]
public class ProgressGame : MonoBehaviour
{
    private TextMeshProUGUI text;
    // private TextMeshProUGUI debug;
    private Button restart;

    private EmotionPredictor predictor;

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


    void OnEnable()
    {
        // var pokeable = restart.GetComponent<PokeInteractable>();
        // pokeable.WhenPointerEventRaised += OnPointerEvent;
    }

    void OnDisable()
    {
        // var pokeable = restart.GetComponent<PokeInteractable>();
        // pokeable.WhenPointerEventRaised -= OnPointerEvent;
    }


    void Start()
    {
        text = GameObject.Find("Instruction").GetComponent<TextMeshProUGUI>();
        // debug = GameObject.Find("Debug").GetComponent<TextMeshProUGUI>();
        restart = GameObject.Find("Restart").GetComponent<Button>();


        predictor = GetComponent<EmotionPredictor>();

        restart.onClick.AddListener(() => RunGame());

        RunGame();
    }


    private async void RunGame()
    {
        // TODO: change to 
        while (true)
        {
            text.fontSize = 36;
            restart.gameObject.SetActive(false);

            // Shuffle emotion list
            for (int i = emotionList.Length - 1; i > 0; --i)
            {
                int j = Random.Range(0, i + 1);
                (emotionList[i], emotionList[j]) = (emotionList[j], emotionList[i]);
            }

            text.text = "You got 10s to act each emotion shown to you. Good luck.";
            await Awaitable.WaitForSecondsAsync(2f);

            Dictionary<Emotion, int> score = new();

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
                text.text = $"{emoji}\n{emotion}";

                // Run prediction loop for this emotion
                int this_score = await RunPredictionCoroutine(emotion);
                score[emotion] = this_score;

                text.text = $"Score: {this_score}/10";
                await Awaitable.WaitForSecondsAsync(1f);
            }

            restart.gameObject.SetActive(true);

            text.text = "Thanks for playing!" + "\n\n" +
                        "Scores:\n" +
                        string.Join("\n", score.Select(kv => $"{kv.Key}: {kv.Value}/10")) + "\n\n" +
                        "Smile to restart";
            text.fontSize = 24;
            Debug.Log("All emotion predictions collected.");

            while (true)
            {
                var pred = await predictor.Predict();

                if (pred.Item1 == Emotion.Happiness) break;
                else await Awaitable.WaitForSecondsAsync(.5f);
            }
        }
    }

    private async Awaitable<int> RunPredictionCoroutine(Emotion emotion)
    {
        predictor.Flush(); // Ensure we start with fresh data
        predictor.Polling = true;

        int score = 0;

        const int intervalMs = 1000;

        int times = 2;
        for (int i = 0; i < times; ++i)
        {
            // Calculate next target time
            float nextTick = Time.time + (intervalMs / 1000f);

            // Await prediction
            var allConfidences = await PredictEmotionAsync();

            if (allConfidences.TryGetValue(emotion, out float confidence))
            {
                // debug.text = $"[{emotion}] Prediction {i + 1}: {confidence:F2}";
                score++;
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

    private void OnPointerEvent(PointerEvent pointerEvent)
    {
        switch (pointerEvent.Type)
        {
            case PointerEventType.Unselect:
                RunGame();
                break;
        }
    }
}