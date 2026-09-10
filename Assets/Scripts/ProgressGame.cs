using System.Collections.Generic;
using System.Linq;

using TMPro;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Android;
using UnityEngine.InputSystem.XR.Haptics;
using Random = UnityEngine.Random;


[RequireComponent(typeof(NewFaceAuModel))]
[RequireComponent(typeof(DeviceManager))]
[RequireComponent(typeof(OVRFaceExpressions))]
public class ProgressGame : MonoBehaviour
{
    private TextMeshProUGUI text;
    // private TextMeshProUGUI debug;

    private NewFaceAuModel predictor;

    [SerializeField]
    private ModelAsset soundModelTemp;


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
    public ProgressBar progressBar;

    void Start()
    {
        text = GameObject.Find("Instruction").GetComponent<TextMeshProUGUI>();
        // debug = GameObject.Find("Debug").GetComponent<TextMeshProUGUI>();

        predictor = GetComponent<NewFaceAuModel>();

        if (progressBar == null)
        {
            progressBar = FindObjectOfType<ProgressBar>();
            if (progressBar != null)
            {
                Debug.Log("Found progressBar: " + progressBar.name);
            }
            else
            {
                Debug.LogError("Could not find ProgressBar component in scene!");
            }
        }

        RunGame();
        RunSoundTest();
    }

    void Update()
    {
        OVRInput.Update();
    }

    void FixedUpdate()
    {
        OVRInput.FixedUpdate();
    }


    private async void RunGame()
    {
        // TODO: change to 
        while (true)
        {
            // Shuffle emotion list
            for (int i = emotionList.Length - 1; i > 0; --i)
            {
                int j = Random.Range(0, i + 1);
                (emotionList[i], emotionList[j]) = (emotionList[j], emotionList[i]);
            }

            text.text = "You got 10s to act each emotion shown to you. Good luck.";
            //hides the progress bar
            if (progressBar != null)
                progressBar.SetProgressBarVisible(false);

            await Awaitable.WaitForSecondsAsync(2f);

            Dictionary<Emotion, int> score = new();

            //hides the progress bar
            if (progressBar != null)
                progressBar.SetProgressBarVisible(false);

            //number of emotions
            for (int n = 0; n < (emotionList.Length); ++n)
            {
                var emotion = emotionList[n];
                var max = emotionList.Length;

                // Countdown before showing emotion
                for (int j = 3; j >= 0; --j)
                {
                    text.text = $"{j}";
                    await Awaitable.WaitForSecondsAsync(1f);
                }

                // Show emoji for current emotion
                string emoji = emotionToEmoji[emotion];
                text.text = $"{n + 1} / {max}\n\n\n{emoji}\n{emotion}";

                //progress bar
                if (progressBar == null)
                {
                    Debug.LogError($"progressbar is null");
                }
                else
                {
                    progressBar.SetMaximum(max);
                    progressBar.SetCurrent(n + 1);
                    progressBar.SetProgressBarVisible(true);
                }

                // Run prediction loop for this emotion
                int this_score = await RunPredictionCoroutine(emotion);
                score[emotion] = this_score;

                //hides progress bar
                if (progressBar != null)
                    progressBar.SetProgressBarVisible(false);

                text.text = $"Score: {this_score}/10";
                await Awaitable.WaitForSecondsAsync(1f);
            }

            text.text = "Thanks for playing! Scores:\n" +
                        string.Join("\n", score.Select(kv => $"{kv.Key}: {kv.Value}/10")) + "\n\n" +
                        "Press any button to restart";

            Debug.Log("All emotion predictions collected.");

            while (!OVRInput.GetDown(OVRInput.Button.Any))
            {
                await Awaitable.NextFrameAsync();
            }
            // Loop back to the top of `while (true)` for the next round instead of
            // recursively calling RunGame() again, which used to stack a second
            // concurrent game loop on top of this one every time the player restarted.
        }
    }

    private async Awaitable<int> RunPredictionCoroutine(Emotion emotion)
    {
        int score = 0;

        const int intervalMs = 1000;

#if UNITY_EDITOR
        int times = 2;
#else
        int times = 10;
#endif
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


    private async void RunSoundTest()
    {
        // TODO: move this to the package side
        if (!Permission.HasUserAuthorizedPermission(Permission.Microphone))
        {
            Permission.RequestUserPermission(Permission.Microphone);
            print("microphone requested");
        }
        else
        {
            print("already have microphone access");
        }

        // TODO: you might want to use loop: true instead
        // var clip = Microphone.Start(null, false, 30, 16100);

        // await Awaitable.WaitForSecondsAsync(31);

        // var data = new float[30 * 16100];
        // if (!clip.GetData(data, 0)) print("Error reading clip!");

        // var model = ModelLoader.Load(soundModelTemp);
        // using var worker = new Worker(model, Unity.InferenceEngine.DeviceType.CPU);
    }
}