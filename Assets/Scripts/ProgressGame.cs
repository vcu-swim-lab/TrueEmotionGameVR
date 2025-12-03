using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

using TMPro;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Android;
using Random = UnityEngine.Random;


[RequireComponent(typeof(FaceAuModelInGame))]
[RequireComponent(typeof(SoundModel))]
public class ProgressGame : MonoBehaviour
{
    private TextMeshProUGUI text;
    // private TextMeshProUGUI debug;

    private FaceAuModelInGame predictor;
    private SoundModel soundModel;

    [SerializeField]
    private AudioClip audio;

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

    // Map emotion name to scenario to display to user
    private static readonly Dictionary<Emotion, string[]> emotionToScenario = new()
    {
        { Emotion.Anger, new string[]
            {
                "You just found out someone pump into your car in the parking lot and did not leave a note.",
                "You find your favorite book torn apart after your friend borrowed it.",
                "You talked to a person who was being rude to you for no reason.",
            }
        },

        {Emotion.Disgust, new string[]
            {
                "You find a moldy sandwich in your bag.",
                "You see a bug crawling on your food.",
                "You see a plate of rotten food filled with maggots.",
            }
        },

        {Emotion.Fear, new string[]
            {
                "You are walking alone at night and hear footsteps behind you.",
                "You are about to give a speech in front of a large audience.",
                "You are lost in a dark forest.",
            }
        },

        {Emotion.Happiness, new string[]
            {
                "You just received a compliment from a stranger.",
                "You achieved a personal goal you set for yourself.",
                "You are spending time with your best friends.",
            }
        },

        {Emotion.Sadness, new string[]
            {
                "You just watched a heartbreaking movie.",
                "You are reminiscing about a lost loved one.",
                "You received some disappointing news.",
            }
        },

        {Emotion.Surprise, new string[]
            {
                "You just found out you your dinner bill was paid by a generous stranger.",
                "You received an unexpected gift from a friend.",
                "You walked into a surprise birthday party thrown for you.",
            }
        },
    };

    private readonly Emotion[] emotionList = emotionToEmoji.Keys.ToArray();
    private readonly Emotion[] scenarioEmotionList = emotionToScenario.Keys.ToArray();


    void Start()
    {
        text = GameObject.Find("Instruction").GetComponent<TextMeshProUGUI>();
        // debug = GameObject.Find("Debug").GetComponent<TextMeshProUGUI>();

        predictor = GetComponent<FaceAuModelInGame>();
        soundModel = GetComponent<SoundModel>();

        audio = Microphone.Start(null, true, 30, 16000);

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

            //Shuffle scenario emotion list
            for (int i = scenarioEmotionList.Length - 1; i > 0; --i)
            {
                int j = Random.Range(0, i + 1);
                (scenarioEmotionList[i], scenarioEmotionList[j]) = (scenarioEmotionList[j], scenarioEmotionList[i]);
                for (int k = emotionToScenario[scenarioEmotionList[i]].Length - 1; k > 0; k--)
                {
                    int l = Random.Range(0, k + 1);
                    (emotionToScenario[scenarioEmotionList[i]][k], emotionToScenario[scenarioEmotionList[i]][l]) = (emotionToScenario[scenarioEmotionList[i]][l], emotionToScenario[scenarioEmotionList[i]][k]);
                }
            }

            // Scenario round
            text.text = "This is the scenario round. You have 10s to act each scenario shown to you. Good luck.";
            await Awaitable.WaitForSecondsAsync(2f);

            foreach (var emotion in scenarioEmotionList)
            {
                for (int j = 3; j >= 0; --j)
                {
                    text.text = $"{j}";
                    await Awaitable.WaitForSecondsAsync(1f);
                }

                // Show scenario for current emotion
                string scenario = emotionToScenario[emotion][0].ToString();
                text.text = $"{scenario}\n({emotion})";

                // Run prediction loop for this emotion
                int this_score = await RunPredictionCoroutine(emotion);
                score[emotion] += this_score; // accumulate score

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

            RunGame();
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

        // var emo = await predictor.Predict();
        var emo = await PredictFromAudioClip(audio);
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

    // Method to predict emotion from an audio file
    private async Awaitable<(Emotion, float)> PredictFromAudioClip(AudioClip audioClip)
    {
        if (audioClip == null)
        {
            Debug.LogError("AudioClip is null");
            return (Emotion.Happiness, 0f);
        }

        var data = new float[audioClip.samples];
        audioClip.GetData(data, 0);

        //Pass this to predict
        var audioInput = new Tensor<float>(new TensorShape(1, audioClip.samples), data);

        var result = await soundModel.Predict(audioInput);

        audioInput.Dispose();

        return result;
    }
}