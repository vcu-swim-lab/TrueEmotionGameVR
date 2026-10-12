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

    [Header("Calibration (optional)")]
    [Tooltip("Before the first round, offer movement calibration: A calibrates (about 90 s), " +
             "B skips, X uses the player's saved calibration if there is one.")]
    [SerializeField] private bool offerCalibration = true;

    [Tooltip("The id the calibration is saved under on this headset: letters, digits, '-', '_'.")]
    [SerializeField] private string playerId = "player1";

    private SessionLogger sessionLogger;

    // Monotonic across rounds, so restarting the game does not reuse trial ids.
    private int trialCounter;

    void Start()
    {
        text = GameObject.Find("Instruction").GetComponent<TextMeshProUGUI>();
        // debug = GameObject.Find("Debug").GetComponent<TextMeshProUGUI>();

        predictor = GetComponent<NewFaceAuModel>();

        StartSessionLogger();

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

    private void StartSessionLogger()
    {
        try
        {
            var modelInfo = new Dictionary<string, string>
            {
                { "face_model_asset", predictor != null ? predictor.FaceModelName ?? "(unassigned)" : "(no predictor)" },
                { "face_model_append_softmax", "false" },
                { "face_model_softmax_reason", "graph already ends in Softmax; output is already a probability distribution" },
                { "sound_model_asset", soundModelTemp != null ? soundModelTemp.name : "(unassigned)" },
                { "sound_model_append_softmax", "true" },
                { "sound_model_softmax_reason", "graph ends in Gemm and emits raw logits" },
                { "logmel_model_append_softmax", "false" },
                { "logmel_model_softmax_reason", "preprocessing transform, loaded with ModelLoader.Load directly" },
            };

            sessionLogger = new SessionLogger(
                NewFaceAuModel.WindowFrames,
                NewFaceAuModel.FeaturesPerFrame,
                modelInfo);
        }
        catch (System.Exception e)
        {
            // A recording failure must not stop the experiment from running.
            sessionLogger = null;
            Debug.LogError($"Could not start session logging ({e.GetType().Name}: {e.Message}). Continuing without it.");
        }
    }

    private void CloseSessionLogger()
    {
        sessionLogger?.Dispose();
        sessionLogger = null;
    }

    // The Quest pauses the app when the headset is removed rather than quitting it, so
    // closing only on quit would lose the tail of a recording.
    void OnApplicationPause(bool paused)
    {
        if (paused)
        {
            CloseSessionLogger();
        }
    }

    void OnApplicationQuit()
    {
        CloseSessionLogger();
    }

    void OnDisable()
    {
        CloseSessionLogger();
    }


    private async void RunGame()
    {
        if (offerCalibration)
        {
            await OfferCalibration();
        }

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
                int this_score = await RunPredictionCoroutine(emotion, trialCounter++);
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

    private async Awaitable<int> RunPredictionCoroutine(Emotion emotion, int trialId)
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
            var prediction = await PredictEmotionAsync();

            // Logged before scoring, and a failed prediction is recorded too, so the CSV
            // holds one row per tick whether or not tracking was available.
            sessionLogger?.LogPrediction(trialId, emotion, i, prediction);

            if (prediction != null)
            {
                // The calibrated emotion when the player calibrated, the model's own otherwise.
                // The confidence is the model's probability for that emotion.
                var value = prediction.Value;
                var predicted = predictor.CalibratedEmotion(value) ?? value.Emotion;
                float confidence = value.Scores[(int)predicted];

                Debug.Log($"[game] round {trialId} {emotion} tick {i}: {predicted} " +
                          $"{confidence:F2} (calibrated: {predictor.IsCalibrated})");

                // A point when the prediction is the prompted emotion.
                if (predicted == emotion)
                {
                    score++;
                }
            }
            else
            {
                Debug.Log($"[game] round {trialId} {emotion} tick {i}: no face tracking");
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

    // Ask once, before the first round. Calibration is optional: skipping it, or a failed
    // calibration, leaves the game on the model's own predictions.
    private async Awaitable OfferCalibration()
    {
        bool hasSaved = predictor.HasSavedCalibration(playerId);

        text.text = "Calibrate the game to your face?\n\n" +
                    "A: calibrate (about 90 seconds)\n" +
                    "B: skip" +
                    (hasSaved ? "\nX: use your saved calibration" : "");

        while (true)
        {
            if (OVRInput.GetDown(OVRInput.Button.One))
            {
                await RunCalibration();
                return;
            }

            if (OVRInput.GetDown(OVRInput.Button.Two))
            {
                predictor.ClearCalibration();
                return;
            }

            if (hasSaved && OVRInput.GetDown(OVRInput.Button.Three))
            {
                text.text = predictor.LoadCalibration(playerId)
                    ? "Saved calibration loaded."
                    : $"Could not load it; playing without calibration.\n{predictor.CalibrationError}";
                await Awaitable.WaitForSecondsAsync(2f);
                return;
            }

            await Awaitable.NextFrameAsync();
        }
    }

    private async Awaitable RunCalibration()
    {
        predictor.StartCalibration();

        foreach (Emotion emotion in System.Enum.GetValues(typeof(Emotion)))
        {
            for (int j = 3; j >= 1; --j)
            {
                text.text = $"Next: {emotion}\n\n{j}";
                await Awaitable.WaitForSecondsAsync(1f);
            }

            // The game font has no Neutral emoji, so Neutral is shown as text only.
            string face = emotion == Emotion.Neutral ? "" : emotionToEmoji[emotion];
            text.text = emotion == Emotion.Neutral
                ? $"{face}\nRest your face and hold it"
                : $"{face}\nMake a {emotion} face and hold it";

            if (!await predictor.CalibrateEmotion(emotion))
            {
                Debug.LogWarning($"ProgressGame: face tracking was lost while calibrating {emotion}.");
            }

            text.text = "Relax your face.";
            await Awaitable.WaitForSecondsAsync(2f);
        }

        if (predictor.FinishCalibration())
        {
            predictor.SaveCalibration(playerId);
            text.text = "Calibration done.";
        }
        else
        {
            Debug.LogWarning($"ProgressGame: calibration failed: {predictor.CalibrationError}");
            text.text = "Calibration did not work; playing without it.";
        }

        await Awaitable.WaitForSecondsAsync(2f);
    }

    // Full prediction: argmax class, all seven scores, and the exact five-frame AU window
    // that produced them. Null when no window of valid tracking frames was available.
    private async Awaitable<FacePrediction?> PredictEmotionAsync()
    {
        return await predictor.PredictDetailed();
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