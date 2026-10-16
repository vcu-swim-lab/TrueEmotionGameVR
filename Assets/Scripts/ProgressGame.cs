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
    [Tooltip("In the calibrated conditions (2 and 4, see ConditionSetup), the game " +
             "offers calibration before the first round: A calibrates (about 90 s), B skips, X " +
             "uses this player's saved calibration. This is the id it is saved under on the " +
             "headset: letters, digits, '-', '_'.")]
    [SerializeField] private string playerId = "player1";

    private SessionLogger sessionLogger;

    // Monotonic across rounds, so restarting the game does not reuse trial ids.
    private int trialCounter;

    void Start()
    {
        text = GameObject.Find("Instruction").GetComponent<TextMeshProUGUI>();
        // debug = GameObject.Find("Debug").GetComponent<TextMeshProUGUI>();

        predictor = GetComponent<NewFaceAuModel>();

        // Which of the four conditions this build runs (backend and calibration) is chosen in
        // ConditionSetup.Configure; the rest of the game reads it back from the predictor. The
        // session logger starts later, in RunGame, once the condition is known to be ready.
        ConditionSetup.Configure(predictor);

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
            SessionLogger.AddFaceBackendInfo(modelInfo, predictor);

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
        // Is the chosen condition able to predict at all (Jev: key and connection)? If not, the
        // screen says why and the game does not start, rather than playing a round of failures.
        if (!await WaitUntilConditionReady())
        {
            return;
        }

        StartSessionLogger();

        // Calibration is on in conditions 2 and 4 (ConditionSetup.Configure sets it); in 1 and 3
        // the game never asks.
        if (predictor.UseCalibration)
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
                        $"What the headset saw this session ({ConditionSetup.Label(predictor)}):\n" +
                        predictor.GetResultsSummary() + "\n\n" +
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

            // One prediction from the library, by whichever backend ConditionSetup chose
            // (SparseVR on the headset, or Jev in the cloud). TryPredict, not PredictDetailed:
            // it also says WHY a prediction is missing (no face tracking, or the backend failed),
            // which matters when the backend is a web service.
            var outcome = await predictor.TryPredict();

            // Logged before scoring, and a failed prediction is recorded too (with its error
            // code), so the CSV holds one row per tick whether or not it succeeded.
            sessionLogger?.LogPrediction(trialId, emotion, i, outcome.PredictionOrNull, outcome.Error);

            if (outcome.Ok)
            {
                // Exactly what the library's own Predict() would return: the calibrated emotion
                // when the player calibrated, the backend's own answer otherwise. The confidence
                // is the backend's score for that emotion (for Jev, its own raw answer).
                var value = outcome.Prediction;
                var predicted = predictor.CalibratedEmotion(value) ?? value.Emotion;
                float confidence = value.ScoreOf(predicted);

                Debug.Log($"[game] round {trialId} {emotion} tick {i}: {predicted} " +
                          $"{confidence:F2} ({ConditionSetup.Id(predictor)}, calibrated: {predictor.IsCalibrated})");

                // A point when the prediction is the prompted emotion.
                if (predicted == emotion)
                {
                    score++;
                }
            }
            else
            {
                // No point and no fallback to another model: a failed tick is a failed tick.
                Debug.Log($"[game] round {trialId} {emotion} tick {i}: no prediction ({outcome.Error})");
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

    // Shows the chosen condition and checks it can predict (Jev: key and connection) before any
    // session file is created or any round starts. Returns false when the game must not start.
    private async Awaitable<bool> WaitUntilConditionReady()
    {
        while (true)
        {
            text.text = $"{ConditionSetup.Label(predictor)}\n\n" +
                        (ConditionSetup.UsesJev(predictor) ? "Checking the Jev connection..." : "Starting...");

            var (ok, message) = await ConditionSetup.CheckReadyAsync(predictor);
            if (ok)
            {
                text.text = $"{ConditionSetup.Label(predictor)}\n\n{message}";
                await Awaitable.WaitForSecondsAsync(2f);
                return true;
            }

            if (ConditionSetup.SetupError != null)
            {
                // A setup problem (no key, no face model) does not go away by retrying: it needs
                // a fix in the project and a rebuild.
                text.text = $"Cannot start.\n\n{message}";
                return false;
            }

            text.text = $"{ConditionSetup.Label(predictor)}\n\n{message}\n\nPress any button to try again";
            while (!OVRInput.GetDown(OVRInput.Button.Any))
            {
                await Awaitable.NextFrameAsync();
            }
        }
    }

    // Ask once, before the first round. Calibration is optional: skipping it leaves the game on
    // the backend's own predictions. A calibration that fails does not end the game; it goes
    // back to this question with the reason, so the player can try again or skip.
    private async Awaitable OfferCalibration()
    {
        // One calibration file per backend: SparseVR's and Jev's must not overwrite each other,
        // and the library refuses to load one backend's file into the other anyway.
        string calibrationId = ConditionSetup.CalibrationId(playerId, predictor);
        string notice = string.Empty;

        while (true)
        {
            bool hasSaved = predictor.HasSavedCalibration(calibrationId);

            text.text = notice +
                        "Calibrate the game to your face?\n\n" +
                        (ConditionSetup.UsesJev(predictor)
                            ? "A: calibrate (about 90 seconds, needs the internet)\n"
                            : "A: calibrate (about 90 seconds)\n") +
                        "B: skip" +
                        (hasSaved ? "\nX: use your saved calibration" : "");

            while (true)
            {
                if (OVRInput.GetDown(OVRInput.Button.One))
                {
                    if (await RunCalibration(calibrationId))
                    {
                        return;
                    }

                    notice = $"Calibration did not work: {CalibrationFailureReason()}\n\n";
                    break;   // ask again
                }

                if (OVRInput.GetDown(OVRInput.Button.Two))
                {
                    predictor.ClearCalibration();
                    return;
                }

                if (hasSaved && OVRInput.GetDown(OVRInput.Button.Three))
                {
                    bool loaded = predictor.LoadCalibration(calibrationId);

                    text.text = loaded
                        ? "Saved calibration loaded."
                        : $"Could not load it; playing without calibration.\n{predictor.CalibrationError}";
                    await Awaitable.WaitForSecondsAsync(2f);
                    return;
                }

                await Awaitable.NextFrameAsync();
            }
        }
    }

    // True when every emotion was collected and the calibration was built and saved.
    private async Awaitable<bool> RunCalibration(string calibrationId)
    {
        // Starts a fresh calibration. After a failure this is also what makes a retry possible:
        // an emotion that already has samples cannot be collected a second time.
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

            // Collects the emotion's samples from the ACTIVE backend (one request each with Jev).
            // False when the library gave up after too many failed attempts: face tracking
            // was lost, or the backend kept failing. No point going on to the next emotion.
            if (!await predictor.CalibrateEmotion(emotion))
            {
                Debug.LogWarning($"ProgressGame: could not calibrate {emotion}: {CalibrationFailureReason()}");
                return false;
            }

            text.text = "Relax your face.";
            await Awaitable.WaitForSecondsAsync(2f);
        }

        if (!predictor.FinishCalibration())
        {
            Debug.LogWarning($"ProgressGame: calibration failed: {predictor.CalibrationError}");
            return false;
        }

        if (!predictor.SaveCalibration(calibrationId))
        {
            // Usable for this run, just not kept for next time.
            Debug.LogWarning($"ProgressGame: calibration built but not saved: {predictor.CalibrationError}");
        }

        text.text = "Calibration done.";
        await Awaitable.WaitForSecondsAsync(2f);
        return true;
    }

    // Why the last calibration step failed, in words for the screen: the library's calibration
    // error if it has one, otherwise the last prediction error (e.g. Jev Unreachable), otherwise
    // lost face tracking.
    private string CalibrationFailureReason()
    {
        if (!string.IsNullOrEmpty(predictor.CalibrationError))
        {
            return predictor.CalibrationError;
        }

        return predictor.LastError.HasValue ? predictor.LastError.Value.ToString() : "face tracking was lost";
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