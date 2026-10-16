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
    [Tooltip("When \"Use Calibration\" is ticked on the New Face Au Model component, the game " +
             "offers calibration before the first round: A calibrates (about 90 s), B skips, X " +
             "uses this player's saved calibration. This is the id it is saved under on the " +
             "headset: letters, digits, '-', '_'.")]
    [SerializeField] private string playerId = "player1";

    private SessionLogger sessionLogger;

    // Jev's answer next to the local model's; see Assets/Scripts/Jev/README.md.
    private JevScorer scorer;

    // What Jev decided this session, for the end screen: how often each emotion won and the
    // sum of Jev's own answers for it. The library's summary below it is the local model's.
    private readonly int[] jevCounts = new int[System.Enum.GetValues(typeof(Emotion)).Length];
    private readonly double[] jevAnswerSums = new double[System.Enum.GetValues(typeof(Emotion)).Length];

    // Monotonic across rounds, so restarting the game does not reuse trial ids.
    private int trialCounter;

    void Start()
    {
        text = GameObject.Find("Instruction").GetComponent<TextMeshProUGUI>();
        // debug = GameObject.Find("Debug").GetComponent<TextMeshProUGUI>();

        predictor = GetComponent<NewFaceAuModel>();
        scorer = JevScorer.GetOrAdd(gameObject);

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
            SessionLogger.AddFaceBackendInfo(modelInfo, scorer);

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
        // The library's "Use Calibration" checkbox; unticked, the game never asks.
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
                        JevSummaryBlock() +
                        "What the headset saw this session" + (JevAnswered() > 0 ? " (local model)" : "") + ":\n" +
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

                // Jev's answer replaces it when Jev is on and answered. When it did not (no key,
                // no network), the local prediction above stands and the round carries on.
                var jevScores = await scorer.Score(value);
                if (scorer.Enabled)
                {
                    sessionLogger?.LogJev(trialId, emotion, i, jevScores, scorer.LastRaw, scorer.LastError, scorer.ProfileSent);
                }

                if (jevScores != null)
                {
                    // Jev's own answer for the winner, not its share of the seven answers: the
                    // shares always sum to 1, so a lone weak answer would read as 100%.
                    predicted = JevScorer.Decide(jevScores);
                    confidence = scorer.LastRaw != null ? scorer.LastRaw[(int)predicted] : jevScores[(int)predicted];

                    jevCounts[(int)predicted]++;
                    jevAnswerSums[(int)predicted] += confidence;
                }

                Debug.Log($"[game] round {trialId} {emotion} tick {i}: {predicted} " +
                          $"{confidence:F2} (calibrated: {predictor.IsCalibrated}, " +
                          $"jev: {(jevScores != null ? "yes" : scorer.Enabled ? "failed: " + scorer.LastError : "off")})");

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

    private int JevAnswered() => jevCounts.Sum();

    // The end-screen block for Jev's decisions, ending in a blank line; empty when Jev gave none.
    private string JevSummaryBlock()
    {
        int total = JevAnswered();
        if (total == 0)
        {
            return string.Empty;
        }

        var lines = new List<string> { $"What Jev decided this session ({total} answers):" };
        foreach (int e in Enumerable.Range(0, jevCounts.Length).OrderByDescending(e => jevCounts[e]))
        {
            if (jevCounts[e] == 0)
            {
                continue;
            }

            lines.Add(string.Format(System.Globalization.CultureInfo.InvariantCulture,
                                    "{0}: {1:0}% (average answer {2:0.00})",
                                    (Emotion)e, 100.0 * jevCounts[e] / total, jevAnswerSums[e] / jevCounts[e]));
        }

        return string.Join("\n", lines) + "\n\n";
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
                scorer.ClearProfile();
                return;
            }

            if (hasSaved && OVRInput.GetDown(OVRInput.Button.Three))
            {
                bool loaded = predictor.LoadCalibration(playerId);
                if (loaded)
                {
                    scorer.LoadProfile(playerId);
                }

                text.text = loaded
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
        scorer.ClearProfile();

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
            if (predictor.SaveCalibration(playerId))
            {
                scorer.LoadProfile(playerId);
            }

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