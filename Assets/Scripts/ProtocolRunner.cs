using System;
using System.Collections.Generic;
using System.Linq;

using TMPro;
using UnityEngine;

using Random = System.Random;

// Finite single-pass session runner. It prompts each of the six performed emotions once,
// shows no score or prediction feedback, and keeps the full hold and relax phases. Neutral
// is not a prompted trial; the relax phase is only a transition and is not scored.
[RequireComponent(typeof(NewFaceAuModel))]
[RequireComponent(typeof(DeviceManager))]
[RequireComponent(typeof(OVRFaceExpressions))]
public class ProtocolRunner : MonoBehaviour
{
    // Bump when anything that changes the meaning of a recording changes: the class set,
    // the block structure, the timings, or the feedback policy. It is written into the
    // session metadata so a pooled analysis can refuse to mix incompatible recordings.
    public const string ProtocolVersion = "single-pass-v2";

    public const int Blocks = 1;
    public const int TicksPerTrial = 10;

    [Header("Scene")]
    [Tooltip("The instruction text. Left empty, the object named 'Instruction' is used.")]
    [SerializeField] private TextMeshProUGUI instructionText;

    [Header("Timings (seconds)")]
    [SerializeField] private float promptSeconds = 2f;
    [SerializeField] private float countdownSeconds = 3f;
    [SerializeField] private float completeSeconds = 1f;
    [SerializeField] private float relaxSeconds = 4f;

    [Header("Session")]
    [Tooltip("Leave at 0 to draw a seed from the clock. The value used is recorded either way.")]
    [SerializeField] private int randomizationSeed = 0;

    [Header("Rehearsal")]
    [Tooltip("Shortens every trial so the flow can be walked through at a desk. " +
             "Recordings made with this on are marked as rehearsals and must not be analysed.")]
    [SerializeField] private bool fastRehearsal = false;

    // The six performed emotions. Neutral remains a model output class, but is not prompted.
    private static readonly Emotion[] PromptedEmotions =
    {
        Emotion.Anger, Emotion.Disgust, Emotion.Fear, Emotion.Happiness,
        Emotion.Sadness, Emotion.Surprise,
    };

    private static readonly Dictionary<Emotion, string> EmotionToEmoji = new()
    {
        { Emotion.Anger, "🤬" },
        { Emotion.Disgust, "🤢" },
        { Emotion.Fear, "😱" },
        { Emotion.Happiness, "😀" },
        { Emotion.Sadness, "☹️" },
        { Emotion.Surprise, "😲" },
    };

    // One prompt, resolved before the session starts so the whole schedule is fixed and
    // recordable in advance rather than decided trial by trial.
    public readonly struct PlannedTrial
    {
        public readonly int TrialId;
        public readonly int BlockIndex;   // 0-based
        public readonly int RepeatIndex;  // 1-based: which presentation of this emotion
        public readonly Emotion Emotion;

        public PlannedTrial(int trialId, int blockIndex, int repeatIndex, Emotion emotion)
        {
            TrialId = trialId;
            BlockIndex = blockIndex;
            RepeatIndex = repeatIndex;
            Emotion = emotion;
        }
    }

    private NewFaceAuModel predictor;
    private SessionLogger sessionLogger;
    private int seedUsed;

    public IReadOnlyList<PlannedTrial> Plan { get; private set; }

    void Start()
    {
        // Explicit null check rather than `??=`: the null-coalescing operators bypass
        // UnityEngine.Object's overloaded `==`, so an unassigned or destroyed reference
        // would not be recognised as missing.
        if (instructionText == null)
        {
            instructionText = GameObject.Find("Instruction")?.GetComponent<TextMeshProUGUI>();
        }

        if (instructionText == null)
        {
            Debug.LogError($"{nameof(ProtocolRunner)}: no instruction text assigned and no " +
                           "GameObject named 'Instruction' in the scene. Cannot run.");
            enabled = false;
            return;
        }

        predictor = GetComponent<NewFaceAuModel>();

        seedUsed = randomizationSeed != 0
            ? randomizationSeed
            : Environment.TickCount & 0x7FFFFFFF;

        Plan = BuildPlan(seedUsed);

        RunSession();
    }

    // One block containing a shuffled pass over all six emotions. Keeping the plan builder
    // block-based makes the recorded block and repeat indices explicit even in a single pass.
    public static List<PlannedTrial> BuildPlan(int seed)
    {
        var random = new Random(seed);
        var plan = new List<PlannedTrial>(Blocks * PromptedEmotions.Length);
        var repeats = PromptedEmotions.ToDictionary(e => e, _ => 0);

        Emotion? previous = null;

        for (int block = 0; block < Blocks; ++block)
        {
            var order = PromptedEmotions.ToArray();

            // Fisher-Yates. The redraw guard is harmless for the first (and currently only)
            // block and prevents adjacent duplicates if more blocks are configured later.
            do
            {
                for (int i = order.Length - 1; i > 0; --i)
                {
                    int j = random.Next(i + 1);
                    (order[i], order[j]) = (order[j], order[i]);
                }
            }
            while (previous.HasValue && order[0] == previous.Value);

            foreach (var emotion in order)
            {
                repeats[emotion]++;
                plan.Add(new PlannedTrial(plan.Count, block, repeats[emotion], emotion));
            }

            previous = order[^1];
        }

        return plan;
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
            };

            var sessionInfo = new Dictionary<string, string>
            {
                { "protocol_version", ProtocolVersion },
                { "randomization_seed", seedUsed.ToString() },
                { "trial_count", Plan.Count.ToString() },
                { "blocks", Blocks.ToString() },
                { "ticks_per_trial", (fastRehearsal ? 3 : TicksPerTrial).ToString() },
                { "fast_rehearsal", fastRehearsal ? "true" : "false" },
                { "self_report", "not collected (deliberate; no controller in this study)" },
            };

            sessionLogger = new SessionLogger(
                NewFaceAuModel.WindowFrames,
                NewFaceAuModel.FeaturesPerFrame,
                modelInfo,
                sessionInfo);
        }
        catch (Exception e)
        {
            // Unlike the game, a recording failure here is fatal: an unrecorded protocol
            // session wastes a participant visit, and running on would hide that.
            sessionLogger = null;
            Debug.LogError($"{nameof(ProtocolRunner)}: could not start logging " +
                           $"({e.GetType().Name}: {e.Message}). Aborting the session.");
            enabled = false;
        }
    }

    private void CloseSessionLogger()
    {
        sessionLogger?.Dispose();
        sessionLogger = null;
    }

    // The Quest pauses rather than quits when the headset comes off mid-session.
    void OnApplicationPause(bool paused)
    {
        if (paused)
        {
            CloseSessionLogger();
        }
    }

    void OnApplicationQuit() => CloseSessionLogger();

    void OnDisable() => CloseSessionLogger();

    private float Scaled(float seconds) => fastRehearsal ? seconds * 0.25f : seconds;

    // Every screen goes through here, so a rehearsal cannot be mistaken for a real session
    // by anyone looking at the headset. The inspector checkbox is easy to leave on by
    // accident and a start-up log line scrolls away; a banner on every frame of the session
    // does not. The recording is marked in its metadata too, and the analysis refuses it.
    private void Show(string message)
    {
        instructionText.text = fastRehearsal
            ? "-- REHEARSAL - NOT VALID DATA --\n\n" + message
            : message;
    }

    private async void RunSession()
    {
        if (fastRehearsal)
        {
            Debug.LogWarning($"{nameof(ProtocolRunner)}: FAST REHEARSAL is on. This recording " +
                             "is marked as a rehearsal in its metadata and must not be analysed.");
        }

        Debug.Log($"{nameof(ProtocolRunner)}: {Plan.Count} trials, " +
                  $"seed {seedUsed}, protocol {ProtocolVersion}");

        StartSessionLogger();
        if (sessionLogger == null)
        {
            return;
        }

        // The recording is identified only by its start time. Which participant that
        // corresponds to is kept by the operator outside this app, so the folder name and
        // the timestamp in the metadata are the only link back to a person -- take care
        // that the operator's note is written at the same moment the session starts.
        Debug.Log($"{nameof(ProtocolRunner)}: recording to {sessionLogger.SessionDirectory}");

        Show("We will show you one expression at a time.\n\n" +
             "Hold each expression until you are told to relax.\n\n" +
             "You will not be scored.");
        await Awaitable.WaitForSecondsAsync(Scaled(6f));

        foreach (var trial in Plan)
        {
            await RunTrial(trial);
        }

        Show("Session complete.\n\nThank you.");
        Debug.Log($"{nameof(ProtocolRunner)}: session complete, {Plan.Count} trials recorded.");

        CloseSessionLogger();
    }

    private async Awaitable RunTrial(PlannedTrial trial)
    {
        // 1. Prompt. Named and pictured for each performed expression.
        Show($"{EmotionToEmoji[trial.Emotion]}\n{trial.Emotion}");
        await Awaitable.WaitForSecondsAsync(Scaled(promptSeconds));

        // 2. Countdown, so the expression starts at a known moment rather than drifting
        //    into the recording window.
        for (int i = Mathf.RoundToInt(countdownSeconds); i > 0; --i)
        {
            Show($"{i}");
            await Awaitable.WaitForSecondsAsync(Scaled(1f));
        }

        // 3. Hold, while the model is polled once a second.
        Show($"{EmotionToEmoji[trial.Emotion]}\n\nHold it");
        await RecordTrial(trial);

        // 4. Non-evaluative acknowledgement -- identical whatever the model predicted.
        Show("Recording complete.");
        await Awaitable.WaitForSecondsAsync(Scaled(completeSeconds));

        // 5. No self-report. This is a deliberate decision, not an omission, and it has a
        //    known cost: when a class scores badly there is no way to tell whether the
        //    participant failed to produce the expression or the model failed to detect
        //    it. Fear scored 0% for both subjects recorded so far, and that result will
        //    stay uninterpretable in exactly this way. Collecting it would have meant an
        //    operator writing down 336 spoken answers by hand, since the study uses no
        //    controllers and there is no in-headset input of any kind.

        // 6. Relax. Fixed length, and never skipped -- it is what keeps the next trial's
        //    first frames from carrying the tail of this expression.
        Show("Relax your face.");
        await Awaitable.WaitForSecondsAsync(Scaled(relaxSeconds));
    }

    // Polls the model once a second for the hold period and logs every tick. Nothing is
    // scored and nothing is shown: the return value the game used to display is simply not
    // computed here.
    private async Awaitable RecordTrial(PlannedTrial trial)
    {
        int ticks = fastRehearsal ? 3 : TicksPerTrial;

        for (int i = 0; i < ticks; ++i)
        {
            float nextTick = Time.time + 1f;

            var prediction = await predictor.PredictDetailed();

            // Logged whether or not tracking held, so a dropout is visible in the file
            // rather than showing up as a missing row.
            sessionLogger?.LogPrediction(trial.TrialId, trial.Emotion, i, prediction);

            float remaining = nextTick - Time.time;
            if (remaining > 0)
            {
                await Awaitable.WaitForSecondsAsync(remaining);
            }
        }
    }
}
