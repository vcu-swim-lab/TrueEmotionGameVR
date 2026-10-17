using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using TMPro;
using UnityEngine;

using VcuSwimLab.Calibration;

using Random = System.Random;

// Finite session runner for the session-local calibration study.
//
// Two blocks, seven states each, no feedback of any kind:
//
//   K -- calibration, one trial per state.   7 trials
//   D -- evaluation, two trials per state.  14 trials
//
// The point of the split is that a calibration profile fitted on K is scored on *separate
// trials* of the same states in D. The earlier within-trial measurement -- fit on the first
// ticks of a held expression, test on its last ticks -- measures consistency within one
// performance, and the gain it reported decays to nothing as those ticks are moved apart.
//
// D has two trials per state rather than one because a seven-trial evaluation cannot support
// a decision: its trial-level metric moves in steps of 1/7, and a sign test over seven paired
// trials reaches significance only on a perfect sweep.
//
// Neutral is a prompted state here, with its own trial, its own countdown and its own ten
// ticks. It is deliberately NOT harvested from the inter-trial relax phase: a transition
// sample carries the tail of the previous expression, and building a Neutral centroid that
// way was measured making things worse rather than better.
[RequireComponent(typeof(NewFaceAuModel))]
[RequireComponent(typeof(DeviceManager))]
[RequireComponent(typeof(OVRFaceExpressions))]
public class ProtocolRunner : MonoBehaviour
{
    // Bump when anything that changes the meaning of a recording changes: the class set,
    // the block structure, the timings, or the feedback policy. It is written into the
    // session metadata so a pooled analysis can refuse to mix incompatible recordings.
    // `single-pass-v2` is a different protocol -- six states, no Neutral -- and keeps its
    // meaning. Sessions recorded under different protocol versions are never pooled, so
    // redefining a version in place would break that guarantee for anything already
    // stamped with it.
    //
    // `session-local-v2` (2026-10-13): the calibration block runs only when calibration is on,
    // each part is named on screen, and the participant sees the predicted emotion every
    // second during the test block. Not pooled with v1, which showed nothing.
    //
    // `session-local-v3` (2026-10-15): the participant also sees a running score during each
    // test trial and a per-emotion results table at the end. Not pooled with v2, which showed
    // the prediction but no score.
    public const string ProtocolVersion = "session-local-v3";

    // The demo (the Demo checkbox) is a separate flow for showing the system to viewers, not a
    // study session: one test pass, a remaining-time bar during the hold, and after each test
    // trial its predicted emotion with the next emotion and a circle countdown. Its recordings
    // carry this version instead, so they are never pooled with study recordings.
    // `demo-v2` (2026-10-17): the result screen also announces the next trial and counts down
    // to it, so the separate get-ready screen is left only before trials that follow no result.
    public const string DemoProtocolVersion = "demo-v2";

    public const int CalibrationPasses = 1;   // K block: 7 trials
    public const int EvaluationPasses = 2;    // D block: 14 trials
    public const int DemoEvaluationPasses = 1; // demo test block: 7 trials
    public const int TicksPerTrial = 10;


    // Demo text colours, on the emotion word only: the predicted emotion when it matches the
    // acted one and when it does not, and the next emotion. Grey for the trial counter.
    private const string DemoCorrectColor = "#9BE15D";
    private const string DemoWrongColor = "#FF5A5A";
    private const string DemoNextColor = "#FFF04D";
    private const string DemoCounterColor = "#969BA5";

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

    [Header("Demo")]
    [Tooltip("Runs the demo instead of the study: each emotion once after calibration, a " +
             "remaining-time bar while acting and the predicted emotion after each trial, " +
             "with a countdown to the next. Recordings are marked as demo and must not be " +
             "analysed with study data.")]
    [SerializeField] private bool demoMode = false;

    // Demo screen lengths, in seconds. Longer than the study's, so viewers can read them. The
    // acting itself stays 10 s: it is the recording, the same as the study's.
    [Tooltip("The start screen.")]
    [SerializeField, Min(0f)] private float demoIntroSeconds = 10f;
    [Tooltip("The \"Part 1: Calibration\" and \"Part 2\" screens.")]
    [SerializeField, Min(0f)] private float demoPartSeconds = 6f;
    [Tooltip("Every get-ready and result screen: how long its text is shown before the " +
             "circle countdown appears.")]
    [SerializeField, Min(0f)] private float demoReadSeconds = 5f;
    [Tooltip("The circle countdown, one number a second, at the end of every get-ready and " +
             "result screen.")]
    [SerializeField, Min(1)] private int demoCountdownSeconds = 5;

    [Tooltip("Saves the 70 face tracking values every frame to face_tracking.csv in the " +
             "session folder, from the first get-ready screen to the end screen, for replaying " +
             "the face on an avatar afterwards. Demo only.")]
    [SerializeField] private bool saveFaceTracking = false;

    [Header("Calibration")]
    [Tooltip("Whether the session calibrates is the \"Use Calibration\" checkbox on the New Face " +
             "Au Model component (VREmotionDetection library). This one only writes each " +
             "evaluation tick's calibrated decision to the Unity console, readable over " +
             "`adb logcat`. Development builds only. Nothing is ever shown to the participant.")]
    [SerializeField] private bool calibrationDebugLogging = false;

    // All seven model output classes are prompted, Neutral included. Leaving Neutral out
    // would let the adapter range over six classes while the raw model ranges over seven,
    // and comparing those two credits the adapter for excluding a class rather than for
    // calibrating anything -- an error this project has already made once.
    public static readonly Emotion[] PromptedEmotions =
    {
        Emotion.Anger, Emotion.Disgust, Emotion.Fear, Emotion.Happiness,
        Emotion.Neutral, Emotion.Sadness, Emotion.Surprise,
    };

    // Neutral is deliberately absent. Every emoji here depicts a performed expression, and
    // showing one for Neutral would prompt the participant to *act* neutral rather than to
    // rest. It is prompted with text instead; see `PromptScreen`.
    public static readonly Dictionary<Emotion, string> EmotionToEmoji = new()
    {
        { Emotion.Anger, "🤬" },
        { Emotion.Disgust, "🤢" },
        { Emotion.Fear, "😱" },
        { Emotion.Happiness, "😀" },
        { Emotion.Sadness, "☹️" },
        { Emotion.Surprise, "😲" },
    };

    // Worded identically for every participant, and asking for rest rather than for an
    // expression. Two lines, matching the shape of the emoji prompts so the Neutral trial
    // does not look like a different kind of screen -- and so it fits the instruction box
    // without overflowing it.
    //
    // Deliberately not "Relax your face": that is exactly what the between-trial transition
    // screen says, and reusing it would blur the line between the part that is recorded and
    // the part that is not.
    //
    // The guidance about where the mouth and eyebrows should sit lives in the intro instead.
    // It is worth saying -- asked to "be neutral", people often produce a deliberately blank
    // or tensed face rather than a resting one -- but it is too long to repeat on a prompt
    // that appears mid-trial, and the intro has six seconds in which to read it.
    private const string NeutralPrompt = "Neutral\nLet your face rest.";

    private static string PromptScreen(Emotion emotion) =>
        emotion == Emotion.Neutral
            ? NeutralPrompt
            : $"{EmotionToEmoji[emotion]}\n{emotion}";

    private static string HoldScreen(Emotion emotion) =>
        emotion == Emotion.Neutral
            ? "Stay relaxed"
            : $"{EmotionToEmoji[emotion]}\n\nHold it";

    // A position marker on the prompt screen, naming the part: "Calibration 3 of 7" in the
    // calibration block, "Trial 5 of 14" in the test block. Shown on the prompt only, never
    // during the hold.
    private string TrialCounter(PlannedTrial trial)
    {
        var block = Plan.Where(p => p.Role == trial.Role).ToList();
        int position = block.FindIndex(p => p.TrialId == trial.TrialId) + 1;
        string name = trial.Role == BlockRole.Calibration ? "Calibration" : "Trial";
        return $"{name} {position} of {block.Count}";
    }

    // The end screen's per-emotion table of the test block, laid out like the analysis
    // script's (emotion, correct seconds / scored seconds, %). Monospaced so the columns line up.
    private string TestResultsTable()
    {
        var table = new StringBuilder("<mspace=0.6em>");
        table.Append($"{"emotion",-11} {"correct",9} {"%",5}");
        for (int c = 0; c < ClassOrder.Length; ++c)
        {
            string percent = testSeconds[c] > 0
                ? $"{100f * testCorrect[c] / testSeconds[c]:F0}%"
                : "-";
            table.Append($"\n{ClassOrder[c],-11} {testCorrect[c],4}/{testSeconds[c],-4} {percent,5}");
        }
        return table.Append("</mspace>").ToString();
    }

    // The demo's end screen: one row per test trial, acted emotion and predicted emotion, the
    // prediction green when it matches and red when it does not.
    private string DemoResultsTable()
    {
        var table = new StringBuilder("<mspace=0.6em>");
        table.Append($"{"Target",-11}{"Predicted",-11}");
        foreach (var (target, predicted) in demoResults)
        {
            string shown = predicted.Scorable
                ? $"<color={(predicted.ClassIndex == (int)target ? DemoCorrectColor : DemoWrongColor)}>" +
                  $"{ClassOrder[predicted.ClassIndex],-11}</color>"
                : $"<color={DemoCounterColor}>{"-",-11}</color>";
            table.Append($"\n{target,-11}{shown}");
        }
        return table.Append("</mspace>").ToString();
    }

    // One prompt, resolved before the session starts so the whole schedule is fixed and
    // recordable in advance rather than decided trial by trial.
    public readonly struct PlannedTrial
    {
        public readonly int TrialId;
        public readonly BlockRole Role;
        public readonly int BlockIndex;   // 0-based pass index within this role
        public readonly int RepeatIndex;  // 1-based presentation of this emotion in this role
        public readonly Emotion Emotion;

        public PlannedTrial(int trialId, BlockRole role, int blockIndex, int repeatIndex, Emotion emotion)
        {
            TrialId = trialId;
            Role = role;
            BlockIndex = blockIndex;
            RepeatIndex = repeatIndex;
            Emotion = emotion;
        }
    }

    private NewFaceAuModel predictor;
    private SessionLogger sessionLogger;
    private int seedUsed;

    private CalibrationSessionConfig calibrationConfig;

    // Movement calibration is the predictor's own (NewFaceAuModel), so the predictor, its
    // results file and this protocol all use the same calibration.
    private bool calibrationSealed;

    // Test-block score per emotion, indexed by `ClassOrder`: scored seconds whose result was
    // the prompted emotion, and scored seconds shown. The same counts as the hold screen's score.
    private readonly int[] testCorrect = new int[ClassOrder.Length];
    private readonly int[] testSeconds = new int[ClassOrder.Length];

    // The demo's test trials in the order acted: the acted emotion and its predicted emotion,
    // for the end screen.
    private readonly List<(Emotion Target, TrialVote Predicted)> demoResults = new();

    public IReadOnlyList<PlannedTrial> Plan { get; private set; }

    // The class order every part of this study agrees on: the `Emotion` enum, the CSV's
    // `score_<Class>` columns, and the seven rows of every centroid
    // matrix. `CheckClassOrder` below refuses to run if they ever stop agreeing, because a
    // permutation would relabel every centroid without changing a single number.
    private static readonly string[] ClassOrder =
    {
        "Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise",
    };

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

        if (!CheckClassOrder())
        {
            enabled = false;
            return;
        }

        seedUsed = randomizationSeed != 0
            ? randomizationSeed
            : Environment.TickCount & 0x7FFFFFFF;

        // The library's "Use Calibration" checkbox: off, the calibration block is skipped.
        Plan = BuildPlan(seedUsed, includeCalibration: predictor.UseCalibration,
                         evaluationPasses: EvaluationPassesThisSession);

        RunSession();
    }

    private int EvaluationPassesThisSession => demoMode ? DemoEvaluationPasses : EvaluationPasses;

    void Update() => demoScreen?.Tick();

    // After every Update, so the face tracking has already been read this frame.
    void LateUpdate() => faceRecorder?.RecordFrame();

    private bool SavingFaceTracking => demoMode && saveFaceTracking;

    // The calibration block, then the evaluation block, each a sequence of shuffled passes
    // over all seven states. Resolved up front so the whole schedule is fixed and recordable
    // in advance rather than decided trial by trial.
    // `includeCalibration`: false leaves out the calibration block, so a session without
    // calibration is the test block only. `evaluationPasses`: the test block's passes (the
    // demo runs one).
    public static List<PlannedTrial> BuildPlan(int seed, bool includeCalibration = true,
                                               int evaluationPasses = EvaluationPasses)
    {
        var random = new Random(seed);
        var plan = new List<PlannedTrial>(
            (CalibrationPasses + evaluationPasses) * PromptedEmotions.Length);

        // Carried across passes *and* across the block boundary, so the first evaluation
        // trial also differs from the last calibration trial.
        Emotion? previous = null;

        void AddPasses(BlockRole role, int passes)
        {
            // Repeat counts restart per block, so `repeat_index` reads as "the nth time this
            // state was shown in this block" and does not depend on how long the preceding
            // block was.
            var repeats = PromptedEmotions.ToDictionary(e => e, _ => 0);

            for (int pass = 0; pass < passes; ++pass)
            {
                var order = PromptedEmotions.ToArray();

                // Fisher-Yates, redrawn if the pass would open on the state the previous
                // pass closed with. Within a pass no state repeats, so a pass boundary is
                // the only place the same state could land back to back.
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
                    plan.Add(new PlannedTrial(plan.Count, role, pass, repeats[emotion], emotion));
                }

                previous = order[^1];
            }
        }

        if (includeCalibration)
        {
            AddPasses(BlockRole.Calibration, CalibrationPasses);
        }
        AddPasses(BlockRole.Evaluation, evaluationPasses);

        return plan;
    }

    // Every centroid is a row in a matrix and every score is a column, and both are indexed by
    // this order. If the `Emotion` enum were ever reordered, the logged columns would keep their
    // names while silently meaning different classes, and the resulting numbers would look
    // entirely ordinary. Cheap to check once; impossible to notice later.
    private static bool CheckClassOrder()
    {
        var declared = (Emotion[])Enum.GetValues(typeof(Emotion));

        if (declared.Length != ClassOrder.Length)
        {
            Debug.LogError($"{nameof(ProtocolRunner)}: the Emotion enum has {declared.Length} " +
                           $"members but the study's class order has {ClassOrder.Length}.");
            return false;
        }

        for (int i = 0; i < ClassOrder.Length; ++i)
        {
            if ((int)declared[i] != i || declared[i].ToString() != ClassOrder[i])
            {
                Debug.LogError($"{nameof(ProtocolRunner)}: Emotion.{declared[i]} is at index " +
                               $"{(int)declared[i]} but the study's class order puts " +
                               $"'{ClassOrder[i]}' there. Every centroid would be mislabelled.");
                return false;
            }
        }

        return true;
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

            // Counted off the plan rather than from the constants, so these can never
            // disagree with the schedule that actually ran.
            int calibrationTrials = Plan.Count(t => t.Role == BlockRole.Calibration);
            int evaluationTrials = Plan.Count(t => t.Role == BlockRole.Evaluation);

            int evaluationPasses = EvaluationPassesThisSession;

            var sessionInfo = new Dictionary<string, string>
            {
                { "protocol_version", demoMode ? DemoProtocolVersion : ProtocolVersion },
                { "randomization_seed", seedUsed.ToString() },
                { "trial_count", Plan.Count.ToString() },

                // Deprecated, kept so readers of older recordings are not broken. Use
                // `block_count` and the per-block fields below.
                { "blocks", "2" },

                { "block_count", "2" },
                { "calibration_trials", calibrationTrials.ToString() },
                { "evaluation_trials", evaluationTrials.ToString() },
                { "calibration_passes", CalibrationPasses.ToString() },
                { "evaluation_passes", evaluationPasses.ToString() },
                { "evaluation_repeats", evaluationPasses.ToString() },

                // What the participant saw, recorded so the difference is machine-detectable.
                // Recordings made before this field exists did not show a counter; a reader
                // comparing them against later ones can tell, without having to know which
                // build produced which folder. The demo shows none.
                { "trial_counter_shown", demoMode ? "false" : "true" },

                { "ticks_per_trial", (fastRehearsal ? 3 : TicksPerTrial).ToString() },
                { "fast_rehearsal", fastRehearsal ? "true" : "false" },
                { "self_report", "not collected (deliberate; no controller in this study)" },
            };

            // Only in demo recordings, so a study recording's metadata is unchanged.
            if (demoMode)
            {
                sessionInfo["demo"] = "true";
                sessionInfo["face_tracking_saved"] = SavingFaceTracking ? "true" : "false";
            }

            sessionLogger = new SessionLogger(
                NewFaceAuModel.WindowFrames,
                NewFaceAuModel.FeaturesPerFrame,
                modelInfo,
                sessionInfo,
                calibrationConfig);
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
        faceRecorder?.Dispose();
        faceRecorder = null;
    }

    // The demo's face tracking file, when Save Face Tracking is on.
    private FaceTrackingRecorder faceRecorder;

    private void StartFaceRecorder()
    {
        if (!SavingFaceTracking)
        {
            return;
        }

        try
        {
            faceRecorder = new FaceTrackingRecorder(GetComponent<OVRFaceExpressions>(),
                                                    sessionLogger.SessionDirectory);
            Debug.Log($"{nameof(ProtocolRunner)}: saving face tracking to {faceRecorder.Path}");
        }
        catch (Exception e)
        {
            // The demo runs on without it: the face file is for the video, not the recording.
            faceRecorder = null;
            Debug.LogError($"{nameof(ProtocolRunner)}: could not save face tracking " +
                           $"({e.GetType().Name}: {e.Message}). The demo continues without it.");
        }
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
        demoScreen?.Hide();
        instructionText.text = fastRehearsal
            ? "-- REHEARSAL - NOT VALID DATA --\n\n" + message
            : message;
    }

    // The demo's own screens replace the instruction text; only a rehearsal banner stays on it.
    private DemoScreen demoScreen;

    private void ClearForDemoScreen()
    {
        instructionText.text = fastRehearsal ? "-- REHEARSAL - NOT VALID DATA --" : "";
    }

    // The canvas is fixed in the room, and the room's forward is the headset's tracking
    // origin, not where the player is looking -- so it could open behind them. Instead it is
    // put in front of the player's gaze (level, at eye height, at its scene distance) once
    // the head is tracked, and again whenever the player recenters.
    private Transform canvasRoot;
    private Transform head;
    private float canvasDistance;

    private async Awaitable PlaceCanvasInFrontOfPlayer()
    {
        canvasRoot = instructionText.canvas.rootCanvas.transform;
        canvasDistance = canvasRoot.position.magnitude;

        var rig = FindAnyObjectByType<OVRCameraRig>();
        var fallback = Camera.main;
        head = rig != null ? rig.centerEyeAnchor
             : fallback != null ? fallback.transform
             : null;
        if (head == null)
        {
            Debug.LogWarning($"{nameof(ProtocolRunner)}: no head found; the screen stays where the scene put it.");
            return;
        }

        // The first frames can still hold the untracked identity pose.
        float giveUp = Time.time + 2f;
        while (head.localRotation == Quaternion.identity && Time.time < giveUp)
        {
            await Awaitable.NextFrameAsync();
        }

        MoveCanvasToGaze();

        if (OVRManager.display != null)
        {
            OVRManager.display.RecenteredPose += MoveCanvasToGaze;
        }
    }

    private void MoveCanvasToGaze()
    {
        if (head == null || canvasRoot == null)
        {
            return;
        }

        var forward = Vector3.ProjectOnPlane(head.forward, Vector3.up);
        forward = forward.sqrMagnitude > 1e-4f ? forward.normalized : Vector3.forward;
        canvasRoot.SetPositionAndRotation(head.position + forward * canvasDistance,
                                          Quaternion.LookRotation(forward, Vector3.up));
    }

    void OnDestroy()
    {
        if (OVRManager.display != null)
        {
            OVRManager.display.RecenteredPose -= MoveCanvasToGaze;
        }
    }

    private async void RunSession()
    {
        await PlaceCanvasInFrontOfPlayer();

        if (demoMode)
        {
            demoScreen = new DemoScreen(instructionText);
        }

        if (fastRehearsal)
        {
            Debug.LogWarning($"{nameof(ProtocolRunner)}: FAST REHEARSAL is on. This recording " +
                             "is marked as a rehearsal in its metadata and must not be analysed.");
        }

        Debug.Log($"{nameof(ProtocolRunner)}: {Plan.Count} trials, " +
                  $"seed {seedUsed}, protocol {(demoMode ? DemoProtocolVersion : ProtocolVersion)}");

        // Read once, here, and never again. Everything downstream holds this object rather
        // than the component, so a checkbox toggled mid-session cannot reach the session.
        // The library's "Use Calibration" checkbox decides whether this session calibrates.
        bool debug = calibrationDebugLogging && Debug.isDebugBuild;
        if (calibrationDebugLogging && !debug)
        {
            Debug.LogWarning($"{nameof(ProtocolRunner)}: calibration debug logging is only " +
                             "available in a development build. It is off for this session " +
                             "and recorded as off.");
        }

        calibrationConfig = CalibrationSessionConfig.Lock(predictor.UseCalibration, debug);

        Debug.Log($"{nameof(ProtocolRunner)}: calibration {calibrationConfig}");

        if (!PrepareCalibration())
        {
            return;
        }

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

        StartFaceRecorder();

        if (demoMode)
        {
            Show("Each trial:\n" +
                 "1. Act the emotion shown (10 s)\n" +
                 "2. See what the system predicted\n" +
                 "3. Get ready for the next one");
            await Awaitable.WaitForSecondsAsync(Scaled(demoIntroSeconds));
        }
        else
        {
            Show("Each trial:\n" +
                 "1. See the emotion\n" +
                 "2. Count down 3-2-1\n" +
                 "3. Hold the face for 10 seconds\n" +
                 "4. Relax");
            await Awaitable.WaitForSecondsAsync(Scaled(6f));
        }

        BlockRole? currentPart = null;
        for (int index = 0; index < Plan.Count; ++index)
        {
            var trial = Plan[index];

            // Name each part when it starts.
            if (trial.Role != currentPart)
            {
                currentPart = trial.Role;
                Show(trial.Role == BlockRole.Calibration
                    ? "Part 1: Calibration\n\nFirst we learn how you make each expression."
                    : (calibrationConfig.MovementEnabled
                        ? "Part 2\n\nNow make each expression again."
                        : "Make each expression as it is shown.") +
                      "\nYou will see what the system thinks.");
                faceRecorder?.SetScreen("part");
                await Awaitable.WaitForSecondsAsync(Scaled(demoMode ? demoPartSeconds : 4f));
            }

            // The profile is sealed on the way into the first evaluation trial, which is the
            // moment the calibration block is complete. Keyed on the role rather than on a
            // trial number so it stays correct if the block sizes ever change.
            if (trial.Role == BlockRole.Evaluation && !calibrationSealed)
            {
                SealCalibration();
            }

            if (demoMode)
            {
                // A test trial's result screen counts down to the next trial, so a get-ready
                // screen is needed only where no result screen came just before.
                bool getReady = index == 0 || Plan[index - 1].Role != BlockRole.Evaluation;
                PlannedTrial? next = index + 1 < Plan.Count ? Plan[index + 1] : null;
                await RunDemoTrial(trial, next, getReady);
            }
            else
            {
                await RunTrial(trial);
            }
        }

        // A plan with no evaluation block would otherwise finish unsealed, leaving the
        // metadata saying the calibration was never closed.
        if (!calibrationSealed)
        {
            SealCalibration();
        }

        Show(demoMode
            ? "Demo complete. Thank you.\n\n" + DemoResultsTable()
            : "Session complete.\n\nThank you.\n\n" + TestResultsTable());
        Debug.Log($"{nameof(ProtocolRunner)}: session complete, {Plan.Count} trials recorded.");

        CloseSessionLogger();
    }

    /// <summary>
    /// Start the predictor's movement calibration when it is on, or refuse to run.
    /// </summary>
    private bool PrepareCalibration()
    {
        // A rehearsal shortens every trial to three ticks, so no class can reach the nine a
        // centroid needs. Rather than let the operator discover that at the seal, after a
        // participant has sat through the whole thing, refuse the combination up front.
        if (fastRehearsal && calibrationConfig.MovementEnabled)
        {
            Abort($"FAST REHEARSAL cannot be combined with {calibrationConfig.ModeName} " +
                  "calibration: a rehearsal trial records three ticks and a centroid needs " +
                  $"{MovementCalibrator.SamplesPerEmotion}. Turn one of the two off.");
            return false;
        }

        if (calibrationConfig.MovementEnabled)
        {
            predictor.StartCalibration();
        }

        return true;
    }

    /// <summary>
    /// Close the calibration block, and record what came of it.
    /// </summary>
    /// <remarks>
    /// A failed seal does not end the session. The raw recording is still worth having -- it
    /// is a complete set of scores and action-unit windows, and the offline analysis can fit
    /// centroids from it afterwards exactly as it always has -- so the session runs on with
    /// the calibrated columns left blank, and the failure is written into the metadata, into
    /// every evaluation row, and into the log. Ending here would throw away a participant's
    /// visit to punish a tracking dropout.
    /// </remarks>
    private void SealCalibration()
    {
        calibrationSealed = true;

        // Raw sessions have nothing to build and nothing to invalidate.
        bool movement = calibrationConfig.MovementEnabled;
        bool valid = !movement || predictor.FinishCalibration();
        string error = movement ? predictor.CalibrationError ?? string.Empty : string.Empty;
        bool built = movement && predictor.IsCalibrated;

        var counts = (movement ? predictor.CalibrationSampleCounts() : null) ?? new int[ClassOrder.Length];
        var perClass = new List<string>();
        for (int c = 0; c < counts.Length; ++c)
        {
            perClass.Add($"{ClassOrder[c]}={counts[c]}");
        }

        var outcome = new Dictionary<string, string>
        {
            { "calibration_valid", valid ? "true" : "false" },
            { "calibration_error", error },
            { "personal_centroid_built", built ? "true" : "false" },
            { "movement_centroid_built", built ? "true" : "false" },
            { "calibration_ticks_per_class", string.Join(", ", perClass) },
        };

        sessionLogger?.RecordCalibrationOutcome(outcome);

        if (valid)
        {
            Debug.Log($"{nameof(ProtocolRunner)}: calibration sealed, mode " +
                      $"{calibrationConfig.ModeName}, ticks {string.Join(", ", perClass)}");
        }
        else
        {
            Debug.LogError($"{nameof(ProtocolRunner)}: CALIBRATION INVALID -- " +
                           $"{error}\nThe session continues and the raw " +
                           "recording stays usable, but no calibrated prediction will be " +
                           "produced and this recording must not be pooled with valid ones.");
        }
    }

    private void Abort(string reason)
    {
        Debug.LogError($"{nameof(ProtocolRunner)}: {reason}");
        Show("Cannot start.\n\nPlease tell the researcher.");
        enabled = false;
    }

    private async Awaitable RunTrial(PlannedTrial trial)
    {
        // 1. Prompt. Named and pictured for the six performed expressions; worded as rest
        //    for Neutral, which has no emoji. Carries the trial counter above it -- see
        //    `TrialCounter` for why it does not name the block.
        Show($"{TrialCounter(trial)}\n\n{PromptScreen(trial.Emotion)}");
        await Awaitable.WaitForSecondsAsync(Scaled(promptSeconds));

        // 2. Countdown, so the expression starts at a known moment rather than drifting
        //    into the recording window. The next emotion stays named above the number.
        for (int i = Mathf.RoundToInt(countdownSeconds); i > 0; --i)
        {
            Show($"Next Emotion: {trial.Emotion}\n\n{i}");
            await Awaitable.WaitForSecondsAsync(Scaled(1f));
        }

        // 3. Hold, while the model is polled once a second.
        Show(HoldScreen(trial.Emotion));
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

    // The demo's trial: a get-ready countdown when no result screen came before it, the hold
    // with the remaining-time bar, and in the test block the trial's predicted emotion with a
    // countdown to the next trial. The recording itself is `RecordTrial`, exactly as in the study.
    private async Awaitable RunDemoTrial(PlannedTrial trial, PlannedTrial? next, bool getReady)
    {
        // 1. Get ready: time to read and relax, then the countdown. Doubles as the relax
        //    period before the trial.
        if (getReady)
        {
            float lead = Scaled(demoReadSeconds);
            ClearForDemoScreen();
            // The face file's time 0 is the first of these screens.
            faceRecorder?.Begin();
            faceRecorder?.SetScreen("get_ready", trial.TrialId, trial.Emotion.ToString());
            demoScreen.ShowCountdown($"<color={DemoCounterColor}>{TrialCounter(trial)}</color>",
                                     DemoScreen.GetReadySize,
                                     $"Relax yourself.\nNext: <color={DemoNextColor}>{trial.Emotion}</color>",
                                     DemoScreen.GetReadySize,
                                     "Starts in", demoCountdownSeconds, Scaled(demoCountdownSeconds), lead);
            await Awaitable.WaitForSecondsAsync(lead + Scaled(demoCountdownSeconds));
        }

        // 2. Hold. No live result and no score: the result comes after the trial.
        ClearForDemoScreen();
        float holdSeconds = fastRehearsal ? 3 : TicksPerTrial;
        if (trial.Emotion == Emotion.Neutral)
        {
            demoScreen.ShowHold("Act Neutral", "Let your face rest.", DemoScreen.BodySize + 6f,
                                "", holdSeconds);
        }
        else
        {
            // Three times the body text, so the emoji reads from a distance.
            demoScreen.ShowHold($"Act {trial.Emotion}", EmotionToEmoji[trial.Emotion],
                                3f * 24f, "Hold it", holdSeconds);
        }

        faceRecorder?.SetScreen("act", trial.TrialId, trial.Emotion.ToString());
        var predicted = await RecordTrial(trial);

        // 3. Test block only: the emotion predicted most often over the trial's scored seconds
        //    (calibrated when calibration is ready), what comes next, and a countdown to it --
        //    to the next trial, or to the end screen after the last.
        if (trial.Role != BlockRole.Evaluation)
        {
            return;
        }

        demoResults.Add((trial.Emotion, predicted));

        string headline = "In this trial,\nyour face was not tracked.";
        if (predicted.Scorable)
        {
            string color = predicted.ClassIndex == (int)trial.Emotion ? DemoCorrectColor : DemoWrongColor;
            headline = "In this trial,\nyour predicted emotion was " +
                       $"<color={color}>{ClassOrder[predicted.ClassIndex]}</color>";
        }

        string upcoming = "That was the last one.";
        string label = "Results in";
        if (next.HasValue)
        {
            string counter = TrialCounter(next.Value);
            upcoming = $"Next: <color={DemoNextColor}>{next.Value.Emotion}</color> " +
                       $"<color={DemoCounterColor}>({char.ToLower(counter[0])}{counter.Substring(1)})</color>";
            label = "Starts in";
        }

        // Time to read first (5 s by default), then "Starts in" and the circle (5 s).
        float readSeconds = Scaled(demoReadSeconds);
        ClearForDemoScreen();
        faceRecorder?.SetScreen("result", trial.TrialId, trial.Emotion.ToString());
        demoScreen.ShowCountdown(headline, DemoScreen.HeadlineSize, upcoming, DemoScreen.BodySize,
                                 label, demoCountdownSeconds, Scaled(demoCountdownSeconds), readSeconds);
        await Awaitable.WaitForSecondsAsync(readSeconds + Scaled(demoCountdownSeconds));
    }

    // Polls the model once a second for the hold period and logs every tick. In the study's
    // test block it also shows each second's result and the running score. Returns the
    // trial's majority vote as it was shown: calibrated when calibration is ready, the
    // model's own otherwise.
    private async Awaitable<TrialVote> RecordTrial(PlannedTrial trial)
    {
        int ticks = fastRehearsal ? 3 : TicksPerTrial;

        // Ticks 1-9 only, on both sides. The offline analysis scores exactly that window, and
        // a runtime vote that also counted tick 0 would produce a different trial prediction
        // from a recomputation of the same recording -- which is the one thing the parity
        // check exists to make impossible.
        var rawVotes = new List<TickVote>();
        var calibratedVotes = new List<TickVote>();
        int correctSeconds = 0;
        int scoredSeconds = 0;

        // Each calibration trial is one emotion of the predictor's movement calibration. The
        // block is recorded the same way in a raw session; only the calibration is skipped.
        bool calibrating = trial.Role == BlockRole.Calibration && calibrationConfig.MovementEnabled;
        if (calibrating)
        {
            predictor.BeginCalibrationEmotion(trial.Emotion);
        }

        for (int i = 0; i < ticks; ++i)
        {
            float nextTick = Time.time + 1f;

            // Calibration-block ticks stay out of the predictor's results file, as calibration
            // samples do everywhere; the detailed session log below still records every tick.
            var prediction = await predictor.PredictDetailed(
                saveResult: trial.Role != BlockRole.Calibration);

            NearestCentroid? calibrated = null;

            if (calibrating)
            {
                // By tick position, so tick 0 is dropped and a missed tick stays missed,
                // exactly as the offline analysis fits the same recording.
                predictor.AddCalibrationSample(i, prediction);
            }
            else if (trial.Role == BlockRole.Evaluation && PersonalCentroidBuilder.IsScoredTick(i)
                     && prediction.HasValue)
            {
                calibrated = predictor.CalibratedDecision(prediction.Value);
                if (calibrated.HasValue)
                {
                    calibratedVotes.Add(new TickVote(i, calibrated.Value.ClassIndex));
                }
            }

            if (prediction.HasValue && PersonalCentroidBuilder.IsScoredTick(i))
            {
                rawVotes.Add(new TickVote(i, (int)prediction.Value.Emotion));
            }

            // Test block: show the final result every tick -- the calibrated emotion when
            // calibration is ready (model scores and face movements against this person's own),
            // the model's own prediction otherwise. The same value as the results file's `emotion`.
            // Below it, the running score over the scored ticks (1-9): seconds whose result was
            // the prompted emotion, out of scored seconds so far. A second without tracking
            // counts as a miss.
            if (trial.Role == BlockRole.Evaluation)
            {
                string seen = "—";
                bool correct = false;
                if (prediction.HasValue)
                {
                    var shown = calibrated.HasValue ? (Emotion)calibrated.Value.ClassIndex
                                                    : prediction.Value.Emotion;
                    seen = shown.ToString();
                    correct = shown == trial.Emotion;
                }

                if (PersonalCentroidBuilder.IsScoredTick(i))
                {
                    scoredSeconds++;
                    testSeconds[(int)trial.Emotion]++;
                    if (correct)
                    {
                        correctSeconds++;
                        testCorrect[(int)trial.Emotion]++;
                    }
                }

                // The demo shows neither during the hold; its result comes after the trial.
                if (!demoMode)
                {
                    string score = scoredSeconds > 0 ? $"\nScore: {correctSeconds}/{scoredSeconds}" : "";
                    Show($"{HoldScreen(trial.Emotion)}\n\nResult: {seen}{score}");
                }
            }

            // Logged whether or not tracking held, so a dropout is visible in the file
            // rather than showing up as a missing row.
            sessionLogger?.LogPrediction(trial.TrialId, trial.Role, trial.RepeatIndex,
                                         trial.Emotion, i, prediction, calibrated);

            LogCalibrationDebug(trial, i, prediction, calibrated);

            float remaining = nextTick - Time.time;
            if (remaining > 0)
            {
                await Awaitable.WaitForSecondsAsync(remaining);
            }
        }

        var rawVote = TrialMajority.Vote(rawVotes, ClassOrder.Length);
        var calibratedVote = TrialMajority.Vote(calibratedVotes, ClassOrder.Length);
        sessionLogger?.LogTrial(trial.TrialId, trial.Role, trial.RepeatIndex, trial.Emotion,
                                rawVote, calibratedVote);

        return calibratedVote.Scorable ? calibratedVote : rawVote;
    }

    /// <summary>
    /// The development-build debug readout, to the Unity console and nowhere else.
    /// </summary>
    /// <remarks>
    /// Deliberately not on the headset panel, and named accordingly. The hold screen is in
    /// front of the participant while their face *is* the measurement, and the existing
    /// protocol goes out of its way never to change text during it -- see `TrialCounter`.
    /// Putting a prediction there would turn the evaluation block into a feedback task and
    /// contradict the intro's promise that nothing is scored.
    ///
    /// The operator reads it over `adb logcat` instead. That keeps the readout available for
    /// debugging while leaving the participant's experience identical in every mode.
    /// </remarks>
    private void LogCalibrationDebug(PlannedTrial trial, int tick,
                                     FacePrediction? prediction, NearestCentroid? calibrated)
    {
        if (!calibrationConfig.DebugLoggingEnabled || trial.Role != BlockRole.Evaluation)
        {
            return;
        }

        string raw = prediction.HasValue ? prediction.Value.Emotion.ToString() : "(no tracking)";
        string calibratedText = calibrated.HasValue
            ? $"{ClassOrder[calibrated.Value.ClassIndex]} (margin " +
              $"{calibrated.Value.Margin:F4})"
            : "(none)";

        Debug.Log($"[calibration debug] trial {trial.TrialId} tick {tick} " +
                  $"prompted {trial.Emotion} | raw {raw} | " +
                  $"{calibrationConfig.ModeName} {calibratedText}");
    }
}
