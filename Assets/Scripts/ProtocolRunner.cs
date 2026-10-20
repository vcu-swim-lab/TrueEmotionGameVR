using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using TMPro;
using UnityEngine;
using UnityEngine.Serialization;

using VcuSwimLab.Calibration;
using VcuSwimLab.VREmotionDetection;

using Random = System.Random;

// The game: the player acts each of the seven emotions and sees what the system predicted.
//
//   Part 1 -- calibration, one trial per emotion (only when the library's Use Calibration is
//             ticked): get ready, act for 10 seconds. The predictor learns this player's face.
//   Part 2 -- one trial per emotion, shuffled: act for 10 seconds, then the predicted emotion
//             (the majority over the trial's scored seconds), the next emotion and a circle
//             countdown to it. The end screen lists every trial's target and prediction.
//
// Every second of acting is logged in full (SessionLogger), and with Save Face Tracking the
// 70 face values of every frame too, for the avatar video (FaceTrackingRecorder).
//
// The earlier study flow (prompt, 3-2-1 countdown, live result and score, 14 test trials) and
// the older emotion game (ProgressGame) were removed on 2026-10-19; both are in git history.
//
// Neutral is acted like the others, with its own trial and its own ten ticks. It is
// deliberately NOT harvested from the time between trials: a transition sample carries the
// tail of the previous expression, and a Neutral centroid built that way was measured making
// things worse rather than better.
[RequireComponent(typeof(NewFaceAuModel))]
[RequireComponent(typeof(DeviceManager))]
[RequireComponent(typeof(OVRFaceExpressions))]
public class ProtocolRunner : MonoBehaviour
{
    // Bump when anything that changes the meaning of a recording changes: the class set,
    // the block structure, the timings, or the feedback policy. It is written into the
    // session metadata so a pooled analysis can refuse to mix incompatible recordings.
    // Recordings of the removed study flow carry `session-local-v1` to `v3`, and are never
    // pooled with these.
    //
    // `demo-v1` (2026-10-16): one test pass, a remaining-time bar while acting, and the
    // predicted emotion after each test trial.
    // `demo-v2` (2026-10-17): the result screen also announces the next trial and counts down
    // to it, so a separate get-ready screen is left only before trials that follow no result.
    // Unchanged when the demo became the only game (2026-10-19): the screens and timings did
    // not change, so these recordings mean the same as the earlier demo-v2 ones.
    //
    // 2026-10-20: the model can be SparseVR or Jev, with calibration on or off (four conditions,
    // see ConditionSetup). The flow is unchanged, so the version is too; every recording carries
    // `condition`, `face_backend` and `backend_id` in its metadata, and a pooled analysis must
    // filter on them rather than assume SparseVR.
    public const string ProtocolVersion = "demo-v2";

    public const int CalibrationPasses = 1;   // Part 1: 7 trials
    public const int EvaluationPasses = 1;    // Part 2: 7 trials
    public const int TicksPerTrial = 10;

    // How long "Condition N: ..." stays up once the condition is ready, before the session starts.
    private const float ConditionScreenSeconds = 3f;

    // Text colours, on the emotion word only: the predicted emotion when it matches the acted
    // one and when it does not, and the next emotion. Grey for the trial counter.
    private const string CorrectColor = "#9BE15D";
    private const string WrongColor = "#FF5A5A";
    private const string NextColor = "#FFF04D";
    private const string CounterColor = "#969BA5";

    [Header("Scene")]
    [Tooltip("The instruction text. Left empty, the object named 'Instruction' is used.")]
    [SerializeField] private TextMeshProUGUI instructionText;

    // The condition (which model predicts, whether the player calibrates) is not set here: it is
    // ConditionSetup.Selected, in code, so no scene value can disagree with it.

    [Header("Screen lengths (seconds)")]
    [Tooltip("The start screen.")]
    [FormerlySerializedAs("demoIntroSeconds")]
    [SerializeField, Min(0f)] private float introScreenSeconds = 10f;
    [Tooltip("The \"Part 1: Calibration\" and \"Part 2\" screens.")]
    [FormerlySerializedAs("demoPartSeconds")]
    [SerializeField, Min(0f)] private float partScreenSeconds = 6f;
    [Tooltip("Every get-ready and result screen: how long its text is shown before the " +
             "circle countdown appears.")]
    [FormerlySerializedAs("demoReadSeconds")]
    [SerializeField, Min(0f)] private float readingSeconds = 5f;
    [Tooltip("The circle countdown, one number a second, at the end of every get-ready and " +
             "result screen.")]
    [FormerlySerializedAs("demoCountdownSeconds")]
    [SerializeField, Min(1)] private int circleCountdownSeconds = 5;

    [Header("Recording")]
    [Tooltip("Saves the 70 face tracking values every frame to face_tracking.csv in the " +
             "session folder, from the first get-ready screen to the end screen, for replaying " +
             "the face on an avatar afterwards.")]
    [SerializeField] private bool saveFaceTracking = false;

    [Header("Session")]
    [Tooltip("Leave at 0 to draw a seed from the clock. The value used is recorded either way.")]
    [SerializeField] private int randomizationSeed = 0;

    [Header("Rehearsal")]
    [Tooltip("Shortens every trial so the flow can be walked through at a desk. " +
             "Recordings made with this on are marked as rehearsals and must not be analysed.")]
    [SerializeField] private bool fastRehearsal = false;

    [Header("Calibration")]
    [Tooltip("Whether the session calibrates is set by ConditionSetup.Selected (2 and 4 " +
             "calibrate, 1 and 3 do not). This one only writes each " +
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
    // showing one for Neutral would prompt the player to *act* neutral rather than to rest.
    // Its acting screen asks them to let the face rest instead.
    public static readonly Dictionary<Emotion, string> EmotionToEmoji = new()
    {
        { Emotion.Anger, "🤬" },
        { Emotion.Disgust, "🤢" },
        { Emotion.Fear, "😱" },
        { Emotion.Happiness, "😀" },
        { Emotion.Sadness, "☹️" },
        { Emotion.Surprise, "😲" },
    };

    // A position marker, naming the part: "Calibration 3 of 7" in Part 1, "Trial 5 of 7" in
    // Part 2. Shown on the get-ready and result screens, never while acting.
    private string TrialCounter(PlannedTrial trial)
    {
        var block = Plan.Where(p => p.Role == trial.Role).ToList();
        int position = block.FindIndex(p => p.TrialId == trial.TrialId) + 1;
        string name = trial.Role == BlockRole.Calibration ? "Calibration" : "Trial";
        return $"{name} {position} of {block.Count}";
    }

    // The end screen: one row per Part 2 trial, acted emotion and predicted emotion, the
    // prediction green when it matches and red when it does not.
    private string ResultsTable()
    {
        var table = new StringBuilder("<mspace=0.6em>");
        table.Append($"{"Target",-11}{"Predicted",-11}");
        foreach (var (target, predicted) in results)
        {
            string shown = predicted.Scorable
                ? $"<color={(predicted.ClassIndex == (int)target ? CorrectColor : WrongColor)}>" +
                  $"{ClassOrder[predicted.ClassIndex],-11}</color>"
                : $"<color={CounterColor}>{"-",-11}</color>";
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

    // Part 2's trials in the order acted: the acted emotion and its predicted emotion, for the
    // end screen.
    private readonly List<(Emotion Target, TrialVote Predicted)> results = new();

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

        // The condition chooses the model and switches calibration on or off, before the plan
        // below is built: whether there is a Part 1 follows the predictor's Use Calibration.
        // A setup problem (no Jev key) is not thrown here; the readiness check in RunSession
        // shows it on the headset.
        ConditionSetup.Configure(predictor, ConditionSetup.Selected);

        if (!CheckClassOrder())
        {
            enabled = false;
            return;
        }

        seedUsed = randomizationSeed != 0
            ? randomizationSeed
            : Environment.TickCount & 0x7FFFFFFF;

        // Conditions 2 and 4 (calibration on) have Part 1; 1 and 3 skip it.
        Plan = BuildPlan(seedUsed, includeCalibration: predictor.UseCalibration);

        RunSession();
    }

    void Update() => gameScreen?.Tick();

    // After every Update, so the face tracking has already been read this frame.
    void LateUpdate() => faceRecorder?.RecordFrame();

    // The calibration block, then the evaluation block, each a sequence of shuffled passes
    // over all seven states. Resolved up front so the whole schedule is fixed and recordable
    // in advance rather than decided trial by trial.
    // `includeCalibration`: false leaves out the calibration block, so a session without
    // calibration is Part 2 only.
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
            SessionLogger.AddFaceBackendInfo(modelInfo, predictor);

            // Counted off the plan rather than from the constants, so these can never
            // disagree with the schedule that actually ran.
            int calibrationTrials = Plan.Count(t => t.Role == BlockRole.Calibration);
            int evaluationTrials = Plan.Count(t => t.Role == BlockRole.Evaluation);

            int evaluationPasses = EvaluationPasses;

            var sessionInfo = new Dictionary<string, string>
            {
                { "protocol_version", ProtocolVersion },
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

                // What the player saw, recorded so the difference is machine-detectable. The
                // removed study flow showed a counter on screen while prompting; this game
                // shows none while acting.
                { "trial_counter_shown", "false" },

                { "ticks_per_trial", (fastRehearsal ? 3 : TicksPerTrial).ToString() },
                { "fast_rehearsal", fastRehearsal ? "true" : "false" },
                { "self_report", "not collected (deliberate; no controller in this study)" },

                // As every demo-v2 recording has carried them, so readers need no special case.
                { "demo", "true" },
                { "face_tracking_saved", saveFaceTracking ? "true" : "false" },
            };

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

    // The face tracking file, when Save Face Tracking is on.
    private FaceTrackingRecorder faceRecorder;

    private void StartFaceRecorder()
    {
        if (!saveFaceTracking)
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
            // The game runs on without it: the face file is for the video, not the recording.
            faceRecorder = null;
            Debug.LogError($"{nameof(ProtocolRunner)}: could not save face tracking " +
                           $"({e.GetType().Name}: {e.Message}). The game continues without it.");
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
        gameScreen?.Hide();
        instructionText.text = fastRehearsal
            ? "-- REHEARSAL - NOT VALID DATA --\n\n" + message
            : message;
    }

    // The trial screens (get ready, act, result) replace the instruction text; only a rehearsal
    // banner stays on it. The start, part and end screens use the instruction text itself.
    private GameScreen gameScreen;

    private void ClearForGameScreen()
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

        gameScreen = new GameScreen(instructionText);

        if (fastRehearsal)
        {
            Debug.LogWarning($"{nameof(ProtocolRunner)}: FAST REHEARSAL is on. This recording " +
                             "is marked as a rehearsal in its metadata and must not be analysed.");
        }

        Debug.Log($"{nameof(ProtocolRunner)}: {Plan.Count} trials, " +
                  $"seed {seedUsed}, protocol {ProtocolVersion}");

        // Read once, here, and never again. Everything downstream holds this object rather
        // than the component, so a checkbox toggled mid-session cannot reach the session.
        // Whether this session calibrates was set by the condition, in Start.
        bool debug = calibrationDebugLogging && Debug.isDebugBuild;
        if (calibrationDebugLogging && !debug)
        {
            Debug.LogWarning($"{nameof(ProtocolRunner)}: calibration debug logging is only " +
                             "available in a development build. It is off for this session " +
                             "and recorded as off.");
        }

        calibrationConfig = CalibrationSessionConfig.Lock(predictor.UseCalibration, debug);

        Debug.Log($"{nameof(ProtocolRunner)}: {ConditionSetup.Label(predictor)}, " +
                  $"calibration {calibrationConfig}");

        // Can the chosen condition predict at all (Jev: key and connection)? Found out here, on
        // the screen, before the player sits through a game of failed seconds. Nothing has been
        // asked of the player yet and no session file exists.
        Show($"{ConditionSetup.Label(predictor)}\n\n" +
             (ConditionSetup.UsesJev(predictor) ? "Checking the Jev connection..." : "Starting..."));
        var (ready, readyMessage) = await ConditionSetup.CheckReadyAsync(predictor);
        if (!ready)
        {
            Abort(readyMessage, showReason: true);
            return;
        }

        // SparseVR is ready at once, which would leave the condition on screen for one frame.
        // Held long enough to read, so whoever starts the build can see which condition it is.
        Show($"{ConditionSetup.Label(predictor)}\n\nReady.");
        await Awaitable.WaitForSecondsAsync(ConditionScreenSeconds);

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

        Show("Each trial:\n" +
             "1. Act the emotion shown (10 s)\n" +
             "2. See what the system predicted\n" +
             "3. Get ready for the next one");
        await Awaitable.WaitForSecondsAsync(Scaled(introScreenSeconds));

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
                await Awaitable.WaitForSecondsAsync(Scaled(partScreenSeconds));
            }

            // The profile is sealed on the way into the first evaluation trial, which is the
            // moment the calibration block is complete. Keyed on the role rather than on a
            // trial number so it stays correct if the block sizes ever change.
            if (trial.Role == BlockRole.Evaluation && !calibrationSealed)
            {
                SealCalibration();
            }

            // A Part 2 trial's result screen counts down to the next trial, so a get-ready
            // screen is needed only where no result screen came just before.
            bool getReady = index == 0 || Plan[index - 1].Role != BlockRole.Evaluation;
            PlannedTrial? next = index + 1 < Plan.Count ? Plan[index + 1] : null;
            await RunTrial(trial, next, getReady);
        }

        // A plan with no evaluation block would otherwise finish unsealed, leaving the
        // metadata saying the calibration was never closed.
        if (!calibrationSealed)
        {
            SealCalibration();
        }

        Show("Demo complete. Thank you.\n\n" + ResultsTable());
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
            // The predictor collects the calibration from the ACTIVE backend: SparseVR's scores
            // in condition 2, Jev's in condition 4 (one request per second).
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

    // `showReason` puts the reason on the headset screen as well as in the log. Used for the
    // condition check (no Jev key, a rejected key, no connection): that happens before the
    // player has been asked to do anything, and whoever is wearing or watching the headset needs
    // to see it without opening `adb logcat`.
    private void Abort(string reason, bool showReason = false)
    {
        Debug.LogError($"{nameof(ProtocolRunner)}: {reason}");
        Show(showReason
            ? $"Cannot start.\n\n{reason}\n\nPlease tell the researcher."
            : "Cannot start.\n\nPlease tell the researcher.");
        enabled = false;
    }

    // One trial: a get-ready countdown when no result screen came before it, acting with the
    // remaining-time bar, and in Part 2 the trial's predicted emotion with a countdown to the
    // next trial. The recording itself is `RecordTrial`.
    //
    // No self-report. This is a deliberate decision, not an omission, and it has a known cost:
    // when a class scores badly there is no way to tell whether the player failed to produce
    // the expression or the model failed to detect it. There are no controllers and no
    // in-headset input of any kind to collect it with.
    private async Awaitable RunTrial(PlannedTrial trial, PlannedTrial? next, bool getReady)
    {
        // 1. Get ready: time to read and relax, then the countdown. Doubles as the relax
        //    period before the trial, which keeps its first frames from carrying the tail of
        //    the previous expression.
        if (getReady)
        {
            float lead = Scaled(readingSeconds);
            ClearForGameScreen();
            // The face file's time 0 is the first of these screens.
            faceRecorder?.Begin();
            faceRecorder?.SetScreen("get_ready", trial.TrialId, trial.Emotion.ToString());
            gameScreen.ShowCountdown($"<color={CounterColor}>{TrialCounter(trial)}</color>",
                                     GameScreen.GetReadySize,
                                     $"Relax yourself.\nNext: <color={NextColor}>{trial.Emotion}</color>",
                                     GameScreen.GetReadySize,
                                     "Starts in", circleCountdownSeconds, Scaled(circleCountdownSeconds), lead);
            await Awaitable.WaitForSecondsAsync(lead + Scaled(circleCountdownSeconds));
        }

        // 2. Act. No live result and no score: the result comes after the trial.
        ClearForGameScreen();
        float holdSeconds = fastRehearsal ? 3 : TicksPerTrial;
        if (trial.Emotion == Emotion.Neutral)
        {
            gameScreen.ShowHold("Act Neutral", "Let your face rest.", GameScreen.BodySize + 6f,
                                "", holdSeconds);
        }
        else
        {
            // Three times the body text, so the emoji reads from a distance.
            gameScreen.ShowHold($"Act {trial.Emotion}", EmotionToEmoji[trial.Emotion],
                                3f * 24f, "Hold it", holdSeconds);
        }

        faceRecorder?.SetScreen("act", trial.TrialId, trial.Emotion.ToString());
        var predicted = await RecordTrial(trial);

        // 3. Part 2 only: the emotion predicted most often over the trial's scored seconds
        //    (calibrated when calibration is ready), what comes next, and a countdown to it --
        //    to the next trial, or to the end screen after the last.
        if (trial.Role != BlockRole.Evaluation)
        {
            return;
        }

        results.Add((trial.Emotion, predicted));

        string headline = "In this trial,\nyour face was not tracked.";
        if (predicted.Scorable)
        {
            string color = predicted.ClassIndex == (int)trial.Emotion ? CorrectColor : WrongColor;
            headline = "In this trial,\nyour predicted emotion was " +
                       $"<color={color}>{ClassOrder[predicted.ClassIndex]}</color>";
        }

        string upcoming = "That was the last one.";
        string label = "Results in";
        if (next.HasValue)
        {
            string counter = TrialCounter(next.Value);
            upcoming = $"Next: <color={NextColor}>{next.Value.Emotion}</color> " +
                       $"<color={CounterColor}>({char.ToLower(counter[0])}{counter.Substring(1)})</color>";
            label = "Starts in";
        }

        // Time to read first (5 s by default), then "Starts in" and the circle (5 s).
        float readSeconds = Scaled(readingSeconds);
        ClearForGameScreen();
        faceRecorder?.SetScreen("result", trial.TrialId, trial.Emotion.ToString());
        gameScreen.ShowCountdown(headline, GameScreen.HeadlineSize, upcoming, GameScreen.BodySize,
                                 label, circleCountdownSeconds, Scaled(circleCountdownSeconds), readSeconds);
        await Awaitable.WaitForSecondsAsync(readSeconds + Scaled(circleCountdownSeconds));
    }

    // Polls the model once a second while the player acts and logs every tick. Returns the
    // trial's majority vote: calibrated when calibration is ready, the model's own otherwise.
    private async Awaitable<TrialVote> RecordTrial(PlannedTrial trial)
    {
        int ticks = fastRehearsal ? 3 : TicksPerTrial;

        // Ticks 1-9 only, on both sides. The offline analysis scores exactly that window, and
        // a runtime vote that also counted tick 0 would produce a different trial prediction
        // from a recomputation of the same recording -- which is the one thing the parity
        // check exists to make impossible.
        var rawVotes = new List<TickVote>();
        var calibratedVotes = new List<TickVote>();

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
            // TryPredict rather than PredictDetailed: it also says why a second has no
            // prediction (no face tracking, or the backend -- Jev -- failed), and that goes in
            // the log. A failed second is never answered by another model.
            var outcome = await predictor.TryPredict(
                saveResult: trial.Role != BlockRole.Calibration);
            FacePrediction? prediction = outcome.PredictionOrNull;

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

            // Logged whether or not tracking held, so a dropout is visible in the file
            // rather than showing up as a missing row.
            sessionLogger?.LogPrediction(trial.TrialId, trial.Role, trial.RepeatIndex,
                                         trial.Emotion, i, prediction, calibrated, outcome.Error);

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
    /// Deliberately not on the headset panel, and named accordingly. The acting screen is in
    /// front of the player while their face *is* the measurement, and never changes its text
    /// during it; the prediction comes after the trial.
    ///
    /// The operator reads it over `adb logcat` instead. That keeps the readout available for
    /// debugging while leaving the player's experience identical in every mode.
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
