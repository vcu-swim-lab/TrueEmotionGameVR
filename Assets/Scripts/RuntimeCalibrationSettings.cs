using System;
using System.IO;

using UnityEngine;
using UnityEngine.Networking;

using VcuSwimLab.Calibration;

// The three operator checkboxes, and the loader for the artifact one of them needs.
//
// Deliberately its own component rather than two more fields on `ProtocolRunner`. The runner
// owns the protocol -- the block structure, the timings, the randomisation -- and those are
// frozen. Which centroid set a session is scored against is a separate decision, made per
// session, and giving it its own component keeps the two from being edited together by
// accident.
//
// Nothing here computes anything. It reads the inspector once, reads a file, and hands both to
// `VcuSwimLab.Calibration`, which is where the arithmetic lives and where it can be tested
// without Unity.
[DisallowMultipleComponent]
public class RuntimeCalibrationSettings : MonoBehaviour
{
    public const string ArtifactFileName = "emojihero_population_centroids_unity.json";

    [Header("Calibration")]
    [Tooltip("Fit seven centroids from this participant's calibration block and score the " +
             "evaluation block against them. Needs all nine scored ticks of all seven " +
             "calibration trials; a session that loses tracking is marked invalid rather " +
             "than calibrated on less.")]
    [SerializeField] private bool enablePersonalCalibration = false;

    [Tooltip("Blend in the frozen EmojiHero population profile. On its own this scores every " +
             "tick against twenty strangers' average; together with personal calibration it " +
             "makes the preregistered hybrid at alpha = 0.40.\n\n" +
             "Measured on the one headset session available: the population term moved the " +
             "result the wrong way. See Docs/VR_SHADOW_EVALUATION_REPORT.md.")]
    [SerializeField] private bool enableEmojiHeroPopulationCalibration = false;

    [Tooltip("Movement calibration. Learns two fingerprints per emotion from the calibration " +
             "block: the player's typical face (the 63 face action units, averaged over each " +
             "window) and the personal score centroid. Each evaluation tick goes to the emotion " +
             "whose two fingerprints it matches best, by the sum of the two cosines.\n\n" +
             "Runs alone: ticking it unticks the two checkboxes above, and ticking either of " +
             "those unticks it. It already contains the personal score centroids.\n\n" +
             "Offline, 9 headset sessions (8 of one person): tick macro-F1 0.730 against 0.647 " +
             "for personal calibration. See Analysis/exploratory/au_calibration.py.")]
    [SerializeField] private bool enableMovementCalibration = false;

    [Header("Debug")]
    [Tooltip("Write each evaluation tick's calibrated decision to the Unity console, " +
             "readable over `adb logcat`. Development builds only, and off by default.\n\n" +
             "Console only. Nothing is ever drawn on the headset panel: the participant sees " +
             "no prediction, no correctness and no score in any mode, because the evaluation " +
             "block is the measurement. Recorded in the session metadata either way.")]
    [SerializeField] private bool enableCalibrationDebugLogging = false;

    // The movement checkbox as OnValidate last saw it, so it can tell which side was just ticked.
    [NonSerialized] private bool seenMovement;

    // Movement runs alone. Whichever side was ticked last wins and the other side is cleared,
    // so the inspector can never show a combination the session would refuse. The session
    // still checks again at start (`CalibrationSessionConfig.Lock`), because a scene file can
    // be edited by hand.
    private void OnValidate()
    {
        bool othersOn = enablePersonalCalibration || enableEmojiHeroPopulationCalibration;

        if (enableMovementCalibration && othersOn)
        {
            bool movementJustTicked = !seenMovement;
            if (movementJustTicked)
            {
                enablePersonalCalibration = false;
                enableEmojiHeroPopulationCalibration = false;
                Debug.LogWarning($"{nameof(RuntimeCalibrationSettings)}: movement calibration runs " +
                                 "alone, so personal and population calibration were unticked.");
            }
            else
            {
                enableMovementCalibration = false;
                Debug.LogWarning($"{nameof(RuntimeCalibrationSettings)}: movement calibration was " +
                                 "unticked, because it cannot run together with personal or " +
                                 "population calibration.");
            }
        }

        seenMovement = enableMovementCalibration;
    }

    /// <summary>
    /// Read the toggles once. Everything afterwards reads the returned object, so changing a
    /// checkbox mid-session cannot reach the running session.
    /// </summary>
    /// <exception cref="CalibrationConfigurationException">
    /// Movement ticked together with personal or population calibration.
    /// </exception>
    public CalibrationSessionConfig LockForSession()
    {
        bool debug = enableCalibrationDebugLogging && Debug.isDebugBuild;

        if (enableCalibrationDebugLogging && !debug)
        {
            Debug.LogWarning($"{nameof(RuntimeCalibrationSettings)}: calibration debug logging " +
                             "is only available in a development build. It is off for this " +
                             "session and recorded as off.");
        }

        return CalibrationSessionConfig.Lock(
            enablePersonalCalibration, enableEmojiHeroPopulationCalibration,
            enableMovementCalibration, debug);
    }

    /// <summary>
    /// Read and validate the population artifact from StreamingAssets.
    /// </summary>
    /// <remarks>
    /// `UnityWebRequest` rather than `File.ReadAllText`, because on the Quest
    /// `Application.streamingAssetsPath` points inside the APK and is not a filesystem path.
    /// The desktop Editor case goes through the same code with a `file://` scheme, so the
    /// path that runs on the headset is the path that gets exercised while developing.
    ///
    /// Throws on every failure. A population or hybrid session that could not load its profile
    /// must not start, because a session that quietly scored raw would be filed as a
    /// calibrated one and pooled with real ones.
    /// </remarks>
    public async Awaitable<PopulationProfile> LoadPopulationProfileAsync()
    {
        string path = Path.Combine(Application.streamingAssetsPath, ArtifactFileName);
        string uri = path.Contains("://") ? path : "file://" + path;

        string json;
        using (var request = UnityWebRequest.Get(uri))
        {
            var operation = request.SendWebRequest();
            while (!operation.isDone)
            {
                await Awaitable.NextFrameAsync();
            }

            if (request.result != UnityWebRequest.Result.Success)
            {
                throw new PopulationProfileException(
                    $"could not read the population artifact at\n  {uri}\n"
                    + $"{request.result}: {request.error}\n"
                    + "Export it with `python Analysis/export_population_artifact_for_unity.py` "
                    + "and make sure it is inside Assets/StreamingAssets before building.");
            }

            json = request.downloadHandler.text;
        }

        // Parse validates everything: schema, source split, class order, epsilon, feature
        // space, transform, metric, model hash, the preregistered semantic hash, the matrix
        // shape, finiteness, and the conversion hash against the file's own contents.
        var profile = PopulationProfile.Parse(json);

        Debug.Log($"{nameof(RuntimeCalibrationSettings)}: population profile loaded, "
                  + $"semantic {profile.SemanticSha256.Substring(0, 12)}..., "
                  + $"{profile.ParticipantCount} participants, split '{profile.SourceSplit}'");

        return profile;
    }
}
