using System;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using VcuSwimLab.Backends;
using VcuSwimLab.Backends.Jev;
using VcuSwimLab.Calibration;
using VcuSwimLab.VREmotionDetection;

using Debug = UnityEngine.Debug;

// Chooses which of the four experiment conditions this build runs.
//
//   Condition 1   SparseVR (on the headset), calibration OFF     <- active in the committed code
//   Condition 2   SparseVR (on the headset), calibration ON
//   Condition 3   Jev (Typesafe's cloud model), calibration OFF
//   Condition 4   Jev (Typesafe's cloud model), calibration ON
//
// HOW TO TEST A CONDITION
//   1. In `Configure` below, put `//` in front of every line of the block that is active now.
//   2. Remove the `//` from every line of the block you want to test (only that one).
//   3. Build and Run on the headset.
//   For conditions 3 and 4, paste your Typesafe key into Assets/Resources/jev_api_key.txt first.
//
// THE PACKAGE DOES THE REAL WORK. This game only asks the VREmotionDetection library for three
// things -- the model (UseSparseVr / UseJev), the calibration (UseCalibration, CalibrateEmotion,
// LoadCalibration ...) and the prediction (TryPredict ...). There is no game-side Jev client,
// no game-side scoring and no fallback: if Jev fails, that prediction fails, it is logged as a
// failure, and nothing quietly asks the local model instead.
//
// Everything else in the game (the game, the study protocol, the session logs, the name a
// calibration is saved under) reads what the library reports -- which backend is active and
// whether calibration is on -- rather than remembering which block was enabled here, so the
// logs can never disagree with what actually ran.
public static class ConditionSetup
{
    // ---- Jev settings (conditions 3 and 4) --------------------------------------------------
    // The library defaults are 10 s and 2 retries, which is about 31 s for a prediction that
    // fails every time. A round asks once a second, so the game uses a shorter wait.
    private const float JevTimeoutSeconds = 5f;
    private const int JevMaxRetries = 1;

    // The library writes one Console line per Jev request -- "[Jev] POST <url> -> 200 in 412 ms
    // (attempt 1 of 2)" -- without the key. On the headset (adb logcat -s Unity) it is the plain
    // proof that conditions 3 and 4 really call Jev, and how fast. Set false for quiet logs.
    private const bool JevLogRequests = true;

    // ---- The key file -----------------------------------------------------------------------
    // Assets/Resources/jev_api_key.txt, loaded with Resources.Load, so it is part of the build and
    // nothing has to be copied to the headset. It is git-ignored: never commit a token.
    private const string KeyResourceName = "jev_api_key";

    // Set by Configure; read by CheckReadyAsync and by the session logger.
    private static JevOptions jevOptions;

    /// <summary>Why the chosen condition could not be set up, or null. Never contains the key.</summary>
    public static string SetupError { get; private set; }

    /// <summary>Where the Jev key came from ("key file" / "environment variable"); never the key.</summary>
    public static string KeySource { get; private set; }

    /// <summary>
    /// Select the backend and calibration for this run. Call once, from <c>Start</c>, before
    /// anything is predicted or calibrated.
    /// </summary>
    /// <remarks>
    /// The order is the library's required order: choose the BACKEND first (choosing a different
    /// one throws away any calibration, because a calibration belongs to the backend that made
    /// it), then switch CALIBRATION on or off, then clear any calibration left over.
    ///
    /// A problem (no face model, no key, a bad Jev setting) is not thrown: it is kept in
    /// <see cref="SetupError"/> and <see cref="CheckReadyAsync"/> reports it on the headset
    /// screen. It must never fall back to another backend, so a "Jev" run is always Jev.
    /// </remarks>
    public static void Configure(NewFaceAuModel predictor)
    {
        if (predictor == null)
        {
            throw new ArgumentNullException(nameof(predictor));
        }

        SetupError = null;
        KeySource = null;
        jevOptions = null;

        try
        {
            // >>> UNCOMMENT EXACTLY ONE of the four blocks below. >>>

            // ===== CONDITION 1: SparseVR, calibration OFF =====================================
            predictor.UseSparseVr();             // the ONNX model that ships in the package, on the headset
            predictor.UseCalibration = false;    // predictions are the model's own answer

            // ===== CONDITION 2: SparseVR, calibration ON ======================================
            // predictor.UseSparseVr();             // same model as condition 1
            // predictor.UseCalibration = true;     // the game / protocol then calibrates (7 faces, about 90 s)

            // ===== CONDITION 3: Jev, calibration OFF ==========================================
            // jevOptions = BuildJevOptions(sendOwnFaces: false);   // reads the key; null (and SetupError) if missing
            // if (jevOptions != null) predictor.UseJev(jevOptions);   // predictions come from Typesafe's cloud model
            // predictor.UseCalibration = false;    // Jev's own answer

            // ===== CONDITION 4: Jev, calibration ON ===========================================
            // jevOptions = BuildJevOptions(sendOwnFaces: false);   // true = also tell Jev how this player makes each face
            // if (jevOptions != null) predictor.UseJev(jevOptions);   // same backend as condition 3
            // predictor.UseCalibration = true;     // calibration is collected from JEV's scores: about 70 requests

            // <<< end of the four blocks <<<

            // Whatever was chosen, start with no calibration loaded; the game or protocol
            // calibrates or loads one afterwards if UseCalibration is on.
            predictor.ClearCalibration();
        }
        catch (Exception e)
        {
            // Typical causes: no face model assigned (SparseVR), or a Jev option the library
            // refuses. Kept for the screen rather than thrown from Start.
            SetupError = $"{e.GetType().Name}: {e.Message}";
        }

        if (SetupError != null)
        {
            Debug.LogError($"[condition] could not set up the condition: {SetupError}");
        }
        else
        {
            Debug.Log($"[condition] {Summary(predictor)}");
        }
    }

    /// <summary>One line describing the run, for the log: condition, backend id, calibration.</summary>
    public static string Summary(NewFaceAuModel predictor)
    {
        return $"{Label(predictor)} | backend={predictor.ActiveBackendId ?? "(none)"} | " +
               $"calibration={(predictor.UseCalibration ? "on" : "off")}" +
               (UsesJev(predictor) ? $" | jev key from {KeySource ?? "?"}" : string.Empty);
    }

    /// <summary>True when the library's active backend is Jev.</summary>
    public static bool UsesJev(NewFaceAuModel predictor)
        => ExperimentCondition.IsJevBackend(predictor.ActiveBackendId);

    /// <summary>"sparsevr_nocal", "sparsevr_cal", "jev_nocal" or "jev_cal".</summary>
    public static string Id(NewFaceAuModel predictor)
        => ExperimentCondition.Id(UsesJev(predictor), predictor.UseCalibration);

    /// <summary>For screens and logs, e.g. "Condition 3: Jev, no calibration".</summary>
    /// <remarks>
    /// When the setup failed (no Jev key, no face model) the predictor still reports the
    /// default SparseVR backend, so the label must not claim a condition that is not running.
    /// </remarks>
    public static string Label(NewFaceAuModel predictor)
        => SetupError != null
            ? "Condition could not be set up"
            : ExperimentCondition.Label(UsesJev(predictor), predictor.UseCalibration);

    /// <summary>"SparseVr" or "Jev".</summary>
    public static string BackendName(NewFaceAuModel predictor)
        => ExperimentCondition.BackendName(UsesJev(predictor));

    /// <summary>The id this player's calibration is saved and loaded under for the active backend.</summary>
    public static string CalibrationId(string playerId, NewFaceAuModel predictor)
        => ExperimentCondition.CalibrationFileId(playerId, UsesJev(predictor));

    // The Jev settings of this run, for the session metadata. Empty unless Jev is active.
    public static string JevModel => jevOptions?.Model;
    public static string JevBaseUrl => jevOptions?.BaseUrl;
    public static string JevTimeout => jevOptions != null
        ? jevOptions.TimeoutSeconds.ToString("0.#", CultureInfo.InvariantCulture) : null;
    public static string JevRetries => jevOptions != null
        ? jevOptions.MaxRetries.ToString(CultureInfo.InvariantCulture) : null;
    public static string JevSendsOwnFaces => jevOptions != null ? (jevOptions.SendOwnFacesInRequest ? "true" : "false") : null;

    /// <summary>
    /// Check, before a session starts, that the chosen condition can really predict.
    /// </summary>
    /// <remarks>
    /// SparseVR: ready as soon as <see cref="Configure"/> succeeded. Jev: sends ONE throwaway
    /// request (a made-up mild smile) through a separate backend object, so a missing or
    /// rejected key or a missing connection is reported here, on the screen, instead of as a
    /// session full of failed predictions. It needs no face tracking.
    ///
    /// Jev may still answer a made-up face with all zeros, which the library reports as
    /// <c>InvalidResponse</c>; that still proves the key and the connection work, so it counts
    /// as ready.
    /// </remarks>
    public static async Awaitable<(bool ok, string message)> CheckReadyAsync(NewFaceAuModel predictor)
    {
        if (SetupError != null)
        {
            return (false, SetupError);
        }

        if (!UsesJev(predictor))
        {
            return (true, "SparseVR model ready");
        }

        if (jevOptions == null)
        {
            // Not reachable through Configure; guards a half-edited block.
            return (false, "Jev is active but no Jev options were built.");
        }

        IEmotionBackend probe = null;
        try
        {
            probe = predictor.CreateJevBackend(jevOptions);
            // A mild smile, so the request has the same shape as a real one (named movements with
            // a strength) rather than an empty list: cheeks raised (4, 5), lip corners pulled (32, 33).
            var testFace = new float[NewFaceAuModel.WindowFrames][];
            for (int f = 0; f < testFace.Length; ++f)
            {
                testFace[f] = new float[NewFaceAuModel.FeaturesPerFrame];
                testFace[f][4] = testFace[f][5] = 0.4f;
                testFace[f][32] = testFace[f][33] = 0.6f;
            }

            var clock = Stopwatch.StartNew();
            var result = await predictor.ScoreWith(probe, testFace);
            clock.Stop();

            if (result.Ok)
            {
                return Report(true, $"Jev OK ({clock.ElapsedMilliseconds} ms)");
            }

            var error = result.Error.Value;
            if (error.Code == PredictionErrorCode.InvalidResponse)
            {
                return Report(true, $"Jev reachable ({clock.ElapsedMilliseconds} ms); " +
                                    "the made-up test face got no usable answer, which is fine");
            }

            return Report(false, $"Jev failed: {error.Code}. {HintFor(error.Code)}");
        }
        catch (Exception e)
        {
            return Report(false, $"Jev check failed: {e.GetType().Name}: {e.Message}");
        }
        finally
        {
            probe?.Dispose();
        }
    }

    private static (bool ok, string message) Report(bool ok, string message)
    {
        if (ok)
        {
            Debug.Log($"[condition] {message}");
        }
        else
        {
            Debug.LogError($"[condition] {message}");
        }

        return (ok, message);
    }

    // What to do about each way Jev can fail, in words for the person holding the headset.
    private static string HintFor(PredictionErrorCode code)
    {
        switch (code)
        {
            case PredictionErrorCode.Unauthorized:
                return "The key was rejected. Check Assets/Resources/jev_api_key.txt and rebuild.";
            case PredictionErrorCode.Unreachable:
            case PredictionErrorCode.Timeout:
                return "No answer from Jev. Is the headset online?";
            case PredictionErrorCode.RateLimited:
                return "Too many requests. Wait a minute and try again.";
            case PredictionErrorCode.ServerError:
                return "Jev had a server problem. Try again later.";
            case PredictionErrorCode.InvalidRequest:
                return "Jev refused the request. Check the model name and URL.";
            default:
                return "See the log (adb logcat -s Unity).";
        }
    }

    // ---- Jev options and the key --------------------------------------------------------------

    // Builds the library's JevOptions from the key file (or, in the Editor, the TYPESAFE_API_KEY
    // environment variable). Returns null and sets SetupError when there is no usable key. The
    // key itself is never logged or stored anywhere but inside the library's JevOptions.
    private static JevOptions BuildJevOptions(bool sendOwnFaces)
    {
        try
        {
            JevOptions options = null;

            string fileKey = ReadKeyFile();
            if (fileKey != null)
            {
                options = new JevOptions(fileKey);   // throws ArgumentException if the key has odd characters
                KeySource = "key file";
            }
            else
            {
                try
                {
                    options = JevOptions.FromEnvironment();   // Editor / desktop only
                    KeySource = "environment variable";
                }
                catch (InvalidOperationException)
                {
                    // Not set either; reported just below.
                }
            }

            if (options == null)
            {
                SetupError = "No Jev key. Paste your Typesafe key into Assets/Resources/jev_api_key.txt " +
                             $"(replace {ExperimentCondition.KeyPlaceholder}) and rebuild, or set TYPESAFE_API_KEY in the Editor.";
                return null;
            }

            options.TimeoutSeconds = JevTimeoutSeconds;
            options.MaxRetries = JevMaxRetries;
            options.LogRequests = JevLogRequests;
            options.SendOwnFacesInRequest = sendOwnFaces;
            return options;
        }
        catch (ArgumentException e)
        {
            SetupError = $"The Jev key is not usable: {e.Message}";
            return null;
        }
    }

    // The first line of the key file that is not blank and not a '#' comment, or null when the
    // file is missing, empty, or still holds the placeholder.
    private static string ReadKeyFile()
    {
        var asset = Resources.Load<TextAsset>(KeyResourceName);
        return asset == null ? null : ExperimentCondition.KeyFromFileText(asset.text);
    }
}
