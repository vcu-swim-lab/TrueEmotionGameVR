using System;
using System.IO;
using UnityEngine;
using VcuSwimLab.Calibration;
using VcuSwimLab.Jev;

// Asks Typesafe's Jev model about the same five-frame window the local model just scored.
//
// TEMPORARY: the Jev classes in this folder are a copy of VREmotionDetection's Runtime/Jev,
// which the library has not published yet. Once it has, this component is the only file that
// changes (it forwards to NewFaceAuModel.Backend instead); see README.md beside it.
//
// The local model still scores every tick. Its scores are what calibration is built from and
// what predictions.csv records, so a session stays replayable through the ONNX graph; Jev's
// answer is returned next to it and is logged separately (SessionLogger.LogJev).
//
// A failed Jev call (no key, no network, rate limit) returns null and the caller keeps the
// local prediction, so a missing key leaves the game exactly as it was.
//
// The player's calibration reaches Jev the way it does in the library: when a finished
// calibration exists, the request carries the player's own movement centroids
// (`players_own_faces`), and Jev judges against how this player makes each emotion. The library
// keeps its calibrator private, so the profile is rebuilt here from the same samples (the study
// protocol) or read from the file the library saved (the game).
public class JevScorer : MonoBehaviour
{
    [Tooltip("Ask Jev about each prediction. On by default while Jev is being tested; untick it " +
             "for the local model alone. Face tracking data leaves the device while this is on.")]
    [SerializeField]
    private bool useJev = true;

    [Tooltip("Leave empty to use defaults with the TYPESAFE_API_KEY environment variable. " +
             "To use a key from an asset instead: Create > VR Emotion > Jev Settings, and do not commit it.")]
    [SerializeField]
    private JevSettings settings;

    private string lastError;
    private string lastWarning;

    // The player's finished calibration, sent with every request when there is one.
    private CalibrationProfile profile;

    // The protocol's calibration in progress: fed the same samples as the predictor's own.
    private MovementCalibrator protocolCalibrator;

    // The layout the library saves a calibration in (NewFaceAuModel.SavedCalibration, which is
    // private there). Matrices are flattened row by row.
    [Serializable]
    private class SavedProfile
    {
        public string implementationVersion;
        public string[] classNames;
        public int scoreWidth;
        public double[] scoreCentroids;
        public int movementWidth;
        public double[] movementCentroids;
    }

    public bool Enabled => useJev;

    // True when the last Score call sent the player's calibration to Jev.
    public bool ProfileSent { get; private set; }

    // Why the last Score call returned null; null when it worked or Jev is off.
    public string LastError => lastError;

    public string Model => Settings.Model;

    public string BaseUrl => Settings.BaseUrl;

    // Explicit null check: JevSettings is a UnityEngine.Object, so `??=` would bypass its ==.
    private JevSettings Settings
    {
        get
        {
            if (settings == null)
            {
                settings = ScriptableObject.CreateInstance<JevSettings>();
            }

            return settings;
        }
    }

    // The scorer on `host`, added when the scene does not have one, so Jev needs no scene edit.
    public static JevScorer GetOrAdd(GameObject host)
    {
        var scorer = host.GetComponent<JevScorer>();
        if (scorer == null)
        {
            scorer = host.AddComponent<JevScorer>();
        }

        return scorer;
    }

    // Jev's seven scores in Emotion order, or null when Jev is off or the call failed.
    public async Awaitable<float[]> Score(FacePrediction prediction)
    {
        lastError = null;

        if (!useJev)
        {
            return null;
        }

        try
        {
            ProfileSent = profile != null;
            var result = await JevClient.Classify(Settings, prediction.Window, profile);

            lastError = result.Error;
            Warn(result.Error);
            return result.Scores;
        }
        catch (Exception e)
        {
            // A failure here must not stop a round or a study session.
            lastError = $"{e.GetType().Name}: {e.Message}";
            Warn(lastError);
            return null;
        }
    }

    // ---- Study protocol: mirrors the predictor's StartCalibration, BeginCalibrationEmotion,
    // AddCalibrationSample and FinishCalibration, with the same scores and windows, so the
    // profile is the one the predictor builds. Call each right beside its predictor call.

    public void StartProtocolCalibration()
    {
        profile = null;
        protocolCalibrator = new MovementCalibrator(Enum.GetNames(typeof(Emotion)));
    }

    public void BeginProtocolEmotion(Emotion emotion)
    {
        protocolCalibrator?.BeginEmotion((int)emotion);
    }

    public void AddProtocolSample(int position, FacePrediction? prediction)
    {
        protocolCalibrator?.ObserveAt(position, prediction?.Scores, prediction?.Window);
    }

    public void FinishProtocolCalibration()
    {
        profile = protocolCalibrator != null && protocolCalibrator.Finish()
            ? protocolCalibrator.ExportProfile()
            : null;
        protocolCalibrator = null;
    }

    // ---- Game: after the predictor saved or loaded this player's calibration.

    public void LoadProfile(string userId)
    {
        profile = null;
        protocolCalibrator = null;

        try
        {
            var saved = JsonUtility.FromJson<SavedProfile>(File.ReadAllText(NewFaceAuModel.CalibrationPath(userId)));
            profile = new CalibrationProfile(saved.implementationVersion, saved.classNames,
                                             Unflatten(saved.scoreCentroids, saved.scoreWidth),
                                             Unflatten(saved.movementCentroids, saved.movementWidth));
        }
        catch (Exception e)
        {
            // Jev then judges without the player's calibration; the local model is unaffected.
            Debug.LogWarning($"{nameof(JevScorer)}: could not read the saved calibration for Jev ({e.Message}).");
        }
    }

    public void ClearProfile()
    {
        profile = null;
        protocolCalibrator = null;
    }

    private static double[][] Unflatten(double[] flat, int width)
    {
        var matrix = new double[flat.Length / width][];
        for (int r = 0; r < matrix.Length; ++r)
        {
            matrix[r] = new double[width];
            Array.Copy(flat, r * width, matrix[r], 0, width);
        }

        return matrix;
    }

    // The emotion with the highest score.
    public static Emotion Decide(float[] scores)
    {
        int best = 0;
        for (int i = 1; i < scores.Length; ++i)
        {
            if (scores[i] > scores[best])
            {
                best = i;
            }
        }

        return (Emotion)best;
    }

    // Once per distinct problem, not once per tick: a missing key would otherwise fill the
    // console and the headset log with the same line every second.
    private void Warn(string error)
    {
        if (error == lastWarning)
        {
            return;
        }

        lastWarning = error;
        if (error != null)
        {
            Debug.LogWarning($"{nameof(JevScorer)}: Jev did not answer ({error}); using the local model.");
        }
    }
}
