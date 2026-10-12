using UnityEngine;

using VcuSwimLab.Calibration;

// The operator checkboxes for calibration.
//
// Deliberately its own component rather than more fields on `ProtocolRunner`. The runner owns
// the protocol -- the block structure, the timings, the randomisation -- and those are frozen.
// Whether a session is calibrated is a separate decision, made per session, and giving it its
// own component keeps the two from being edited together by accident.
//
// Nothing here computes anything. The calibration itself is the VREmotionDetection library's
// `MovementCalibrator`.
[DisallowMultipleComponent]
public class RuntimeCalibrationSettings : MonoBehaviour
{
    [Header("Calibration")]
    [Tooltip("Movement calibration, from the VREmotionDetection library. Learns two " +
             "fingerprints per emotion from the calibration block: the player's typical face " +
             "(the 63 face action units, averaged over each window) and the player's typical " +
             "model scores. Each evaluation tick goes to the emotion whose two fingerprints it " +
             "matches best, by the sum of the two cosines.\n\n" +
             "Off: the session records raw predictions only.\n\n" +
             "Headset, 10 sessions (mostly one person): tick macro-F1 0.737 against 0.584 raw.")]
    [SerializeField] private bool enableMovementCalibration = false;

    [Header("Debug")]
    [Tooltip("Write each evaluation tick's calibrated decision to the Unity console, " +
             "readable over `adb logcat`. Development builds only, and off by default.\n\n" +
             "Console only. Nothing is ever drawn on the headset panel: the participant sees " +
             "no prediction, no correctness and no score, because the evaluation block is the " +
             "measurement. Recorded in the session metadata either way.")]
    [SerializeField] private bool enableCalibrationDebugLogging = false;

    /// <summary>
    /// Read the toggles once. Everything afterwards reads the returned object, so changing a
    /// checkbox mid-session cannot reach the running session.
    /// </summary>
    public CalibrationSessionConfig LockForSession()
    {
        bool debug = enableCalibrationDebugLogging && Debug.isDebugBuild;

        if (enableCalibrationDebugLogging && !debug)
        {
            Debug.LogWarning($"{nameof(RuntimeCalibrationSettings)}: calibration debug logging " +
                             "is only available in a development build. It is off for this " +
                             "session and recorded as off.");
        }

        return CalibrationSessionConfig.Lock(enableMovementCalibration, debug);
    }
}
