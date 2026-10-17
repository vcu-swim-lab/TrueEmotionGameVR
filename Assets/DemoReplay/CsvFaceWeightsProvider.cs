using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

using Meta.XR.Movement.FaceTracking.Samples;
using UnityEngine;

// Plays a demo's face_tracking.csv back on a Movement SDK character, in place of the live
// headset (OVRWeightsProvider): the same 70 weights, by the same names, frame by frame at the
// recorded times. The 8 eye-direction weights, which the character's face shapes do not use,
// turn its eye bones instead. Used by the avatar replay (VR Emotion > Avatar Replay), never
// on the headset.
//
// Runs before the face driver each frame, so the face shows this frame's weights.
[DefaultExecutionOrder(-100)]
public class CsvFaceWeightsProvider : WeightsProvider
{
    [Tooltip("The demo's face_tracking.csv.")]
    [SerializeField] private string csvPath;

    [Header("Eyes")]
    [SerializeField] private Transform head;
    [SerializeField] private Transform leftEye;
    [SerializeField] private Transform rightEye;
    [Tooltip("Degrees an eye turns at a look weight of 1.")]
    [SerializeField] private float maxEyeYaw = 30f;
    [SerializeField] private float maxEyePitch = 25f;

    private string[] names;
    private float[] times;
    private float[][] frames;
    private float[] current;
    private int cursor;

    private float startTime;
    private bool playing;

    private Quaternion leftEyeRest;
    private Quaternion rightEyeRest;
    private int lookLeftL, lookRightL, lookUpL, lookDownL;
    private int lookLeftR, lookRightR, lookUpR, lookDownR;

    public string CsvPath
    {
        get => csvPath;
        set => csvPath = value;
    }

    public void SetEyes(Transform headBone, Transform left, Transform right)
    {
        head = headBone;
        leftEye = left;
        rightEye = right;
    }

    // Seconds of face data: the end of the replay.
    public float Duration
    {
        get
        {
            EnsureLoaded();
            return times.Length > 0 ? times[^1] : 0f;
        }
    }

    public bool Finished => playing && Time.time - startTime > Duration;

    // Time 0 of the file, from now. Until then the face holds the first frame.
    public void Play()
    {
        startTime = Time.time;
        playing = true;
    }

    public override bool IsValid
    {
        get
        {
            EnsureLoaded();
            return names != null;
        }
    }

    public override float[] GetWeights()
    {
        EnsureLoaded();
        return current;
    }

    public override string[] GetWeightNames()
    {
        EnsureLoaded();
        return names;
    }

    private void Start()
    {
        if (leftEye != null) leftEyeRest = leftEye.localRotation;
        if (rightEye != null) rightEyeRest = rightEye.localRotation;
    }

    private void Update()
    {
        EnsureLoaded();
        if (frames.Length == 0)
        {
            return;
        }

        float t = playing ? Time.time - startTime : 0f;
        while (cursor + 1 < times.Length && times[cursor + 1] <= t)
        {
            ++cursor;
        }

        // Between two recorded frames, blend them, so a render at another frame rate stays smooth.
        int next = Mathf.Min(cursor + 1, times.Length - 1);
        float span = times[next] - times[cursor];
        float k = span > 0f ? Mathf.Clamp01((t - times[cursor]) / span) : 0f;
        var a = frames[cursor];
        var b = frames[next];
        for (int i = 0; i < current.Length; ++i)
        {
            current[i] = Mathf.Lerp(a[i], b[i], k);
        }
    }

    private void LateUpdate()
    {
        TurnEye(leftEye, leftEyeRest, lookLeftL, lookRightL, lookUpL, lookDownL);
        TurnEye(rightEye, rightEyeRest, lookLeftR, lookRightR, lookUpR, lookDownR);
    }

    // The look weights are the person's own directions: "left" is their left, which faces the
    // character's left too.
    private void TurnEye(Transform eye, Quaternion rest, int left, int right, int up, int down)
    {
        if (eye == null || current == null || left < 0 || right < 0 || up < 0 || down < 0)
        {
            return;
        }

        var basis = head != null ? head : transform;
        float yaw = (current[right] - current[left]) * maxEyeYaw;
        float pitch = (current[up] - current[down]) * maxEyePitch;
        eye.localRotation = rest;
        eye.rotation = Quaternion.AngleAxis(yaw, basis.up) * Quaternion.AngleAxis(-pitch, basis.right) * eye.rotation;
    }

    private void EnsureLoaded()
    {
        if (names != null)
        {
            return;
        }

        var rowTimes = new List<float>();
        var rowWeights = new List<float[]>();
        string[] header = null;
        int timeColumn = -1, validColumn = -1, firstWeight = -1;
        float[] lastValid = null;

        if (!string.IsNullOrEmpty(csvPath) && File.Exists(csvPath))
        {
            foreach (var line in File.ReadLines(csvPath))
            {
                var cells = line.Split(',');
                if (header == null)
                {
                    header = cells;
                    timeColumn = Array.IndexOf(header, "time_s");
                    validColumn = Array.IndexOf(header, "valid");
                    firstWeight = validColumn + 1;
                    continue;
                }

                if (cells.Length < header.Length)
                {
                    continue;
                }

                // A frame without tracking holds the last tracked face rather than snapping to zero.
                float[] weights;
                if (cells[validColumn] == "true")
                {
                    weights = new float[header.Length - firstWeight];
                    for (int i = 0; i < weights.Length; ++i)
                    {
                        float.TryParse(cells[firstWeight + i], NumberStyles.Float, CultureInfo.InvariantCulture,
                                       out weights[i]);
                    }
                    lastValid = weights;
                }
                else
                {
                    weights = lastValid ?? new float[header.Length - firstWeight];
                }

                float.TryParse(cells[timeColumn], NumberStyles.Float, CultureInfo.InvariantCulture, out var time);
                rowTimes.Add(time);
                rowWeights.Add(weights);
            }
        }
        else
        {
            Debug.LogError($"{nameof(CsvFaceWeightsProvider)}: face file not found: '{csvPath}'");
        }

        names = header != null && firstWeight > 0 ? header[firstWeight..] : Array.Empty<string>();
        times = rowTimes.ToArray();
        frames = rowWeights.ToArray();
        current = frames.Length > 0 ? (float[])frames[0].Clone() : new float[names.Length];

        lookLeftL = Array.IndexOf(names, "EyesLookLeftL");
        lookRightL = Array.IndexOf(names, "EyesLookRightL");
        lookUpL = Array.IndexOf(names, "EyesLookUpL");
        lookDownL = Array.IndexOf(names, "EyesLookDownL");
        lookLeftR = Array.IndexOf(names, "EyesLookLeftR");
        lookRightR = Array.IndexOf(names, "EyesLookRightR");
        lookUpR = Array.IndexOf(names, "EyesLookUpR");
        lookDownR = Array.IndexOf(names, "EyesLookDownR");

        Debug.Log($"{nameof(CsvFaceWeightsProvider)}: {frames.Length} frames, {names.Length} weights, " +
                  $"{Duration:F1} s from {csvPath}");
    }
}
