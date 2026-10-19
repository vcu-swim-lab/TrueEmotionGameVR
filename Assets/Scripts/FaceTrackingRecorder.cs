using System;
using System.Globalization;
using System.IO;
using System.Text;

using UnityEngine;

// The game's face tracking file (ProtocolRunner's Save Face Tracking checkbox): the 70 Meta
// face-tracking weights every frame, for replaying the face on an avatar afterwards. Time 0 is
// the first get-ready screen, which is also visible in a recording of the headset view, so the
// two can be lined up without a sync marker.
//
// One row per frame: the UTC time of day (as the session log writes it), time since the
// start, which screen was showing, the trial, whether
// tracking held, and the weights in `OVRFaceExpressions.FaceExpression` order -- the same
// values and order the model reads. A frame without tracking keeps its row, with the weights
// left blank, so a dropout shows as a gap rather than as a jump in time.
public class FaceTrackingRecorder : IDisposable
{
    public const string FileName = "face_tracking.csv";

    // Buffered, not flushed every row as the session logger is: about 70 rows a second would
    // otherwise mean a disk write every frame. Flushed every few seconds and on close, so a
    // crash loses at most that much.
    private const int BufferBytes = 64 * 1024;
    private const float FlushSeconds = 5f;

    private readonly OVRFaceExpressions face;
    private readonly float[] weights = new float[(int)OVRFaceExpressions.FaceExpression.Max];
    private readonly StringBuilder row = new();
    private StreamWriter writer;

    private float startTime;
    private DateTime startUtc;
    private float nextFlush;
    private bool started;

    private string screen = "";
    private string trialId = "";
    private string emotion = "";

    public string Path { get; }

    public FaceTrackingRecorder(OVRFaceExpressions face, string directory)
    {
        this.face = face;
        Path = System.IO.Path.Combine(directory, FileName);

        // UTF-8 without a BOM, as the session logger writes, so a plain CSV reader sees the
        // first column's name as written.
        writer = new StreamWriter(Path, append: false, new UTF8Encoding(false), BufferBytes);

        var header = new StringBuilder("timestamp_utc,time_s,frame,screen,trial_id,emotion,valid");
        for (int i = 0; i < weights.Length; ++i)
        {
            header.Append(',').Append((OVRFaceExpressions.FaceExpression)i);
        }
        writer.WriteLine(header.ToString());
    }

    // Time 0, and the first row. Called once, when the first get-ready screen appears; later
    // calls change nothing.
    public void Begin()
    {
        if (started || writer == null)
        {
            return;
        }

        started = true;
        startTime = Time.time;
        startUtc = DateTime.UtcNow;
        nextFlush = startTime + FlushSeconds;
    }

    // What is on screen from now on: get_ready, act, result or part, and the trial it belongs to.
    public void SetScreen(string screenName, int? trial = null, string emotionName = null)
    {
        screen = screenName;
        trialId = trial?.ToString(CultureInfo.InvariantCulture) ?? "";
        emotion = emotionName ?? "";
    }

    // Once a frame, after the face tracking has updated.
    public void RecordFrame()
    {
        if (!started || writer == null)
        {
            return;
        }

        bool valid = face != null && face.ValidExpressions;

        // The wall clock, in the session log's format, so the file lines up with a video by the
        // time of day; counted on from the start rather than read each frame, so it never
        // disagrees with time_s.
        float elapsed = Time.time - startTime;
        row.Clear();
        row.Append(startUtc.AddSeconds(elapsed).ToString("yyyy-MM-dd'T'HH:mm:ss.fff'Z'", CultureInfo.InvariantCulture)).Append(',')
           .Append(elapsed.ToString("F4", CultureInfo.InvariantCulture)).Append(',')
           .Append(Time.frameCount).Append(',')
           .Append(screen).Append(',')
           .Append(trialId).Append(',')
           .Append(emotion).Append(',')
           .Append(valid ? "true" : "false");

        if (valid)
        {
            face.CopyTo(weights);
            for (int i = 0; i < weights.Length; ++i)
            {
                row.Append(',').Append(weights[i].ToString("F4", CultureInfo.InvariantCulture));
            }
        }
        else
        {
            row.Append(',', weights.Length);
        }

        writer.WriteLine(row);

        if (Time.time >= nextFlush)
        {
            writer.Flush();
            nextFlush = Time.time + FlushSeconds;
        }
    }

    public void Dispose()
    {
        writer?.Dispose();
        writer = null;
    }
}
