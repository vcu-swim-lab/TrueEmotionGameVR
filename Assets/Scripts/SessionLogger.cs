using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

using UnityEngine;

// Writes one self-contained CSV row per face prediction, plus a metadata sidecar
// describing the session.
//
// The point of this file is offline replay. Each row carries the complete five-frame AU
// window that produced the prediction, so a recorded session can be pushed back through
// the same ONNX graph in Python without a headset. Logging only the newest frame would
// not be enough: the model is a temporal TCN and consumes all five.
//
// All numbers are formatted with the invariant culture. On a Windows machine with a
// Turkish (or any comma-decimal) locale the default formatting would emit "0,4213",
// which silently corrupts a comma-separated file.
public sealed class SessionLogger : IDisposable
{
    public const string StatusOk = "ok";
    public const string StatusNoTracking = "no_tracking";

    private static readonly Emotion[] ClassOrder =
    {
        Emotion.Anger, Emotion.Disgust, Emotion.Fear, Emotion.Happiness,
        Emotion.Neutral, Emotion.Sadness, Emotion.Surprise,
    };

    private readonly string sessionId;
    private readonly string sessionDirectory;
    private readonly int windowFrames;
    private readonly int featuresPerFrame;
    private readonly IReadOnlyDictionary<string, string> sessionInfo;

    private StreamWriter writer;
    private readonly StringBuilder row = new();

    public string SessionDirectory => sessionDirectory;
    public string CsvPath { get; }
    public string MetadataPath { get; }

    // `sessionInfo` is written at the top level of the metadata, alongside `session_id`
    // and `utc_start`. It is separate from `modelInfo` on purpose: how a session was run --
    // which protocol, which randomisation seed, how many trials, whether it was a rehearsal
    // -- is not a property of the model, and a reader looking for it would not think to
    // open a "models" block.
    public SessionLogger(
        int windowFrames,
        int featuresPerFrame,
        IReadOnlyDictionary<string, string> modelInfo,
        IReadOnlyDictionary<string, string> sessionInfo = null)
    {
        this.windowFrames = windowFrames;
        this.featuresPerFrame = featuresPerFrame;
        this.sessionInfo = sessionInfo;

        var startedUtc = DateTime.UtcNow;

        // Timestamped id per session: nothing is ever overwritten, and a session that
        // starts in the same second as another still gets its own directory.
        sessionId = startedUtc.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture)
                    + "_" + Guid.NewGuid().ToString("N").Substring(0, 6);

        sessionDirectory = Path.Combine(Application.persistentDataPath, "sessions", "session_" + sessionId);
        Directory.CreateDirectory(sessionDirectory);

        CsvPath = Path.Combine(sessionDirectory, "predictions.csv");
        MetadataPath = Path.Combine(sessionDirectory, "session_metadata.json");

        // UTF-8 *without* a BOM. `Encoding.UTF8` emits one, which ends up prepended to the
        // first header cell, so a plain `csv.DictReader` sees a column named "﻿timestamp_utc"
        // and every lookup of "timestamp_utc" fails. Pandas has the same problem unless the
        // caller knows to pass encoding="utf-8-sig".
        writer = new StreamWriter(CsvPath, append: false, new UTF8Encoding(false))
        {
            // Sessions are short and a lost recording costs a headset session to redo,
            // so flushing every row is the right trade.
            AutoFlush = true,
        };
        writer.WriteLine(BuildHeader());

        WriteMetadata(startedUtc, modelInfo);

        Debug.Log($"SessionLogger: writing to {CsvPath}");
    }

    private string BuildHeader()
    {
        var sb = new StringBuilder();
        sb.Append("timestamp_utc,session_id,trial_id,prompted_emotion,prediction_index,")
          .Append("predicted_emotion,correct,status,generation");

        foreach (var emotion in ClassOrder)
        {
            sb.Append(",score_").Append(emotion);
        }

        for (int f = 0; f < windowFrames; ++f)
        {
            sb.Append(",frame_").Append(f).Append("_valid");
        }

        for (int f = 0; f < windowFrames; ++f)
        {
            for (int a = 0; a < featuresPerFrame; ++a)
            {
                sb.Append(",frame_").Append(f).Append("_au_").Append(a);
            }
        }

        return sb.ToString();
    }

    // One prediction. `prediction` is null when no window of valid frames could be
    // collected, which is logged as a row with status `no_tracking` rather than dropped:
    // knowing when tracking failed is part of the diagnosis.
    public void LogPrediction(int trialId, Emotion promptedEmotion, int predictionIndex, FacePrediction? prediction)
    {
        if (writer == null)
        {
            return;
        }

        try
        {
            row.Clear();

            row.Append(DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fff", CultureInfo.InvariantCulture)).Append('Z');
            row.Append(',').Append(Escape(sessionId));
            row.Append(',').Append(trialId.ToString(CultureInfo.InvariantCulture));
            row.Append(',').Append(Escape(promptedEmotion.ToString()));
            row.Append(',').Append(predictionIndex.ToString(CultureInfo.InvariantCulture));

            if (prediction == null)
            {
                row.Append(",,0,").Append(StatusNoTracking).Append(',');

                // scores, per-frame validity, and AU values are all absent
                int blanks = ClassOrder.Length + windowFrames + (windowFrames * featuresPerFrame);
                row.Append(',', blanks);

                writer.WriteLine(row.ToString());
                return;
            }

            var value = prediction.Value;
            bool correct = value.Emotion == promptedEmotion;

            row.Append(',').Append(Escape(value.Emotion.ToString()));
            row.Append(',').Append(correct ? '1' : '0');
            row.Append(',').Append(StatusOk);
            row.Append(',').Append(value.Generation.ToString(CultureInfo.InvariantCulture));

            for (int i = 0; i < ClassOrder.Length; ++i)
            {
                row.Append(',');
                if (value.Scores != null && i < value.Scores.Length)
                {
                    row.Append(Format(value.Scores[i]));
                }
            }

            // Every frame that reaches the ring buffer is valid by construction: invalid
            // readings are dropped in DeviceManager.PollAndDispatch and never stored. The
            // columns are kept so the schema stays stable, and because `generation` is the
            // signal that actually reveals dropouts (a step smaller than expected between
            // consecutive rows means frames were discarded).
            for (int f = 0; f < windowFrames; ++f)
            {
                row.Append(",1");
            }

            for (int f = 0; f < windowFrames; ++f)
            {
                var frame = value.Window != null && f < value.Window.Length ? value.Window[f] : null;
                for (int a = 0; a < featuresPerFrame; ++a)
                {
                    row.Append(',');
                    if (frame != null && a < frame.Length)
                    {
                        row.Append(Format(frame[a]));
                    }
                }
            }

            writer.WriteLine(row.ToString());
        }
        catch (Exception e)
        {
            // Logging must never take the game down mid-session.
            Debug.LogError($"SessionLogger: failed to write row ({e.GetType().Name}: {e.Message})");
        }
    }

    private void WriteMetadata(DateTime startedUtc, IReadOnlyDictionary<string, string> modelInfo)
    {
        try
        {
            var sb = new StringBuilder();
            sb.AppendLine("{");
            sb.Append("  \"session_id\": \"").Append(JsonEscape(sessionId)).AppendLine("\",");
            sb.Append("  \"utc_start\": \"")
              .Append(startedUtc.ToString("yyyy-MM-ddTHH:mm:ss.fff", CultureInfo.InvariantCulture))
              .AppendLine("Z\",");
            sb.Append("  \"app_version\": \"").Append(JsonEscape(Application.version)).AppendLine("\",");
            sb.Append("  \"application_identifier\": \"").Append(JsonEscape(Application.identifier)).AppendLine("\",");
            sb.Append("  \"unity_version\": \"").Append(JsonEscape(Application.unityVersion)).AppendLine("\",");
            sb.Append("  \"platform\": \"").Append(JsonEscape(Application.platform.ToString())).AppendLine("\",");
            sb.Append("  \"device_model\": \"").Append(JsonEscape(SystemInfo.deviceModel)).AppendLine("\",");
            sb.Append("  \"window_frames\": ").Append(windowFrames).AppendLine(",");
            sb.Append("  \"features_per_frame\": ").Append(featuresPerFrame).AppendLine(",");

            // How this session was run, at the top level where a reader will find it.
            // Written as strings without exception, so a value never arrives as a number
            // in one recording and a string in another.
            if (sessionInfo != null)
            {
                foreach (var kv in sessionInfo)
                {
                    sb.Append("  \"").Append(JsonEscape(kv.Key)).Append("\": \"")
                      .Append(JsonEscape(kv.Value)).AppendLine("\",");
                }
            }

            sb.AppendLine("  \"class_order\": [");
            for (int i = 0; i < ClassOrder.Length; ++i)
            {
                sb.Append("    \"").Append(ClassOrder[i]).Append('"');
                sb.AppendLine(i < ClassOrder.Length - 1 ? "," : "");
            }
            sb.AppendLine("  ],");

            sb.AppendLine("  \"models\": {");
            int written = 0;
            foreach (var kv in modelInfo)
            {
                sb.Append("    \"").Append(JsonEscape(kv.Key)).Append("\": \"")
                  .Append(JsonEscape(kv.Value)).Append('"');
                sb.AppendLine(++written < modelInfo.Count ? "," : "");
            }
            sb.AppendLine("  },");

            sb.AppendLine("  \"columns\": {");
            sb.AppendLine("    \"timestamp_utc\": \"ISO-8601 UTC time the row was written\",");
            sb.AppendLine("    \"session_id\": \"identifier shared by every row of this session\",");
            sb.AppendLine("    \"trial_id\": \"0-based index of the prompt within the session\",");
            sb.AppendLine("    \"prompted_emotion\": \"emotion the player was asked to act\",");
            sb.AppendLine("    \"prediction_index\": \"0-based tick within the trial\",");
            sb.AppendLine("    \"predicted_emotion\": \"argmax class, empty when status is no_tracking\",");
            sb.AppendLine("    \"correct\": \"1 when predicted_emotion equals prompted_emotion\",");
            sb.AppendLine("    \"status\": \"ok, or no_tracking when no valid window was available\",");
            sb.AppendLine("    \"generation\": \"monotonic count of valid frames pushed into the ring buffer\",");
            sb.AppendLine("    \"score_<Class>\": \"model output for that class, in class_order\",");
            sb.AppendLine("    \"frame_<f>_valid\": \"always 1 for logged windows; invalid frames are dropped upstream\",");
            sb.AppendLine("    \"frame_<f>_au_<a>\": \"action unit a of window frame f, frame 0 oldest\"");
            sb.AppendLine("  },");
            sb.AppendLine("  \"notes\": [");
            sb.AppendLine("    \"Face scores are probabilities: the face graph already ends in Softmax and no second one is appended.\",");
            sb.AppendLine("    \"Frame 0 is the oldest frame of the window and frame 4 the newest.\",");
            sb.AppendLine("    \"The window is an exact copy of the tensor input for that prediction.\",");
            sb.AppendLine("    \"All decimals use invariant culture, so '.' is the decimal separator.\"");
            sb.AppendLine("  ]");
            sb.AppendLine("}");

            // No BOM here either: json.load rejects a leading BOM.
            File.WriteAllText(MetadataPath, sb.ToString(), new UTF8Encoding(false));
        }
        catch (Exception e)
        {
            Debug.LogError($"SessionLogger: failed to write metadata ({e.GetType().Name}: {e.Message})");
        }
    }

    private static string Format(float value)
    {
        // G9 round-trips a Single exactly.
        return value.ToString("G9", CultureInfo.InvariantCulture);
    }

    private static string Escape(string value)
    {
        if (string.IsNullOrEmpty(value))
        {
            return string.Empty;
        }

        if (value.IndexOfAny(new[] { ',', '"', '\n', '\r' }) < 0)
        {
            return value;
        }

        return "\"" + value.Replace("\"", "\"\"") + "\"";
    }

    private static string JsonEscape(string value)
    {
        if (string.IsNullOrEmpty(value))
        {
            return string.Empty;
        }

        return value.Replace("\\", "\\\\").Replace("\"", "\\\"")
                    .Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t");
    }

    public void Dispose()
    {
        if (writer == null)
        {
            return;
        }

        try
        {
            writer.Flush();
            writer.Dispose();
        }
        catch (Exception e)
        {
            Debug.LogError($"SessionLogger: failed to close ({e.GetType().Name}: {e.Message})");
        }
        finally
        {
            writer = null;
        }
    }
}
