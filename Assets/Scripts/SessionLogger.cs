using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

using UnityEngine;

using VcuSwimLab.Backends;
using VcuSwimLab.Calibration;

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
// Which block of the session a trial belongs to. The analysis splits on this: centroids are
// fitted on Calibration trials and scored on Evaluation trials, which are separate trials of
// the same state rather than later ticks of the same held expression.
public enum BlockRole
{
    Calibration,
    Evaluation,
}

public sealed class SessionLogger : IDisposable
{
    public const string StatusOk = "ok";
    public const string StatusNoTracking = "no_tracking";

    // The backend itself failed (Jev unreachable, key rejected, ...), as opposed to the headset
    // not delivering a face. Kept apart from `no_tracking` so a service outage is never
    // mistaken for lost face tracking. `error_code` says which failure.
    public const string StatusBackendError = "backend_error";

    // Canonical on-disk spellings, written instead of `blockRole.ToString()`. The enum member
    // names are capitalised and renaming one would silently change the file vocabulary that
    // the Python side matches on; going through this map makes that a compile error instead.
    public const string RoleCalibration = "calibration";
    public const string RoleEvaluation = "evaluation";

    private static string Canonical(BlockRole role) => role switch
    {
        BlockRole.Calibration => RoleCalibration,
        BlockRole.Evaluation => RoleEvaluation,
        _ => throw new ArgumentOutOfRangeException(nameof(role), role, "unmapped block role"),
    };

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

    // Null for the scored game, which has no calibration and no blocks. When it is null the
    // CSV, the metadata and the file set are exactly what they were before runtime calibration
    // existed -- a raw-mode *protocol* session still gets the columns, carrying
    // `calibration_mode=raw`, because "this session was not calibrated" is a fact worth
    // recording rather than one to be inferred from missing columns.
    private readonly CalibrationSessionConfig calibration;

    // Only known once the calibration block closes, so the metadata is rewritten then.
    private readonly DateTime startedUtc;
    private readonly IReadOnlyDictionary<string, string> modelInfo;
    private IReadOnlyDictionary<string, string> calibrationOutcome;

    private StreamWriter writer;
    private StreamWriter trialWriter;
    private readonly StringBuilder row = new();

    public string SessionDirectory => sessionDirectory;
    public string CsvPath { get; }
    public string MetadataPath { get; }

    /// <summary>The trial-level file, or null when this session records no calibration.</summary>
    public string TrialCsvPath { get; }

    // `sessionInfo` is written at the top level of the metadata, alongside `session_id`
    // and `utc_start`. It is separate from `modelInfo` on purpose: how a session was run --
    // which protocol, which randomisation seed, how many trials, whether it was a rehearsal
    // -- is not a property of the model, and a reader looking for it would not think to
    // open a "models" block.
    public SessionLogger(
        int windowFrames,
        int featuresPerFrame,
        IReadOnlyDictionary<string, string> modelInfo,
        IReadOnlyDictionary<string, string> sessionInfo = null,
        CalibrationSessionConfig calibration = null)
    {
        this.windowFrames = windowFrames;
        this.featuresPerFrame = featuresPerFrame;
        this.sessionInfo = sessionInfo;
        this.calibration = calibration;
        this.modelInfo = modelInfo;

        var startedUtc = DateTime.UtcNow;
        this.startedUtc = startedUtc;

        // Timestamped id per session: nothing is ever overwritten, and a session that
        // starts in the same second as another still gets its own directory.
        sessionId = startedUtc.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture)
                    + "_" + Guid.NewGuid().ToString("N").Substring(0, 6);

        sessionDirectory = Path.Combine(Application.persistentDataPath, "sessions", "session_" + sessionId);
        Directory.CreateDirectory(sessionDirectory);

        CsvPath = Path.Combine(sessionDirectory, "predictions.csv");
        MetadataPath = Path.Combine(sessionDirectory, "session_metadata.json");
        TrialCsvPath = calibration != null
            ? Path.Combine(sessionDirectory, "trials.csv")
            : null;

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

        if (TrialCsvPath != null)
        {
            trialWriter = new StreamWriter(TrialCsvPath, append: false, new UTF8Encoding(false))
            {
                AutoFlush = true,
            };
            trialWriter.WriteLine(BuildTrialHeader());
        }

        WriteMetadata();

        Debug.Log($"SessionLogger: writing to {CsvPath}");
    }

    private string BuildHeader()
    {
        var sb = new StringBuilder();
        sb.Append("timestamp_utc,session_id,trial_id,block_role,repeat_index,")
          .Append("prompted_emotion,prediction_index,")
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

        // Appended after the action units rather than inserted next to `predicted_emotion`,
        // where a reader would rather find them. Every existing consumer looks columns up by
        // name -- `session_analysis.load_session` uses a DictReader -- so position does not
        // matter to a program, but inserting would move 350 columns for no gain and any tool
        // that did depend on position would break silently. `trials.csv` is the file meant to
        // be read by eye.
        foreach (string column in CalibrationColumns)
        {
            sb.Append(',').Append(column);
        }

        // Always last, after the calibration columns, for the same reason: appended, so no
        // existing column moves. Which backend scored the row, why it failed if it did, and
        // the backend's own answers before any normalisation (Jev; blank for SparseVR).
        sb.Append(",backend_id,error_code");
        foreach (var emotion in ClassOrder)
        {
            sb.Append(",raw_").Append(emotion);
        }

        return sb.ToString();
    }

    /// <summary>
    /// The per-tick calibration columns, in order. Empty when the session records no
    /// calibration, which is how the scored game keeps its original schema.
    /// </summary>
    private string[] CalibrationColumns => calibration == null
        ? Array.Empty<string>()
        : new[]
        {
            "calibration_mode",
            "personal_calibration_enabled",
            "population_calibration_enabled",
            "movement_calibration_enabled",
            "hybrid_alpha",
            "raw_predicted_emotion",
            "calibrated_predicted_emotion",
            "calibrated_winner_similarity",
            "calibrated_runner_up_similarity",
            "calibrated_margin",
            "calibration_valid",
            "calibration_error",
        };

    private string BuildTrialHeader()
        => "session_id,trial_id,block_role,repeat_index,prompted_emotion,"
           + "calibration_mode,hybrid_alpha,"
           + "raw_trial_prediction,raw_trial_correct,raw_scored_ticks,raw_trial_tie,"
           + "calibrated_trial_prediction,calibrated_trial_correct,"
           + "calibrated_scored_ticks,calibrated_trial_tie,"
           + "calibration_valid,calibration_error";

    // One prediction from a study session, tagged with the block it belongs to.
    //
    // `prediction` is null when no window of valid frames could be collected, which is
    // logged as a row with status `no_tracking` rather than dropped: knowing when tracking
    // failed is part of the diagnosis.
    /// <param name="calibrated">
    /// The calibrated decision for this tick, or null when there is none: raw mode, the
    /// calibration block, an invalid profile, or an untracked tick. A null is written as blank
    /// calibrated columns rather than as a fallback prediction, so a reader can never mistake
    /// an uncalibrated tick for a calibrated one.
    /// </param>
    /// <param name="error">
    /// Why `prediction` is null, when the predictor said so (<c>PredictionOutcome.Error</c>).
    /// Null with a null prediction is written as plain `no_tracking`, as before.
    /// </param>
    public void LogPrediction(int trialId, BlockRole blockRole, int repeatIndex,
                              Emotion promptedEmotion, int predictionIndex,
                              FacePrediction? prediction,
                              NearestCentroid? calibrated = null,
                              PredictionError? error = null)
        => LogRow(trialId,
                  Canonical(blockRole),
                  repeatIndex.ToString(CultureInfo.InvariantCulture),
                  promptedEmotion, predictionIndex, prediction, calibrated, error);

    // One prediction from a session that has no calibration and no evaluation block -- the
    // scored game. `block_role` and `repeat_index` are left empty rather than guessed, so an
    // analysis that splits on them refuses the file outright instead of silently treating a
    // game recording as study data.
    public void LogPrediction(int trialId, Emotion promptedEmotion, int predictionIndex,
                              FacePrediction? prediction, PredictionError? error = null)
        => LogRow(trialId, string.Empty, string.Empty,
                  promptedEmotion, predictionIndex, prediction, null, error);

    // What a session's metadata says about which condition produced the predictions. Both game
    // scripts call this instead of each spelling it out. Everything is read from the predictor
    // (what the library says is active), not from which block of ConditionSetup was enabled,
    // and the API key is never written.
    public static void AddFaceBackendInfo(IDictionary<string, string> modelInfo, NewFaceAuModel predictor)
    {
        bool jev = ConditionSetup.UsesJev(predictor);

        modelInfo["condition"] = ConditionSetup.Id(predictor);
        modelInfo["condition_label"] = ConditionSetup.Label(predictor);
        modelInfo["face_backend"] = ConditionSetup.BackendName(predictor);
        modelInfo["backend_id"] = predictor.ActiveBackendId ?? string.Empty;
        modelInfo["use_calibration"] = predictor.UseCalibration ? "true" : "false";

        if (jev)
        {
            modelInfo["jev_model"] = ConditionSetup.JevModel ?? string.Empty;
            modelInfo["jev_base_url"] = ConditionSetup.JevBaseUrl ?? string.Empty;
            modelInfo["jev_timeout_s"] = ConditionSetup.JevTimeout ?? string.Empty;
            modelInfo["jev_max_retries"] = ConditionSetup.JevRetries ?? string.Empty;
            modelInfo["jev_send_own_faces"] = ConditionSetup.JevSendsOwnFaces ?? string.Empty;
            modelInfo["jev_note"] = "score_<Class> are Jev's seven answers divided by their sum; raw_<Class> are the answers as Jev gave them. A failed Jev call is a backend_error row; it is never answered by another model.";
        }
    }

    private void LogRow(int trialId, string blockRole, string repeatIndex,
                        Emotion promptedEmotion, int predictionIndex,
                        FacePrediction? prediction,
                        NearestCentroid? calibrated,
                        PredictionError? error)
    {
        if (writer == null)
        {
            return;
        }

        try
        {
            row.Clear();

            // `block_role` and `repeat_index` come from the plan, so they are written even
            // on a no_tracking row: a dropout must still be attributable to the block and
            // repeat it happened in. That also keeps the blank count below unchanged.
            row.Append(DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fff", CultureInfo.InvariantCulture)).Append('Z');
            row.Append(',').Append(Escape(sessionId));
            row.Append(',').Append(trialId.ToString(CultureInfo.InvariantCulture));
            row.Append(',').Append(blockRole);
            row.Append(',').Append(repeatIndex);
            row.Append(',').Append(Escape(promptedEmotion.ToString()));
            row.Append(',').Append(predictionIndex.ToString(CultureInfo.InvariantCulture));

            if (prediction == null)
            {
                // `no_tracking` unless the predictor named a different failure; that one is
                // `backend_error`, with its code in `error_code`.
                bool backendFailed = error.HasValue && error.Value.Code != PredictionErrorCode.NoTracking;
                row.Append(",,0,").Append(backendFailed ? StatusBackendError : StatusNoTracking).Append(',');

                // scores, per-frame validity, and AU values are all absent
                int blanks = ClassOrder.Length + windowFrames + (windowFrames * featuresPerFrame);
                row.Append(',', blanks);

                // The session-level calibration fields are known even here; an untracked tick
                // still belongs to a session that was configured one way. Only the per-tick
                // decision is missing, and it is missing because there was nothing to decide.
                AppendCalibration(null, null);
                AppendBackend(error?.BackendId, error, null);

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

            AppendCalibration(value.Emotion, calibrated);
            AppendBackend(value.BackendId, null, value.Raw);

            writer.WriteLine(row.ToString());
        }
        catch (Exception e)
        {
            // Logging must never take the game down mid-session.
            Debug.LogError($"SessionLogger: failed to write row ({e.GetType().Name}: {e.Message})");
        }
    }

    // The backend tail of a prediction row: backend_id, error_code, raw_<Class> x 7.
    // `raw` is the backend's own answers (Jev); null for SparseVR, written as blanks.
    private void AppendBackend(string backendId, PredictionError? error, float[] raw)
    {
        row.Append(',').Append(Escape(backendId));
        row.Append(',').Append(error.HasValue ? error.Value.Code.ToString() : string.Empty);

        for (int i = 0; i < ClassOrder.Length; ++i)
        {
            row.Append(',');
            if (raw != null && i < raw.Length)
            {
                row.Append(Format(raw[i]));
            }
        }
    }

    // ------------------------------------------------------------------- calibration

    private string CalibrationValidText
        => calibrationOutcome != null && calibrationOutcome.TryGetValue("calibration_valid", out string v)
            ? v
            : string.Empty;

    private string CalibrationErrorText
        => calibrationOutcome != null && calibrationOutcome.TryGetValue("calibration_error", out string v)
            ? v
            : string.Empty;

    /// <summary>
    /// The calibration tail of a prediction row.
    /// </summary>
    /// <remarks>
    /// Writes nothing at all when the session records no calibration, which is what keeps the
    /// scored game's CSV byte-identical to the one it wrote before any of this existed.
    ///
    /// `calibration_valid` and `calibration_error` are blank on calibration-block rows and
    /// filled on evaluation-block rows, because that is literally when they became known: the
    /// profile is sealed between the two blocks. The metadata carries the final answer for a
    /// reader who wants it without scanning rows.
    /// </remarks>
    private void AppendCalibration(Emotion? rawPrediction, NearestCentroid? calibrated)
    {
        if (calibration == null)
        {
            return;
        }

        row.Append(',').Append(calibration.ModeName);
        // Personal and population calibration were removed; their columns stay, always
        // "false", so recordings from before and after read the same way. Likewise alpha.
        row.Append(",false,false");
        row.Append(',').Append(calibration.MovementEnabled ? "true" : "false");
        row.Append(',');

        // An explicit alias for `predicted_emotion`. Once calibrated columns sit beside it,
        // "predicted" stops being self-explanatory, and a reader comparing methods should not
        // have to know which of the two columns is the untouched model.
        row.Append(',').Append(rawPrediction.HasValue
            ? Escape(rawPrediction.Value.ToString())
            : string.Empty);

        if (calibrated.HasValue)
        {
            var decision = calibrated.Value;
            row.Append(',').Append(Escape(ClassOrder[decision.ClassIndex].ToString()));
            row.Append(',').Append(Format(decision.WinnerSimilarity));
            row.Append(',').Append(Format(decision.RunnerUpSimilarity));
            row.Append(',').Append(Format(decision.Margin));
        }
        else
        {
            // Four blank fields: no calibrated class, no similarities, no margin. Deliberately
            // not the raw prediction repeated -- that would make an uncalibrated tick
            // indistinguishable from one the calibration happened to agree with.
            row.Append(",,,,");
        }

        row.Append(',').Append(CalibrationValidText);
        row.Append(',').Append(Escape(CalibrationErrorText));
    }

    /// <summary>
    /// Record what happened when the calibration block closed, and rewrite the metadata.
    /// </summary>
    /// <remarks>
    /// Called once, between the two blocks. Everything here is only knowable then: whether
    /// nine ticks arrived for all seven states, whether the centroids came out finite, and
    /// what went wrong if not. Rewriting the whole metadata file rather than appending keeps
    /// it a single valid JSON document at every moment, which matters because a session that
    /// is interrupted mid-way still has to be readable.
    /// </remarks>
    public void RecordCalibrationOutcome(IReadOnlyDictionary<string, string> outcome)
    {
        calibrationOutcome = outcome;
        WriteMetadata();
    }

    /// <summary>One trial's raw and calibrated majority votes.</summary>
    /// <remarks>
    /// A second file rather than more columns on the prediction rows. A trial-level fact
    /// repeated across ten tick rows is a fact that can disagree with itself, and this is the
    /// file a person actually opens to see what a session did.
    /// </remarks>
    public void LogTrial(int trialId, BlockRole blockRole, int repeatIndex,
                         Emotion promptedEmotion,
                         TrialVote raw, TrialVote calibrated)
    {
        if (trialWriter == null)
        {
            return;
        }

        try
        {
            row.Clear();
            row.Append(Escape(sessionId));
            row.Append(',').Append(trialId.ToString(CultureInfo.InvariantCulture));
            row.Append(',').Append(Canonical(blockRole));
            row.Append(',').Append(repeatIndex.ToString(CultureInfo.InvariantCulture));
            row.Append(',').Append(Escape(promptedEmotion.ToString()));
            row.Append(',').Append(calibration.ModeName);
            row.Append(',');

            AppendTrialVote(raw, promptedEmotion);
            AppendTrialVote(calibrated, promptedEmotion);

            row.Append(',').Append(CalibrationValidText);
            row.Append(',').Append(Escape(CalibrationErrorText));

            trialWriter.WriteLine(row.ToString());
        }
        catch (Exception e)
        {
            Debug.LogError($"SessionLogger: failed to write trial row " +
                           $"({e.GetType().Name}: {e.Message})");
        }
    }

    private void AppendTrialVote(TrialVote vote, Emotion prompted)
    {
        if (!vote.Scorable)
        {
            // prediction, correct, scored ticks, tie
            row.Append(",,,").Append(vote.ScoredTicks.ToString(CultureInfo.InvariantCulture))
               .Append(',');
            return;
        }

        var predicted = ClassOrder[vote.ClassIndex];
        row.Append(',').Append(Escape(predicted.ToString()));
        row.Append(',').Append(predicted == prompted ? '1' : '0');
        row.Append(',').Append(vote.ScoredTicks.ToString(CultureInfo.InvariantCulture));
        row.Append(',').Append(vote.TieOccurred ? "true" : "false");
    }

    private void WriteMetadata()
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

            WriteCalibrationMetadata(sb);

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
            sb.AppendLine("    \"block_role\": \"calibration or evaluation; which block this trial belongs to\",");
            sb.AppendLine("    \"repeat_index\": \"1-based presentation of this emotion within its block role\",");
            sb.AppendLine("    \"prompted_emotion\": \"emotion the player was asked to act\",");
            sb.AppendLine("    \"prediction_index\": \"0-based tick within the trial\",");
            sb.AppendLine("    \"predicted_emotion\": \"argmax class, empty when status is no_tracking\",");
            sb.AppendLine("    \"correct\": \"1 when predicted_emotion equals prompted_emotion\",");
            sb.AppendLine("    \"status\": \"ok; no_tracking when no valid window was available; backend_error when the backend failed (see error_code)\",");
            sb.AppendLine("    \"generation\": \"monotonic count of valid frames pushed into the ring buffer\",");
            sb.AppendLine("    \"score_<Class>\": \"model output for that class, in class_order\",");
            sb.AppendLine("    \"frame_<f>_valid\": \"always 1 for logged windows; invalid frames are dropped upstream\",");
            sb.AppendLine("    \"frame_<f>_au_<a>\": \"action unit a of window frame f, frame 0 oldest\",");
            if (calibration != null)
            {
                sb.AppendLine("    \"calibration_mode\": \"raw or movement; fixed at session start\",");
                sb.AppendLine("    \"personal_calibration_enabled\": \"always false; personal calibration was removed, the column is kept for older readers\",");
                sb.AppendLine("    \"population_calibration_enabled\": \"always false; population calibration was removed, the column is kept for older readers\",");
                sb.AppendLine("    \"movement_calibration_enabled\": \"the operator checkbox, as it was at session start\",");
                sb.AppendLine("    \"hybrid_alpha\": \"always empty; hybrid calibration was removed\",");
                sb.AppendLine("    \"raw_predicted_emotion\": \"the untouched model argmax; equal to predicted_emotion, named so the two methods are distinguishable\",");
                sb.AppendLine("    \"calibrated_predicted_emotion\": \"movement calibration's decision; blank in raw mode, in the calibration block, on an untracked tick, and when the profile is invalid\",");
                sb.AppendLine("    \"calibrated_winner_similarity\": \"sum of the score and movement cosines of the winning emotion (0 to 2)\",");
                sb.AppendLine("    \"calibrated_runner_up_similarity\": \"the same sum for the second-placed emotion\",");
                sb.AppendLine("    \"calibrated_margin\": \"winner minus runner-up; a small value marks a tick that nearly went the other way\",");
                sb.AppendLine("    \"calibration_valid\": \"true once a usable profile was sealed; blank on calibration-block rows, which were written before the seal\",");
                sb.AppendLine("    \"calibration_error\": \"why the profile is unusable, empty when it is fine\",");
            }
            sb.AppendLine("    \"backend_id\": \"the library's id of the backend that scored the row (the model asset's name for SparseVR, jev:<model>:p<prompt version> for Jev)\",");
            sb.AppendLine("    \"error_code\": \"the library's PredictionErrorCode when the row failed, empty otherwise\",");
            sb.AppendLine("    \"raw_<Class>\": \"the backend's own answer for that class before normalisation (Jev); blank for SparseVR\"");
            sb.AppendLine("  },");
            sb.AppendLine("  \"notes\": [");
            sb.AppendLine("    \"SparseVR face scores are probabilities: the face graph already ends in Softmax and no second one is appended.\",");
            sb.AppendLine("    \"Jev scores (models.face_backend = Jev) are its seven yes/no answers divided by their sum; raw_<Class> holds the answers themselves.\",");
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

    /// <summary>
    /// The calibration block of the metadata: how this session was configured and how the
    /// calibration block turned out.
    /// </summary>
    /// <remarks>
    /// Provenance and decisions only. The centroid vectors are deliberately absent from every
    /// file this project writes: a centroid is a biometric description of one identifiable
    /// participant. What a reader needs in order to interpret a result -- which mode, whether
    /// the profile was valid -- is all here, and none of it is a face.
    /// </remarks>
    private void WriteCalibrationMetadata(StringBuilder sb)
    {
        if (calibration == null)
        {
            return;
        }

        sb.AppendLine("  \"calibration\": {");

        var fields = new List<KeyValuePair<string, string>>
        {
            new("mode", calibration.ModeName),
            new("personal_calibration_enabled", "false"),
            new("population_calibration_enabled", "false"),
            new("movement_calibration_enabled", calibration.MovementEnabled ? "true" : "false"),
            new("hybrid_alpha", string.Empty),
            new("calibration_source", "VREmotionDetection library, MovementCalibrator"),
            new("epsilon", CalibrationMath.Epsilon.ToString("R", CultureInfo.InvariantCulture)),
            new("feature_space", CalibrationMath.FeatureSpaceName),
            new("transform", CalibrationMath.TransformName),
            new("metric", CalibrationMath.MetricName),
            new("estimator", "nearest centroid; centroid is the mean of the scored ticks' "
                             + "log-score vectors"),
            new("class_order", string.Join(", ", ClassOrder)),
            new("personal_centroid_tick_indices",
                $"{PersonalCentroidBuilder.FirstScoredTick}-{PersonalCentroidBuilder.LastScoredTick} "
                + "inclusive; tick 0 is a transition sample and is excluded"),
            new("personal_centroid_required_ticks",
                PersonalCentroidBuilder.RequiredTicksPerCentroid.ToString(CultureInfo.InvariantCulture)),
            new("runtime_calibration_version", MovementCalibrator.ImplementationVersion),
            // Console logging, separate from what the participant sees on the headset.
            new("debug_logging_enabled", calibration.DebugLoggingEnabled ? "true" : "false"),
            new("participant_feedback_shown",
                "test block: the predicted emotion every second (calibrated when calibration "
                + "is ready); calibration block: none; never a score"),
            new("trial_majority_tie_rule",
                "most ticks -> the trial's final tracked tick -> first tied class in class order"),
        };

        if (calibration.MovementEnabled)
        {
            fields.Add(new("movement_feature", CalibrationMath.MovementFeatureName));
            fields.Add(new("movement_rule", CalibrationMath.MovementRuleName));
        }

        fields.Add(new("population_artifact", "not loaded; population calibration was removed"));

        if (calibrationOutcome != null)
        {
            foreach (var kv in calibrationOutcome) fields.Add(kv);
        }
        else
        {
            fields.Add(new("calibration_valid",
                           "not yet sealed when this file was written; the calibration block "
                           + "had not finished"));
        }

        for (int i = 0; i < fields.Count; ++i)
        {
            sb.Append("    \"").Append(JsonEscape(fields[i].Key)).Append("\": \"")
              .Append(JsonEscape(fields[i].Value)).Append('"');
            sb.AppendLine(i < fields.Count - 1 ? "," : "");
        }

        sb.AppendLine("  },");
    }

    private static string Format(float value)
    {
        // G9 round-trips a Single exactly.
        return value.ToString("G9", CultureInfo.InvariantCulture);
    }

    private static string Format(double value)
    {
        // G17 round-trips a Double exactly. Similarities and margins are computed in double,
        // and a parity check that compares the runtime against a recomputation cannot afford
        // to lose bits on the way to the file.
        return value.ToString("G17", CultureInfo.InvariantCulture);
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
        // The trial file is closed even if the prediction file was already closed, so an
        // interrupted session does not leave one of the two open.
        Close(ref trialWriter, "trials.csv");
        Close(ref writer, "predictions.csv");
    }

    private static void Close(ref StreamWriter handle, string what)
    {
        if (handle == null)
        {
            return;
        }

        try
        {
            handle.Flush();
            handle.Dispose();
        }
        catch (Exception e)
        {
            Debug.LogError($"SessionLogger: failed to close {what} " +
                           $"({e.GetType().Name}: {e.Message})");
        }
        finally
        {
            handle = null;
        }
    }
}
