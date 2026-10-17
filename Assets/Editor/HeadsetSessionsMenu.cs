using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build;
using UnityEngine;

using Debug = UnityEngine.Debug;

// Unity menu: VR Emotion > Get Sessions From Headset.
//
// Copies the game's session folders (predictions.csv, trials.csv, session_metadata.json and,
// for a demo with Save Face Tracking on, face_tracking.csv) from the connected headset into
// <project>/HeadsetResults/sessions and opens that folder. Sits next to the library's
// "Get Results From Headset", which copies only the library's own results files; this one is
// in the game because the session folders are the game's. The adb helpers follow the library's.
public static class HeadsetSessionsMenu
{
    private const string LocalFolderName = "HeadsetResults";
    private const string SessionsFolderName = "sessions";

    [MenuItem("VR Emotion/Get Sessions From Headset")]
    public static void GetSessionsFromHeadset()
    {
        string adb = FindAdb();
        if (adb == null)
        {
            Tell("Could not find adb.\n\nInstall Android Build Support for this Unity version " +
                 "(Unity Hub > Installs > Add modules), then try again.");
            return;
        }

        var (_, devicesText) = Run(adb, "devices");
        var lines = devicesText.Split('\n').Skip(1).Select(l => l.Trim()).Where(l => l.Length > 0).ToArray();
        var ready = lines.Where(l => l.EndsWith("\tdevice")).Select(l => l.Split('\t')[0]).ToArray();

        if (ready.Length == 0)
        {
            Tell(lines.Any(l => l.EndsWith("unauthorized"))
                ? "The headset is connected but has not allowed this computer.\n\nPut the headset " +
                  "on and accept \"Allow USB debugging\", then try again."
                : "No headset found.\n\nConnect the Quest with a USB cable, turn it on, and make " +
                  "sure developer mode is enabled.");
            return;
        }

        string package = PlayerSettings.GetApplicationIdentifier(NamedBuildTarget.Android);
        string remote = $"/sdcard/Android/data/{package}/files/{SessionsFolderName}";
        string results = Path.Combine(Directory.GetParent(Application.dataPath).FullName, LocalFolderName);
        string local = Path.Combine(results, SessionsFolderName);

        // The recordings describe real people: the whole results folder stays out of git.
        Directory.CreateDirectory(local);
        string gitignore = Path.Combine(results, ".gitignore");
        if (!File.Exists(gitignore))
        {
            File.WriteAllText(gitignore, "# Results describe real people: never commit them.\n*\n");
        }

        try
        {
            EditorUtility.DisplayProgressBar("VR Emotion", "Copying sessions from the headset...", 0.5f);
            var (ok, output) = Run(adb, $"-s {ready[0]} pull \"{remote}/.\" \"{local}\"", 600000);

            if (!ok)
            {
                Tell(output.Contains("does not exist") || output.Contains("No such file")
                    ? $"There are no sessions on the headset yet for {package}.\n\nRun a session, " +
                      "then try again."
                    : $"Copying failed:\n\n{output.Trim()}");
                return;
            }
        }
        finally
        {
            EditorUtility.ClearProgressBar();
        }

        var sessions = Directory.GetDirectories(local, "session_*");
        int withFace = sessions.Count(s => File.Exists(Path.Combine(s, "face_tracking.csv")));
        string newest = sessions.OrderBy(s => s, StringComparer.Ordinal).LastOrDefault();

        EditorUtility.RevealInFinder(newest ?? local);
        Tell($"Done. {sessions.Length} session folder(s) are in:\n{local}\n\n" +
             $"{withFace} of them have face_tracking.csv." +
             (newest != null ? $"\n\nNewest: {Path.GetFileName(newest)}" : ""));
    }

    private static void Tell(string message)
    {
        Debug.Log($"VR Emotion: {message}");
        EditorUtility.DisplayDialog("VR Emotion", message, "OK");
    }

    // Unity's own copy first, then the Android SDK Unity is set to, then the usual places.
    private static string FindAdb()
    {
        string exe = Application.platform == RuntimePlatform.WindowsEditor ? "adb.exe" : "adb";
        var candidates = new[]
        {
            Path.Combine(EditorApplication.applicationContentsPath, "PlaybackEngines", "AndroidPlayer",
                         "SDK", "platform-tools", exe),
            Path.Combine(EditorPrefs.GetString("AndroidSdkRoot"), "platform-tools", exe),
            Path.Combine(Environment.GetEnvironmentVariable("ANDROID_HOME") ?? "", "platform-tools", exe),
            Path.Combine(Environment.GetEnvironmentVariable("ANDROID_SDK_ROOT") ?? "", "platform-tools", exe),
        };

        string found = candidates.FirstOrDefault(File.Exists);
        if (found != null) return found;

        // Last resort: adb on PATH.
        var (ok, _) = Run(exe, "version");
        return ok ? exe : null;
    }

    private static (bool ok, string output) Run(string file, string arguments, int timeoutMs = 20000)
    {
        try
        {
            using var process = Process.Start(new ProcessStartInfo(file, arguments)
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            });

            var stdout = process.StandardOutput.ReadToEndAsync();
            var stderr = process.StandardError.ReadToEndAsync();
            if (!process.WaitForExit(timeoutMs))
            {
                process.Kill();
                return (false, "adb did not answer in time.");
            }

            return (process.ExitCode == 0, stdout.Result + stderr.Result);
        }
        catch (Exception e)
        {
            return (false, e.Message);
        }
    }
}
