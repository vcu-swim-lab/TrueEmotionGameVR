using System;
using System.IO;
using System.Linq;
using UnityEngine;

#if UNITY_ANDROID && !UNITY_EDITOR 
using UnityEngine.Android;
#endif

public class QuestMicRecorder : MonoBehaviour
{
    [SerializeField] private AudioSource monitorSource; // optional: to monitor live input
    private AudioClip recordedClip;
    private string directoryPath;
    private string selectedDevice;
    private float startTime;
    private float recordingLengthSec;
    private int sampleRate;

    private void Awake()
    {
#if UNITY_ANDROID && !UNITY_EDITOR 
        // Primary writable location on Android/Quest
        string basePath = Application.persistentDataPath;

        // Fallback in the unlikely event persistentDataPath is weird
        string manualFallback = $"/sdcard/Android/data/{Application.identifier}/files";

        try
        {
            directoryPath = Path.Combine(basePath, "Recordings");
            Directory.CreateDirectory(directoryPath); // safe if exists
        }
        catch (Exception e)
        {
            Debug.LogError($"[Recorder] Failed to create dir at persistentDataPath: {basePath}\n{e.Message}\nUsing fallback...");
            directoryPath = Path.Combine(manualFallback, "Recordings");
            try
            {
                Directory.CreateDirectory(directoryPath);
            }
            catch (Exception e2)
            {
                Debug.LogError($"[Recorder] Fallback also failed: {manualFallback}\n{e2.Message}");
                // As a last resort, keep using persistentDataPath without subfolder
                directoryPath = basePath;
            }
        }

        Debug.Log($"[Recorder] Saving to: {directoryPath}\n" +
                  $"identifier={Application.identifier}\n" +
                  $"persistentDataPath={Application.persistentDataPath}\n" +
                  $"temporaryCachePath={Application.temporaryCachePath}\n" +
                  $"dataPath={Application.dataPath} (READ-ONLY ON ANDROID)");
#else
        directoryPath = Path.Combine(Application.dataPath, "Recordings");
#endif
        Debug.Log("Awake!!"); 

        if (!Directory.Exists(directoryPath))
            Directory.CreateDirectory(directoryPath);

        // Use output sample rate or 48000 (Quest uses 48 kHz typically)
        sampleRate = AudioSettings.outputSampleRate > 0 ? AudioSettings.outputSampleRate : 48000;

        Debug.Log($"[QuestMicRecorder] Save dir: {directoryPath}");
#if UNITY_ANDROID && !UNITY_EDITOR 
        EnsureAndroidMicPermission();
#else
        // In Editor/PC, pick a sensible device (Quest mic if connected via Link; else default)
        PickDesktopDevice();
#endif
    }

#if UNITY_ANDROID && !UNITY_EDITOR 
    private void EnsureAndroidMicPermission()
    {
        if (!Permission.HasUserAuthorizedPermission(Permission.Microphone))
        {
            Permission.RequestUserPermission(Permission.Microphone);
        }
    }
#endif

    private void PickDesktopDevice()
    {
        var devices = Microphone.devices;
        Debug.Log($"[QuestMicRecorder] Devices: {string.Join(", ", devices)}");

        // Prefer an Oculus/Quest device if present (Link/Air Link case)
        selectedDevice = devices.FirstOrDefault(d =>
            d.IndexOf("oculus", StringComparison.OrdinalIgnoreCase) >= 0 ||
            d.IndexOf("quest", StringComparison.OrdinalIgnoreCase) >= 0);

        // Otherwise, default to the first device (or null → default device)
        if (string.IsNullOrEmpty(selectedDevice))
            selectedDevice = devices.Length > 0 ? devices[0] : null;

        Debug.Log($"[QuestMicRecorder] Selected device: {(selectedDevice ?? "<default>")}");
    }

    public void StartRecording()
    {
#if UNITY_ANDROID && !UNITY_EDITOR 
        // On Android/Quest, passing null/empty uses the default headset mic reliably.
        string deviceName = null;
#else
        string deviceName = selectedDevice; // Desktop/Editor: use a chosen device
#endif
        if (Microphone.devices.Length == 0 && string.IsNullOrEmpty(deviceName))
        {
            Debug.LogError("[QuestMicRecorder] No microphone available.");
            return;
        }

        int lengthSec = 30; // max duration cap; we'll trim on stop
        recordedClip = Microphone.Start(deviceName, false, lengthSec, sampleRate);
        startTime = Time.realtimeSinceStartup;

        // (Optional) monitor mic live through speakers/headphones
        if (monitorSource != null)
        {
            monitorSource.Stop();
            monitorSource.clip = recordedClip;
            monitorSource.loop = true;
            // Wait a moment until mic starts to avoid zero-position glitch
            StartCoroutine(BeginMonitorWhenReady(deviceName));
        }

        Debug.Log($"[QuestMicRecorder] Recording started. Device={(deviceName ?? "<default>")}, SR={sampleRate}");
    }

    private System.Collections.IEnumerator BeginMonitorWhenReady(string deviceName)
    {
        // Wait until the microphone actually starts producing samples
        while (Microphone.GetPosition(deviceName) <= 0) yield return null;
        monitorSource.Play();
    }

    public void StopRecording()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        string deviceName = null;
#else
        string deviceName = selectedDevice;
#endif
        Microphone.End(deviceName);

        recordingLengthSec = Mathf.Max(0f, Time.realtimeSinceStartup - startTime);
        if (monitorSource != null) monitorSource.Stop();

        if (recordedClip == null || recordingLengthSec < 0.05f)
        {
            Debug.LogWarning("[QuestMicRecorder] Recording too short or null; not saving.");
            return;
        }

        // Trim to actual duration (so your WAV isn't full of silence)
        recordedClip = TrimClip(recordedClip, recordingLengthSec);
        SaveWav(recordedClip);
    }

    private void SaveWav(AudioClip clip)
    {
        try
        {
            string fileName = $"recording_{DateTime.Now:yyyyMMdd_HHmmss}.wav";
            string fullPath = Path.Combine(directoryPath, fileName);

            // Use your existing WavUtility
            WavUtility.Save(fullPath, clip);

            Debug.Log($"[QuestMicRecorder] Saved WAV → {fullPath}");
        }
        catch (Exception e)
        {
            Debug.LogError($"[QuestMicRecorder] Failed to save WAV: {e.Message}");
        }
    }

    private static AudioClip TrimClip(AudioClip clip, float lengthSec)
    {
        int channels = clip.channels;
        int freq = clip.frequency;
        int targetSamples = Mathf.Min(clip.samples, Mathf.FloorToInt(freq * lengthSec));

        if (targetSamples <= 0) return clip;

        float[] data = new float[targetSamples * channels];
        clip.GetData(data, 0);

        var trimmed = AudioClip.Create(clip.name + "_trimmed", targetSamples, channels, freq, false);
        trimmed.SetData(data, 0);
        return trimmed;
    }
}