using System;
using System.IO;
using System.Linq;

using UnityEditor;
using UnityEditor.Recorder;
using UnityEditor.Recorder.Encoder;
using UnityEditor.Recorder.Input;
using UnityEditor.SceneManagement;
using UnityEngine;

using Meta.XR.Movement.FaceTracking.Samples;

// Unity menu: VR Emotion > Avatar Replay > Render Face File...
//
// Renders a demo's face_tracking.csv on the Movement SDK's HighFidelity character to
// avatar.mp4 (1280 x 720, 30 fps) next to the file, for the bottom-right box of the demo video.
// It builds a small scene (the character, a camera on its face, lights), enters Play mode,
// records with Unity Recorder until the face data ends, then stops by itself.
//
// The face uses retargeting_exact_1to1.json: each tracked weight drives only the face shape of
// its own name, at full strength, so the face shows the values the model received. Meta's own
// mapping amplifies and mixes several of them.
[InitializeOnLoad]
public static class AvatarReplayMenu
{
    private const string ScenePath = "Assets/DemoReplay/AvatarReplay.unity";
    private const string ConfigPath = "Assets/DemoReplay/retargeting_exact_1to1.json";
    private const string CharacterName = "HighFidelityCharacterMirrored";
    private const string OutputName = "avatar";

    // The sky of the Movement SDK's MovementBody sample scene, used for its ambient light.
    private const string SampleSkyboxGuid = "69c55b26f7a9bef4eb62a052fd5f09d8";

    // The webcam's own resolution (Logitech C270, 720p), so the avatar is as sharp as the
    // webcam beside it; the overlay script scales it into its 640 x 360 box.
    private const int Width = 1280;
    private const int Height = 720;
    private const float FrameRate = 30f;

    // Kept across the domain reload that entering Play mode causes.
    private const string PendingKey = "AvatarReplay.Pending";
    private const string OutputKey = "AvatarReplay.Output";

    // The body tracking parts of the sample character, which need a live headset.
    private static readonly string[] BodyComponents =
    {
        "CharacterRetargeter", "MetaSourceDataProvider", "MirrorTransforms", "OVRBody",
        "OVRCustomSkeleton", "OVRSkeleton", "HandDeformation", "OVRWeightsProvider",
    };

    private static RecorderController recorder;

    static AvatarReplayMenu()
    {
        EditorApplication.playModeStateChanged += OnPlayModeChanged;
    }

    [MenuItem("VR Emotion/Avatar Replay/Render Face File...")]
    public static void RenderFaceFile()
    {
        string sessions = Path.Combine(Directory.GetParent(Application.dataPath).FullName, "HeadsetResults", "sessions");
        string csv = EditorUtility.OpenFilePanel("Choose a demo's face_tracking.csv", sessions, "csv");
        if (string.IsNullOrEmpty(csv))
        {
            return;
        }

        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
        {
            return;
        }

        if (!BuildScene(csv))
        {
            return;
        }

        string output = Path.Combine(Path.GetDirectoryName(csv), OutputName);
        SessionState.SetBool(PendingKey, true);
        SessionState.SetString(OutputKey, output);
        EditorApplication.EnterPlaymode();
    }

    private static bool BuildScene(string csv)
    {
        var characterGuid = AssetDatabase.FindAssets($"{CharacterName} t:Prefab").FirstOrDefault();
        var config = AssetDatabase.LoadAssetAtPath<TextAsset>(ConfigPath);
        if (characterGuid == null || config == null)
        {
            EditorUtility.DisplayDialog("VR Emotion",
                characterGuid == null
                    ? $"{CharacterName} not found.\n\nImport Meta XR Movement SDK > Samples > Body Tracking Samples, then try again."
                    : $"{ConfigPath} not found.", "OK");
            return false;
        }

        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

        var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(AssetDatabase.GUIDToAssetPath(characterGuid));
        var character = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
        character.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);

        foreach (var component in character.GetComponentsInChildren<MonoBehaviour>(true))
        {
            if (component != null && BodyComponents.Contains(component.GetType().Name))
            {
                component.enabled = false;
            }
        }

        // The face is drawn as a plain skinned mesh. Meta's RecalculateNormals draws a corrected
        // copy through a shader keyword and a GPU pass that do not work in this Editor: the face
        // came out flat, then not at all. Its "use Unity's function" switch skips the keyword and
        // the copy, and with the component off it does nothing else; the face keeps the normals
        // its shapes were imported with.
        foreach (var component in character.GetComponentsInChildren<MonoBehaviour>(true))
        {
            if (component != null && component.GetType().Name == "RecalculateNormals")
            {
                var normals = new SerializedObject(component);
                normals.FindProperty("_useUnityFunction").boolValue = true;
                normals.ApplyModifiedPropertiesWithoutUndo();
                component.enabled = false;
            }
        }

        // The face: the CSV in place of the headset, through the exact mapping.
        var retargeter = character.GetComponentInChildren<FaceRetargeterComponent>(true);
        if (retargeter == null)
        {
            EditorUtility.DisplayDialog("VR Emotion", $"{CharacterName} has no face retargeter.", "OK");
            return false;
        }

        var provider = retargeter.gameObject.AddComponent<CsvFaceWeightsProvider>();
        provider.CsvPath = csv;

        var serialized = new SerializedObject(retargeter);
        serialized.FindProperty("_retargeterConfig").objectReferenceValue = config;
        serialized.FindProperty("_retargeterConfigOverride").stringValue = "";
        serialized.FindProperty("_weightsProvider").objectReferenceValue = provider;
        serialized.ApplyModifiedPropertiesWithoutUndo();

        var bones = character.GetComponentsInChildren<Transform>(true);
        Transform Find(string name) => bones.FirstOrDefault(b => b.name == name);
        var headBone = Find("Head");
        provider.SetEyes(headBone, Find("LeftEye"), Find("RightEye"));

        // The camera, straight in front of the face, framing head and neck.
        var focus = headBone != null ? headBone.position + new Vector3(0f, 0.05f, 0f) : new Vector3(0f, 1.6f, 0f);
        var cameraObject = new GameObject("Replay Camera") { tag = "MainCamera" };
        var camera = cameraObject.AddComponent<Camera>();
        camera.clearFlags = CameraClearFlags.SolidColor;
        camera.backgroundColor = new Color32(0x14, 0x16, 0x1C, 0xFF);
        camera.fieldOfView = 20f;
        camera.nearClipPlane = 0.05f;

        // Nothing should be on the HiddenMesh layer now (RecalculateNormals is off), but if
        // anything is, it is a hidden duplicate and must not be drawn.
        int hidden = LayerMask.NameToLayer("HiddenMesh");
        if (hidden >= 0)
        {
            camera.cullingMask &= ~(1 << hidden);
        }
        // The way the face points: from the head to between the eyes, level. The model's own
        // forward axis is not relied on, since sample characters are not all exported facing +Z.
        var forward = character.transform.forward;
        var leftEye = Find("LeftEye");
        var rightEye = Find("RightEye");
        if (headBone != null && leftEye != null && rightEye != null)
        {
            var toEyes = Vector3.ProjectOnPlane((leftEye.position + rightEye.position) / 2f - headBone.position, Vector3.up);
            if (toEyes.sqrMagnitude > 1e-6f)
            {
                forward = toEyes.normalized;
            }
        }
        cameraObject.transform.position = focus + forward * 1.1f;
        cameraObject.transform.LookAt(focus);

        // Lit as the Movement SDK's own body sample lights this character, so the skin keeps its
        // colour: warm key and fill lights from the front, a cool rim from behind, and the
        // sample's sky for ambient light and reflections. The background stays plain.
        var warm = new Color(1f, 0.914f, 0.675f);
        var cool = new Color(0.44f, 0.66f, 0.89f);
        AddLight("Key Light", warm, 1.2f, Quaternion.LookRotation(-forward + new Vector3(-0.5f, -0.4f, 0f)));
        AddLight("Fill Light", warm, 0.5f, Quaternion.LookRotation(-forward + new Vector3(0.6f, -0.1f, 0f)));
        AddLight("Rim Light", cool, 1.0f, Quaternion.LookRotation(forward + new Vector3(0f, -0.3f, 0f)));

        var sky = AssetDatabase.LoadAssetAtPath<Material>(AssetDatabase.GUIDToAssetPath(SampleSkyboxGuid));
        if (sky != null)
        {
            RenderSettings.skybox = sky;
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Skybox;
            RenderSettings.ambientIntensity = 1f;
            DynamicGI.UpdateEnvironment();
        }
        else
        {
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = new Color(0.212f, 0.227f, 0.259f);
            RenderSettings.ambientEquatorColor = new Color(0.114f, 0.125f, 0.133f);
            RenderSettings.ambientGroundColor = new Color(0.047f, 0.043f, 0.035f);
        }

        EditorSceneManager.SaveScene(scene, ScenePath);
        return true;
    }

    private static void AddLight(string name, Color color, float intensity, Quaternion rotation)
    {
        var light = new GameObject(name).AddComponent<Light>();
        light.type = LightType.Directional;
        light.color = color;
        light.intensity = intensity;
        light.transform.rotation = rotation;
    }

    private static void OnPlayModeChanged(PlayModeStateChange change)
    {
        if (change != PlayModeStateChange.EnteredPlayMode || !SessionState.GetBool(PendingKey, false))
        {
            return;
        }

        SessionState.SetBool(PendingKey, false);
        var provider = UnityEngine.Object.FindAnyObjectByType<CsvFaceWeightsProvider>();
        if (provider == null || provider.Duration <= 0f)
        {
            Debug.LogError("Avatar replay: no face data to render.");
            EditorApplication.ExitPlaymode();
            return;
        }

        var movie = ScriptableObject.CreateInstance<MovieRecorderSettings>();
        movie.name = "Avatar";
        movie.Enabled = true;
        movie.EncoderSettings = new CoreEncoderSettings
        {
            Codec = CoreEncoderSettings.OutputCodec.MP4,
            EncodingQuality = CoreEncoderSettings.VideoEncodingQuality.High,
        };
        movie.ImageInputSettings = new CameraInputSettings
        {
            Source = ImageSource.MainCamera,
            OutputWidth = Width,
            OutputHeight = Height,
        };
        movie.CaptureAudio = false;
        movie.OutputFile = SessionState.GetString(OutputKey, OutputName);

        var settings = ScriptableObject.CreateInstance<RecorderControllerSettings>();
        settings.AddRecorderSettings(movie);
        settings.SetRecordModeToManual();
        settings.FrameRate = FrameRate;
        // The recorder steps game time by exactly one frame (1/30 s) per frame it saves, so the
        // video keeps the face file's timing however fast it renders. Not capping it lets a short
        // render finish faster than real time.
        settings.CapFrameRate = false;

        recorder = new RecorderController(settings);
        recorder.PrepareRecording();
        recorder.StartRecording();
        provider.Play();
        Debug.Log($"Avatar replay: rendering {provider.Duration:F1} s to {movie.OutputFile}.mp4");

        EditorApplication.update += WaitForEnd;
    }

    private static void WaitForEnd()
    {
        var provider = UnityEngine.Object.FindAnyObjectByType<CsvFaceWeightsProvider>();
        if (EditorApplication.isPlaying && provider != null && !provider.Finished)
        {
            return;
        }

        EditorApplication.update -= WaitForEnd;
        if (recorder != null && recorder.IsRecording())
        {
            recorder.StopRecording();
        }
        recorder = null;

        string output = SessionState.GetString(OutputKey, "") + ".mp4";
        if (EditorApplication.isPlaying)
        {
            EditorApplication.ExitPlaymode();
        }

        Debug.Log($"Avatar replay: done, {output}");
        if (File.Exists(output))
        {
            EditorUtility.RevealInFinder(output);
        }
    }
}
