using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

// Adds `RuntimeCalibrationSettings` to whatever GameObject carries `ProtocolRunner`, in every
// scene that has one.
//
// `[RequireComponent]` does not reach a GameObject that was already serialised into a scene --
// it only fires when the component is added in the editor -- so a scene saved before that
// attribute existed opens without the component and with no warning. This is the one-off fixup
// for that, kept in the repository rather than run once and deleted so the next scene, or the
// next person who reorganises this one, has something to run.
//
//   Editor:     Tools > Calibration > Wire calibration settings into the open scene
//   Batch mode: -executeMethod CalibrationSceneWiring.WireMainScene
//
// It is idempotent, and it changes nothing except adding a missing component: its movement
// calibration checkbox keeps its default, off, which is raw mode.
public static class CalibrationSceneWiring
{
    private const string MainScenePath = "Assets/Scenes/Main.unity";

    [MenuItem("Tools/Calibration/Wire calibration settings into the open scene")]
    public static void WireOpenScene()
    {
        var scene = EditorSceneManager.GetActiveScene();
        int added = Wire(scene);

        if (added > 0)
        {
            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene);
        }

        Debug.Log($"{nameof(CalibrationSceneWiring)}: {scene.path} -- "
                  + (added > 0 ? $"added {added} component(s)" : "already wired, nothing to do"));
    }

    /// <summary>Batch-mode entry point. Non-zero exit if the scene could not be wired.</summary>
    public static void WireMainScene()
    {
        // Whether the component is *serialised*, which is a different question from whether
        // it is present once the scene is open.
        //
        // Loading a scene in the editor applies `RequireComponent` in memory: the missing
        // component appears on the GameObject, `Wire` below finds nothing to do, the scene is
        // never marked dirty, and the file on disk is left exactly as broken as it was. The
        // first attempt at this fixup did precisely that and reported success. So the decision
        // to save is taken from the file's own bytes, before the scene is opened at all.
        bool serialised = SceneAlreadyReferences(CalibrationSettingsScriptGuid());

        var scene = EditorSceneManager.OpenScene(MainScenePath, OpenSceneMode.Single);
        int added = Wire(scene);

        if (added > 0 || !serialised)
        {
            EditorSceneManager.MarkSceneDirty(scene);
            if (!EditorSceneManager.SaveScene(scene))
            {
                Debug.LogError($"{nameof(CalibrationSceneWiring)}: could not save {MainScenePath}");
                EditorApplication.Exit(1);
                return;
            }

            Debug.Log($"{nameof(CalibrationSceneWiring)}: saved {MainScenePath} "
                      + $"(added in memory: {added}, was serialised: {serialised})");
        }

        if (!SceneAlreadyReferences(CalibrationSettingsScriptGuid()))
        {
            Debug.LogError($"{nameof(CalibrationSceneWiring)}: {MainScenePath} still does not "
                           + $"reference {nameof(RuntimeCalibrationSettings)} after saving. The "
                           + "component exists only in memory, which is the failure this "
                           + "script exists to fix.");
            EditorApplication.Exit(1);
            return;
        }

        // Read back from the saved scene rather than from the objects still in memory, so the
        // success message is about the file rather than about the intention.
        var reopened = EditorSceneManager.OpenScene(MainScenePath, OpenSceneMode.Single);
        foreach (var root in reopened.GetRootGameObjects())
        {
            foreach (var runner in root.GetComponentsInChildren<ProtocolRunner>(true))
            {
                var settings = runner.GetComponent<RuntimeCalibrationSettings>();
                if (settings == null)
                {
                    Debug.LogError($"{nameof(CalibrationSceneWiring)}: '{runner.name}' still has "
                                   + $"no {nameof(RuntimeCalibrationSettings)} after saving");
                    EditorApplication.Exit(1);
                    return;
                }

                var config = settings.LockForSession();
                Debug.Log($"{nameof(CalibrationSceneWiring)}: '{runner.name}' carries "
                          + $"{nameof(RuntimeCalibrationSettings)}; mode is '{config.ModeName}' "
                          + $"(movement calibration {config.MovementEnabled})");
            }
        }

        Debug.Log($"{nameof(CalibrationSceneWiring)}: {MainScenePath} wired "
                  + $"({added} component(s) added)");
        EditorApplication.Exit(0);
    }

    /// <summary>
    /// Report what the inspector will actually show for the calibration settings, without
    /// opening the inspector.
    /// </summary>
    /// <remarks>
    /// `SerializedProperty.displayName` is the string Unity draws beside each checkbox. It is
    /// derived from the field name rather than written anywhere, so the only way to know a
    /// label is right is to ask for it. Changing nothing and saving nothing.
    ///
    ///   Batch mode: -executeMethod CalibrationSceneWiring.ReportMainSceneWiring
    /// </remarks>
    public static void ReportMainSceneWiring()
    {
        var scene = EditorSceneManager.OpenScene(MainScenePath, OpenSceneMode.Single);
        int found = 0;

        foreach (var root in scene.GetRootGameObjects())
        {
            foreach (var runner in root.GetComponentsInChildren<ProtocolRunner>(true))
            {
                found++;
                var settings = runner.GetComponent<RuntimeCalibrationSettings>();

                Debug.Log($"[wiring] GameObject '{runner.name}': ProtocolRunner + "
                          + (settings == null ? "NO RuntimeCalibrationSettings"
                                              : nameof(RuntimeCalibrationSettings)));

                if (settings == null) continue;

                var serialized = new SerializedObject(settings);
                var property = serialized.GetIterator();
                property.NextVisible(true);

                while (property.NextVisible(false))
                {
                    if (property.propertyType != SerializedPropertyType.Boolean) continue;

                    Debug.Log($"[wiring]   checkbox \"{property.displayName}\" "
                              + $"(field {property.name}) = {property.boolValue}");
                }

                var config = settings.LockForSession();
                Debug.Log($"[wiring]   resolved mode: {config.ModeName}");
            }
        }

        Debug.Log($"[wiring] {found} ProtocolRunner(s) in {MainScenePath}");
        EditorApplication.Exit(found > 0 ? 0 : 1);
    }

    /// <summary>The GUID Unity assigned to `RuntimeCalibrationSettings.cs`.</summary>
    private static string CalibrationSettingsScriptGuid()
    {
        var script = AssetDatabase.FindAssets(
            $"t:MonoScript {nameof(RuntimeCalibrationSettings)}");

        foreach (string guid in script)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            if (System.IO.Path.GetFileNameWithoutExtension(path)
                == nameof(RuntimeCalibrationSettings))
            {
                return guid;
            }
        }

        throw new System.InvalidOperationException(
            $"no MonoScript asset named {nameof(RuntimeCalibrationSettings)}");
    }

    /// <summary>Whether the scene *file* mentions a script, independent of what is loaded.</summary>
    private static bool SceneAlreadyReferences(string scriptGuid)
        => System.IO.File.ReadAllText(MainScenePath).Contains(scriptGuid);

    private static int Wire(UnityEngine.SceneManagement.Scene scene)
    {
        int added = 0;

        foreach (var root in scene.GetRootGameObjects())
        {
            // `true` includes inactive objects: a disabled rig is still a rig, and leaving it
            // unwired would produce the same failure the next time it was enabled.
            foreach (var runner in root.GetComponentsInChildren<ProtocolRunner>(true))
            {
                if (runner.GetComponent<RuntimeCalibrationSettings>() != null)
                {
                    continue;
                }

                Undo.AddComponent<RuntimeCalibrationSettings>(runner.gameObject);
                added++;

                Debug.Log($"{nameof(CalibrationSceneWiring)}: added "
                          + $"{nameof(RuntimeCalibrationSettings)} to '{runner.name}' "
                          + "(movement calibration off, which is raw mode)");
            }
        }

        return added;
    }
}
