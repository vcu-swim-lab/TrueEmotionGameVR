using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

using NUnit.Framework;

namespace VcuSwimLab.Calibration.Tests
{
    /// <summary>
    /// Checks that the scene actually carries the components the runtime assumes.
    /// </summary>
    /// <remarks>
    /// `[RequireComponent]` is not a guarantee about a scene. It adds the required component
    /// when the requiring one is *added* in the editor, and does nothing at all to a GameObject
    /// that was serialised before the attribute existed. A scene saved earlier therefore opens
    /// with the component missing and no warning anywhere, and the first symptom is a
    /// NullReferenceException partway through a session -- in front of a participant. That is
    /// precisely what happened here, and it was caught by a person reading the scene file
    /// rather than by anything that ran.
    ///
    /// So this reads the scene YAML directly, as text, and asserts the invariant. Parsing the
    /// file rather than loading it through `EditorSceneManager` is deliberate:
    ///
    ///   - it needs no reference to Assembly-CSharp, which an assembly definition cannot have
    ///     in any case, so the test can live beside the rest of the calibration tests;
    ///   - it runs under `dotnet test` as well as in the Unity Editor, so the check is part of
    ///     the same sweep as the Python parity tests rather than something that only happens
    ///     when someone opens Unity;
    ///   - and it asserts what is *on disk*, which is what gets built and deployed, rather than
    ///     what some editor session might have reconstructed in memory.
    /// </remarks>
    public class SceneWiringTests
    {
        private const string ScenePath = "Assets/Scenes/Main.unity";
        private const string ProtocolRunnerScript = "Assets/Scripts/ProtocolRunner.cs";
        private const string CalibrationSettingsScript =
            "Assets/Scripts/RuntimeCalibrationSettings.cs";

        /// <summary>One `--- !u!114 &id MonoBehaviour` block of a scene file.</summary>
        private sealed class SceneComponent
        {
            public string FileId;
            public string GameObjectId;
            public string ScriptGuid;
            public readonly Dictionary<string, string> Fields = new(StringComparer.Ordinal);
        }

        // `--- !u!114 &964688081` : a MonoBehaviour, with its file id.
        private static readonly Regex BlockHeader =
            new(@"^--- !u!(?<class>\d+) &(?<id>\d+)", RegexOptions.Compiled);

        private static readonly Regex GameObjectRef =
            new(@"^\s+m_GameObject: \{fileID: (?<id>-?\d+)\}", RegexOptions.Compiled);

        private static readonly Regex ScriptRef =
            new(@"^\s+m_Script: \{fileID: -?\d+, guid: (?<guid>[0-9a-f]+)", RegexOptions.Compiled);

        private static readonly Regex Field =
            new(@"^  (?<key>\w+): (?<value>.*)$", RegexOptions.Compiled);

        private static string Root => TestPaths.RepositoryRoot;

        /// <summary>The GUID Unity assigns a script, read from its `.meta` sidecar.</summary>
        private static string ScriptGuid(string relativePath)
        {
            string meta = Path.Combine(Root, relativePath.Replace('/', Path.DirectorySeparatorChar))
                          + ".meta";

            Assert.IsTrue(File.Exists(meta), $"no .meta beside {relativePath}");

            foreach (string line in File.ReadLines(meta))
            {
                if (line.StartsWith("guid: ", StringComparison.Ordinal))
                {
                    return line.Substring("guid: ".Length).Trim();
                }
            }

            throw new AssertionException($"{meta} has no guid line");
        }

        private static List<SceneComponent> MonoBehaviours()
        {
            string path = Path.Combine(Root, ScenePath.Replace('/', Path.DirectorySeparatorChar));
            Assert.IsTrue(File.Exists(path), $"no scene at {path}");

            var components = new List<SceneComponent>();
            SceneComponent current = null;

            foreach (string line in File.ReadLines(path))
            {
                var header = BlockHeader.Match(line);
                if (header.Success)
                {
                    // 114 is MonoBehaviour. Every other class id starts a block this test has
                    // no interest in, and `current` going null is what stops its fields being
                    // attributed to the previous component.
                    current = header.Groups["class"].Value == "114"
                        ? new SceneComponent { FileId = header.Groups["id"].Value }
                        : null;

                    if (current != null) components.Add(current);
                    continue;
                }

                if (current == null) continue;

                var gameObject = GameObjectRef.Match(line);
                if (gameObject.Success)
                {
                    current.GameObjectId = gameObject.Groups["id"].Value;
                    continue;
                }

                var script = ScriptRef.Match(line);
                if (script.Success)
                {
                    current.ScriptGuid = script.Groups["guid"].Value;
                    continue;
                }

                var field = Field.Match(line);
                if (field.Success)
                {
                    current.Fields[field.Groups["key"].Value] = field.Groups["value"].Value.Trim();
                }
            }

            return components;
        }

        private static List<SceneComponent> With(string guid)
            => MonoBehaviours().Where(c => c.ScriptGuid == guid).ToList();

        // ------------------------------------------------------------------- the invariant

        [Test]
        public void The_scene_is_text_serialised_so_this_test_can_read_it()
        {
            string settings = Path.Combine(Root, "ProjectSettings", "EditorSettings.asset");
            Assert.IsTrue(File.Exists(settings), settings);

            // 2 is ForceText. Under ForceBinary the scene would be unreadable here and this
            // whole file would silently stop checking anything, so the mode is pinned.
            StringAssert.Contains("m_SerializationMode: 2", File.ReadAllText(settings),
                                  "the scene must stay text-serialised");
        }

        [Test]
        public void The_scene_has_a_protocol_runner()
        {
            var runners = With(ScriptGuid(ProtocolRunnerScript));

            Assert.AreEqual(1, runners.Count,
                            $"expected exactly one ProtocolRunner in {ScenePath}");
            Assert.IsNotNull(runners[0].GameObjectId, "the ProtocolRunner has no host GameObject");
        }

        [Test]
        public void Every_protocol_runner_shares_its_GameObject_with_calibration_settings()
        {
            string runnerGuid = ScriptGuid(ProtocolRunnerScript);
            string settingsGuid = ScriptGuid(CalibrationSettingsScript);

            var components = MonoBehaviours();
            var hostsWithSettings = components
                .Where(c => c.ScriptGuid == settingsGuid)
                .Select(c => c.GameObjectId)
                .ToHashSet(StringComparer.Ordinal);

            var runners = components.Where(c => c.ScriptGuid == runnerGuid).ToList();
            Assert.IsNotEmpty(runners, "no ProtocolRunner in the scene at all");

            foreach (var runner in runners)
            {
                Assert.IsTrue(
                    hostsWithSettings.Contains(runner.GameObjectId),
                    $"the GameObject (fileID {runner.GameObjectId}) carrying ProtocolRunner in "
                    + $"{ScenePath} has no RuntimeCalibrationSettings. RequireComponent does not "
                    + "add one to a GameObject that was already serialised; run "
                    + "Tools > Calibration > Wire calibration settings into the open scene, or "
                    + "add it by hand and save the scene.");
            }
        }

        [Test]
        public void The_calibration_settings_serialise_both_checkboxes()
        {
            var settings = With(ScriptGuid(CalibrationSettingsScript));
            Assert.IsNotEmpty(settings, "no RuntimeCalibrationSettings in the scene");

            foreach (var component in settings)
            {
                // Named exactly as the fields are declared. A rename that did not reach the
                // scene would leave the old key here holding the operator's choice while the
                // new field silently took its default.
                CollectionAssert.Contains(component.Fields.Keys, "enablePersonalCalibration");
                CollectionAssert.Contains(component.Fields.Keys,
                                          "enableEmojiHeroPopulationCalibration");
                CollectionAssert.Contains(component.Fields.Keys,
                                          "enableCalibrationDebugLogging");
            }
        }

        [Test]
        public void Both_checkboxes_are_serialised_off_which_is_raw_mode()
        {
            var settings = With(ScriptGuid(CalibrationSettingsScript));
            Assert.IsNotEmpty(settings);

            foreach (var component in settings)
            {
                bool personal = component.Fields["enablePersonalCalibration"] == "1";
                bool population = component.Fields["enableEmojiHeroPopulationCalibration"] == "1";

                Assert.IsFalse(personal, "enablePersonalCalibration is checked in the scene");
                Assert.IsFalse(population,
                               "enableEmojiHeroPopulationCalibration is checked in the scene");
                Assert.IsFalse(component.Fields["enableCalibrationDebugLogging"] == "1",
                               "enableCalibrationDebugLogging is checked in the scene");

                Assert.AreEqual(CalibrationMode.Raw,
                                CalibrationModes.FromToggles(personal, population),
                                "the committed scene must record raw mode, so a session run "
                                + "without deliberately enabling calibration behaves exactly as "
                                + "this protocol did before runtime calibration existed");
            }
        }

        [Test]
        public void The_settings_component_is_listed_on_its_host_GameObject()
        {
            // The `m_GameObject` back-reference alone is not the whole link: Unity drives the
            // inspector from the host's `m_Component` list, and a component missing from it
            // would not appear even though it points at the object.
            string settingsGuid = ScriptGuid(CalibrationSettingsScript);
            var settings = With(settingsGuid);
            Assert.IsNotEmpty(settings);

            string path = Path.Combine(Root, ScenePath.Replace('/', Path.DirectorySeparatorChar));
            string scene = File.ReadAllText(path);

            foreach (var component in settings)
            {
                int at = scene.IndexOf($"--- !u!1 &{component.GameObjectId}\n",
                                       StringComparison.Ordinal);
                if (at < 0)
                {
                    at = scene.IndexOf($"--- !u!1 &{component.GameObjectId}\r\n",
                                       StringComparison.Ordinal);
                }

                Assert.GreaterOrEqual(at, 0,
                                      $"no GameObject block for fileID {component.GameObjectId}");

                int end = scene.IndexOf("\n--- ", at + 1, StringComparison.Ordinal);
                string block = end < 0 ? scene.Substring(at) : scene.Substring(at, end - at);

                StringAssert.Contains($"component: {{fileID: {component.FileId}}}", block,
                                      "the settings component is not in its host's m_Component "
                                      + "list, so it would not show in the inspector");
            }
        }
    }
}
