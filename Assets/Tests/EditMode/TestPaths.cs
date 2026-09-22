using System;
using System.IO;
using System.Reflection;

namespace VcuSwimLab.Calibration.Tests
{
    /// <summary>
    /// Locates repository files from a test, in Unity and outside it.
    /// </summary>
    /// <remarks>
    /// These tests run in two places: the Unity Editor's EditMode runner, and `dotnet test`
    /// against `Analysis/CsharpTests/Calibration.Tests.csproj`, which links the same source
    /// files. The second is what makes the Python/C# parity check runnable without opening
    /// Unity -- and therefore what makes it runnable in the same breath as the Python tests.
    ///
    /// Neither `Application.dataPath` nor a relative path works in both, so the root is found
    /// by walking up from the assembly on disk looking for the markers a checkout has. In
    /// Unity the assembly sits in `Library/ScriptAssemblies`; under `dotnet test` it sits in
    /// `Analysis/CsharpTests/bin/...`. Both are inside the project.
    /// </remarks>
    public static class TestPaths
    {
        public static string RepositoryRoot
        {
            get
            {
                var directory = new DirectoryInfo(
                    Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)
                    ?? Directory.GetCurrentDirectory());

                while (directory != null)
                {
                    if (Directory.Exists(Path.Combine(directory.FullName, "Assets"))
                        && Directory.Exists(Path.Combine(directory.FullName, "Analysis")))
                    {
                        return directory.FullName;
                    }

                    directory = directory.Parent;
                }

                throw new InvalidOperationException(
                    "could not find the project root: no ancestor of "
                    + Assembly.GetExecutingAssembly().Location
                    + " holds both an Assets and an Analysis directory");
            }
        }

        public static string PopulationArtifact => Path.Combine(
            RepositoryRoot, "Assets", "StreamingAssets",
            "emojihero_population_centroids_unity.json");

        public static string ParityFixture => Path.Combine(
            RepositoryRoot, "Analysis", "fixtures", "runtime_calibration_parity.json");

        public static string ReadPopulationArtifact()
        {
            string path = PopulationArtifact;
            if (!File.Exists(path))
            {
                throw new FileNotFoundException(
                    "no exported population artifact. Run "
                    + "`python Analysis/export_population_artifact_for_unity.py`.", path);
            }

            return File.ReadAllText(path);
        }

        public static string ReadParityFixture()
        {
            string path = ParityFixture;
            if (!File.Exists(path))
            {
                throw new FileNotFoundException(
                    "no parity fixture. Run "
                    + "`python Analysis/make_runtime_parity_fixture.py`.", path);
            }

            return File.ReadAllText(path);
        }
    }
}
