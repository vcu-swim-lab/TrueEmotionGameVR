
using System;
using System.Collections.Generic;
using System.Linq;
using Unity.InferenceEngine;
using UnityEngine;

// TODO: only initialize `ModelInput` entries for specified input types

#region helpers
internal class Async
{
    internal static async Awaitable<T[]> WhenAll<T>(params Awaitable<T>[] awaitables)
    {
        int completed = 0;
        int total = awaitables.Length;

        var completion = new AwaitableCompletionSource();
        var results = new T[awaitables.Length];

        for (int i = 0; i < awaitables.Length; ++i)
        {
            _ = AwaitAndCount(awaitables[i], onDone: res =>
            {
                ++completed;
                results[i] = res;

                if (completed == total)
                {
                    completion.SetResult();
                }
            });
        }

        await completion.Awaitable;
        return results;
    }

    internal static async Awaitable AwaitAndCount<T>(Awaitable<T> awaitable, Action<T> onDone)
    {
        var res = await awaitable;
        onDone.Invoke(res);
    }
}

#endregion

public enum Emotion
{
    Anger,
    Disgust,
    Fear,
    Happiness,
    Neutral,
    Sadness,
    Surprise,
}

public enum InputType
{
    FaceAU,
    Sound, // not implemented yet

    Count, // metadata
}

class ModelInput
{
    internal ModelInput()
    {
        data = new();
        SetupArray(InputType.FaceAU, 70, 5);
    }

    internal float[] NextFrame(InputType type)
    {
        int len = data[type].Length;

        // If this part is skipped, the new frame overwrites the last one.
        // We want the frame to be added to the end and push the older frames backwards, so we simply swap move the frames first.
        /*
            So think like this:
            1 2 3 4 5
                    ^--- cursor here
            Ok no problem so far. But:

            1 2 3 4 6
                    ^
            Oops, the last frame is in the wrong place now. Just swap move the first frame: (6, 2, 3, 4, 5) -> (2, 6, 3, 4, 5) -> (2, 3, 6, 4, 5) -> (2, 3, 4, 6, 5) -> (2, 3, 4, 5, 6)
        */
        for (int i = 0; i < len - 1; ++i)
        {
            // TODO: does this work?
            (data[type][i], data[type][i + 1]) = (data[type][i + 1], data[type][i]);
        }

        return data[type][len - 1];
    }

    internal Dictionary<InputType, Tensor> Prepare()
    {
        Dictionary<InputType, Tensor> tensors = new();
        foreach (var (k, v) in data)
        {
            // flatten the array
            var flat = v.SelectMany(sub => sub).ToArray();
            var t = new Tensor<float>(new TensorShape(1, v.Length, v[0].Length), flat);

            tensors.Add(k, t);
        }

        return tensors;
    }

    private void SetupArray(InputType type, int w, int h)
    {
        data[type] = new float[h][];
        for (int i = 0; i < h; ++i)
        {
            data[type][i] = new float[w];
        }
    }

    private readonly Dictionary<InputType, float[][]> data;
}


#region device

public interface Device
{
    InputType InputType { get; }
    void Write(float[] data);
}

public class AUDevice : Device
{
    public AUDevice(OVRFaceExpressions faceExpressions)
    {
        this.faceExpressions = faceExpressions;
    }

    public InputType InputType { get => InputType.FaceAU; }

    // TODO: implement
    public void Write(float[] data)
    {
        faceExpressions.CopyTo(data);
    }

    private readonly OVRFaceExpressions faceExpressions;
}

public class SoundDevice : Device
{
    public InputType InputType { get => InputType.Sound; }

    public void Write(float[] data)
    {
        throw new NotImplementedException();
    }
}

#endregion


public class EmotionPredictor : MonoBehaviour
{
    public void Setup(Device[] devices, ModelAsset[][] models)
    {
        this.devices = devices;

        for (int i = 0; i < models.Length; ++i)
        {
            foreach (var modelAsset in models[i])
            {
                var type = (InputType)i;
                ++i;

                var model = ModelLoader.Load(modelAsset);
                var worker = new Worker(model, BackendType.CPU);

                entries.TryAdd(type, new());

                entries[type].Add(new Entry
                {
                    asset = modelAsset,
                    model = model,
                    worker = worker,
                });
            }
        }
    }

    public async Awaitable<Emotion> Predict()
    {
        if (devices.Length == 0)
        {
            Debug.LogWarning("No devices attached to EmotionPredictor; prediction will always return `Neutral`. Make sure to call `EmotionPredictor.Listen` before calling `Predict`.");
            return Emotion.Neutral;
        }

        foreach (var device in devices)
        {
            device.Write(inputs.NextFrame(device.InputType));
        }

        var data = inputs.Prepare();
        var pred = Predict(data);

        foreach ((_, var tensor) in data)
        {
            tensor.Dispose();
        }

        return await pred;
    }

    private async Awaitable<Emotion> Predict(Dictionary<InputType, Tensor> data)
    {
        int count = 0;
        foreach (var (type, tensor) in data)
        {
            if (!entries.ContainsKey(type))
            {
                Debug.LogWarningFormat("Data from {} is passed for prediction but no model that supports this data was registered.", type);
            }
            else
            {
                count += entries[type].Count;
            }
        }

        Debug.Assert(count != 0, "No model was registered for prediction! Did you forget to setup?");


        List<Awaitable<Tensor<float>>> outputsAwaiters = new();

        foreach (var (type, tensor) in data)
        {
            if (entries.TryGetValue(type, out var list))
            {
                foreach (var entry in list)
                {
                    entry.worker.Schedule(tensor);

                    var t = entry.worker.PeekOutput() as Tensor<float>;
                    outputsAwaiters.Add(t.ReadbackAndCloneAsync());
                }
            }
        }

        var outputs = await Async.WhenAll(outputsAwaiters.ToArray());

        // TODO: decide here by voting or something

        var output = outputs[0].DownloadToArray();

        var maxProb = output.Max();
        var iEmo = Array.IndexOf(output, maxProb);

        return (Emotion)iEmo;
    }

    void OnDestroy()
    {
        foreach (var list in entries.Values)
        {
            foreach (var entry in list)
            {
                entry.worker.Dispose();
            }
        }
    }


    internal struct Entry
    {
        internal ModelAsset asset; // used by the inspector/editor
        public Model model;
        public Worker worker;
    }

    private Device[] devices;
    private readonly ModelInput inputs = new();

    // invariant: if `entries[type]` exists, it has at least 1 entry
    internal readonly Dictionary<InputType, List<Entry>> entries = new();
}