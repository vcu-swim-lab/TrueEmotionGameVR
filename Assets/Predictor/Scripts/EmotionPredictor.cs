
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
            int capturedIndex = i;
            AwaitAndCount(awaitables[i], onDone: res =>
            {
                ++completed;
                results[capturedIndex] = res;

                if (completed == total)
                {
                    completion.SetResult();
                }
            });
        }

        await completion.Awaitable;
        return results;
    }

    internal static async void AwaitAndCount<T>(Awaitable<T> awaitable, Action<T> onDone)
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


#region device

public abstract class Device
{
    public abstract (InputType, int, int) InputType { get; }
    public abstract void Write(float[] data);

    public static implicit operator DeviceReader(Device d)
    {
        return new DeviceReader(d);
    }
}

public class AUDevice : Device
{
    public AUDevice(OVRFaceExpressions faceExpressions) => this.faceExpressions = faceExpressions;

    public override (InputType, int, int) InputType => (global::InputType.FaceAU, 70, 5);

    public override void Write(float[] data)
    {
        if (faceExpressions.ValidExpressions)
        {
            faceExpressions.CopyTo(data);
        }
    }

    private readonly OVRFaceExpressions faceExpressions;
}

public class SoundDevice : Device
{
    public override (InputType, int, int) InputType => throw new NotImplementedException();

    public override void Write(float[] data)
    {
        throw new NotImplementedException();
    }
}


public class DeviceReader
{
    public DeviceReader(Device device)
        : this(device, device.InputType.Item3) { }

    // Frequency is in frames per second, defaults to number of frames the device needs.
    public DeviceReader(Device device, int frequency)
    {
        Step = 1.0f / frequency;
        this.device = device;

        var (_, w, h) = device.InputType;
        data = new float[h][];
        for (int i = 0; i < h; ++i)
        {
            data[i] = new float[w];
        }

        max = h;
    }

    // dummy API
    internal void Poll()
    {
        if (index == max) return;
        device.Write(data[index++]);
    }

    internal void Flush()
    {
        index = 0;
    }

    public bool IsReady => index == max;

    public (InputType, Tensor<float>) Data()
    {
        var (ty, w, h) = device.InputType;
        var flat = data.SelectMany(sub => sub).ToArray();
        index = 0;
        return (ty, new(new TensorShape(1, h, w), flat));
    }

    internal float Step { get; }
    internal readonly Device device;

    private readonly float[][] data;
    private int index = 0;
    private readonly int max;
}

#endregion


public class EmotionPredictor : MonoBehaviour
{
    [SerializeField]
    private PredictorConfig config;

    void Start()
    {
        var face = gameObject.GetComponentInParent<OVRFaceExpressions>();
        var readers = new DeviceReader[] { new AUDevice(face) };
        var models = new ModelAsset[][] { new[] { config.auModel } };

        Setup(readers, models);
        Polling = config.pollByDefault;
    }

    void Setup(DeviceReader[] readers, ModelAsset[][] models)
    {
        this.readers = readers;
        nextPollTime = new float[readers.Length];

        for (int i = 0; i < readers.Length; ++i)
        {
            nextPollTime[i] = Time.time + readers[i].Step;
        }

        for (int i = 0; i < models.Length; ++i)
        {
            foreach (var modelAsset in models[i])
            {
                var type = (InputType)i;
                ++i;

                var model = ModelLoader.Load(modelAsset);
                var worker = new Worker(model, BackendType.GPUCompute);

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

    public async Awaitable<(Emotion, float)> Predict()
    {
        if (readers.Length == 0)
        {
            Debug.LogWarning("No devices attached to EmotionPredictor; prediction will always return `Neutral`. Make sure to call `EmotionPredictor.Listen` before calling `Predict`.");
            return (Emotion.Neutral, 1f);
        }

        // aPredIsWaitingData checks if another prediction is waiting for data to be ready
        while (aPredIsWaitingData)
        {
            await Awaitable.NextFrameAsync();
        }

        aPredIsWaitingData = true;

        // Wait until all readers are ready
        // This is to ensure that all data is collected before making a prediction
        bool ready;
        do
        {
            ready = true;

            foreach (var reader in readers)
            {
                if (!reader.IsReady)
                {
                    await Awaitable.NextFrameAsync();
                    ready = false;
                    break;
                }
            }
        } while (!ready);

        aPredIsWaitingData = false;

        Dictionary<InputType, Tensor<float>> data = new();
        foreach (var reader in readers)
        {
            var (ty, tensor) = reader.Data();
            data[ty] = tensor;
        }

        var pred = Predict(data);

        foreach ((_, var tensor) in data)
        {
            tensor.Dispose();
        }

        return await pred;
    }

    private async Awaitable<(Emotion, float)> Predict(Dictionary<InputType, Tensor<float>> data)
    {
        int count = 0;
        foreach (var (type, tensor) in data)
        {
            if (!entries.ContainsKey(type))
            {
                Debug.LogWarning($"Data from {type} is passed for prediction but no model that supports this data was registered.");
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

        foreach (var outT in outputs)
        {
            outT.Dispose();
        }


        return ((Emotion)iEmo, maxProb);
    }

    public void Flush()
    {
        foreach (var reader in readers)
        {
            reader.Flush();
        }
    }

    public bool Polling { get; set; } = true;

    void Update()
    {
        if (!Polling) return;

        for (int i = 0; i < readers.Length; ++i)
        {
            if (Time.time >= nextPollTime[i])
            {
                readers[i].Poll();
                nextPollTime[i] += readers[i].Step;
            }
        }
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

    private DeviceReader[] readers;
    private float[] nextPollTime;

    private bool aPredIsWaitingData = false;

    // invariant: if `entries[type]` exists, it has at least 1 entry
    internal readonly Dictionary<InputType, List<Entry>> entries = new();
}