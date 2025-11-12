using System.Diagnostics;
using Unity.InferenceEngine;
using Unity.Collections;
using UnityEngine;
using System.Collections.Generic;

using Debug = UnityEngine.Debug;

[RequireComponent(typeof(OVRFaceExpressions))]
[RequireComponent(typeof(DeviceManager))]
public class NewFaceAuModel : MonoBehaviour
{
    [SerializeField]
    private ModelAsset faceModel;

    private Model faceModelObject;
    private Worker faceWorker;

    private RingBuffer<float[]> inputBuffer;

    public void Start()
    {
        Debug.Assert(faceModel != null, $"Face model not assigned in ${GetType().Name}");

        faceModelObject = ModelLoader.Load(faceModel);
        faceWorker = new Worker(faceModelObject, BackendType.CPU);

        var deviceManager = GetComponent<DeviceManager>();

        var auDevice = deviceManager.Require(InputType.FaceAU);
        if (auDevice == null)
        {
            Debug.LogError("Could not create action units device!");
        }

        inputBuffer = new(30, () => new float[64]);
        inputBuffer.Listen(auDevice);

        Debug.Log("NewFaceAuModel initialized.");
    }

    async Awaitable<Tensor<float>> Infer(Tensor<float> input)
    {
        faceWorker.Schedule(input);

        var faceTensor = await faceWorker.PeekOutput().ReadbackAndCloneAsync();
        return faceTensor as Tensor<float>;
    }

    // TODO: `Predict` that doesn't wait on new data, just uses whatever is in the buffer.

    public async Awaitable<Tensor<float>> PredictRaw()
    {
        inputBuffer.Clear();

        while (!inputBuffer.Full)
        {
            await Awaitable.NextFrameAsync();
        }

        // TODO: should you dispose the tensor?
        var input = inputBuffer.ToTensor();

        return await Infer(input);
    }

    private static readonly Dictionary<Emotion, float> naturalWeightMap = new()
    {
        { Emotion.Neutral, 0.5f },
        { Emotion.Happiness, 0.3f },
        { Emotion.Sadness, 0.7f },
        { Emotion.Anger, 0.5f },
        { Emotion.Fear, 0.5f },
        { Emotion.Disgust, 1f },
        { Emotion.Surprise, 0.3f },
    };
    public async Awaitable<(Emotion, float)> Predict()
    {
        var face = await PredictRaw();
        var faceArr = face.AsReadOnlyNativeArray();

        var outputArray = new NativeArray<float>(faceArr.Length, Allocator.Temp);
        for (int i = 0; i < faceArr.Length; ++i)
            outputArray[i] = faceArr[i] * naturalWeightMap[(Emotion)i];

        int maxIndex = 0;
        float maxValue = outputArray[0];
        for (int i = 1; i < outputArray.Length; ++i)
        {
            if (outputArray[i] > maxValue)
            {
                maxValue = outputArray[i];
                maxIndex = i;
            }
        }

        face.Dispose();

        return ((Emotion)maxIndex, maxValue);
    }


    public void Dispose()
    {
        faceWorker.Dispose();
    }
}