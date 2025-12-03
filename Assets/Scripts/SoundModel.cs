using System.Diagnostics;
using Unity.InferenceEngine;
using Unity.Collections;
using UnityEngine;
using System.Collections.Generic;

using Debug = UnityEngine.Debug;
using Mono.Cecil;

// [RequireComponent(typeof(DeviceManager))]
public class SoundModel : MonoBehaviour
{
    // logmel

    [SerializeField]
    private ModelAsset voiceModel;

    [SerializeField]
    private ModelAsset logmelVoiceModel;

    private Model voiceModelObject;
    private Model logmelVoiceModelObject;

    private Worker voiceWorker;
    private Worker logmelVoiceWorker;

    // private RingBuffer<float[]> inputBuffer;

    void Start()
    {
        Debug.Assert(voiceModel != null, $"Voice model not assigned in ${GetType().Name}");

        voiceModelObject = ModelLoader.Load(voiceModel);
        voiceWorker = new Worker(voiceModelObject, BackendType.CPU);


        logmelVoiceModelObject = ModelLoader.Load(logmelVoiceModel);
        logmelVoiceWorker = new Worker(logmelVoiceModelObject, BackendType.CPU);

        // var deviceManager = GetComponent<DeviceManager>();

        // var voiceDevice = deviceManager.Require(InputType.Sound);
        // if (voiceDevice == null)
        // {
        //     Debug.LogError("Could not create microphone!");
        // }

        // inputBuffer = new(30, () => new float[16000]);
        // inputBuffer.Listen(voiceDevice);

        Debug.Log("SoundModel initialized.");
    }

    async Awaitable<Tensor<float>> Infer(Tensor<float> input)
    {
        logmelVoiceWorker.Schedule(input);

        var logMelTensor = logmelVoiceWorker.PeekOutput();

        voiceWorker.Schedule(logMelTensor);

        var voiceTensor = await voiceWorker.PeekOutput().ReadbackAndCloneAsync();
        return voiceTensor as Tensor<float>;
    }

    // TODO: `Predict` that doesn't wait on new data, just uses whatever is in the buffer.

    public async Awaitable<Tensor<float>> PredictRaw(Tensor<float> audioInput = null)
    {
        Tensor<float> input;

        // if (audioInput != null)
        // {
            input = audioInput;
        // }
        // else
        // {
        //     inputBuffer.Clear();

        //     while (!inputBuffer.Full)
        //     {
        //         await Awaitable.NextFrameAsync();
        //     }

        //     // TODO: should you dispose the tensor?
        //     input = inputBuffer.ToTensor();
        // }

        return await Infer(input);
    }


    public async Awaitable<(Emotion, float)> Predict(Tensor<float> audioInput = null)
    {
        var voiceTensor = await PredictRaw(audioInput);
        var voiceArr = voiceTensor.AsReadOnlyNativeArray();

        int maxIndex = 0;
        float maxValue = voiceArr[0];
        for (int i = 1; i < voiceArr.Length; ++i)
        {
            if (voiceArr[i] > maxValue)
            {
                maxValue = voiceArr[i];
                maxIndex = i;
            }
        }

        voiceTensor.Dispose();

        return ((Emotion)maxIndex, maxValue);
    }


    public void Dispose()
    {
        voiceWorker.Dispose();
    }
}