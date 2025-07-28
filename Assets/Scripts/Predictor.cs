using UnityEngine;
using Unity.InferenceEngine;


// Step 1. Require EmotionPredictor; make sure to set it up accordingly in the inspector
[RequireComponent(typeof(EmotionPredictor))]
public class Predictor : MonoBehaviour
{
    // Step 2. Define EmotionPredictor member.
    private EmotionPredictor predictor;

    void Awake()
    {
        // Step 3. Initialize predictor on Awake or Start, your choice
        predictor = GetComponent<EmotionPredictor>();
    }

    void Update()
    {
        // Step 4. Get a prediction based on the sources configured from the inspector and use it.
        var emo = predictor.Predict();
        print(emo);
    }
}
