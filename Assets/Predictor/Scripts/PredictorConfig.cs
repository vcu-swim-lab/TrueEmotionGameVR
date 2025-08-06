using Unity.InferenceEngine;
using UnityEngine;

[CreateAssetMenu(fileName = "PredictorConfig", menuName = "Scriptable Objects/PredictorConfig")]
public class PredictorConfig : ScriptableObject
{
    [SerializeField]
    internal ModelAsset auModel;
    [SerializeField]
    internal bool pollByDefault = false;
}
