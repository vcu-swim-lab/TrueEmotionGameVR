using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(EmotionPredictor))]
public class EmotionPredictor_Editor : Editor
{
    ModelAsset auModel;

    void OnEnable()
    {
        var pred = (EmotionPredictor)target;
        // TODO: this is not always true
        auModel = pred.entries[InputType.FaceAU][0].asset;
    }

    public override void OnInspectorGUI()
    {
        serializedObject.Update();

        EditorGUI.BeginChangeCheck();

        auModel = (ModelAsset)EditorGUILayout.ObjectField(
            "AU Model",
            auModel,
            typeof(ModelAsset),
            false
        );

        var changed = EditorGUI.EndChangeCheck();

        serializedObject.ApplyModifiedProperties();

        if (changed)
        {
            Debug.Assert(auModel != null, "You should specify an AU model for now.");

            var obj = ((EmotionPredictor)target).gameObject;

            if (!obj.TryGetComponent<OVRFaceExpressions>(out var face))
            {
                face = obj.AddComponent<OVRFaceExpressions>();
            }

            // TODO: add more models
            obj.GetComponent<EmotionPredictor>().Setup(
                new[] { new AUDevice(face) },
            new ModelAsset[][]{
                new[] { auModel },
            }//
            );

            EditorUtility.SetDirty(target);
        }
    }
}
