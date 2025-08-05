// using Unity.InferenceEngine;
// using UnityEditor;
// using UnityEngine;

// [CustomEditor(typeof(EmotionPredictor))]
// public class EmotionPredictor_Editor : Editor
// {
//     private SerializedProperty setupValueProp;
//     private bool isInit = false;

//     void OnEnable()
//     {
//         setupValueProp = serializedObject.FindProperty("AU Model");
//         Debug.Log($"Type of property: {setupValueProp.type}");

//         if (!isInit)
//         {
//             doSetup(target, (ModelAsset)setupValueProp.boxedValue);

//             isInit = true;
//         }

//         // var pred = (EmotionPredictor)target;
//         // TODO: this is not always true
//         // auModel = pred.entries[InputType.FaceAU][0].asset;
//     }

//     public override void OnInspectorGUI()
//     {
//         serializedObject.Update();

//         EditorGUI.BeginChangeCheck();

//         EditorGUILayout.PropertyField(setupValueProp, new GUIContent("FACS"));

//         var changed = EditorGUI.EndChangeCheck();


//         if (changed)
//         {
//             serializedObject.ApplyModifiedProperties();

//             doSetup(target, (ModelAsset)setupValueProp.boxedValue);

//             EditorUtility.SetDirty(target);
//         }
//     }

//     private static void doSetup(Object target, ModelAsset auModel)
//     {
//         var pred = (EmotionPredictor)target;
//         var obj = pred.gameObject;

//         if (!obj.TryGetComponent<OVRFaceExpressions>(out var face))
//         {
//             face = obj.AddComponent<OVRFaceExpressions>();
//         }

//         face.enabled = true;

//         pred.Setup(
//             new DeviceReader[] { new AUDevice(face) },
//             new ModelAsset[][]{
//                 new []{auModel }
//             }
//         );
//     }
// }
