using UnityEngine;

// TODO:
// https://discussions.unity.com/t/increase-draw-distance-of-ovrcamerarig/252138

public class AdjustVideoDistance : MonoBehaviour
{
    public Camera vrCamera; // Assign the CenterEyeAnchor camera
    public float videoDistance = 10f; // Default distance for the far plane

    void Update()
    {
        if (vrCamera != null)
        {
            // Dynamically adjust the far clipping plane to control the video distance
            vrCamera.farClipPlane = videoDistance;
        }
    }
}
