using System.Collections;
using UnityEngine;


public class ViewpointCamera : MonoBehaviour
{
    [SerializeField] private float distance = 0.5f;
    [SerializeField] private Vector3 offset = Vector3.zero;

    void Start()
    {
        StartCoroutine(FindViewpoint());
    }

    private IEnumerator FindViewpoint()
    {
        yield return new WaitForSeconds(0.1f);

        while (true)
        {
            // Find the Viewpoint object
            GameObject viewpointObj = GameObject.Find("Viewpoint");
            if (viewpointObj != null)
            {
                var viewpoint = viewpointObj.transform;

                while (true)
                {
                    yield return null;

                    // Position camera in front of Aura's face (negative forward direction)
                    transform.position = viewpoint.position + offset + (-viewpoint.forward * distance);
                    // Look at Aura's face
                    transform.LookAt(viewpoint.position + offset);
                }
            }
            else
            {
                Debug.LogWarning("Could not find Viewpoint. Will try again in 0.5 seconds.");
                yield return new WaitForSeconds(0.5f); // try again after a delay
            }
        }
    }

    public void SetDistance(float newDistance)
    {
        distance = newDistance;
    }

    public void SetOffset(Vector3 newOffset)
    {
        offset = newOffset;
    }
}