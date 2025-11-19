using UnityEngine;
using System.Collections;

public class FlyingObject : MonoBehaviour
{
    public float horizontalSpeed = 0.08f;
    public float verticalSpeed = 2.0f;
    public float height = 2.0f; 

    public Vector3 tempPosition; 
    void Start()
    {
        tempPosition = transform.position;
    }

    void FixedUpdate()
    {
        tempPosition.x += horizontalSpeed;
        tempPosition.y = Mathf.Sin(Time.realtimeSinceStartup * verticalSpeed) * height;
        transform.position = tempPosition;
    }
}
