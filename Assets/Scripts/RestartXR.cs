using UnityEngine;
using UnityEngine.XR.Management;

public class XRSubsystemManager : MonoBehaviour
{
    void OnApplicationQuit()
    {
        var xrManager = XRGeneralSettings.Instance.Manager;
        if (xrManager.isInitializationComplete)
        {
            xrManager.StopSubsystems();
            xrManager.DeinitializeLoader();
        }
    }
}
