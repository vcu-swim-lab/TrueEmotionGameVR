using UnityEngine;
using UnityEngine.UI;

public class ProgressBar : MonoBehaviour
{
    public int maximum;
    public int current;
    public Image progressFill;
    public Image progressBarBack;


    [ExecuteInEditMode()]
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        current = 0;
    }

    // Update is called once per frame
    void Update()
    {
        GetCurrentFill();
    }

    public void SetCurrent(int newCurrent)
    {
        current = newCurrent;
        GetCurrentFill(); 
    }

    public void SetMaximum(int newMaximum)
    {
        maximum = newMaximum;
        GetCurrentFill(); 
    }
    public void SetProgressBarVisible(bool isVisible)
    {
        if (progressBarBack != null)
            progressBarBack.gameObject.SetActive(isVisible);

        if (progressFill != null)
            progressFill.gameObject.SetActive(isVisible);
    }

    void GetCurrentFill()
    {
        if (progressFill == null)
        {
            float fillAmount = (float)current / (float)maximum;
            progressFill.fillAmount = fillAmount;
        }
    }
}
