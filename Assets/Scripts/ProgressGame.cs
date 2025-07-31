using System.Collections;
using UnityEngine;
using TMPro;



public class ProgressGame : MonoBehaviour
{
    private TextMeshProUGUI text;
    private static readonly string[] emojis = new string[]
    {
        "🤬",
        "🤢",
        "😱",
        "😀",
        "☹️",
        "😲",
    };

    void Start()
    {
        text = GameObject.Find("Instructions").GetComponent<TextMeshProUGUI>();

        StartCoroutine(RunGame());
    }

    private IEnumerator RunGame()
    {
        for (int i = emojis.Length - 1; i > 0; --i)
        {
            int j = Random.Range(0, i);
            (emojis[i], emojis[j]) = (emojis[j], emojis[i]);
        }

        text.text = "You got 1 minute to act each emotion shown to you. Good luck.";
        yield return new WaitForSeconds(0.5f);


        for (int i = 0; i < 6; ++i)
        {
            for (int j = 3; j >= 0; --j)
            {
                text.text = $"{j}";
                yield return new WaitForSeconds(1.0f);
            }

            text.text = emojis[i];
            // TODO: start recording here
            yield return new WaitForSeconds(10);
            // TODO: stop recording here
        }
    }
}
