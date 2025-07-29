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

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        text = GameObject.Find("Instructions").GetComponent<TextMeshProUGUI>();

        StartCoroutine(RunGame());
    }

    /*
        sched:
            tasks: Task[]
    
        update():
            for task in tasks:
                resume(task)


        task.yield(value): // yield value;
            if value.ready:
                continue with task
            else:
                sched.push(task)
    */

    private IEnumerator RunGame()
    {
        for (int i = emojis.Length - 1; i > 0; --i)
        {
            int j = Random.Range(0, i);
            (emojis[i], emojis[j]) = (emojis[j], emojis[i]);
        }

        text.text = "You got 1 minute to act each emotion shown to you. Good luck.";
        yield return new WaitForSeconds(0.5f); // <-- stopped here

        /*
            yield return null; // wait for next frame
            yield return new WaitForSeconds(...); // can also wait for ms
        */

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
