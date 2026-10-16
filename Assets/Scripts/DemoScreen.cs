using TMPro;
using UnityEngine;
using UnityEngine.UI;

// The demo's screens (ProtocolRunner's Demo checkbox): a title, a large centre text (the
// emoji), a line under it and a remaining-time bar. Built at run time as copies of the scene's
// instruction text -- same font, emoji sprites and render layer -- so the demo needs no scene
// changes and the study's own screens are left exactly as they are.
public class DemoScreen
{
    // Canvas units. The canvas is 800 x 447 with its origin at the centre; the content is
    // kept within about +-120 of it, so it stays near eye level.
    private const float TextWidth = 700f;
    private const float TextHeight = 100f;
    private const float BarWidth = 420f;
    private const float BarHeight = 22f;
    private const float BarBorder = 2f;

    // The remaining-time bar: green for the time left, black for the time gone, white border.
    private static readonly Color BarBorderColor = Color.white;
    private static readonly Color BarBackColor = Color.black;
    private static readonly Color BarFillColor = new Color(0.18f, 0.49f, 0.20f);

    // The canvas background during the demo (the study keeps the scene's grey).
    private static readonly Color BackgroundColor = new Color32(0x14, 0x16, 0x1C, 0xFF);

    public const float BodySize = 22f;
    private const float TitleSize = 30f;
    private const float HeadlineSize = 28f;

    private readonly GameObject root;
    private readonly TextMeshProUGUI title;
    private readonly TextMeshProUGUI center;
    private readonly TextMeshProUGUI below;
    private readonly RectTransform bar;
    private readonly RectTransform barFill;

    private float timerStart;
    private float timerLength;
    private bool timerRunning;

    public DemoScreen(TextMeshProUGUI template)
    {
        var parent = template.rectTransform.parent;
        int layer = template.gameObject.layer;

        var background = parent.Find("Background");
        var backgroundImage = background != null ? background.GetComponent<Image>() : null;
        if (backgroundImage != null)
        {
            backgroundImage.color = BackgroundColor;
        }

        root = NewUiObject("Demo", parent, layer);
        var rootRect = root.GetComponent<RectTransform>();
        rootRect.anchorMin = Vector2.zero;
        rootRect.anchorMax = Vector2.one;
        rootRect.offsetMin = rootRect.offsetMax = Vector2.zero;

        title = CopyText(template, "DemoTitle");
        center = CopyText(template, "DemoCenter");
        below = CopyText(template, "DemoBelow");

        bar = NewUiObject("DemoTimeBar", root.transform, layer).GetComponent<RectTransform>();
        Center(bar, 0f, new Vector2(BarWidth + 2 * BarBorder, BarHeight + 2 * BarBorder));
        AddImage(bar.gameObject, BarBorderColor);

        var back = NewUiObject("Back", bar, layer).GetComponent<RectTransform>();
        back.anchorMin = Vector2.zero;
        back.anchorMax = Vector2.one;
        back.offsetMin = new Vector2(BarBorder, BarBorder);
        back.offsetMax = new Vector2(-BarBorder, -BarBorder);
        AddImage(back.gameObject, BarBackColor);

        barFill = NewUiObject("Fill", back, layer).GetComponent<RectTransform>();
        barFill.anchorMin = Vector2.zero;
        barFill.anchorMax = Vector2.one;
        barFill.offsetMin = barFill.offsetMax = Vector2.zero;
        AddImage(barFill.gameObject, BarFillColor);

        Hide();
    }

    // The hold phase: "Act <emotion>" at the top, the emoji under it, a line under that and
    // the remaining-time bar at the bottom.
    public void ShowHold(string titleText, string centerText, float centerSize, string belowText,
                         float seconds)
    {
        Set(title, 95f, TitleSize, titleText);
        Set(center, 20f, centerSize, centerText);
        Set(below, -45f, BodySize, belowText);
        ShowBar(-85f, seconds);
        root.SetActive(true);
    }

    // A text-only screen, centred, with no bar.
    public void ShowText(string text)
    {
        Set(title, 0f, BodySize, "");
        Set(center, 0f, BodySize, text);
        Set(below, 0f, BodySize, "");
        HideBar();
        root.SetActive(true);
    }

    // A headline, a smaller text under it and the remaining-time bar at the bottom.
    public void ShowResult(string headline, string belowText, float seconds)
    {
        Set(title, 60f, HeadlineSize, headline);
        Set(center, 0f, BodySize, "");
        Set(below, -35f, BodySize, belowText);
        ShowBar(-105f, seconds);
        root.SetActive(true);
    }

    public void Hide()
    {
        HideBar();
        root.SetActive(false);
    }

    // Called every frame; drains the bar from the left as the time runs out.
    public void Tick()
    {
        if (!timerRunning)
        {
            return;
        }

        float left = 1f - Mathf.Clamp01((Time.time - timerStart) / timerLength);
        barFill.anchorMin = new Vector2(1f - left, 0f);
    }

    private void ShowBar(float y, float seconds)
    {
        bar.anchoredPosition = new Vector2(0f, y);
        bar.gameObject.SetActive(true);
        timerStart = Time.time;
        timerLength = Mathf.Max(seconds, 0.01f);
        timerRunning = true;
        Tick();
    }

    private void HideBar()
    {
        timerRunning = false;
        bar.gameObject.SetActive(false);
    }

    private static void Set(TextMeshProUGUI text, float y, float size, string value)
    {
        text.rectTransform.anchoredPosition = new Vector2(0f, y);
        text.fontSize = size;
        text.text = value;
    }

    private TextMeshProUGUI CopyText(TextMeshProUGUI template, string name)
    {
        var copy = Object.Instantiate(template, root.transform);
        copy.name = name;
        copy.text = "";
        Center(copy.rectTransform, 0f, new Vector2(TextWidth, TextHeight));
        return copy;
    }

    private static void Center(RectTransform rect, float y, Vector2 size)
    {
        rect.anchorMin = rect.anchorMax = rect.pivot = new Vector2(0.5f, 0.5f);
        rect.anchoredPosition = new Vector2(0f, y);
        rect.sizeDelta = size;
    }

    private static GameObject NewUiObject(string name, Transform parent, int layer)
    {
        var go = new GameObject(name, typeof(RectTransform));
        go.layer = layer;
        go.transform.SetParent(parent, false);
        return go;
    }

    private static void AddImage(GameObject go, Color color)
    {
        var image = go.AddComponent<Image>();
        image.color = color;
        image.raycastTarget = false;
    }
}
