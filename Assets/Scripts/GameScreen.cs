using TMPro;
using UnityEngine;
using UnityEngine.UI;

// The game's trial screens (ProtocolRunner): a title, a centre text, a line under it, and
// either a remaining-time bar (acting) or a circle countdown (get ready, result). Built at run
// time as copies of the scene's instruction text -- same font, emoji sprites and render
// layer -- so they need no scene changes. The start, part and end screens use the instruction
// text itself.
public class GameScreen
{
    // Canvas units. The canvas is 800 x 447 with its origin at the centre; the content is
    // kept within about +-120 of it, so it stays near eye level.
    // Narrow, so no line is wide enough to need a head turn to read.
    private const float TextWidth = 480f;
    private const float TextHeight = 100f;
    private const float BarWidth = 420f;
    private const float BarHeight = 22f;
    private const float BarBorder = 2f;

    // The remaining-time bar: green for the time left, black for the time gone, white border.
    private static readonly Color BarBorderColor = Color.white;
    private static readonly Color BarBackColor = Color.black;
    private static readonly Color BarFillColor = new Color(0.18f, 0.49f, 0.20f);

    // The circle countdown: a number inside a ring that empties once a second, green for the
    // part of the second left, dark grey for the part gone.
    private const float RingSize = 60f;
    private const float RingNumberSize = 26f;
    private const int RingTexture = 128;
    private const float RingThickness = 0.23f; // of the radius
    private static readonly Color RingBackColor = new Color32(0x3C, 0x40, 0x4A, 0xFF);
    private static readonly Color RingFillColor = new Color32(0x2E, 0xA0, 0x46, 0xFF);

    // The canvas background (the scene's own is grey).
    private static readonly Color BackgroundColor = new Color32(0x14, 0x16, 0x1C, 0xFF);

    public const float BodySize = 22f;
    public const float GetReadySize = 26f;
    public const float HeadlineSize = 28f;
    private const float TitleSize = 30f;

    private readonly GameObject root;
    private readonly TextMeshProUGUI title;
    private readonly TextMeshProUGUI center;
    private readonly TextMeshProUGUI below;
    private readonly RectTransform bar;
    private readonly RectTransform barFill;
    private readonly GameObject ring;
    private readonly Image ringFill;
    private readonly TextMeshProUGUI ringNumber;

    private float timerStart;
    private float timerLength;
    private bool timerRunning;

    private float countdownStart;
    private float countdownStep;
    private int countdownFrom;
    private bool countdownRunning;
    private string countdownLabel = "";

    public GameScreen(TextMeshProUGUI template)
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

        var ringSprite = RingSprite();
        var ringRect = NewUiObject("DemoCountdown", root.transform, layer).GetComponent<RectTransform>();
        Center(ringRect, 0f, new Vector2(RingSize, RingSize));
        AddImage(ringRect.gameObject, RingBackColor).sprite = ringSprite;
        ring = ringRect.gameObject;

        var fillRect = NewUiObject("Fill", ringRect, layer).GetComponent<RectTransform>();
        fillRect.anchorMin = Vector2.zero;
        fillRect.anchorMax = Vector2.one;
        fillRect.offsetMin = fillRect.offsetMax = Vector2.zero;
        ringFill = AddImage(fillRect.gameObject, RingFillColor);
        ringFill.sprite = ringSprite;
        ringFill.type = Image.Type.Filled;
        ringFill.fillMethod = Image.FillMethod.Radial360;
        ringFill.fillOrigin = (int)Image.Origin360.Top;
        // Anticlockwise from the top, so as it shrinks the gone part grows clockwise.
        ringFill.fillClockwise = false;

        ringNumber = Object.Instantiate(template, ringRect);
        ringNumber.name = "Number";
        Center(ringNumber.rectTransform, 0f, new Vector2(RingSize, RingSize));
        ringNumber.fontSize = RingNumberSize;
        ringNumber.alignment = TextAlignmentOptions.Center;
        ringNumber.text = "";

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
        HideCountdown();
        ShowBar(-85f, seconds);
        root.SetActive(true);
    }

    // Between trials: a title, a text under it, a label ("Starts in") and the circle
    // counting down from `count`, one number per `seconds / count`. With a `lead`, the label
    // and the circle appear only after it, so there is time to read first; the text stays put.
    public void ShowCountdown(string titleText, float titleSize, string centerText, float centerSize,
                              string label, int count, float seconds, float lead = 0f)
    {
        Set(title, 95f, titleSize, titleText);
        Set(center, 15f, centerSize, centerText);
        Set(below, -40f, BodySize, "");
        countdownLabel = label;
        HideBar();
        ShowRing(-100f, count, seconds, lead);
        root.SetActive(true);
    }

    public void Hide()
    {
        HideBar();
        HideCountdown();
        root.SetActive(false);
    }

    // Called every frame: drains the bar from the left as the time runs out, and runs the
    // circle -- each number's ring empties over its second, then the next number shows.
    public void Tick()
    {
        if (timerRunning)
        {
            float left = 1f - Mathf.Clamp01((Time.time - timerStart) / timerLength);
            barFill.anchorMin = new Vector2(1f - left, 0f);
        }

        if (countdownRunning)
        {
            if (Time.time < countdownStart)
            {
                return;
            }

            if (!ring.activeSelf)
            {
                below.text = countdownLabel;
                ring.SetActive(true);
            }

            float steps = (Time.time - countdownStart) / countdownStep;
            int done = Mathf.Min(Mathf.FloorToInt(steps), countdownFrom - 1);
            ringNumber.text = (countdownFrom - done).ToString();
            ringFill.fillAmount = Mathf.Clamp01(1f - (steps - done));
        }
    }

    private void ShowRing(float y, int count, float seconds, float lead)
    {
        ((RectTransform)ring.transform).anchoredPosition = new Vector2(0f, y);
        ring.SetActive(false);
        countdownStart = Time.time + Mathf.Max(lead, 0f);
        countdownFrom = Mathf.Max(count, 1);
        countdownStep = Mathf.Max(seconds, 0.01f) / countdownFrom;
        countdownRunning = true;
        Tick();
    }

    private void HideCountdown()
    {
        countdownRunning = false;
        ring.SetActive(false);
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

    private static Image AddImage(GameObject go, Color color)
    {
        var image = go.AddComponent<Image>();
        image.color = color;
        image.raycastTarget = false;
        return image;
    }

    // A white ring with a soft edge, drawn once at start-up; the Images tint it.
    private static Sprite RingSprite()
    {
        var texture = new Texture2D(RingTexture, RingTexture, TextureFormat.RGBA32, false)
        {
            wrapMode = TextureWrapMode.Clamp,
            filterMode = FilterMode.Bilinear,
        };
        float half = RingTexture / 2f;
        float outer = half - 1f;
        float inner = outer * (1f - RingThickness);
        var pixels = new Color32[RingTexture * RingTexture];
        for (int y = 0; y < RingTexture; ++y)
        {
            for (int x = 0; x < RingTexture; ++x)
            {
                float r = Vector2.Distance(new Vector2(x + 0.5f, y + 0.5f), new Vector2(half, half));
                float alpha = Mathf.Clamp01(outer - r + 0.5f) * Mathf.Clamp01(r - inner + 0.5f);
                pixels[y * RingTexture + x] = new Color32(255, 255, 255, (byte)(alpha * 255f));
            }
        }
        texture.SetPixels32(pixels);
        texture.Apply();
        return Sprite.Create(texture, new Rect(0, 0, RingTexture, RingTexture), new Vector2(0.5f, 0.5f));
    }
}
