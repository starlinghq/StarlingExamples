using System.Diagnostics;
using System.Globalization;
using System.Numerics.Tensors;
using System.Text.Json;
using Microsoft.Playwright;
using Starlings.Models.EmbeddingGemma2;
using Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp;

// Looks at each crawled page the way a visitor does: rendered in a real browser at desktop size, split into the
// blocks a person sees (banners, menus, tables, forms, cards), in the order they appear on screen. Each block is
// cropped and embedded with EmbeddingGemma 2, which puts images and text in one space, so a block can be labelled
// by comparing it with short descriptions and found later with a text query.
internal static class Visual
{
    // Short descriptions of common page blocks. Labels are relative: each block takes the description it matches
    // better than the site's blocks match it on average, which removes the pull of generic descriptions.
    public static readonly string[] RegionLabels =
    [
        "a website navigation menu bar", "a large welcome banner with a heading and a button", "a content card",
        "a table of information", "a red warning alert banner", "a map", "a form with text fields",
        "a page footer with contact links", "a paragraph of text", "a photograph", "a list of links",
    ];

    public const int ViewportWidth = 1280;
    public const int ViewportHeight = 800;

    // Finds the visible blocks of the rendered page: landmarks (header, nav, main parts, aside, footer), tables,
    // forms, figures, images, alerts, and any block with its own background or border. Nested candidates collapse
    // to the outermost one unless it is <main> or a large container, which is replaced by its parts.
    private const string FindRegionsScript = """
        () => {
          const bodyBg = getComputedStyle(document.body).backgroundColor;
          const boxed = el => {
            const s = getComputedStyle(el);
            const bg = s.backgroundColor;
            return (bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent' && bg !== bodyBg) || parseFloat(s.borderTopWidth) > 0;
          };
          const tags = new Set(['HEADER','NAV','MAIN','FOOTER','ASIDE','FORM','TABLE','FIGURE','IMG','SECTION','ARTICLE']);
          const all = [...document.body.querySelectorAll('*')];
          const visible = el => {
            const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
            return r.width >= 120 && r.height >= 32 && s.visibility !== 'hidden' && s.display !== 'none' && parseFloat(s.opacity) > 0.05;
          };
          const isCandidate = el => (tags.has(el.tagName) || el.getAttribute('role') === 'alert' || boxed(el)) && visible(el);
          const pageHeight = document.documentElement.scrollHeight;
          const pick = (root) => {
            const out = [];
            for (const child of root.children) {
              if (isCandidate(child)) {
                const r = child.getBoundingClientRect();
                const parts = pick(child);
                const split = child.tagName === 'MAIN' ? parts.length > 0 : r.height > 0.6 * pageHeight && parts.length > 1;
                if (split) out.push(...parts); else out.push(child);
              } else out.push(...pick(child));
            }
            return out;
          };
          return JSON.stringify(pick(document.body).map(el => {
            const r = el.getBoundingClientRect();
            return {
              sourceOrder: all.indexOf(el), tag: el.tagName.toLowerCase(), role: el.getAttribute('role') || '',
              x: r.left + scrollX, y: r.top + scrollY, width: r.width, height: r.height,
              text: (el.innerText || el.getAttribute('alt') || '').replace(/\s+/g, ' ').trim().slice(0, 240)
            };
          }));
        }
        """;

    public static async Task<VisualIndex> LookAsync(SiteIndex index, string shotsDir, int maxRegions,
        Func<EmbeddingGemma2MultimodalEmbedder> getEmbedder, Action<string> log)
    {
        Directory.CreateDirectory(shotsDir);
        using IPlaywright playwright = await Playwright.CreateAsync();
        await using IBrowser browser = await playwright.Chromium.LaunchAsync();
        IPage page = await browser.NewPageAsync(new()
        {
            ViewportSize = new() { Width = ViewportWidth, Height = ViewportHeight },
            UserAgent = Crawler.UserAgent,
        });

        var pages = new List<VisualPage>();
        // Site furniture (the same header and footer on every page) is cropped and embedded once; later copies point
        // at the first crop. New crops are embedded in batches after the pages are rendered.
        var firstCrop = new Dictionary<(string Tag, string Text, double Width, double Height), string>();
        var toEmbed = new List<string>();
        var render = new Stopwatch();
        var shoot = new Stopwatch();
        string[] urls = [.. index.Passages.Select(p => p.Url).Distinct()];
        for (int n = 0; n < urls.Length; n++)
        {
            string url = urls[n];
            render.Start();
            // The load event, not network idle: network idle waits at least 500 ms after the last request on every page.
            await page.GotoAsync(url, new() { WaitUntil = WaitUntilState.Load });
            var found = JsonSerializer.Deserialize<List<FoundRegion>>(await page.EvaluateAsync<string>(FindRegionsScript),
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true }) ?? [];
            render.Stop();
            string slug = $"page{n + 1:D3}";
            string viewportPath = Path.Combine(shotsDir, $"{slug}.png");
            shoot.Start();
            await page.ScreenshotAsync(new() { Path = viewportPath });
            shoot.Stop();
            // Keep the blocks a visitor is most likely to notice: the largest ones, nearer the top first.
            var kept = found
                .Where(r => r.Width * r.Height >= 8000)
                .OrderByDescending(r => r.Width * Math.Min(r.Height, ViewportHeight) / (1 + r.Y / ViewportHeight))
                .Take(maxRegions)
                .ToList();
            var regions = new List<VisualRegion>();
            foreach (FoundRegion r in kept)
            {
                var key = (r.Tag, r.Text, Math.Round(r.Width), Math.Round(r.Height));
                if (!firstCrop.TryGetValue(key, out string? crop))
                {
                    crop = $"{slug}-r{regions.Count + 1}.png";
                    shoot.Start();
                    await page.ScreenshotAsync(new()
                    {
                        Path = Path.Combine(shotsDir, crop),
                        FullPage = true,
                        Clip = new() { X = (float)r.X, Y = (float)r.Y, Width = (float)r.Width, Height = (float)Math.Min(r.Height, 2000) },
                    });
                    shoot.Stop();
                    firstCrop[key] = crop;
                    toEmbed.Add(crop);
                }
                regions.Add(new VisualRegion(r.Tag, r.Role, r.SourceOrder, r.X, r.Y, r.Width, r.Height,
                    r.Y < ViewportHeight, r.Text, crop, [], ""));
            }
            // The viewport screenshot is kept for people to look at; nothing searches it, so it is not embedded.
            pages.Add(new VisualPage(url, Path.GetFileName(viewportPath), [], regions));
            log($"look  {new Uri(url).PathAndQuery} ({regions.Count} regions)");
        }

        // Rendering needs no model, so it runs while EmbeddingGemma 2 may still be loading; wait for it only here.
        EmbeddingGemma2MultimodalEmbedder embedder = getEmbedder();
        var embed = Stopwatch.StartNew();
        var vectors = new Dictionary<string, float[]>();
        foreach (string[] batch in toEmbed.Chunk(8))
        {
            float[][] batchVectors = embedder.EmbedBatch([.. batch.Select(c => EmbeddingGemma2ImageDecoder.DecodeFile(Path.Combine(shotsDir, c)))]);
            for (int i = 0; i < batch.Length; i++)
                vectors[batch[i]] = batchVectors[i];
        }
        foreach (VisualPage p in pages)
        {
            for (int i = 0; i < p.Regions.Count; i++)
                p.Regions[i] = p.Regions[i] with { Vector = vectors[p.Regions[i].Crop] };
        }
        embed.Stop();
        log(string.Create(CultureInfo.InvariantCulture,
            $"(rendering {render.Elapsed.TotalSeconds:F1} s, screenshots {shoot.Elapsed.TotalSeconds:F1} s for {toEmbed.Count + urls.Length}, " +
            $"embedding {embed.Elapsed.TotalSeconds:F1} s for {toEmbed.Count} distinct blocks)"));

        // Label every region against the descriptions, relative to how the whole site matches each description.
        float[][] labelVectors = embedder.EmbedBatch(
            [.. RegionLabels.Select(EmbeddingGemma2MultimodalInput.FromText)], EmbeddingGemma2Prompt.Query);
        // A whole <main> is a container of many kinds of block, so it is not labelled as one.
        var all = pages.SelectMany(p => p.Regions).Where(r => r.Tag != "main").ToList();
        if (all.Count > 0)
        {
            float[][] scores = [.. all.Select(r => labelVectors.Select(l => TensorPrimitives.CosineSimilarity(r.Vector, l)).ToArray())];
            double[] mean = [.. Enumerable.Range(0, RegionLabels.Length).Select(j => scores.Average(s => s[j]))];
            foreach (var p in pages)
            {
                for (int i = 0; i < p.Regions.Count; i++)
                {
                    if (p.Regions[i].Tag == "main")
                    {
                        p.Regions[i] = p.Regions[i] with { Label = "the main content" };
                        continue;
                    }
                    float[] s = scores[all.IndexOf(p.Regions[i])];
                    int best = Enumerable.Range(0, RegionLabels.Length).MaxBy(j => s[j] - mean[j]);
                    p.Regions[i] = p.Regions[i] with { Label = RegionLabels[best] };
                }
            }
        }
        return new VisualIndex(ViewportWidth, ViewportHeight, pages);
    }

    // Screen order (top to bottom, then left to right) is the order a visitor reads in; source order is what a
    // text extractor reads in. Where they disagree, layout is telling the reader something the HTML does not.
    public static IEnumerable<VisualRegion> ScreenOrder(IEnumerable<VisualRegion> regions) =>
        regions.OrderBy(r => Math.Round(r.Y / 40)).ThenBy(r => r.X);

    private sealed record FoundRegion(int SourceOrder, string Tag, string Role, double X, double Y, double Width, double Height, string Text);
}

internal sealed record VisualRegion(string Tag, string Role, int SourceOrder, double X, double Y, double Width, double Height,
    bool AboveFold, string Text, string Crop, float[] Vector, string Label);

internal sealed record VisualPage(string Url, string Screenshot, float[] Vector, List<VisualRegion> Regions);

internal sealed record VisualIndex(int ViewportWidth, int ViewportHeight, List<VisualPage> Pages)
{
    public static VisualIndex? TryLoad(string path) =>
        File.Exists(path) ? JsonSerializer.Deserialize<VisualIndex>(File.ReadAllText(path)) : null;

    public void Save(string path) => File.WriteAllText(path, JsonSerializer.Serialize(this));
}
