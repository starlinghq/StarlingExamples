using System.Net;
using System.Text;
using System.Text.RegularExpressions;
using AngleSharp.Dom;
using AngleSharp.Html.Parser;

// A polite breadth-first crawler for one site: same scheme, host and port as the start URL, robots.txt honoured,
// one request at a time with a delay between them, a page cap, and a named user agent.
internal sealed class Crawler(HttpClient http, CrawlerOptions options)
{
    public const string UserAgent = "StarlingExamples-SiteReader/1.0 (+https://github.com/starlinghq/StarlingExamples)";

    private static readonly HashSet<string> SkippedExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".pdf", ".zip", ".gz", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".mp3", ".mp4", ".webm",
        ".css", ".js", ".json", ".xml", ".woff", ".woff2", ".ttf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    };

    public async Task<List<CrawledPage>> CrawlAsync(Uri start, Action<string> log, CancellationToken cancellationToken = default)
    {
        RobotsRules robots = await RobotsRules.FetchAsync(http, start, cancellationToken);
        TimeSpan delay = robots.CrawlDelay is { } crawlDelay && crawlDelay > options.Delay ? crawlDelay : options.Delay;

        var pages = new List<CrawledPage>();
        var seen = new HashSet<string> { Normalize(start) };
        var queue = new Queue<Uri>([start]);
        var parser = new HtmlParser();
        bool first = true;
        while (queue.Count > 0 && pages.Count < options.MaxPages)
        {
            Uri url = queue.Dequeue();
            if (!robots.IsAllowed(url.PathAndQuery))
            {
                log($"skip  {url.PathAndQuery} (robots.txt)");
                continue;
            }
            if (!first)
                await Task.Delay(delay, cancellationToken);
            first = false;

            using var response = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
            if (response.StatusCode != HttpStatusCode.OK
                || response.Content.Headers.ContentType?.MediaType is not ("text/html" or "application/xhtml+xml"))
            {
                log($"skip  {url.PathAndQuery} ({(int)response.StatusCode} {response.Content.Headers.ContentType?.MediaType})");
                continue;
            }
            // Redirects may leave the site; keep only pages that end up on it.
            Uri final = response.RequestMessage?.RequestUri ?? url;
            if (!SameSite(start, final))
                continue;
            string html = await ReadLimitedAsync(response.Content, options.MaxPageBytes, cancellationToken);

            IDocument document = await parser.ParseDocumentAsync(html, cancellationToken);
            foreach (IElement anchor in document.QuerySelectorAll("a[href]"))
            {
                if (!Uri.TryCreate(final, anchor.GetAttribute("href"), out Uri? link)
                    || link.Scheme is not ("http" or "https")
                    || !SameSite(start, link)
                    || SkippedExtensions.Contains(Path.GetExtension(link.AbsolutePath)))
                    continue;
                link = new UriBuilder(link) { Fragment = "" }.Uri;
                if (seen.Add(Normalize(link)))
                    queue.Enqueue(link);
            }

            var page = Extract(final, document, options.MaxChunkWords);
            if (page.Chunks.Count > 0)
            {
                pages.Add(page);
                log($"read  {final.PathAndQuery} ({page.Chunks.Count} passages)");
            }
        }
        return pages;
    }

    private static bool SameSite(Uri a, Uri b) =>
        a.Scheme == b.Scheme && a.Port == b.Port && string.Equals(a.Host, b.Host, StringComparison.OrdinalIgnoreCase);

    private static string Normalize(Uri url) => url.GetLeftPart(UriPartial.Query).TrimEnd('/');

    private static async Task<string> ReadLimitedAsync(HttpContent content, int maxBytes, CancellationToken cancellationToken)
    {
        await using Stream stream = await content.ReadAsStreamAsync(cancellationToken);
        var buffer = new MemoryStream();
        byte[] chunk = new byte[16 * 1024];
        int read;
        while ((read = await stream.ReadAsync(chunk, cancellationToken)) > 0 && buffer.Length < maxBytes)
            buffer.Write(chunk, 0, Math.Min(read, maxBytes - (int)buffer.Length));
        Encoding encoding = content.Headers.ContentType?.CharSet is { } charset
            ? Encoding.GetEncoding(charset.Trim('"'))
            : Encoding.UTF8;
        return encoding.GetString(buffer.GetBuffer(), 0, (int)buffer.Length);
    }

    // Reads the page the way a person skims it: navigation, headers, footers and scripts are dropped, then the
    // remaining headings, paragraphs and list items become passages. Each passage stays under the heading it
    // belongs to and is capped in words so it fits both models' 512-token inputs.
    internal static CrawledPage Extract(Uri url, IDocument document, int maxChunkWords)
    {
        string title = Clean(document.Title ?? "");
        foreach (IElement noise in document.QuerySelectorAll("script, style, noscript, nav, header, footer, aside, form, template").ToArray())
            noise.Remove();
        IElement root = document.QuerySelector("main") ?? document.QuerySelector("article") ?? document.Body ?? document.DocumentElement;

        var chunks = new List<string>();
        string heading = "";
        var current = new StringBuilder();
        int words = 0;
        void Flush()
        {
            if (words > 0)
                chunks.Add(heading.Length > 0 ? $"{heading}: {current}" : current.ToString());
            current.Clear();
            words = 0;
        }

        const string Blocks = "p, li, dd, blockquote, pre, tr, caption, div";
        foreach (IElement element in root.QuerySelectorAll("h1, h2, h3, h4, " + Blocks))
        {
            // A block inside another block is read with its parent; a div only counts when it holds text itself
            // (an alert or a notice written without a <p>), not when it only wraps other blocks.
            if (element.ParentElement?.Closest(Blocks.Replace(", div", "")) is not null)
                continue;
            if (element.LocalName == "div"
                && (element.QuerySelector("h1, h2, h3, h4, " + Blocks) is not null
                    || !element.ChildNodes.Any(n => n.NodeType == NodeType.Text && !string.IsNullOrWhiteSpace(n.TextContent))
                       && element.Children.All(c => c.LocalName is not ("strong" or "em" or "b" or "i" or "a" or "span"))))
                continue;
            // A table row reads as "Saturday: 10:00 to 16:00", not as loose cells.
            string text = element.LocalName == "tr"
                ? string.Join(": ", element.Children.Select(c => Clean(c.TextContent)).Where(t => t.Length > 0))
                : Clean(element.TextContent);
            if (text.Length == 0)
                continue;
            if (element.LocalName is "h1" or "h2" or "h3" or "h4")
            {
                Flush();
                heading = text;
                continue;
            }
            foreach (string sentence in SplitLong(text, maxChunkWords))
            {
                int count = CountWords(sentence);
                if (words > 0 && words + count > maxChunkWords)
                    Flush();
                current.Append(current.Length > 0 ? " " : "").Append(sentence);
                words += count;
            }
        }
        Flush();
        return new CrawledPage(url.ToString(), title.Length > 0 ? title : url.AbsolutePath, chunks);
    }

    private static IEnumerable<string> SplitLong(string text, int maxWords)
    {
        if (CountWords(text) <= maxWords)
        {
            yield return text;
            yield break;
        }
        var part = new StringBuilder();
        int words = 0;
        foreach (string sentence in Regex.Split(text, @"(?<=[.!?])\s+"))
        {
            foreach (string word in sentence.Split(' '))
            {
                if (words == maxWords)
                {
                    yield return part.ToString();
                    part.Clear();
                    words = 0;
                }
                part.Append(part.Length > 0 ? " " : "").Append(word);
                words++;
            }
        }
        if (words > 0)
            yield return part.ToString();
    }

    private static int CountWords(string text) => text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;

    private static string Clean(string text) => Regex.Replace(text, @"\s+", " ").Trim();
}

internal sealed record CrawlerOptions(int MaxPages, TimeSpan Delay, int MaxPageBytes = 2 * 1024 * 1024, int MaxChunkWords = 120);

internal sealed record CrawledPage(string Url, string Title, List<string> Chunks);

// The subset of robots.txt that matters to a single polite crawler: the group for our user agent (or "*"),
// Allow and Disallow by longest matching prefix (with "*" and "$" wildcards), and Crawl-delay.
internal sealed class RobotsRules
{
    private readonly List<(string Pattern, bool Allow)> rules = [];

    public TimeSpan? CrawlDelay { get; private set; }

    public static async Task<RobotsRules> FetchAsync(HttpClient http, Uri site, CancellationToken cancellationToken)
    {
        try
        {
            using var response = await http.GetAsync(new Uri(site, "/robots.txt"), cancellationToken);
            if (!response.IsSuccessStatusCode)
                return new RobotsRules();
            return Parse(await response.Content.ReadAsStringAsync(cancellationToken), "StarlingExamples-SiteReader");
        }
        catch (HttpRequestException)
        {
            return new RobotsRules();
        }
    }

    public static RobotsRules Parse(string text, string agent)
    {
        // Collect the rules of the most specific group: one naming our agent wins over "*".
        var groups = new List<(List<string> Agents, RobotsRules Rules)>();
        (List<string> Agents, RobotsRules Rules)? group = null;
        bool lastWasAgent = false;
        foreach (string raw in text.Split('\n'))
        {
            string line = raw.Split('#')[0].Trim();
            int colon = line.IndexOf(':');
            if (colon < 0)
                continue;
            string key = line[..colon].Trim().ToLowerInvariant();
            string value = line[(colon + 1)..].Trim();
            if (key == "user-agent")
            {
                if (group is null || !lastWasAgent)
                {
                    group = ([], new RobotsRules());
                    groups.Add(group.Value);
                }
                group.Value.Agents.Add(value.ToLowerInvariant());
                lastWasAgent = true;
                continue;
            }
            lastWasAgent = false;
            if (group is null)
                continue;
            if (key is "allow" or "disallow" && value.Length > 0)
                group.Value.Rules.rules.Add((value, key == "allow"));
            else if (key == "crawl-delay" && double.TryParse(value, System.Globalization.CultureInfo.InvariantCulture, out double seconds))
                group.Value.Rules.CrawlDelay = TimeSpan.FromSeconds(Math.Min(seconds, 30));
        }
        string lower = agent.ToLowerInvariant();
        return groups.FirstOrDefault(g => g.Agents.Any(a => a != "*" && lower.Contains(a))).Rules
            ?? groups.FirstOrDefault(g => g.Agents.Contains("*")).Rules
            ?? new RobotsRules();
    }

    public bool IsAllowed(string path)
    {
        (string Pattern, bool Allow)? best = null;
        foreach (var rule in rules)
        {
            if (Matches(rule.Pattern, path) && (best is null || rule.Pattern.Length > best.Value.Pattern.Length
                || rule.Pattern.Length == best.Value.Pattern.Length && rule.Allow))
                best = rule;
        }
        return best?.Allow ?? true;
    }

    private static bool Matches(string pattern, string path)
    {
        string regex = "^" + Regex.Escape(pattern).Replace(@"\*", ".*");
        if (regex.EndsWith(@"\$", StringComparison.Ordinal))
            regex = regex[..^2] + "$";
        return Regex.IsMatch(path, regex);
    }
}
