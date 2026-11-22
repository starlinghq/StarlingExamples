using System.Net;

// Serves the bundled sample-site/ folder on localhost so the crawl runs without touching anyone else's site.
internal sealed class SampleServer : IDisposable
{
    private readonly HttpListener listener = new();
    private readonly string root;
    private readonly Task loop;

    public SampleServer(string root, int port)
    {
        this.root = Path.GetFullPath(root);
        Url = new Uri($"http://localhost:{port}/");
        listener.Prefixes.Add(Url.ToString());
        listener.Start();
        loop = Task.Run(ServeAsync);
    }

    public static Uri DefaultUrl { get; } = new("http://localhost:8787/");

    public Uri Url { get; }

    private async Task ServeAsync()
    {
        while (listener.IsListening)
        {
            HttpListenerContext context;
            try
            {
                context = await listener.GetContextAsync();
            }
            catch (Exception ex) when (ex is HttpListenerException or ObjectDisposedException)
            {
                return;
            }
            using HttpListenerResponse response = context.Response;
            string relative = Uri.UnescapeDataString(context.Request.Url?.AbsolutePath ?? "/").TrimStart('/');
            string path = Path.GetFullPath(Path.Combine(root, relative));
            if (Directory.Exists(path))
                path = Path.Combine(path, "index.html");
            if (!path.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.Ordinal) || !File.Exists(path))
            {
                response.StatusCode = 404;
                continue;
            }
            response.ContentType = Path.GetExtension(path) switch
            {
                ".html" => "text/html; charset=utf-8",
                ".css" => "text/css; charset=utf-8",
                _ => "text/plain; charset=utf-8",
            };
            byte[] body = await File.ReadAllBytesAsync(path);
            await response.OutputStream.WriteAsync(body);
        }
    }

    public void Dispose()
    {
        listener.Stop();
        listener.Close();
        loop.Wait(TimeSpan.FromSeconds(5));
    }
}
