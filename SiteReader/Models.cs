using System.Diagnostics;
using System.Globalization;
using Microsoft.Extensions.AI;
using Starlings.Acquisition;
using Starlings.Generation.OnnxGenAI;
using Starlings.Models.E5;
using Starlings.Models.EmbeddingGemma2;
using Starlings.Models.Gliner2;
using Starlings.Models.Qwen35;
using Starlings.Models.Qwen35.Extensions.AI;
using Starlings.Runtime.Onnx;

// Loads each model from the verified local cache, on the CPU or (with --cuda, in a build with
// -p:NativeRuntime=cuda) on CUDA. CUDA_VISIBLE_DEVICES picks the GPU. Each load reports how long verifying the
// files and creating the session took, which is most of a short command's time.
internal static class Models
{
    public static bool Cuda { get; set; }

    private static OnnxSessionOptions Session => Cuda ? new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cuda, CudaDeviceId = 0 } : new OnnxSessionOptions();

    // One instance of each model per process, created on first use and kept until the process exits: loading is the
    // expensive part (seconds), using a loaded model is not. Preload starts several loads at once on the thread pool,
    // so their SHA-256 checks run on separate cores and overlap with the crawl; a step that needs a model waits for it.
    private static Task<E5TextEmbedder>? e5;
    private static Task<Gliner2EntityExtractor>? gliner2;
    private static Task<EmbeddingGemma2MultimodalEmbedder>? embeddingGemma2;
    private static Task<IChatClient>? qwen;

    public static void Preload(ModelAcquisitionOptions acquisition, bool visual)
    {
        StartE5(acquisition);
        StartGliner2(acquisition);
        StartQwen(acquisition);
        if (visual)
            StartEmbeddingGemma2(acquisition);
    }

    public static void PreloadVisual(ModelAcquisitionOptions acquisition) => StartEmbeddingGemma2(acquisition);

    public static E5TextEmbedder E5(ModelAcquisitionOptions acquisition) => StartE5(acquisition).GetAwaiter().GetResult();

    public static Gliner2EntityExtractor Gliner2(ModelAcquisitionOptions acquisition) => StartGliner2(acquisition).GetAwaiter().GetResult();

    public static EmbeddingGemma2MultimodalEmbedder EmbeddingGemma2(ModelAcquisitionOptions acquisition) =>
        StartEmbeddingGemma2(acquisition).GetAwaiter().GetResult();

    public static IChatClient Qwen(ModelAcquisitionOptions acquisition) => StartQwen(acquisition).GetAwaiter().GetResult();

    private static Task<E5TextEmbedder> StartE5(ModelAcquisitionOptions acquisition) =>
        e5 ??= Task.Run(() => Load("e5", () => E5ModelAssets.OpenLocal(acquisition),
            files => E5TextEmbedder.Create(files, new E5EncoderOptions { Session = Session })));

    private static Task<Gliner2EntityExtractor> StartGliner2(ModelAcquisitionOptions acquisition) =>
        gliner2 ??= Task.Run(() => Load("GLiNER2", () => Gliner2ModelAssets.OpenLocal(acquisition),
            files => Gliner2EntityExtractor.Create(files, new Gliner2ExtractorOptions { Session = Session })));

    private static Task<EmbeddingGemma2MultimodalEmbedder> StartEmbeddingGemma2(ModelAcquisitionOptions acquisition) =>
        embeddingGemma2 ??= Task.Run(() => Load("EmbeddingGemma 2", () => EmbeddingGemma2ModelAssets.OpenLocalImageAssets(acquisition),
            files => EmbeddingGemma2MultimodalEmbedder.Create(files, new EmbeddingGemma2MultimodalOptions { Session = Session })));

    // Qwen3.5 fp32x runs on both providers; on CUDA it goes through ONNX Runtime GenAI's CUDA build.
    private static Task<IChatClient> StartQwen(ModelAcquisitionOptions acquisition) =>
        qwen ??= Task.Run(() => Load("Qwen3.5", () => Qwen35ModelAssets.OpenLocal(Qwen35Variant.Fp32x, acquisition),
            files => (IChatClient)Qwen35ChatClient.Create(files,
                new Qwen35ModelOptions { Provider = Cuda ? GenAIExecutionProvider.Cuda : GenAIExecutionProvider.Cpu })));

    private static TModel Load<TFiles, TModel>(string name, Func<TFiles> open, Func<TFiles, TModel> create)
    {
        var timer = Stopwatch.StartNew();
        TFiles files = open();
        TimeSpan verify = timer.Elapsed;
        TModel model = create(files);
        Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"(loaded {name}: files checked in {verify.TotalSeconds:F1} s, model ready {(timer.Elapsed - verify).TotalSeconds:F1} s later)"));
        return model;
    }

    // glibc reads MALLOC_MMAP_THRESHOLD_ only at process start; GLiNER2 and EmbeddingGemma 2 on CUDA need it.
    public static void WarnIfCudaEnvironmentMissing()
    {
        if (Cuda && OperatingSystem.IsLinux() && Environment.GetEnvironmentVariable("MALLOC_MMAP_THRESHOLD_") != "65536")
            Console.Error.WriteLine("warning: start the process with MALLOC_MMAP_THRESHOLD_=65536 for GLiNER2 and EmbeddingGemma 2 on CUDA.");
    }
}
