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

    public static E5TextEmbedder E5(ModelAcquisitionOptions acquisition) =>
        Load("e5", () => E5ModelAssets.OpenLocal(acquisition), files => E5TextEmbedder.Create(files, new E5EncoderOptions { Session = Session }));

    public static Gliner2EntityExtractor Gliner2(ModelAcquisitionOptions acquisition) =>
        Load("GLiNER2", () => Gliner2ModelAssets.OpenLocal(acquisition),
            files => Gliner2EntityExtractor.Create(files, new Gliner2ExtractorOptions { Session = Session }));

    public static EmbeddingGemma2MultimodalEmbedder EmbeddingGemma2(ModelAcquisitionOptions acquisition) =>
        Load("EmbeddingGemma 2", () => EmbeddingGemma2ModelAssets.OpenLocalImageAssets(acquisition),
            files => EmbeddingGemma2MultimodalEmbedder.Create(files, new EmbeddingGemma2MultimodalOptions { Session = Session }));

    // Qwen3.5 fp32x runs on both providers; on CUDA it goes through ONNX Runtime GenAI's CUDA build.
    public static IChatClient Qwen(ModelAcquisitionOptions acquisition) =>
        Load("Qwen3.5", () => Qwen35ModelAssets.OpenLocal(Qwen35Variant.Fp32x, acquisition),
            files => Qwen35ChatClient.Create(files,
                new Qwen35ModelOptions { Provider = Cuda ? GenAIExecutionProvider.Cuda : GenAIExecutionProvider.Cpu }));

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
