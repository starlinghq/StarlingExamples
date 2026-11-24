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
// -p:NativeRuntime=cuda) on CUDA. CUDA_VISIBLE_DEVICES picks the GPU.
internal static class Models
{
    public static bool Cuda { get; set; }

    private static OnnxSessionOptions Session => Cuda ? new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cuda, CudaDeviceId = 0 } : new OnnxSessionOptions();

    public static E5TextEmbedder E5(ModelAcquisitionOptions acquisition) =>
        E5TextEmbedder.Create(E5ModelAssets.OpenLocal(acquisition), new E5EncoderOptions { Session = Session });

    public static Gliner2EntityExtractor Gliner2(ModelAcquisitionOptions acquisition) =>
        Gliner2EntityExtractor.Create(Gliner2ModelAssets.OpenLocal(acquisition), new Gliner2ExtractorOptions { Session = Session });

    public static EmbeddingGemma2MultimodalEmbedder EmbeddingGemma2(ModelAcquisitionOptions acquisition) =>
        EmbeddingGemma2MultimodalEmbedder.Create(EmbeddingGemma2ModelAssets.OpenLocalImageAssets(acquisition),
            new EmbeddingGemma2MultimodalOptions { Session = Session });

    // Qwen3.5 fp32x runs on both providers; on CUDA it goes through ONNX Runtime GenAI's CUDA build.
    public static IChatClient Qwen(ModelAcquisitionOptions acquisition) =>
        Qwen35ChatClient.Create(Qwen35ModelAssets.OpenLocal(Qwen35Variant.Fp32x, acquisition),
            new Qwen35ModelOptions { Provider = Cuda ? GenAIExecutionProvider.Cuda : GenAIExecutionProvider.Cpu });

    // glibc reads MALLOC_MMAP_THRESHOLD_ only at process start; GLiNER2 and EmbeddingGemma 2 on CUDA need it.
    public static void WarnIfCudaEnvironmentMissing()
    {
        if (Cuda && OperatingSystem.IsLinux() && Environment.GetEnvironmentVariable("MALLOC_MMAP_THRESHOLD_") != "65536")
            Console.Error.WriteLine("warning: start the process with MALLOC_MMAP_THRESHOLD_=65536 for GLiNER2 and EmbeddingGemma 2 on CUDA.");
    }
}
