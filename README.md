# StarlingExamples

Example programs that use [Starling](https://github.com/starlinghq/starling) to solve real problems in .NET: reading a website, organising photos, finding a moment in a video, triaging feedback. Each one is a small console program that downloads its models once; after that the models run on your machine and never use the network (an example that reads a website still fetches the site). Nothing is sent to a hosted API.

They are written as package consumers: every example references the published `Starlings.*` packages on nuget.org (0.5.0) and builds on its own, outside this repository. For the API itself, the [Starling README](https://github.com/starlinghq/starling#readme) and its [samples](https://github.com/starlinghq/starling/tree/main/ai-dotnet/samples) are the reference. The examples here combine several models, and ML.NET where classical machine learning fits, into one complete job each.

## Requirements

- The [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0).
- Disk for the models (each example says how much) and a few GB of free memory. Linux x64 on CPU is what these are run on; CUDA works where the example says so.

## Examples

| Example | What it does | Models | Status |
| --- | --- | --- | --- |
| [SiteReader](SiteReader) | Crawls a website politely, works out its sections, and answers questions with the page each answer came from; optionally renders the pages to see their layout (banners, tables, maps, forms) and search them by description | e5, GLiNER2, Qwen3.5, EmbeddingGemma 2 (image); ML.NET k-means; Playwright | Available |
| PhotoAlbums | Sorts a folder of photos into albums and names them; finds photos from a description | EmbeddingGemma 2 (image); ML.NET k-means | Planned |
| VideoMoments | Finds where in a video something happens ("the dog jumps into the lake") and prints timestamps | EmbeddingGemma 2 (video, FFmpeg 8) | Planned |
| FeedbackRadar | Groups customer reviews or tickets by topic, flags near-duplicates and unusual ones, pulls out the products and places mentioned | e5 in an ML.NET pipeline (clustering, anomaly detection), GLiNER2 | Planned |
| LogPatterns | Clusters log lines into recurring patterns, flags patterns not seen before, and names each one | e5 in ML.NET, Qwen3.5 | Planned |
| ReceiptReader | Reads photographed receipts and invoices into a CSV of merchant, date, items and total | PaddleOCR (PP-OCRv5), GLiNER2 | Waiting for a published OCR package |

The planned examples are described below so the intent is clear before the code lands. Each will follow the same shape as SiteReader: a `download` command, commands that run offline, a README with the output of a real run, and a section on what it gets wrong.

### PhotoAlbums

EmbeddingGemma 2 puts images and text in one embedding space. The example embeds every photo in a folder, clusters the vectors with ML.NET k-means into albums, and names each album by comparing its centre with a list of candidate descriptions ("a beach", "a birthday party", "a city at night"). The same index answers text searches such as "two people on a bridge". Vision takes seconds per photo on a CPU, so the index is cached and only new photos are embedded.

### VideoMoments

The EmbeddingGemma 2 video adapter decodes a clip with your own LGPL build of FFmpeg 8 and samples it the way the upstream model does. The example cuts a long video into short overlapping windows, embeds each one, and ranks the windows against a text description, printing start and end times you can jump to.

### FeedbackRadar

Classical machine learning over modern features. e5 embeddings enter an ordinary ML.NET pipeline through `Starlings.Models.E5.MLNet`; k-means groups the feedback into topics, randomized-PCA anomaly detection flags reviews unlike the rest, and cosine similarity finds near-duplicates. GLiNER2 extracts the products, features and locations each topic mentions. Because e5 is multilingual, reviews in different languages land in the same topics.

### LogPatterns

Log lines are normalised (numbers, ids and paths masked), embedded with e5 and clustered in ML.NET, so "connection reset by peer" and "Connection to 10.0.0.4 reset" become one pattern. New lines are scored against the known patterns to surface the ones that have never appeared before. Qwen3.5 gives each pattern a short name from a few of its lines.

### ReceiptReader

`Starlings.Models.PaddleOcr` (PP-OCRv5 detection and recognition) is in the Starling repository but not yet on nuget.org. Once it is published, this example reads receipts and invoices: OCR finds and reads each line, GLiNER2 labels the merchant, date, line items and total, and the rows go to a CSV. OCR would also let SiteReader read text that is only in images, such as banners and scanned notices.

## Licences

The example code is MIT licensed ([LICENSE](LICENSE)). The models keep their own licences: e5 (MIT), GLiNER2 (Apache-2.0), Qwen3.5 (Apache-2.0), EmbeddingGemma 2 (Apache-2.0, with the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)), PP-OCRv5 (Apache-2.0). Each example lists the ones it downloads.
