# SiteReader

Reads a website the way a new visitor would. It crawls the site, works out what its sections are about, answers questions with the page each answer came from, and, optionally, looks at the rendered pages to see what a text extractor misses: which blocks are banners, tables, maps or forms, and what appears first on screen.

Everything runs on your machine. The crawl and the optional browser step fetch the site's pages; the models never send anything anywhere.

| Step | What it uses |
| --- | --- |
| `crawl` | A polite crawler (same site only, robots.txt, one request at a time, 500 ms apart, at most 50 pages, a named user agent). Each page becomes passages; [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) embeds them and [GLiNER2](https://huggingface.co/fastino/gliner2-base-v1) pulls out people, places, organisations, dates, events and products. |
| `map` | ML.NET k-means groups the pages by what they say, not by their URLs. [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) writes a short overview. |
| `ask` | e5 retrieves passages; Qwen3.5 answers from them and cites them. |
| `look` (optional) | Headless Chromium renders each page at 1280x800 and splits it into the blocks a person sees. [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) embeds each block's screenshot in the same space as text, which labels the block and makes it searchable. |
| `find` (optional) | Text-to-image search over those blocks. |

## Run it

Requires the .NET 10 SDK. The repository includes a small fictional website (`sample-site/`, a public library) that `crawl sample` serves on `localhost:8787`, so the output below is reproducible.

```sh
cd SiteReader
dotnet run -c Release -- download          # e5, GLiNER2, Qwen3.5: about 4.3 GB, once
dotnet run -c Release -- crawl sample      # or a URL you are allowed to crawl
dotnet run -c Release -- map
dotnet run -c Release -- ask "Can I borrow a tile cutter, and what do I need to do first?"
```

For the visual steps:

```sh
dotnet run -c Release -- download-visual   # Chromium (Playwright) and the EmbeddingGemma 2 image graphs: about 3 GB, once
dotnet run -c Release -- look
dotnet run -c Release -- find "a map of the library branches"
```

The output below is from a run on Linux x64 on an Intel Core i7-5820K (AVX2), on the published 0.5.0 packages.

### crawl

```text
Crawling http://localhost:8787/ (at most 50 pages, 500 ms apart)
read  / (1 passages)
read  /visit/ (1 passages)
...
skip  /staff/ (robots.txt)
...
Read 18 pages, 26 passages, 200 entity mentions. Saved site-index/index.json.
```

About 20 seconds, most of it the delay between requests.

### map

```text
http://localhost:8787/: 18 pages in 4 sections

## Printing and scanning (7 pages)
   mentions: laser cutter, tile cutter, 3D printers, Alder Point Makers group, Alder Point district
   Printing and scanning  http://localhost:8787/services/printing.html
   Maker space  http://localhost:8787/services/maker-space.html
   Membership  http://localhost:8787/services/membership.html
   Tool library  http://localhost:8787/services/tool-library.html
   Services  http://localhost:8787/services/
   Room booking  http://localhost:8787/services/room-booking.html
   Contact  http://localhost:8787/visit/contact.html

## Events (5 pages)
   ...
## Branches (4 pages)
   mentions: Central Library, Harbour Branch, Closed, Quay Street, harbour master's office
   Branches  http://localhost:8787/visit/branches.html
   Accessibility  http://localhost:8787/visit/accessibility.html
   Opening hours  http://localhost:8787/visit/hours.html
   Visit us  http://localhost:8787/visit/

## Summer reading challenge (2 pages)
   mentions: Central Library, Children, Rhyme Time for babies and toddlers, September celebration, Story hour
   ...

Overview: The website is for the Alder Point Public Library, designed to guide visitors through its diverse collections, services, and events. Visitors can browse the Maker space, access the tool library, and participate in summer reading challenges, while also finding information on branch locations, repair cafés, and local history talks.
```

Every page on a site shares its name, address and tone, so before clustering `map` subtracts the site's mean page vector and clusters what is left. Without that step, 13 of the 18 sample pages fell into one section in an earlier run.

### ask

```text
[1] http://localhost:8787/services/tool-library.html (cosine 0.872)
...
Question: Can I borrow a tile cutter, and what do I need to do first?
Answer: [1] Safety induction is required for the tile cutter.
```

```text
Question: Does the library have a café that sells coffee?
Answer: The website does not mention a café selling coffee.
```

## Why look at the pages

The hours page of the sample site has a red banner at the top: the Harbour Branch is closed for roof repairs. In the HTML it is the last element of the page; CSS moves it to the top. A text extractor reads it last, after the holiday closures, as one more sentence. A visitor reads it first.

`look` renders each page, finds its blocks (landmarks, tables, forms, figures, images, alerts, and anything with its own background or border), crops and embeds each one, and reports where the order on screen differs from the order in the HTML:

```text
Looked at 18 pages, 60 regions in 4.5 min. Saved site-index/visual.json.

http://localhost:8787/visit/hours.html
  first on screen:   a red warning alert banner (div, number 3 of 3 in the HTML): "Harbour Branch closed for roof repairs until 30 November. Reserved ..."
  first in the HTML: a table of information (table): "Central Library, Quay Street Monday to Friday 9:00 to 19:00 Saturda..."
```

`ask` uses that. Without `look`, the same question gets the timetable:

```text
Question: Can I go to the Harbour Branch on Saturday morning?
Answer: [1] Opening hours: Harbour Branch, Ferry Lane Tuesday, Thursday, Saturday: 10:00 to 14:00.
```

After `look`, a block that the layout moved to the top of a retrieved page goes into the context with each passage of that page, and is printed for you as well:

```text
Banner at the top of http://localhost:8787/visit/hours.html: Harbour Branch closed for roof repairs until 30 November. Reserved items can be collected from the Central Library.
Question: Can I go to the Harbour Branch on Saturday morning?
Answer: [1] The Harbour Branch is closed for roof repairs until 30 November.
[2] Holiday closures: All branches close from 24 December to 1 January.
[3] The Harbour Branch is on the first floor and has no lift.
[4] The Harbour Branch is located at 2 Ferry Lane, above the harbour master's office.
```

`map` adds what each section looks like (`looks like: a table of information, a map, ...`), and `find` searches the blocks by description:

```text
Looking for: a map of the library branches
  0.699  http://localhost:8787/visit/branches.html  a map, 816x381 px at y=178 (above the fold)
         crop: site-index/shots/page010-r1.png

Looking for: a notice that a branch is closed
  0.724  http://localhost:8787/visit/hours.html  a red warning alert banner, 896x74 px at y=78 (above the fold)
         crop: site-index/shots/page009-r4.png

Looking for: a form to sign up
  0.774  http://localhost:8787/services/membership.html  a form with text fields, 420x360 px at y=332 (above the fold)
         crop: site-index/shots/page006-r1.png
```

Labels come from comparing each block's image embedding with short descriptions ("a table of information", "a form with text fields", ...). A raw comparison favours a few generic descriptions, so each block takes the description it matches better than the site's blocks match it on average.

## On a GPU

Build with the CUDA natives and pass `--cuda` to each command. Linux x64 only; the CUDA 13 runtime and cuDNN 9 must be on the loader path, and GLiNER2 and EmbeddingGemma 2 on CUDA need `MALLOC_MMAP_THRESHOLD_=65536` in the environment the process starts with (the program warns if it is missing). `CUDA_VISIBLE_DEVICES` picks the GPU.

```sh
export MALLOC_MMAP_THRESHOLD_=65536
dotnet run -c Release -p:NativeRuntime=cuda -- crawl sample --cuda
dotnet run -c Release -p:NativeRuntime=cuda -- look --cuda
dotnet run -c Release -p:NativeRuntime=cuda -- ask "Can I go to the Harbour Branch on Saturday morning?" --cuda
```

Each command prints how long it took. On the sample site, on one machine: an Intel Core i7-5820K for the CPU column and a consumer NVIDIA GPU for the CUDA column:

| Step | CPU | CUDA |
| --- | --- | --- |
| `crawl` (18 pages) | 18.4 s | 17.0 s |
| `look` (60 regions) | 4.5 min | 41.8 s |
| `map` | 19.3 s | 12.5 s |
| `ask` | 20 to 23 s | 14 to 15 s |
| `find` | 13.8 s | 14.3 s |

The answers, block labels and `find` scores were the same on both. `look` is where the GPU matters in this table: it embeds one image per block. The other steps embed a handful of texts or generate one short answer, so most of their time is loading the models when the process starts, and `crawl` mostly waits between requests.

Each model load prints where its time went, and `ask -` reads questions from standard input so the models load once. Three questions in one process:

| Stage | CPU | CUDA |
| --- | --- | --- |
| Checking the files of e5 and Qwen3.5 (SHA-256 of 3.6 GB) | 9.4 s | 8.9 s |
| Creating the two models | 5.9 s | 4.5 s |
| Answer of 76 tokens (prompt included) | 10.5 s | 2.1 s |
| Answers of 10 to 12 tokens (prompt included) | 3.1 to 3.9 s | 0.3 to 0.4 s |

Once the models are loaded, the GPU answers four to ten times faster; a one-question `ask` mostly measures the load. The SHA-256 check is the library's default and protects against a corrupted or altered cache. After `download` has verified the files once, `--verify size` checks only their sizes, which takes the check to zero on later runs (`ask` with three questions on CUDA: 16.7 s with SHA-256, 8.8 s with sizes only). Starling tests its CUDA profiles on an RTX 3090 only; the GPU used here is outside that contract.

## What it gets wrong

- Qwen3.5-0.8B is a small model. It sometimes lists every passage instead of answering, as in the Harbour Branch answer above, and it can be wrong; the sources are printed so you can check.
- The sections are clusters, not a site map. In the run above, Contact lands with the services pages and the home page with the events, because that is what their text is closest to.
- Block labels are right for menus, footers, tables, forms, banners and the map on the sample site, and wrong for cards: two of the three event cards were labelled as paragraphs and the third as a warning banner, and one card on the home page as a welcome banner. `ask` therefore acts only on structure (a block moved to the top, or marked `role="alert"`), never on a label alone.
- Images are embedded, not read: there is no OCR yet, so text that exists only inside an image is not in the index. EmbeddingGemma 2 appears to pick up some rendered text (the closure banner ranks first for "a notice that a branch is closed", though its colour alone could explain that), but it is not a transcript.
- `look` takes about 5 s per block on CPU. Repeated blocks (the same header and footer on every page) are embedded once; the sample site took 4.5 minutes. Use `--max-regions` to keep fewer blocks per page, or a GPU (below 1 minute on the sample site; see above).
- The crawler reads server-rendered HTML. Pages built entirely by JavaScript have no text for `crawl` and are skipped, so `look`, which visits the pages `crawl` kept, never sees them either.

## Crawling other sites

Crawl only sites you are allowed to. The crawler stays on the start URL's scheme, host and port, follows robots.txt (Allow, Disallow, wildcards and Crawl-delay for `*` or its own agent), waits between requests, and stops at `--max-pages`. Its user agent names this repository.

## Models and licences

| Model | Licence | Download |
| --- | --- | --- |
| multilingual-e5-small | MIT | about 0.5 GB |
| GLiNER2 base | Apache-2.0 | about 0.8 GB |
| Qwen3.5-0.8B (fp32) | Apache-2.0 | about 3 GB |
| EmbeddingGemma 2 (text and image, for `look` and `find`) | Apache-2.0, with the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) | about 2.9 GB |

Chromium is downloaded by Playwright's installer. The sample site is fictional; any resemblance to a real library is accidental.
