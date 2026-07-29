# Keural Mid Vision Encoder — Dataset Documentation

## Overview

The Keural Mid Vision Encoder (183.5M params) is trained on two public, large-scale image-text pair datasets. Both datasets are widely used in published vision-language research and are freely available under permissive licenses.

| Dataset | Total Pairs | Pairs Used (Stage 1) | Disk Size | Source |
|---|---|---|---|---|
| DataComp-1B Medium | ~89M downloaded | **20M curated** (`--max-samples`) | 3.6 TB | [HuggingFace](https://huggingface.co/datasets/mlfoundations/datacomp_1b) |
| COYO-700M Filtered | ~32M downloaded | Reserved for Stage 2 | 2.0 TB | [HuggingFace](https://huggingface.co/datasets/kakaobrain/coyo-700m) |
| **Total** | | **20M (Stage 1)** | **5.6 TB** | |

---

## Dataset 1: DataComp-1B Medium

- **Full Name:** DataComp-1B (Medium Subset)
- **Authors:** Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, et al. (UW, Google, Apple, Stanford, Columbia)
- **License:** Subject to original image licenses (CommonCrawl sourced)
- **Paper:** [DataComp: In Search of the Next Generation of Multimodal Datasets (NeurIPS 2023)](https://arxiv.org/abs/2304.14108)
- **HuggingFace:** [mlfoundations/datacomp_1b](https://huggingface.co/datasets/mlfoundations/datacomp_1b)
- **GitHub:** [mlfoundations/datacomp](https://github.com/mlfoundations/datacomp)

### Description

DataComp is a benchmark and dataset designed specifically for training CLIP-style vision-language models. The "medium" subset contains ~128M image-text pairs pre-filtered for quality from the full 1.4B candidate pool. The filtering was based on CLIP similarity scores, text quality, NSFW filtering, and deduplication.

### Why DataComp?

- Pre-filtered for quality — not random web scrapes
- Designed specifically for vision encoder training
- Used as the standard benchmark in the DataComp competition
- Diverse content: objects, scenes, text, diagrams, art, photography
- Many published models (OpenCLIP, MetaCLIP) trained on DataComp subsets

### Our Usage

- Downloaded metadata (parquet URL lists) from HuggingFace
- Downloaded images using `img2dataset` at 512px resolution
- ~65.6% URL survival rate (34.4% dead links — normal for web-sourced datasets)
- Output format: WebDataset `.tar` shards (10,000 samples per shard)
- Storage path: `/home/work/keural-model/vision-encoder-dataset/datacomp_medium/`

---

## Dataset 2: COYO-700M

- **Full Name:** COYO-700M
- **Authors:** Kakao Brain (Byeon et al.)
- **License:** CC-BY-4.0
- **Paper:** [COYO-700M: Image-Text Pair Dataset](https://github.com/kakaobrain/coyo-dataset)
- **HuggingFace:** [kakaobrain/coyo-700m](https://huggingface.co/datasets/kakaobrain/coyo-700m)
- **GitHub:** [kakaobrain/coyo-dataset](https://github.com/kakaobrain/coyo-dataset)

### Description

COYO-700M contains 747M image-text pairs collected from CommonCrawl (Oct 2020 — Aug 2021). Starting from ~10B alt-text/image pairs in HTML documents, Kakao Brain applied image-level and text-level filtering to produce the final 747M pairs. Each sample includes rich metadata: CLIP similarity scores, image dimensions, watermark/NSFW scores, and more.

### Why COYO?

- High caption quality (alt-text from real web pages)
- Rich metadata enables quality filtering (CLIP scores, watermark detection)
- CC-BY-4.0 license — fully open for research and commercial use
- Complements DataComp with different source distribution and time period
- Used by published models including ViT-L and various CLIP variants

### Our Usage

- Streamed full dataset from HuggingFace (747M rows)
- Filtered to `clip_similarity_vitb32 >= 0.28` — kept 60M high-quality pairs
- Downloaded images using `img2dataset` at 512px resolution
- ~54% URL survival rate (higher dead link rate — dataset is older)
- Output format: WebDataset `.tar` shards (10,000 samples per shard)
- Storage path: `/home/work/keural-model/vision-encoder-dataset/coyo_filtered/`

---

## Data Processing Pipeline

```
Raw URLs (HuggingFace)
    ↓
[Filter] COYO: clip_similarity >= 0.28, DataComp: pre-filtered medium subset
    ↓
[img2dataset] Download + resize to 512px + JPEG encode (quality=95)
    ↓
[WebDataset] Pack into .tar shards (10K samples each)
    ↓
[Index Builder] Scan .tar files → TSV index (caption, shard, sample_id, w, h)
    ↓
[Training] CC3MDataset reads .tar via index, applies transforms on-the-fly
```

### Download Configuration

```
--image_size 512
--resize_mode keep_ratio_largest
--encode_format jpg
--encode_quality 95
--output_format webdataset
--number_sample_per_shard 10000
--processes_count 16
--thread_count 64
```

---

## Comparison with Other Vision Encoder Training Datasets

| Dataset | Pairs | Public | Used by |
|---|---|---|---|
| **DataComp-1B** | 1.4B (128M medium) | Yes | OpenCLIP, MetaCLIP |
| **COYO-700M** | 747M | Yes | ViT-L, various CLIP |
| LAION-5B | 5.8B | Taken down | Stable Diffusion, OpenCLIP |
| WebLI (Google) | 10B+ | No (private) | SigLIP, PaLI |
| WIT (OpenAI) | 400M | No (private) | CLIP |
| DFN-5B (Apple) | 5B | Partially | DFN-CLIP |

---

## References

1. Gadre, S.Y., Ilharco, G., Fang, A., et al. "DataComp: In Search of the Next Generation of Multimodal Datasets." NeurIPS 2023. [arXiv:2304.14108](https://arxiv.org/abs/2304.14108)

2. Byeon, M., Park, B., Kim, H., et al. "COYO-700M: Image-Text Pair Dataset." Kakao Brain, 2022. [GitHub](https://github.com/kakaobrain/coyo-dataset)

---

## Storage Summary

| Path | Contents | Size |
|---|---|---|
| `vision-encoder-dataset/datacomp_medium/` | DataComp image tar shards | 3.6 TB |
| `vision-encoder-dataset/coyo_filtered/` | COYO image tar shards | ~1.5 TB |
| `vision-encoder-dataset/coyo_meta/` | COYO filtered parquet | 7.7 GB |
| `vision-encoder-dataset/indexes/` | TSV index files | ~1 GB |

All data stored on NFS: `/home/work/keural-model/vision-encoder-dataset/`
