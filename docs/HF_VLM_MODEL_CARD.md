---
language:
  - en
  - ko
license: other
license_name: proprietary
license_link: LICENSE
tags:
  - vision-language-model
  - multimodal
  - visual-question-answering
  - image-to-text
  - bilingual
  - korean
  - lora
  - keural
library_name: peft
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen2.5-7B-Instruct
model-index:
  - name: keural-mid-vlm-bilingual
    results:
      - task:
          type: visual-question-answering
          name: Visual Question Answering
        dataset:
          type: lmms-lab/VQAv2
          name: VQAv2 (validation, n=5000)
        metrics:
          - type: accuracy
            value: 53.04
            name: Strict soft accuracy
          - type: accuracy
            value: 13.21
            name: Gain over blind baseline
      - task:
          type: visual-question-answering
          name: Object Hallucination
        dataset:
          type: lmms-lab/POPE
          name: POPE (n=2000)
        metrics:
          - type: accuracy
            value: 70.50
            name: Accuracy
          - type: f1
            value: 70.59
            name: F1
          - type: accuracy
            value: 25.95
            name: Gain over blind baseline
      - task:
          type: visual-question-answering
          name: Multimodal Evaluation
        dataset:
          type: lmms-lab/MME
          name: MME (n=2374, all 14 categories)
        metrics:
          - type: accuracy
            value: 1258.9
            name: MME total score
          - type: accuracy
            value: 446.9
            name: Gain over blind baseline
      - task:
          type: visual-question-answering
          name: Multimodal Benchmark
        dataset:
          type: lmms-lab/MMBench
          name: MMBench (n=4377)
        metrics:
          - type: accuracy
            value: 60.20
            name: Accuracy
---

# Keural Mid VLM — Bilingual (English + Korean)

A vision-language model built on **Keural Mid**, a 183.6M-parameter vision encoder
trained from scratch — not a CLIP or ViT fine-tune. This repository holds only the
**trained bridge**: a projector (18 MB) and LoRA adapters (20 MB). The frozen
encoder and the decoder are pulled from their own repositories at load time.

| Component | Source | Trained here | Size |
|---|---|---|---|
| Vision encoder | [`mkd-hika/keural-vision-encoder-mid`](https://huggingface.co/mkd-hika/keural-vision-encoder-mid) | ❄️ frozen | 700 MB |
| Decoder | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | ❄️ frozen | 15 GB |
| **LevelAwareProjector** | this repo | ✅ **8.9M params** | 18 MB |
| **LoRA adapters** (r=16, α=32) | this repo | ✅ **5.0M params** | 20 MB |

## What the vision encoder actually contributes

A vision-language model can score respectably while barely using the image, because
many questions are answerable from language priors alone. Every benchmark below is
therefore reported **twice** — once normally, and once with the visual tokens removed
and everything else held constant. The difference is the measured contribution of the
vision pipeline, and it is the only figure a language model cannot produce on its own.

| Benchmark | With image | Blind | **Vision contributes** |
|---|---|---|---|
| POPE accuracy (n=2,000) | **70.50%** | 44.55% | **+25.95** |
| POPE F1 (n=2,000) | **70.59%** | 23.46% | **+47.13** |
| VQAv2 strict (n=5,000) | **53.04%** | 39.83% | **+13.21** |
| VQAv2 lenient (n=5,000) | **53.58%** | 40.27% | **+13.31** |
| MME total, 14 categories (n=2,374) | **1258.9** | 812.0 | **+446.9** |
| MMBench (n=4,377) | 60.20% | — | — |
| Held-out perplexity, English | 2.99 | — | — |
| Held-out perplexity, Korean | **2.15** | — | — |

Two properties are worth reading alongside the headline numbers. The **POPE yes-ratio
is 0.492 against a gold ratio of 0.500**, meaning the model genuinely discriminates on
object-existence questions rather than defaulting to one answer — a model replying
"yes" indiscriminately scores near 50% accuracy while being useless, so accuracy alone
is not interpretable without it. And the **blind VQAv2 score of 39.83% lands where
language-prior performance is expected to**, which is corroborating evidence that the
measurement itself is sound.

## Usage

```python
import torch, torchvision.transforms as T
from PIL import Image
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DT = torch.bfloat16

# 1. frozen vision encoder (trust_remote_code: custom architecture)
vision = AutoModel.from_pretrained(
    "mkd-hika/keural-vision-encoder-mid", trust_remote_code=True, dtype=DT
).cuda().eval()

# 2. decoder + the LoRA adapters from this repo
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
llm = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", dtype=DT, device_map="auto")
llm = PeftModel.from_pretrained(llm, "mkd-hika/keural-mid-vlm-bilingual").eval()

# 3. the projector from this repo
from huggingface_hub import hf_hub_download
from alignment.projectors import LevelAwareProjector          # see the GitHub repo
proj = LevelAwareProjector(encoder_dim=768, llm_dim=3584, hidden_dim=2048).cuda().to(DT)
proj.load_state_dict(torch.load(
    hf_hub_download("mkd-hika/keural-mid-vlm-bilingual", "projector.pt")))
proj.eval()

tf = T.Compose([T.Resize((512, 512)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

@torch.no_grad()
def ask(image_path, question):
    px = tf(Image.open(image_path).convert("RGB")).unsqueeze(0).cuda().to(DT)
    vis = proj(vision(px, token_budget=512))                  # (1, 512, 3584)
    emb = llm.get_input_embeddings()
    ids = tok.apply_chat_template([{"role": "user", "content": question}],
                                  tokenize=True, add_generation_prompt=True,
                                  return_tensors="pt").to(vis.device)
    inp = torch.cat([vis, emb(ids).to(DT)], dim=1)
    out = llm.generate(inputs_embeds=inp,
                       attention_mask=torch.ones(inp.shape[:2], dtype=torch.long,
                                                 device=vis.device),
                       max_new_tokens=128, do_sample=False,
                       pad_token_id=tok.pad_token_id)
    return tok.decode(out[0], skip_special_tokens=True)

print(ask("photo.jpg", "What is in this image?"))
print(ask("photo.jpg", "이 이미지에 무엇이 있나요?"))   # Korean
```

**Input resolution must be 512×512 and the token budget must be 512.** Both are what
Stage 2 trained at, and neither is auto-detected — inferring at a different size is
off-distribution and degrades output with no error message.

## Training

Three phases, each warm-starting from the previous with a fresh optimizer and
learning-rate schedule. The vision encoder is frozen throughout.

| Phase | Data | Records | Steps | LR | Held-out perplexity |
|---|---|---|---|---|---|
| 2a — caption alignment | LLaVA-Pretrain | 557,128 | 5,000 | 5e-4 | 31.81 → 17.30 (EN) |
| 2b — instruction tuning (SFT) | LLaVA-Instruct-150K | 156,712 | 8,000 | 2e-5 | 3.22 → 2.90 (EN) |
| 2b — bilingual SFT | Merged EN/KO | 603,255 | 12,000 | 2e-5 | 2.90 → **2.14** (KO) |

2× NVIDIA H200 under DistributedDataParallel, bfloat16, effective batch 64, seed 42.
Label masking supervises **assistant turns only**, so the model is trained to produce
answers rather than to reproduce questions.

The bilingual mixture is 60% Korean / 40% English. It was rebuilt before the final run
in response to a measured failure: the English-only model scored 1.73% strict against
48.60% lenient on VQAv2 — a 47-point divergence caused by answer *format*, not
perception. Asked "Is this rice noodle soup?" with gold answer "yes", it replied "Yes,
this is a bowl of rice noodle soup", which exact-match scoring credits as zero. Adding
84,590 English short-answer records raised strict accuracy to 53.04%, and strict and
lenient now agree within 0.54 points.

## Limitations

- **Long-form output fabricates detail.** Asked for a detailed description the model
  invents plausible objects that are not present. Short answers are markedly more
  reliable, so answer length is a reliability control, not just a verbosity setting.
- **Text recognition is partial.** No OCR data was used in any training stage. The
  model scores 110.0 on the MME OCR category against a blind 50.0 and reads large,
  high-contrast signage, but cannot read documents or small text.
- **Sensitive to prompt phrasing.** On POPE the instruction "Answer yes or no only" —
  absent from training — collapsed the yes-ratio to 0.050, while the trained phrasing
  gave 0.492. Prompts resembling the training format behave very differently from
  those that do not.
- **The encoder is the ceiling.** At 37.2% zero-shot ImageNet it is undertrained
  relative to CLIP ViT-L/14 (~75%), and it bounds every downstream number here.
- **No temporal modelling.** Video works by frame aggregation only; motion, ordering
  and causality are not perceived.

## Related checkpoints

Three further checkpoints were trained and are **not** recommended as the general
model, but are documented for completeness:

| Checkpoint | MME total | MME OCR | POPE acc | POPE yes-ratio |
|---|---|---|---|---|
| **This model (2b bilingual)** | 1258.9 | 110.0 | **70.50%** | **0.503** |
| 2c v1 — OCR specialist | 1299.7 | **137.5** | 59.20% | 0.886 |
| 2c v2 — OCR + rebalanced | **1337.8** | 115.0 | 64.40% | 0.773 |
| Stage 3 — DPO | — | — | 61.00% | 0.116 |

Every stage after supervised fine-tuning traded one capability for another. OCR
training raised MME but drifted toward answering "yes"; DPO optimized its own
objective (held-out reward accuracy 0.890) but collapsed toward answering "no". This
checkpoint is the only calibrated one, which is why it is the recommended model.

## Citation

```bibtex
@misc{keural_mid_vlm_2026,
  title  = {Keural Mid VLM: A Bilingual Vision-Language Model on an Adaptive-Tokenization Encoder},
  author = {Barki, Hika},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/mkd-hika/keural-mid-vlm-bilingual}},
  note   = {MKD Co., Ltd.}
}
```

## License

Proprietary — MKD Co., Ltd. The base decoder (Qwen2.5-7B-Instruct) is Apache-2.0 and
is not redistributed here.
