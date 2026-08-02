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
  - image-text-to-text
  - bilingual
  - korean
  - lora
  - peft
  - keural
library_name: peft
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen2.5-7B-Instruct
model-index:
  - name: keural-mid-vlm-bilingual
    results:
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
          name: Multimodal Evaluation
        dataset:
          type: lmms-lab/MME
          name: MME (n=2374, 14 categories)
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

# Keural Mid VLM — Bilingual (EN / KO)

A bilingual vision-language model built on **Keural Mid**, a 183.6M-parameter vision
encoder trained from random initialization — not a CLIP, ViT or DINO derivative.

This repository contains the trained bridge only: a projector and LoRA adapters,
**37 MB total**. The encoder and decoder load from their own repositories.

---

## Measured vision contribution

A vision-language model can post respectable scores while barely using the image —
"what colour is the banana?" is answerable from language priors alone. Every benchmark
below was therefore run **twice**: once normally, once with the visual tokens removed
and everything else held constant.

**The delta is the encoder's contribution.** It is the one number a language model
cannot manufacture.

| Benchmark | With image | Blind | **Δ** |
|:---|---:|---:|---:|
| **POPE** accuracy · n=2,000 | **70.50%** | 44.55% | **+25.95** |
| **POPE** F1 · n=2,000 | **70.59%** | 23.46% | **+47.13** |
| **VQAv2** strict · n=5,000 | **53.04%** | 39.83% | **+13.21** |
| **VQAv2** lenient · n=5,000 | **53.58%** | 40.27% | **+13.31** |
| **MME** total, 14 categories · n=2,374 | **1258.9** | 812.0 | **+446.9** |
| **MMBench** · n=4,377 | 60.20% | — | — |
| Held-out perplexity · KO / EN | **2.15** / 2.99 | — | — |

![Vision contribution](stage2_vision_contribution.png)

Two figures qualify the headline numbers and are reported because they are what a
careful reader will check:

**POPE yes-ratio is 0.492 against a gold ratio of 0.500.** The model discriminates on
object existence rather than defaulting to one answer. A model replying "yes"
indiscriminately scores ~50% accuracy while being useless, so accuracy without the
yes-ratio is uninterpretable.

**The blind VQAv2 score of 39.83% lands where language-prior performance is expected.**
That the baseline behaves as theory predicts is corroborating evidence that the
measurement itself is sound.

---

## Composition

| Component | Source | Trained here | Size |
|:---|:---|:---:|---:|
| Vision encoder | [`keural-vision-encoder-mid`](https://huggingface.co/mkd-hika/keural-vision-encoder-mid) | frozen | 700 MB |
| Decoder | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | frozen | 15 GB |
| LevelAwareProjector | **this repo** | **8.9M** | 17 MB |
| LoRA adapters · r=16 α=32 | **this repo** | **5.0M** | 20 MB |

14.0M trainable parameters against 7.8B frozen. The encoder is never fine-tuned at any
learning rate, so every result above is attributable to the Stage 1 representation plus
a trained bridge.

---

## Usage

```python
import torch, torchvision.transforms as T
from PIL import Image
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import hf_hub_download
from alignment.projectors import LevelAwareProjector   # github.com/mkd-hika/Keural-VLM-Mid-level

DT = torch.bfloat16
REPO = "mkd-hika/keural-mid-vlm-bilingual"

vision = AutoModel.from_pretrained(
    "mkd-hika/keural-vision-encoder-mid", trust_remote_code=True, dtype=DT).cuda().eval()

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
llm = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", dtype=DT, device_map="auto")
llm = PeftModel.from_pretrained(llm, REPO).eval()

proj = LevelAwareProjector(encoder_dim=768, llm_dim=3584, hidden_dim=2048).cuda().to(DT)
proj.load_state_dict(torch.load(hf_hub_download(REPO, "projector.pt")))
proj.eval()

tf = T.Compose([T.Resize((512, 512)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

@torch.no_grad()
def ask(image_path: str, question: str) -> str:
    px  = tf(Image.open(image_path).convert("RGB")).unsqueeze(0).cuda().to(DT)
    vis = proj(vision(px, token_budget=512))                    # (1, 512, 3584)
    emb = llm.get_input_embeddings()
    ids = tok.apply_chat_template([{"role": "user", "content": question}],
                                  tokenize=True, add_generation_prompt=True,
                                  return_tensors="pt").to(vis.device)
    inp = torch.cat([vis, emb(ids).to(DT)], dim=1)
    out = llm.generate(
        inputs_embeds=inp,
        attention_mask=torch.ones(inp.shape[:2], dtype=torch.long, device=vis.device),
        max_new_tokens=128, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0], skip_special_tokens=True)

ask("photo.jpg", "What is in this image?")
ask("photo.jpg", "이 이미지에 무엇이 있나요?")
```

> **Inference must run at 512×512 with a 512-token budget.** Both are what Stage 2
> trained at, neither is auto-detected, and a mismatch degrades output silently with no
> error raised.

---

## Training

| Phase | Objective | Data | Records | Steps | LR | Held-out ppl |
|:---|:---|:---|---:|---:|---:|:---|
| 2a | Projector alignment | LLaVA-Pretrain | 557,128 | 5,000 | 5e-4 | 31.81 → 17.30 (EN) |
| 2b | Supervised instruction tuning | LLaVA-Instruct-150K | 156,712 | 8,000 | 2e-5 | 3.22 → 2.90 (EN) |
| 2b | Bilingual SFT | Merged EN/KO | 603,255 | 12,000 | 2e-5 | 2.90 → **2.14** (KO) |

![Held-out perplexity](stage2_val_perplexity.png)

2× NVIDIA H200 · DistributedDataParallel · bfloat16 · effective batch 64 · seed 42.
Label masking supervises **assistant turns only**, so the model learns to answer rather
than to reproduce the question.

**On the bilingual mixture.** It is 60% Korean / 40% English, rebalanced from an
original 70/30 in response to a measured failure rather than a hunch. The English-only
model scored 1.73% strict against 48.60% lenient on VQAv2 — a 47-point divergence
caused by answer *format*, not perception. Asked "Is this rice noodle soup?" with gold
answer "yes", it replied *"Yes, this is a bowl of rice noodle soup"*, which exact-match
scoring credits as zero. The mixture carried only verbose conversational data and no
English short-answer supervision. Adding 84,590 short-answer records raised strict
accuracy to 53.04%, and strict and lenient now agree to within 0.54 points.

---

## Scope and limitations

- **Long-form generation fabricates detail.** Asked for a detailed description the model
  invents plausible objects that are not present. Short answers are substantially more
  reliable, so answer length functions as a reliability control rather than a stylistic
  one.
- **Text recognition is partial.** No OCR data was used in training. The model scores
  110.0 on the MME OCR category against a blind 50.0 and reads large, high-contrast
  signage, but not documents or small text.
- **Output is sensitive to prompt phrasing.** On POPE, the instruction *"Answer yes or
  no only"* — absent from training — collapsed the yes-ratio to 0.050, while the trained
  phrasing gave 0.492. Prompts resembling the training format behave materially
  differently from those that do not.
- **The encoder bounds everything downstream.** At 37.2% zero-shot ImageNet it is
  undertrained relative to CLIP ViT-L/14 (~75%), and its Stage 1 loss was still
  declining when the run ended.
- **Single images only.** No temporal modelling.

---

## Citation

```bibtex
@misc{keural_mid_vlm_2026,
  title  = {Keural Mid VLM: A Bilingual Vision-Language Model on an
            Adaptive-Tokenization Encoder},
  author = {Barki, Hika},
  year   = {2026},
  url    = {https://huggingface.co/mkd-hika/keural-mid-vlm-bilingual},
  note   = {MKD Co., Ltd.}
}
```

**License** — Proprietary, MKD Co., Ltd. Qwen2.5-7B-Instruct (Apache-2.0) is used as the
decoder and is not redistributed here.
