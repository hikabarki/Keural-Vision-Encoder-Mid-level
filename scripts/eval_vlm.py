"""
Keural Mid VLM — benchmark evaluation (POPE, VQAv2).

Evaluates the Stage 2 VLM (encoder -> projector -> LLM+LoRA) on standard
vision-language benchmarks. Datasets stream from the HuggingFace hub, so no
manual data preparation is needed.

  POPE  — object-hallucination benchmark. Yes/no questions about whether an
          object is present. Reports accuracy / precision / recall / F1, plus
          the yes-ratio (a model that answers "yes" to everything scores ~50%
          accuracy with a yes-ratio near 1.0 — always check both).
  VQAv2 — open-ended visual question answering, scored with the standard VQA
          soft accuracy: min(#matching annotators / 3, 1).

Ported from the PoC evals with the same fixes as train_projector.py:
  * encoder_dim and llm_dim read from the models, not hardcoded 384/4096
  * prompts built with the LLM's own chat template (Qwen), not Mistral [INST]

Usage
-----
    python3 scripts/eval_vlm.py \
        --benchmark pope \
        --projector-ckpt outputs/projector_2b_bilingual/checkpoint-12000/projector.pt \
        --lora-ckpt      outputs/projector_2b_bilingual/checkpoint-12000/lora_adapter \
        --limit 500

    python3 scripts/eval_vlm.py --benchmark vqav2 --projector-ckpt ... --limit 1000

Start with --limit a few hundred for a quick signal; drop it for a full run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

import alignment.hf_wrapper  # registers the encoder with AutoModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from alignment.projectors import LevelAwareProjector

parser = argparse.ArgumentParser(description="Keural Mid VLM benchmark evaluation")
parser.add_argument("--benchmark", choices=["pope", "vqav2"], required=True)
parser.add_argument("--vision-ckpt", default="outputs/run/final")
parser.add_argument("--projector-ckpt", required=True)
parser.add_argument("--lora-ckpt", default=None)
parser.add_argument("--llm-name", default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--token-budget", type=int, default=512)
parser.add_argument("--image-size", type=int, default=384)
parser.add_argument("--hidden-dim", type=int, default=2048)
parser.add_argument("--limit", type=int, default=None,
                    help="Evaluate only the first N samples (quick signal)")
parser.add_argument("--max-new-tokens", type=int, default=16)
parser.add_argument("--output", default=None, help="Write results JSON here")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if device.type == "cuda" else torch.float32

# -- load the pipeline ---------------------------------------------------------
print(f"Loading vision encoder: {args.vision_ckpt}")
vision = AutoModel.from_pretrained(args.vision_ckpt, dtype=DTYPE).to(device).eval()
try:
    vision.encoder.pool_metadata.data[:] = torch.tensor(
        [[0.5, 0.5, 1.0, 1.0]], device=vision.encoder.pool_metadata.device)
    for block in vision.encoder.blocks:
        rope = block.attn.rope
        i = torch.arange(0, rope.half_dim, 2, dtype=torch.float32)
        rope.inv_freq.data = (1.0 / (rope.base ** (i / rope.half_dim))).to(rope.inv_freq.device)
except AttributeError as e:
    print(f"  WARNING: buffer reinit skipped ({e})")
for p in vision.parameters():
    p.requires_grad_(False)
encoder_dim = vision.encoder.embed_dim

print(f"Loading LLM: {args.llm_name}")
tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
llm = AutoModelForCausalLM.from_pretrained(args.llm_name, torch_dtype=DTYPE, device_map="auto")
llm_dim = llm.config.hidden_size
if args.lora_ckpt and os.path.isdir(args.lora_ckpt):
    from peft import PeftModel
    llm = PeftModel.from_pretrained(llm, args.lora_ckpt)
    print(f"  LoRA loaded: {args.lora_ckpt}")
llm.eval()

projector = LevelAwareProjector(encoder_dim=encoder_dim, llm_dim=llm_dim,
                                 hidden_dim=args.hidden_dim).to(device).to(DTYPE)
if not os.path.exists(args.projector_ckpt):
    raise SystemExit(f"ERROR: projector checkpoint not found: {args.projector_ckpt}")
projector.load_state_dict(torch.load(args.projector_ckpt, map_location=device))
projector.eval()
print(f"  projector: {encoder_dim} -> {args.hidden_dim} -> {llm_dim}")

transform = T.Compose([
    T.Resize((args.image_size, args.image_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
embed_fn = llm.get_input_embeddings()
EMBED_DEV = next(embed_fn.parameters()).device


@torch.no_grad()
def ask(pil_img: Image.Image, question: str) -> str:
    pv = transform(pil_img.convert("RGB")).unsqueeze(0).to(device, dtype=DTYPE)
    visual = projector(vision(pv, token_budget=args.token_budget))
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt").to(EMBED_DEV)
    text = embed_fn(ids).to(DTYPE)
    inp = torch.cat([visual.to(EMBED_DEV), text], dim=1)
    attn = torch.ones(inp.shape[:2], dtype=torch.long, device=EMBED_DEV)
    out = llm.generate(inputs_embeds=inp, attention_mask=attn,
                       max_new_tokens=args.max_new_tokens, do_sample=False,
                       pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\b(a|an|the)\b", " ", s).strip()


def yes_no(text: str) -> str:
    t = text.lower()
    if re.search(r"\byes\b", t):
        return "yes"
    if re.search(r"\bno\b", t):
        return "no"
    return "unknown"


from datasets import load_dataset  # noqa: E402

# -- POPE ----------------------------------------------------------------------
if args.benchmark == "pope":
    ds = load_dataset("lmms-lab/POPE", split="test")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    preds, labels, unknown = [], [], 0
    for it in tqdm(ds, desc="POPE"):
        q = it["question"] + "\nAnswer yes or no only."
        p = yes_no(ask(it["image"], q))
        if p == "unknown":
            unknown += 1
            p = "no"  # count unparseable as the negative class rather than skipping
        preds.append(p)
        labels.append(norm(it["answer"]))

    tp = sum(p == "yes" and l == "yes" for p, l in zip(preds, labels))
    fp = sum(p == "yes" and l == "no" for p, l in zip(preds, labels))
    fn = sum(p == "no" and l == "yes" for p, l in zip(preds, labels))
    acc = sum(p == l for p, l in zip(preds, labels)) / len(preds)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    yes_ratio = preds.count("yes") / len(preds)
    results = {"benchmark": "POPE", "n": len(preds), "accuracy": acc,
               "precision": prec, "recall": rec, "f1": f1,
               "yes_ratio": yes_ratio, "unparseable": unknown}
    print(f"\nPOPE  n={len(preds)}")
    print(f"  accuracy : {acc*100:.2f}%")
    print(f"  precision: {prec*100:.2f}%   recall: {rec*100:.2f}%   F1: {f1*100:.2f}%")
    print(f"  yes-ratio: {yes_ratio:.3f}  (near 1.0 => model says yes to everything)")
    print(f"  unparseable answers: {unknown}")

# -- VQAv2 ---------------------------------------------------------------------
else:
    ds = load_dataset("lmms-lab/VQAv2", split="validation")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    total = 0.0
    for it in tqdm(ds, desc="VQAv2"):
        q = it["question"] + "\nAnswer the question using a single word or phrase."
        pred = norm(ask(it["image"], q))
        answers = [norm(a) for a in it["answers"]]
        total += min(answers.count(pred) / 3.0, 1.0)   # standard VQA soft accuracy
    acc = total / len(ds)
    results = {"benchmark": "VQAv2", "n": len(ds), "accuracy": acc}
    print(f"\nVQAv2  n={len(ds)}")
    print(f"  soft accuracy: {acc*100:.2f}%")

out_path = args.output or f"outputs/eval_{args.benchmark}.json"
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {out_path}")
