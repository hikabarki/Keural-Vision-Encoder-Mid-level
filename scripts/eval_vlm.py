"""
Keural Mid VLM — benchmark evaluation.

Benchmarks
----------
  pope     Object-hallucination (yes/no). Reports accuracy / precision / recall
           / F1 PLUS the yes-ratio — a model that answers "yes" to everything
           scores ~50% accuracy, so accuracy alone is not interpretable.
  vqav2    Open-ended VQA, standard soft accuracy: min(#matching / 3, 1).
  mme      14 perception/cognition subtasks (yes/no). Reports per-category
           accuracy and the standard MME score (acc + acc_plus).
  mmbench  Multiple choice (A/B/C/D) — robust to answer-parsing noise.
  heldout  Teacher-forced loss + perplexity on our own held-out split,
           reported SEPARATELY FOR ENGLISH AND KOREAN. This is the only
           Korean measurement available: no public Korean VLM benchmark
           exists, and the bilingual run is ~70% Korean, so without this the
           majority of training would be unmeasured.

The --no-vision blind baseline
------------------------------
VQAv2 and MME are substantially answerable from language priors alone — a model
that never sees the image can score ~40% on VQAv2 by exploiting question
statistics ("what color is the banana?"). Run any benchmark with --no-vision to
get the blind score, then compare:

    vision_score - blind_score  =  what the vision pipeline actually contributes

If that gap is near zero, the encoder/projector are not helping, regardless of
how respectable the headline number looks. Given this encoder is undertrained
(37.2% zero-shot ImageNet), running the blind baseline is not optional.

Usage
-----
    P=outputs/projector_2b_bilingual/checkpoint-12000
    python3 scripts/eval_vlm.py --benchmark pope \
        --projector-ckpt $P/projector.pt --lora-ckpt $P/lora_adapter --limit 500

    # the diagnostic pair
    python3 scripts/eval_vlm.py --benchmark vqav2 --projector-ckpt $P/projector.pt --limit 1000
    python3 scripts/eval_vlm.py --benchmark vqav2 --projector-ckpt $P/projector.pt --limit 1000 --no-vision

    # Korean + English held-out quality
    python3 scripts/eval_vlm.py --benchmark heldout --projector-ckpt $P/projector.pt --limit 500
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
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

DATA_ROOT = "/home/work/keural-model/vlm-datasets"

parser = argparse.ArgumentParser(description="Keural Mid VLM benchmark evaluation")
parser.add_argument("--benchmark", required=True,
                    choices=["pope", "vqav2", "mme", "mmbench", "heldout"])
parser.add_argument("--vision-ckpt", default="outputs/run/final")
parser.add_argument("--projector-ckpt", required=True)
parser.add_argument("--lora-ckpt", default=None)
parser.add_argument("--llm-name", default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--token-budget", type=int, default=512)
parser.add_argument("--image-size", type=int, default=512,
                    help="MUST match Stage 2 training (512). The encoder was\n                         pretrained at 384/448, but Stage 2 trains the projector\n                         at 512 — inferring at a different size is off-distribution.")
parser.add_argument("--hidden-dim", type=int, default=2048)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--max-new-tokens", type=int, default=16)
parser.add_argument("--no-vision", action="store_true",
                    help="BLIND BASELINE: drop the visual tokens entirely and answer "
                         "from text alone. Compare against the normal run to measure "
                         "what vision actually contributes.")
parser.add_argument("--output", default=None)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if device.type == "cuda" else torch.float32

# -- load pipeline -------------------------------------------------------------
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
if args.no_vision:
    print("\n  *** BLIND BASELINE: visual tokens disabled (text-only) ***\n")

transform = T.Compose([
    T.Resize((args.image_size, args.image_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
embed_fn = llm.get_input_embeddings()
EMBED_DEV = next(embed_fn.parameters()).device


@torch.no_grad()
def visual_tokens(pil_img):
    """Projected visual tokens, or an empty tensor under --no-vision."""
    if args.no_vision:
        return torch.zeros(1, 0, llm_dim, device=EMBED_DEV, dtype=DTYPE)
    pv = transform(pil_img.convert("RGB")).unsqueeze(0).to(device, dtype=DTYPE)
    return projector(vision(pv, token_budget=args.token_budget)).to(EMBED_DEV)


@torch.no_grad()
def ask(pil_img, question: str) -> str:
    vis = visual_tokens(pil_img)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt").to(EMBED_DEV)
    inp = torch.cat([vis, embed_fn(ids).to(DTYPE)], dim=1)
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


def yn_metrics(preds, labels, unknown):
    tp = sum(p == "yes" and l == "yes" for p, l in zip(preds, labels))
    fp = sum(p == "yes" and l == "no" for p, l in zip(preds, labels))
    fn = sum(p == "no" and l == "yes" for p, l in zip(preds, labels))
    acc = sum(p == l for p, l in zip(preds, labels)) / max(len(preds), 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "yes_ratio": preds.count("yes") / max(len(preds), 1),
            "unparseable": unknown, "n": len(preds)}


from datasets import load_dataset  # noqa: E402

results = {"benchmark": args.benchmark, "blind_baseline": args.no_vision}

# ---------------------------------------------------------------- POPE --------
if args.benchmark == "pope":
    ds = load_dataset("lmms-lab/POPE", split="test")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    preds, labels, unk = [], [], 0
    for it in tqdm(ds, desc="POPE"):
        p = yes_no(ask(it["image"], it["question"] + "\nAnswer yes or no only."))
        if p == "unknown":
            unk += 1; p = "no"
        preds.append(p); labels.append(norm(it["answer"]))
    results.update(yn_metrics(preds, labels, unk))
    print(f"\nPOPE  n={results['n']}")
    print(f"  accuracy {results['accuracy']*100:.2f}%  F1 {results['f1']*100:.2f}%")
    print(f"  yes-ratio {results['yes_ratio']:.3f}  (near 1.0 => answers yes to everything)")

# --------------------------------------------------------------- VQAv2 --------
elif args.benchmark == "vqav2":
    ds = load_dataset("lmms-lab/VQAv2", split="validation")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    total, lenient, samples = 0.0, 0.0, []
    for it in tqdm(ds, desc="VQAv2"):
        q = it["question"] + "\nAnswer the question using a single word or phrase."
        raw = ask(it["image"], q)
        pred = norm(raw)
        # lmms-lab/VQAv2 stores each annotation as a dict
        # {"answer", "answer_confidence", "answer_id"} -- not a bare string.
        # Passing the dict straight to norm() raised AttributeError and killed
        # the whole benchmark after the model had already loaded.
        golds = [norm(a["answer"] if isinstance(a, dict) else a)
                 for a in it["answers"]]
        total += min(golds.count(pred) / 3.0, 1.0)

        # Lenient score: a gold answer counts if it appears as a whole word in
        # the prediction. Stage 2b trained only on verbose LLaVA-Instruct
        # conversations -- no short-answer VQA data -- so the model replies
        # "Yes, this is a bowl of rice noodle soup." where VQAv2 expects "yes".
        # Strict exact match scores that 0, which measures output FORMAT rather
        # than whether the answer was right. Reporting both separates
        # "got it wrong" from "got it right but wouldn't shut up".
        hits = sum(1 for g in golds
                   if g and re.search(rf"\b{re.escape(g)}\b", pred))
        lenient += min(hits / 3.0, 1.0)
        if len(samples) < 30:
            samples.append({"question": it["question"],
                            "gold": it.get("multiple_choice_answer"),
                            "prediction": raw})

    results.update({"accuracy": total / len(ds),
                    "lenient_accuracy": lenient / len(ds),
                    "n": len(ds), "samples": samples})
    print(f"\nVQAv2  n={len(ds)}")
    print(f"  strict soft accuracy  {results['accuracy']*100:.2f}%   "
          "(exact match — penalises verbose answers)")
    print(f"  lenient accuracy      {results['lenient_accuracy']*100:.2f}%   "
          "(gold appears in the answer)")

# ----------------------------------------------------------------- MME --------
elif args.benchmark == "mme":
    ds = load_dataset("lmms-lab/MME", split="test")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    per_cat = defaultdict(lambda: {"preds": [], "labels": [], "unk": 0})
    for it in tqdm(ds, desc="MME"):
        cat = it.get("category", "unknown")
        q = it["question"]
        if "yes or no" not in q.lower():
            q += "\nAnswer yes or no only."
        p = yes_no(ask(it["image"], q))
        b = per_cat[cat]
        if p == "unknown":
            b["unk"] += 1; p = "no"
        b["preds"].append(p); b["labels"].append(norm(it["answer"]))
    cats, total_score = {}, 0.0
    for cat, b in sorted(per_cat.items()):
        m = yn_metrics(b["preds"], b["labels"], b["unk"])
        # MME "acc+" credits a pair only when both of its questions are right
        pairs = list(zip(b["preds"], b["labels"]))
        acc_plus = sum(
            pairs[i][0] == pairs[i][1] and pairs[i + 1][0] == pairs[i + 1][1]
            for i in range(0, len(pairs) - 1, 2)) / max(len(pairs) // 2, 1)
        m["acc_plus"] = acc_plus
        m["score"] = (m["accuracy"] + acc_plus) * 100
        total_score += m["score"]
        cats[cat] = m
    results.update({"categories": cats, "total_score": total_score,
                    "n": sum(len(b["preds"]) for b in per_cat.values())})
    print(f"\nMME  n={results['n']}  total score {total_score:.1f}")
    for c, m in cats.items():
        print(f"  {c:<28s} acc {m['accuracy']*100:5.1f}%  score {m['score']:6.1f}")

# ------------------------------------------------------------- MMBench --------
elif args.benchmark == "mmbench":
    ds = load_dataset("lmms-lab/MMBench_EN", split="dev")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    correct, n = 0, 0
    for it in tqdm(ds, desc="MMBench"):
        opts = [(k, it.get(k)) for k in ["A", "B", "C", "D"] if it.get(k)]
        q = it["question"] + "\n" + "\n".join(f"{k}. {v}" for k, v in opts)
        q += "\nAnswer with the option letter only."
        out = ask(it["image"], q).strip().upper()
        m = re.search(r"\b([ABCD])\b", out)
        pred = m.group(1) if m else None
        gold = str(it.get("answer", "")).strip().upper()
        if gold:
            n += 1
            correct += int(pred == gold)
    results.update({"accuracy": correct / max(n, 1), "n": n})
    print(f"\nMMBench  n={n}  accuracy {results['accuracy']*100:.2f}%")

# ------------------------------------------------------------- heldout --------
else:
    from data.vlm_dataset import _normalize_conversation, build_llava_batch, IGNORE_INDEX
    splits = {
        "en": (f"{DATA_ROOT}/llava-instruct/train_150k_val.jsonl",
               f"{DATA_ROOT}/coco/train2017"),
        "ko": (f"{DATA_ROOT}/kollava-instruct/train_ko_coco_val.jsonl", DATA_ROOT),
    }
    per_lang = {}
    for lang, (jsonl, root) in splits.items():
        if not os.path.exists(jsonl):
            print(f"  skip {lang}: {jsonl} not found"); continue
        recs = [json.loads(l) for l in open(jsonl)]
        if args.limit:
            recs = recs[:args.limit]
        tot_loss, tot_tok = 0.0, 0
        for r in tqdm(recs, desc=f"heldout-{lang}"):
            msgs = _normalize_conversation(r)
            img_p = Path(root) / r["image"]
            if not msgs or not img_p.exists():
                continue
            with torch.no_grad():
                vis = visual_tokens(Image.open(img_p))
                ids, attn, labels = build_llava_batch(
                    [msgs], tokenizer, 512, EMBED_DEV)
                inp = torch.cat([vis, embed_fn(ids).to(DTYPE)], dim=1)
                full_attn = torch.cat(
                    [torch.ones(vis.shape[:2], dtype=torch.long, device=EMBED_DEV), attn], dim=1)
                vis_lab = torch.full(vis.shape[:2], IGNORE_INDEX,
                                      dtype=torch.long, device=EMBED_DEV)
                out = llm(inputs_embeds=inp, attention_mask=full_attn,
                          labels=torch.cat([vis_lab, labels], dim=1))
                ntok = int((labels != IGNORE_INDEX).sum())
                if ntok:
                    tot_loss += out.loss.item() * ntok; tot_tok += ntok
        if tot_tok:
            loss = tot_loss / tot_tok
            per_lang[lang] = {"loss": loss, "perplexity": float(torch.exp(torch.tensor(loss))),
                              "n": len(recs), "tokens": tot_tok}
    results["languages"] = per_lang
    print("\nHeld-out (teacher-forced, assistant tokens only)")
    for lang, m in per_lang.items():
        print(f"  {lang.upper()}: loss {m['loss']:.4f}  perplexity {m['perplexity']:.2f}  "
              f"({m['n']} samples, {m['tokens']} scored tokens)")

suffix = "_blind" if args.no_vision else ""
out_path = args.output or f"outputs/eval_{args.benchmark}{suffix}.json"
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved -> {out_path}")
if not args.no_vision and args.benchmark in ("vqav2", "mme", "mmbench"):
    print("\nNOTE: re-run with --no-vision to get the blind baseline. The DIFFERENCE "
          "is what your vision pipeline actually contributes.")
