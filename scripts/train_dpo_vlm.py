"""
Stage 3 — Direct Preference Optimization for the Keural Mid VLM.

    python3 scripts/train_dpo_vlm.py \
        --base outputs/projector_2b_bilingual/checkpoint-12000 \
        --output-dir outputs/dpo_stage3 --max-steps 2000

What trains and what does not
-----------------------------
Only the LoRA adapters train. The vision encoder and the projector are both
frozen: preference over responses is a property of the decoder, and the
projector is already aligned by Stage 2.

The reference policy
--------------------
DPO requires pi_ref to be the SFT policy training starts from -- here Qwen2.5-7B
PLUS the Stage 2b LoRA. It is NOT base Qwen. Disabling the adapter would give
base Qwen and constrain the policy toward the wrong model, so instead the
reference log-probabilities are PRECOMPUTED once, before any weight update, with
the Stage 2b adapter active. Because pi_ref is frozen by definition, those values
never change, so caching them is exact rather than an approximation -- and it
removes two forward passes from every training step. The cache is keyed by the
base checkpoint so it cannot be silently reused across different starting
models.

The objective
-------------
    L = -log sigma( beta * [ (log pi_t(y_w|x) - log pi_ref(y_w|x))
                           - (log pi_t(y_l|x) - log pi_ref(y_l|x)) ] )

y_w is the human-corrected response, y_l the hallucinated one. Only response
tokens contribute to the log-probabilities; the visual prefix and the prompt are
excluded, because including them would let the model lower its loss by changing
how it scores the QUESTION rather than the ANSWER.

Metrics
-------
Loss alone is a poor progress signal for DPO. The script also reports:
  reward_acc  fraction of pairs where the chosen response scores above the
              rejected one. This is the number that matters -- it starts near
              0.5 and should climb.
  margin      mean implicit reward gap. Rising margin with flat reward_acc means
              the model is growing confident on pairs it already had right.
A held-out split is evaluated periodically, because DPO overfits small
preference sets quickly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import alignment.hf_wrapper  # registers the encoder  # noqa: F401
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from alignment.projectors import LevelAwareProjector

DATA = "/home/work/keural-model/vlm-datasets"

p = argparse.ArgumentParser(description="DPO for Keural Mid VLM")
p.add_argument("--base", default="outputs/projector_2b_bilingual/checkpoint-12000",
               help="Stage 2 checkpoint supplying projector + starting LoRA")
p.add_argument("--vision-ckpt", default="outputs/run/final")
p.add_argument("--llm-name", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--train-jsonl", default=f"{DATA}/dpo_rlhfv.jsonl")
p.add_argument("--val-jsonl", default=f"{DATA}/dpo_rlhfv_val.jsonl")
p.add_argument("--image-root", default=DATA)
p.add_argument("--output-dir", default="outputs/dpo_stage3")
p.add_argument("--beta", type=float, default=0.1,
               help="KL strength. Larger keeps the policy closer to reference.")
p.add_argument("--lr", type=float, default=5e-6,
               help="DPO is far more sensitive than SFT; 5e-6 is already brisk.")
p.add_argument("--max-steps", type=int, default=2000)
p.add_argument("--grad-accum", type=int, default=8)
p.add_argument("--warmup-steps", type=int, default=100)
p.add_argument("--max-len", type=int, default=640)
p.add_argument("--token-budget", type=int, default=512)
p.add_argument("--image-size", type=int, default=512)
p.add_argument("--hidden-dim", type=int, default=2048)
p.add_argument("--val-every", type=int, default=200)
p.add_argument("--val-samples", type=int, default=100)
p.add_argument("--save-every", type=int, default=500)
p.add_argument("--log-every", type=int, default=10)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)
log_path = out_dir / "training_log.jsonl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.bfloat16 if device.type == "cuda" else torch.float32

# -- inherit the exact inference config the base checkpoint was trained with ---
cfg_p = Path(args.base) / "config.json"
if cfg_p.exists():
    cfg = json.loads(cfg_p.read_text())
    for k in ("token_budget", "image_size", "hidden_dim"):
        if k in cfg and getattr(args, k) != cfg[k]:
            print(f"[config] {k}: {getattr(args, k)} -> {cfg[k]} (from base checkpoint)")
            setattr(args, k, cfg[k])

print(f"[1/3] Vision encoder: {args.vision_ckpt}")
vision = AutoModel.from_pretrained(args.vision_ckpt, dtype=DT).to(device).eval()
try:
    vision.encoder.pool_metadata.data[:] = torch.tensor(
        [[0.5, 0.5, 1.0, 1.0]], device=vision.encoder.pool_metadata.device)
    for blk in vision.encoder.blocks:
        r = blk.attn.rope
        i = torch.arange(0, r.half_dim, 2, dtype=torch.float32)
        r.inv_freq.data = (1.0 / (r.base ** (i / r.half_dim))).to(r.inv_freq.device)
except AttributeError as e:
    print(f"  WARNING: buffer reinit skipped ({e})")
for q in vision.parameters():
    q.requires_grad_(False)

print(f"[2/3] LLM: {args.llm_name}")
tok = AutoTokenizer.from_pretrained(args.llm_name)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
llm = AutoModelForCausalLM.from_pretrained(args.llm_name, dtype=DT, device_map="auto")

from peft import PeftModel
adapter = Path(args.base) / "lora_adapter"
llm = PeftModel.from_pretrained(llm, str(adapter), is_trainable=True)
llm.train()
trainable = [q for q in llm.parameters() if q.requires_grad]
print(f"  LoRA trainable: {sum(q.numel() for q in trainable):,} params")
if not trainable:
    raise SystemExit("ERROR: no trainable LoRA parameters — check is_trainable")

print(f"[3/3] Projector (FROZEN): {args.base}/projector.pt")
proj = LevelAwareProjector(encoder_dim=vision.encoder.embed_dim,
                           llm_dim=llm.config.hidden_size,
                           hidden_dim=args.hidden_dim).to(device).to(DT)
proj.load_state_dict(torch.load(Path(args.base) / "projector.pt", map_location=device))
proj.eval()
for q in proj.parameters():
    q.requires_grad_(False)

transform = T.Compose([
    T.Resize((args.image_size, args.image_size)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
emb_fn = llm.get_input_embeddings()
ED = next(emb_fn.parameters()).device


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


train_rows = load_rows(args.train_jsonl)
val_rows = load_rows(args.val_jsonl)[: args.val_samples]
print(f"\npairs: {len(train_rows):,} train, {len(val_rows):,} val")


@torch.no_grad()
def visual_prefix(img_rel):
    """Projected visual tokens. Under no_grad because both the encoder and the
    projector are frozen — gradients only ever reach the LoRA adapters."""
    img = Image.open(os.path.join(args.image_root, img_rel)).convert("RGB")
    pv = transform(img).unsqueeze(0).to(device, dtype=DT)
    return proj(vision(pv, token_budget=args.token_budget)).to(ED)


def build(question, response):
    """Return (prompt_ids, response_ids) using the decoder's own chat template."""
    pid = tok.apply_chat_template([{"role": "user", "content": question}],
                                  tokenize=True, add_generation_prompt=True,
                                  return_tensors="pt").to(ED)
    rid = tok(response, return_tensors="pt",
              add_special_tokens=False).input_ids.to(ED)
    keep = max(args.max_len - pid.shape[1], 16)
    return pid, rid[:, :keep]


def seq_logprob(vis, prompt_ids, resp_ids):
    """Sum of log p(response token | everything before it).

    Indexing: the input is [vis (V) | prompt (P) | response (R)]. The logit that
    predicts response token j sits at input position V+P+j-1, so the slice that
    lines up with resp_ids is [V+P-1 : V+P+R-1].
    """
    V = vis.shape[1]
    P = prompt_ids.shape[1]
    R = resp_ids.shape[1]
    inp = torch.cat([vis,
                     emb_fn(prompt_ids).to(DT),
                     emb_fn(resp_ids).to(DT)], dim=1)
    attn = torch.ones(inp.shape[:2], dtype=torch.long, device=ED)
    logits = llm(inputs_embeds=inp, attention_mask=attn).logits
    sl = logits[:, V + P - 1: V + P + R - 1, :]
    logp = F.log_softmax(sl.float(), dim=-1)
    return logp.gather(-1, resp_ids.unsqueeze(-1)).squeeze(-1).sum(-1)


REF: dict[str, tuple[float, float]] = {}


@torch.no_grad()
def precompute_reference(rows, tag):
    """Reference log-probs under the STAGE 2 model, computed before any update.

    pi_ref is frozen by definition, so these are exact for the whole run."""
    cache = out_dir / f"ref_cache_{tag}.json"
    key = str(Path(args.base).resolve())
    if cache.exists():
        blob = json.loads(cache.read_text())
        if blob.get("base") == key:
            print(f"  reusing {cache.name} ({len(blob['ref']):,} pairs)")
            return {k: tuple(v) for k, v in blob["ref"].items()}
        print(f"  {cache.name} was built from a different base — recomputing")
    llm.eval()
    ref, t0 = {}, time.time()
    for i, row in enumerate(rows):
        try:
            vis = visual_prefix(row["image"])
            pc, rc = build(row["question"], row["chosen"])
            pr, rr = build(row["question"], row["rejected"])
            ref[row["id"]] = (float(seq_logprob(vis, pc, rc)),
                              float(seq_logprob(vis, pr, rr)))
        except Exception:
            continue
        if (i + 1) % 250 == 0:
            el = time.time() - t0
            print(f"    {i+1:,}/{len(rows):,}  eta {(el/(i+1)*(len(rows)-i-1))/60:.0f}m",
                  flush=True)
    llm.train()
    cache.write_text(json.dumps({"base": key, "ref": ref}))
    print(f"  cached {len(ref):,} reference pairs -> {cache.name}")
    return ref


def dpo_terms(row, with_grad=True):
    r = REF.get(row["id"])
    if r is None:
        raise KeyError("no reference logprob for this pair")
    ref_c, ref_r = r
    vis = visual_prefix(row["image"])
    pc, rc = build(row["question"], row["chosen"])
    pr, rr = build(row["question"], row["rejected"])
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        pol_c = seq_logprob(vis, pc, rc)
        pol_r = seq_logprob(vis, pr, rr)
    d_c = pol_c - ref_c            # implicit reward, chosen
    d_r = pol_r - ref_r            # implicit reward, rejected
    logits = args.beta * (d_c - d_r)
    loss = -F.logsigmoid(logits)
    return loss.mean(), float(logits.item()), float((d_c - d_r).item())


@torch.no_grad()
def run_val():
    llm.eval()
    losses, correct, margins = [], 0, []
    for row in val_rows:
        try:
            loss, lg, margin = dpo_terms(row, with_grad=False)
        except Exception:
            continue
        losses.append(float(loss))
        correct += int(lg > 0)
        margins.append(margin)
    llm.train()
    n = max(len(losses), 1)
    return sum(losses) / n, correct / n, sum(margins) / n


print("\nPrecomputing reference log-probabilities (pi_ref = the Stage 2 model)…")
REF.update(precompute_reference(train_rows, "train"))
REF.update(precompute_reference(val_rows, "val"))
# Sanity: at step 0 the policy IS the reference, so the implicit reward margin
# must be ~0. A large value here means the reference is the wrong model.
_probe = [r for r in val_rows[:8] if r["id"] in REF]
if _probe:
    ms = []
    for r in _probe:
        try:
            _, _, m = dpo_terms(r, with_grad=False)
            ms.append(m)
        except Exception:
            pass
    if ms:
        mm = sum(abs(x) for x in ms) / len(ms)
        print(f"  step-0 |margin| = {mm:.4f} over {len(ms)} pairs "
              f"({'OK — policy == reference' if mm < 0.05 else 'WARNING — reference mismatch'})")

opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))


def lr_at(step):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / max(1, args.warmup_steps)
    prog = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))


print(f"\nStarting DPO: beta={args.beta}  lr={args.lr}  steps={args.max_steps}  "
      f"grad-accum={args.grad_accum}  image={args.image_size}px/{args.token_budget}tok")
print("  reward_acc is the metric to watch — it starts near 0.500\n")

rng = random.Random(args.seed)
order = list(range(len(train_rows)))
rng.shuffle(order)
ptr = 0
t0 = time.time()
run_loss, run_acc, run_margin, seen = 0.0, 0, 0.0, 0

for step in range(args.max_steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    opt.zero_grad(set_to_none=True)

    for _ in range(args.grad_accum):
        if ptr >= len(order):
            rng.shuffle(order)
            ptr = 0
        row = train_rows[order[ptr]]
        ptr += 1
        try:
            loss, lg, margin = dpo_terms(row, with_grad=True)
        except Exception as e:
            if seen < 3:
                print(f"  skip pair: {type(e).__name__}: {str(e)[:90]}")
            continue
        (loss / args.grad_accum).backward()
        run_loss += float(loss)
        run_acc += int(lg > 0)
        run_margin += margin
        seen += 1

    gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()

    if step % args.log_every == 0 and seen:
        el = time.time() - t0
        eta = el / (step + 1) * (args.max_steps - step - 1)
        line = (f"[dpo] step={step:5d}/{args.max_steps} "
                f"({step/args.max_steps*100:.1f}%) ETA={int(eta//3600)}h{int(eta%3600//60):02d}m "
                f"lr={lr_at(step):.2e} loss={run_loss/seen:.4f} "
                f"reward_acc={run_acc/seen:.3f} margin={run_margin/seen:+.3f} "
                f"grad_norm={float(gn):.2e}")
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(json.dumps({"step": step, "loss": run_loss / seen,
                                "reward_acc": run_acc / seen,
                                "margin": run_margin / seen,
                                "lr": lr_at(step)}) + "\n")
        run_loss, run_acc, run_margin, seen = 0.0, 0, 0.0, 0

    if step > 0 and step % args.val_every == 0:
        vl, va, vm = run_val()
        print(f"[val]  step={step:5d}  loss={vl:.4f}  reward_acc={va:.3f}  "
              f"margin={vm:+.3f}", flush=True)
        with log_path.open("a") as f:
            f.write(json.dumps({"step": step, "type": "val", "val_loss": vl,
                                "val_reward_acc": va, "val_margin": vm}) + "\n")

    if step > 0 and step % args.save_every == 0:
        d = out_dir / f"checkpoint-{step}"
        d.mkdir(exist_ok=True)
        llm.save_pretrained(d / "lora_adapter")
        torch.save(proj.state_dict(), d / "projector.pt")   # frozen copy, for eval convenience
        (d / "config.json").write_text(json.dumps({
            "step": step, "stage": "dpo_stage3", "beta": args.beta, "lr": args.lr,
            "base": args.base, "token_budget": args.token_budget,
            "image_size": args.image_size, "hidden_dim": args.hidden_dim,
            "llm_name": args.llm_name, "encoder_dim": vision.encoder.embed_dim,
            "llm_dim": llm.config.hidden_size, "seed": args.seed,
            "note": "projector and vision encoder frozen; only LoRA trained",
        }, indent=2))
        print(f"[dpo] saved {d}", flush=True)

d = out_dir / f"checkpoint-{args.max_steps}"
d.mkdir(exist_ok=True)
llm.save_pretrained(d / "lora_adapter")
torch.save(proj.state_dict(), d / "projector.pt")
vl, va, vm = run_val()
print(f"\nFinal held-out: loss={vl:.4f}  reward_acc={va:.3f}  margin={vm:+.3f}")
print(f"Saved {d}")
