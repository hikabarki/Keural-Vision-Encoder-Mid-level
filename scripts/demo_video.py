"""
Keural Mid — video understanding probe (no training required).

    python3 scripts/demo_video.py --video clip.mp4 \
        --projector-ckpt outputs/projector_2b/checkpoint-8000/projector.pt \
        --lora-ckpt      outputs/projector_2b/checkpoint-8000/lora_adapter \
        --question "What is happening in this video?"

What this is
------------
Samples N frames, encodes them in ONE batched encoder pass, projects them, and
concatenates the visual tokens into a single LLM prompt. It is a capability
probe, not a video model.

Why it is cheap here specifically
--------------------------------
Two properties of this architecture make it a ~40-line extension rather than a
new model:

  * token_budget is a RUNTIME argument, so each frame can be encoded at a
    reduced budget. A fixed-grid encoder like CLIP emits 576 tokens per frame
    whether you want them or not (8 frames = 4,608 tokens); ATB lets you spend
    128 on each frame and put them where the saliency predictor says they
    matter, so an 8-frame clip costs about what two still images cost.
  * The projector is a per-token MLP, so it is indifferent to how many tokens
    arrive.

What it genuinely CANNOT do
---------------------------
Be honest when reading the output. The projector and LoRA were trained
exclusively on SINGLE images, and the encoder's RoPE is spatial only -- there is
no temporal position signal anywhere in the stack. So this aggregates per-frame
understanding; it does not perceive motion. Questions whose answer depends on
order, direction or causality ("is the door opening or closing?", "what happened
first?") will not work reliably, and a confident answer to one of those is the
language model guessing. Real temporal reasoning needs a Stage 2c: per-frame
temporal embeddings plus training on video instruction data.

--frame-markers (on by default) interleaves a literal "Frame k:" text token
between each frame's visual tokens. It is the cheapest available substitute for
a temporal encoding: it gives the LLM explicit ordering without any training.
Turn it off to see how much it is contributing.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import alignment.hf_wrapper  # registers the encoder with AutoModel  # noqa: F401
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from alignment.projectors import LevelAwareProjector

p = argparse.ArgumentParser(description="Keural Mid video understanding probe")
src = p.add_mutually_exclusive_group(required=True)
src.add_argument("--video", help="Video file (mp4/avi/mov/webm)")
src.add_argument("--frames-dir", help="Directory of pre-extracted frames "
                                      "(sorted by filename) — lets you test "
                                      "without a video file")
p.add_argument("--vision-ckpt", default="outputs/run/final")
p.add_argument("--projector-ckpt", required=True)
p.add_argument("--lora-ckpt", default=None)
p.add_argument("--llm-name", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--question", default="What is happening in this video?")
p.add_argument("--num-frames", type=int, default=8,
               help="Frames sampled uniformly across the clip")
p.add_argument("--frame-budget", type=int, default=128,
               help="Vision tokens PER FRAME. num_frames x frame_budget is the "
                    "total visual context; 8x128=1024 costs about what two "
                    "still images do at the default 512.")
p.add_argument("--image-size", type=int, default=512)
p.add_argument("--hidden-dim", type=int, default=2048)
p.add_argument("--max-new-tokens", type=int, default=256)
p.add_argument("--temperature", type=float, default=0.2)
p.add_argument("--no-frame-markers", action="store_true",
               help="Do not interleave 'Frame k:' text between frames")
p.add_argument("--no-vision", action="store_true",
               help="BLIND: drop all visual tokens. Whatever the model still "
                    "says came from language priors, not from the video.")
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if device.type == "cuda" else torch.float32


# -- frame sampling ------------------------------------------------------------
def sample_frames(n: int) -> list[Image.Image]:
    if args.frames_dir:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        paths = sorted(q for q in Path(args.frames_dir).iterdir()
                       if q.suffix.lower() in exts)
        if not paths:
            raise SystemExit(f"ERROR: no images in {args.frames_dir}")
        idx = [round(i * (len(paths) - 1) / max(n - 1, 1)) for i in range(n)] \
            if len(paths) > 1 else [0]
        picked = [paths[i] for i in dict.fromkeys(idx)]
        print(f"[frames] {len(picked)} of {len(paths)} from {args.frames_dir}")
        return [Image.open(q).convert("RGB") for q in picked]

    import cv2
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if total <= 0:
        raise SystemExit(f"ERROR: {args.video} reports 0 frames")
    # Uniform sampling across the whole clip. Uniform rather than keyframe-based
    # on purpose: keyframe selection would bias toward scene cuts and quietly
    # skip steady action.
    idx = sorted({round(i * (total - 1) / max(n - 1, 1)) for i in range(n)})
    out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok:
            out.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    cap.release()
    dur = total / fps if fps else 0
    print(f"[frames] {len(out)} sampled from {total} frames "
          f"({dur:.1f}s @ {fps:.1f}fps)" if fps else f"[frames] {len(out)} sampled")
    if not out:
        raise SystemExit("ERROR: no frames could be decoded")
    return out


frames = sample_frames(args.num_frames)

# -- load pipeline -------------------------------------------------------------
print(f"[1/3] Vision encoder: {args.vision_ckpt}")
vision = AutoModel.from_pretrained(args.vision_ckpt, dtype=DTYPE).to(device).eval()
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
llm = AutoModelForCausalLM.from_pretrained(args.llm_name, dtype=DTYPE,
                                           device_map="auto")
if args.lora_ckpt and os.path.isdir(args.lora_ckpt):
    from peft import PeftModel
    llm = PeftModel.from_pretrained(llm, args.lora_ckpt)
    print(f"  LoRA: {args.lora_ckpt}")
llm.eval()
llm_dim = llm.config.hidden_size

print(f"[3/3] Projector: {args.projector_ckpt}")
proj = LevelAwareProjector(encoder_dim=vision.encoder.embed_dim, llm_dim=llm_dim,
                           hidden_dim=args.hidden_dim).to(device).to(DTYPE)
if not os.path.exists(args.projector_ckpt):
    raise SystemExit(f"ERROR: projector not found: {args.projector_ckpt}")
proj.load_state_dict(torch.load(args.projector_ckpt, map_location=device))
proj.eval()

transform = T.Compose([
    T.Resize((args.image_size, args.image_size)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
embed_fn = llm.get_input_embeddings()
ED = next(embed_fn.parameters()).device


def text_embeds(s: str) -> torch.Tensor:
    ids = tok(s, return_tensors="pt", add_special_tokens=False).input_ids.to(ED)
    return embed_fn(ids).to(DTYPE)


@torch.no_grad()
def encode_video(imgs: list[Image.Image]) -> torch.Tensor:
    """All frames in ONE batched encoder pass -> (1, total_tokens, llm_dim)."""
    if args.no_vision:
        return torch.zeros(1, 0, llm_dim, device=ED, dtype=DTYPE)
    px = torch.stack([transform(im) for im in imgs]).to(device, dtype=DTYPE)
    out = vision(px, token_budget=args.frame_budget)   # batch-first: (F, N, D)
    projected = proj(out).to(ED)                       # (F, N, llm_dim)
    mask = out.attention_mask.to(ED).bool()            # (F, N) 1 = real token

    pieces = []
    for f in range(projected.shape[0]):
        if not args.no_frame_markers:
            pieces.append(text_embeds(f"\nFrame {f + 1}: "))
        # Drop padding: the contract says only mask==1 positions are real.
        pieces.append(projected[f][mask[f]].unsqueeze(0))
    vis = torch.cat(pieces, dim=1)
    real = int(mask.sum().item())
    print(f"  {projected.shape[0]} frames x {args.frame_budget} budget "
          f"-> {real} visual tokens ({vis.shape[1]} incl. frame markers)")
    return vis


@torch.no_grad()
def answer(imgs, question: str) -> str:
    vis = encode_video(imgs)
    preamble = (f"The following are {len(imgs)} frames sampled in order from a "
                f"video.\n") if not args.no_vision else ""
    ids = tok.apply_chat_template([{"role": "user", "content": preamble + question}],
                                  tokenize=True, add_generation_prompt=True,
                                  return_tensors="pt").to(ED)
    inp = torch.cat([vis, embed_fn(ids).to(DTYPE)], dim=1)
    out = llm.generate(
        inputs_embeds=inp,
        attention_mask=torch.ones(inp.shape[:2], dtype=torch.long, device=ED),
        max_new_tokens=args.max_new_tokens, do_sample=args.temperature > 0,
        temperature=max(args.temperature, 1e-5), top_p=0.9,
        pad_token_id=tok.pad_token_id)
    return tok.decode(out[0], skip_special_tokens=True).strip()


print()
if args.no_vision:
    print("*** BLIND: no visual tokens — this is the language prior alone ***\n")
print(f"Q: {args.question}")
print(f"A: {answer(frames, args.question)}")
