"""
Keural Mid VLM — interactive / single-shot inference.

Loads the full Stage 2 pipeline and answers questions about an image:

    image -> [frozen encoder 183.5M] -> 512 x 768 tokens
          -> [trained LevelAwareProjector] -> 512 x <llm_dim>
          -> [LLM + trained LoRA] -> answer text

This is the "did training actually work?" check. Run it as soon as Stage 2a or
2b produces a checkpoint — a loss curve alone will not tell you whether the
model emits coherent text.

Ported from the PoC demo with the same fixes applied to train_projector.py:
  * encoder_dim read from the encoder (768), not hardcoded 384
  * llm_dim read from llm.config.hidden_size, not hardcoded 4096
  * prompt built with the LLM's own chat template (Qwen), not Mistral's
    [INST] markers

Usage
-----
    # single question
    python3 scripts/demo_vlm.py \
        --projector-ckpt outputs/projector_2b_bilingual/checkpoint-12000/projector.pt \
        --lora-ckpt      outputs/projector_2b_bilingual/checkpoint-12000/lora_adapter \
        --image path/to/image.jpg \
        --question "What is in this image?"

    # interactive chat about one image
    python3 scripts/demo_vlm.py --projector-ckpt ... --lora-ckpt ... --image x.jpg --chat

    # Korean
    python3 scripts/demo_vlm.py ... --question "이 이미지에 무엇이 있나요?"

After Stage 2a (captions only) the model captions but does not follow
instructions — that is expected; instruction-following comes from 2b.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torchvision.transforms as T
from PIL import Image

import alignment.hf_wrapper  # registers KeuralVisionConfig/Model with AutoModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from alignment.projectors import LevelAwareProjector

parser = argparse.ArgumentParser(description="Keural Mid VLM inference")
parser.add_argument("--vision-ckpt", default="outputs/run/final")
parser.add_argument("--projector-ckpt", required=True,
                    help="Trained projector weights (.pt) from Stage 2a/2b")
parser.add_argument("--lora-ckpt", default=None,
                    help="LoRA adapter dir from Stage 2 (omit to use the base LLM)")
parser.add_argument("--llm-name", default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--image", required=True)
parser.add_argument("--question", default="Describe this image in detail.")
parser.add_argument("--chat", action="store_true",
                    help="Interactive loop: keep asking about the same image")
parser.add_argument("--token-budget", type=int, default=512,
                    help="Must match training (512) — see docs/TRAINING_GUIDE.md")
parser.add_argument("--image-size", type=int, default=384)
parser.add_argument("--hidden-dim", type=int, default=2048,
                    help="Projector MLP hidden dim — must match the trained checkpoint")
parser.add_argument("--max-new-tokens", type=int, default=256)
parser.add_argument("--temperature", type=float, default=0.2)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if device.type == "cuda" else torch.float32

# -- load vision encoder -------------------------------------------------------
print(f"[1/3] Vision encoder: {args.vision_ckpt}")
vision = AutoModel.from_pretrained(args.vision_ckpt, dtype=DTYPE).to(device).eval()
# Non-persistent buffers lose fp32 precision under a bf16 load — reinitialize.
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
print(f"  encoder_dim = {encoder_dim}")

# -- load LLM (+ LoRA) ---------------------------------------------------------
print(f"[2/3] LLM: {args.llm_name}")
tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
llm = AutoModelForCausalLM.from_pretrained(
    args.llm_name, torch_dtype=DTYPE, device_map="auto")
llm_dim = llm.config.hidden_size
print(f"  llm_dim = {llm_dim}")

if args.lora_ckpt:
    if os.path.isdir(args.lora_ckpt):
        from peft import PeftModel
        llm = PeftModel.from_pretrained(llm, args.lora_ckpt)
        print(f"  LoRA adapter loaded: {args.lora_ckpt}")
    else:
        print(f"  WARNING: LoRA dir not found ({args.lora_ckpt}) — using base LLM")
llm.eval()

# -- load projector ------------------------------------------------------------
print(f"[3/3] Projector: {args.projector_ckpt}")
projector = LevelAwareProjector(
    encoder_dim=encoder_dim, llm_dim=llm_dim, hidden_dim=args.hidden_dim,
).to(device).to(DTYPE)
if os.path.exists(args.projector_ckpt):
    state = torch.load(args.projector_ckpt, map_location=device)
    projector.load_state_dict(state)
    n = sum(p.numel() for p in projector.parameters())
    print(f"  {n:,} params loaded ({encoder_dim} -> {args.hidden_dim} -> {llm_dim})")
else:
    raise SystemExit(
        f"ERROR: projector checkpoint not found: {args.projector_ckpt}\n"
        "Train Stage 2a first (scripts/run_stage2a_caption.sh). Running with a "
        "random projector would produce meaningless output.")
projector.eval()

transform = T.Compose([
    T.Resize((args.image_size, args.image_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def encode_image(pil_img: Image.Image) -> torch.Tensor:
    """image -> projected visual tokens (1, N_vis, llm_dim)."""
    pv = transform(pil_img).unsqueeze(0).to(device, dtype=DTYPE)
    vis_out = vision(pv, token_budget=args.token_budget)
    return projector(vis_out)


@torch.no_grad()
def answer(pil_img: Image.Image, question: str) -> str:
    visual_embeds = encode_image(pil_img)

    # Build the text prompt with the LLM's OWN chat template — this is what the
    # model was trained on in Stage 2b, so it must match exactly.
    msgs = [{"role": "user", "content": question}]
    prompt_ids = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")

    embed_fn = llm.get_input_embeddings()
    ed = next(embed_fn.parameters()).device
    prompt_ids = prompt_ids.to(ed)
    text_embeds = embed_fn(prompt_ids).to(DTYPE)

    # Vision tokens come FIRST, matching training.
    inputs_embeds = torch.cat([visual_embeds.to(ed), text_embeds], dim=1)
    attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=ed)

    out = llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=max(args.temperature, 1e-5),
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
    )
    # With inputs_embeds, generate() returns ONLY the new tokens.
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


img = Image.open(args.image).convert("RGB")
print(f"\nImage: {args.image}  ({img.size[0]}x{img.size[1]})")

if args.chat:
    print("Interactive mode — Ctrl-C or empty line to quit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        print(f"Model: {answer(img, q)}\n")
else:
    print(f"Q: {args.question}")
    print(f"A: {answer(img, args.question)}")
