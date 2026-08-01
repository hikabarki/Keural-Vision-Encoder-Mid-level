"""
Publish the Keural Mid artifacts to Hugging Face.

Two repositories, deliberately:

  mkd-hika/keural-vision-encoder-mid   the foundation model  (exists)
  mkd-hika/keural-mid-vlm-bilingual    the VLM bridge        (created here)

They are separate because they are different artifacts with different lifecycles,
dependencies and audiences. The encoder is reusable for retrieval, detection or
any downstream head and depends on nothing else. The VLM is one application of it
and requires Qwen2.5-7B at load time. Someone who wants the 38 MB bridge should
not have to download 700 MB of encoder, and someone who wants the encoder should
not inherit a 7B dependency. A repository also carries exactly one pipeline_tag,
so bundling would force a choice between being discoverable as
image-feature-extraction or as image-text-to-text -- it cannot be both.

    python3 scripts/publish_to_hf.py --dry-run    # inspect, upload nothing
    python3 scripts/publish_to_hf.py              # upload
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "outputs/projector_2b_bilingual/checkpoint-12000"
CARD = ROOT / "docs/HF_VLM_MODEL_CARD.md"
FIGS = ROOT / "docs/figures"

ENCODER_REPO = "mkd-hika/keural-vision-encoder-mid"
VLM_REPO = "mkd-hika/keural-mid-vlm-bilingual"

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--vlm-repo", default=VLM_REPO)
ap.add_argument("--skip-cleanup", action="store_true",
                help="Do not delete optimizer state from the encoder repo")
args = ap.parse_args()

from huggingface_hub import HfApi, whoami

api = HfApi()
try:
    me = whoami()
    user = me.get("name")
except Exception as e:
    print(f"NOT AUTHENTICATED ({type(e).__name__}). Run:  huggingface-cli login")
    sys.exit(1)

owner = args.vlm_repo.split("/")[0]
print(f"authenticated as : {user}")
print(f"target owner     : {owner}")
if user != owner:
    orgs = [o.get("name") for o in me.get("orgs", [])]
    if owner not in orgs:
        print(f"\nWARNING: you are '{user}' but the target namespace is '{owner}'.")
        print("Pushing will fail unless that account has write access. Re-auth with:")
        print("  huggingface-cli logout && huggingface-cli login")
        if not args.dry_run:
            sys.exit(1)

# -- what goes in the VLM repo -------------------------------------------------
FILES = [
    (CKPT / "projector.pt", "projector.pt"),
    (CKPT / "config.json", "config.json"),
    (CKPT / "lora_adapter/adapter_model.safetensors", "adapter_model.safetensors"),
    (CKPT / "lora_adapter/adapter_config.json", "adapter_config.json"),
    (CARD, "README.md"),
    (FIGS / "stage2_vision_contribution.png", "stage2_vision_contribution.png"),
    (FIGS / "stage2_val_perplexity.png", "stage2_val_perplexity.png"),
    (FIGS / "model_selection.png", "model_selection.png"),
]
# NOTE: training_state.pt is deliberately absent. It holds Adam moments, is 73 MB,
# and is useless to anyone loading the model for inference.

print(f"\n=== {args.vlm_repo} ===")
total = 0
missing = []
for src, dst in FILES:
    if src.exists():
        mb = src.stat().st_size / 2**20
        total += mb
        print(f"  {dst:38s} {mb:>8.1f} MB")
    else:
        missing.append(str(src))
        print(f"  {dst:38s}    MISSING  <- {src}")
print(f"  {'total':38s} {total:>8.1f} MB")
if missing:
    print(f"\nERROR: {len(missing)} file(s) missing; not uploading.")
    sys.exit(1)

# -- optimizer state sitting in the encoder repo -------------------------------
try:
    info = api.model_info(ENCODER_REPO, files_metadata=True)
    dead = [f for f in info.siblings if "training_state" in f.rfilename]
    dead_gb = sum(f.size or 0 for f in dead) / 2**30
    all_gb = sum(f.size or 0 for f in info.siblings) / 2**30
    if dead:
        print(f"\n=== {ENCODER_REPO} cleanup ===")
        print(f"  {len(dead)} optimizer-state files = {dead_gb:.2f} GB of {all_gb:.2f} GB "
              f"({dead_gb/all_gb*100:.0f}% of the repo)")
        for f in dead:
            print(f"    {f.rfilename}")
        print("  These hold Adam moments for resuming training. Nobody downloading")
        print("  the model needs them, and they dominate the repo size.")
except Exception as e:
    dead = []
    print(f"\n(could not inspect {ENCODER_REPO}: {type(e).__name__})")

if args.dry_run:
    print("\n--dry-run: nothing uploaded.")
    sys.exit(0)

# -- upload --------------------------------------------------------------------
print(f"\ncreating {args.vlm_repo} …")
api.create_repo(args.vlm_repo, repo_type="model", private=True, exist_ok=True)
for src, dst in FILES:
    print(f"  uploading {dst} …")
    api.upload_file(path_or_fileobj=str(src), path_in_repo=dst,
                    repo_id=args.vlm_repo, repo_type="model")
print(f"done -> https://huggingface.co/{args.vlm_repo}")

if dead and not args.skip_cleanup:
    print(f"\ndeleting optimizer state from {ENCODER_REPO} …")
    for f in dead:
        try:
            api.delete_file(path_in_repo=f.rfilename, repo_id=ENCODER_REPO,
                            repo_type="model")
            print(f"  removed {f.rfilename}")
        except Exception as e:
            print(f"  FAILED {f.rfilename}: {type(e).__name__}")
