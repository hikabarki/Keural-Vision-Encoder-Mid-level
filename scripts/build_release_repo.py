"""
Build a clean release repository: encoder + VLM through the bilingual model.

The working repository contains the full research history — the OCR
specialization that traded calibration for MME, the DPO run that failed, the
report-generation and publishing utilities. That history is valuable internally
and is preserved in the original repo. This builds a separate, focused
repository containing only the pipeline that produced the recommended model and
the results that support it.

Excluded on purpose:
  * Stage 2c (OCR) and Stage 3 (DPO) code and results — experiments that did not
    produce the recommended checkpoint
  * report/card generation and Hugging Face publishing utilities — internal
    tooling, not part of the model
  * the model_selection figure — it exists to compare against the excluded runs

    python3 scripts/build_release_repo.py            # build
    python3 scripts/build_release_repo.py --dry-run  # list only
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
DST = SRC.parent / "keural-mid-vlm"

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--dest", default=str(DST))
args = ap.parse_args()
DST = Path(args.dest)

DIRS = ["architecture", "alignment", "tokenizer", "training", "data",
        "models", "configs", "assets"]

SCRIPTS = [
    # Stage 1
    "train_mid.py", "smoke_test_mid.py", "evaluate.py", "viz_saliency.py",
    # Stage 2 through bilingual
    "train_projector.py",
    "run_stage2a_caption.sh", "run_stage2b_instruct.sh", "run_stage2b_bilingual.sh",
    "prepare_llava_jsonl.py", "build_bilingual_mix_v2.py",
    # inference + evaluation
    "demo_vlm.py", "eval_vlm.py", "run_eval_2b.sh", "vlm_studio.py",
    # figures
    "make_stage2_figures.py", "make_stage2_result_figures.py",
]

DOCS = ["ARCHITECTURE.md", "PIPELINE_TECHNICAL_SPEC.md", "BENCHMARK_COMPARISON.md",
        "TRAINING_GUIDE.md", "DATASET_INFO.md", "DATASET_DOWNLOAD.md",
        "KEURAL_TINY_VS_MID.md", "HF_MODEL_CARD.md", "HF_VLM_MODEL_CARD.md"]

FIGURES = ["training_loss_curves.png", "benchmark_comparison_chart.png",
           "benchmark_radar_chart.png", "benchmark_recall_curve_chart.png",
           "saliency_evolution_across_training.png",
           "stage2a_val_perplexity.png", "stage2a_grad_norm.png",
           "stage2b_training.png", "stage2_val_perplexity.png",
           "stage2_vision_contribution.png"]

# Root-level modules matter: keural_mid_config.py is imported as a TOP-LEVEL
# module by architecture/, alignment/ and the training scripts, so omitting it
# breaks every import in the package. Caught by the smoke test, not by review.
ROOT_FILES = ["requirements.txt", "pyproject.toml", "LICENSE", ".gitignore"]
ROOT_FILES += [p.name for p in SRC.glob("*.py") if p.name != "setup.py"]

EXCLUDED = [
    "scripts/train_dpo_vlm.py", "scripts/prepare_dpo_data.py",
    "scripts/prepare_ocr_data.py", "scripts/prepare_ocr_data_v2.py",
    "scripts/run_stage2c_v2.sh", "scripts/gpu_queue.sh",
    "scripts/queue_followup.sh", "scripts/demo_video.py",
    "scripts/publish_to_hf.py", "scripts/update_*.py",
    "scripts/make_weekly_report.py", "scripts/make_model_comparison_figure.py",
    "docs/OCR_ROADMAP.md", "docs/figures/model_selection.png",
    "docs/reports/", "outputs/",
]

plan = []
for d in DIRS:
    if (SRC / d).is_dir():
        plan.append((SRC / d, DST / d, "dir"))
for s in SCRIPTS:
    if (SRC / "scripts" / s).exists():
        plan.append((SRC / "scripts" / s, DST / "scripts" / s, "file"))
for d in DOCS:
    if (SRC / "docs" / d).exists():
        plan.append((SRC / "docs" / d, DST / "docs" / d, "file"))
for f in FIGURES:
    if (SRC / "docs/figures" / f).exists():
        plan.append((SRC / "docs/figures" / f, DST / "docs/figures" / f, "file"))
for f in ROOT_FILES:
    if (SRC / f).exists():
        plan.append((SRC / f, DST / f, "file"))
# The release README lives in the source repo so a rebuild cannot lose it.
RELEASE_README = SRC / "docs/release/README.md"
if RELEASE_README.exists():
    plan.append((RELEASE_README, DST / "README.md", "file"))

if (SRC / "test_images/README.md").exists():
    plan.append((SRC / "test_images/README.md", DST / "test_images/README.md", "file"))

print(f"source : {SRC}")
print(f"dest   : {DST}\n")
print(f"INCLUDING {len(plan)} items:")
for s, d, kind in plan:
    print(f"  {kind:4s} {d.relative_to(DST)}")
print(f"\nEXCLUDING (kept in the research repo):")
for e in EXCLUDED:
    print(f"       {e}")

if args.dry_run:
    print("\n--dry-run: nothing written.")
    sys.exit(0)

if DST.exists():
    shutil.rmtree(DST)
DST.mkdir(parents=True)

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache",
                                "*.egg-info", ".ipynb_checkpoints")
for s, d, kind in plan:
    d.parent.mkdir(parents=True, exist_ok=True)
    if kind == "dir":
        shutil.copytree(s, d, ignore=IGNORE, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

# A release repo should not inherit rules for paths it does not contain.
(DST / ".gitignore").write_text(
    "__pycache__/\n*.py[cod]\n*.egg-info/\n.venv/\nvenv*/\n"
    "outputs/\n*.pt\n*.safetensors\n*.ckpt\n"
    ".DS_Store\n.ipynb_checkpoints/\n"
    "test_images/*.jpg\ntest_videos/*.mp4\n")

n = sum(1 for _ in DST.rglob("*") if _.is_file())
size = sum(f.stat().st_size for f in DST.rglob("*") if f.is_file()) / 2**20
print(f"\nbuilt {DST}")
print(f"  {n} files, {size:.1f} MB")

# Verify nothing excluded leaked in.
leaked = []
for pat in ("*dpo*", "*ocr*", "*2c*", "model_selection*", "gpu_queue*",
            "publish_to_hf*", "update_*report*", "*weekly*"):
    leaked += [str(p.relative_to(DST)) for p in DST.rglob(pat) if p.is_file()]
print(f"  excluded-content leak check: {len(leaked)} hits"
      + ("" if not leaked else f" -> {leaked}"))
