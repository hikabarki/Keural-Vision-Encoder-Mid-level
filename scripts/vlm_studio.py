"""
Keural Mid VLM Studio — a Streamlit GUI for judging VLM quality.

    streamlit run scripts/vlm_studio.py --server.port 8501 --server.address 0.0.0.0

Built around the failure modes this model actually has, not generic demo chrome:

  * Sighted vs BLIND side by side. The single most informative view. Dropping the
    visual tokens and re-asking shows what the answer owes to the image versus to
    Qwen's language priors. Stage 2b answered "WELCOME TO THE CITY OF NEW YORK"
    for a NO PARKING sign -- blind mode is how you catch that class of error in
    one glance instead of by reading captions carefully.

  * Checkpoint switching without reloading the 7B. The base LLM loads once; only
    the projector (18 MB) and LoRA adapter (20 MB) swap. Comparing checkpoint-6000
    against 8000 costs seconds, not two full model loads.

  * Prompt-length presets. Long-form answers hallucinate far more than short ones
    (VQAv2 strict 1.73% vs lenient 48.60%), so "one sentence" and "single word"
    are one click away.

Streamlit reruns the whole script on every interaction, so everything expensive
is behind @st.cache_resource.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

st.set_page_config(page_title="Keural Mid VLM Studio", page_icon="🔍", layout="wide")

DEFAULT_VISION = str(_ROOT / "outputs/run/final")
TEST_IMAGES = _ROOT / "test_images"


def find_checkpoints():
    """Every checkpoint dir with usable weights, newest stage first."""
    out = []
    for d in sorted((_ROOT / "outputs").glob("projector_*/checkpoint-*")):
        if (d / "projector.pt").exists():
            out.append(d)
    return sorted(out, key=lambda p: (p.parent.name, int(p.name.split("-")[-1])),
                  reverse=True)


def ckpt_config(ckpt: Path) -> dict:
    """token_budget / image_size must match training or output degrades with no
    error. Checkpoints written after 2026-07-31 carry config.json; older ones
    fall back to the values Stage 2 actually used."""
    p = ckpt / "config.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"token_budget": 512, "image_size": 512, "hidden_dim": 2048,
            "_fallback": True}


@st.cache_resource(show_spinner="Loading vision encoder + Qwen2.5-7B (once)…")
def load_base(vision_ckpt: str, llm_name: str, hidden_dim: int):
    import alignment.hf_wrapper  # registers the encoder with AutoModel  # noqa
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    from alignment.projectors import LevelAwareProjector

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if dev.type == "cuda" else torch.float32

    vision = AutoModel.from_pretrained(vision_ckpt, dtype=dt).to(dev).eval()
    # Non-persistent buffers lose fp32 precision under a bf16 load.
    try:
        vision.encoder.pool_metadata.data[:] = torch.tensor(
            [[0.5, 0.5, 1.0, 1.0]], device=vision.encoder.pool_metadata.device)
        for blk in vision.encoder.blocks:
            r = blk.attn.rope
            i = torch.arange(0, r.half_dim, 2, dtype=torch.float32)
            r.inv_freq.data = (1.0 / (r.base ** (i / r.half_dim))).to(r.inv_freq.device)
    except AttributeError:
        pass
    for p in vision.parameters():
        p.requires_grad_(False)

    tok = AutoTokenizer.from_pretrained(llm_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(llm_name, dtype=dt, device_map="auto")

    # Wrap with LoRA once using any checkpoint's adapter config; the weights are
    # swapped per checkpoint below, so which one seeds the wrapper is irrelevant.
    from peft import PeftModel
    seed = next((c for c in find_checkpoints() if (c / "lora_adapter").is_dir()), None)
    if seed is not None:
        llm = PeftModel.from_pretrained(llm, str(seed / "lora_adapter"))
    llm.eval()

    proj = LevelAwareProjector(
        encoder_dim=vision.encoder.embed_dim,
        llm_dim=llm.config.hidden_size,
        hidden_dim=hidden_dim).to(dev).to(dt)
    return {"vision": vision, "llm": llm, "tok": tok, "proj": proj,
            "dev": dev, "dt": dt, "llm_dim": llm.config.hidden_size}


def load_weights(B, ckpt: Path):
    """Swap projector + LoRA weights in place. Avoids reloading the 7B."""
    if st.session_state.get("_loaded") == str(ckpt):
        return
    B["proj"].load_state_dict(torch.load(ckpt / "projector.pt", map_location=B["dev"]))
    B["proj"].eval()

    adir = ckpt / "lora_adapter"
    afile = next((adir / n for n in ("adapter_model.safetensors", "adapter_model.bin")
                  if (adir / n).exists()), None)
    if afile is not None:
        if afile.suffix == ".safetensors":
            from safetensors.torch import load_file
            sd = load_file(str(afile))
        else:
            sd = torch.load(afile, map_location="cpu")
        # peft 0.19.1's set_peft_model_state_dict imports EmbeddingParallel from
        # transformers.integrations.tensor_parallel, which does not exist in
        # transformers 4.57.6. Remap the keys ourselves:
        # save_pretrained writes "...lora_A.weight"; the live model names it
        # "...lora_A.default.weight".
        remap = {}
        for k, v in sd.items():
            for tag in ("lora_A.", "lora_B.", "lora_embedding_A.", "lora_embedding_B."):
                if tag in k:
                    k = k.replace(tag, f"{tag}default.")
                    break
            remap[k] = v
        keys = set(B["llm"].state_dict().keys())
        matched = {k: v for k, v in remap.items() if k in keys}
        if len(matched) != len(remap):
            st.error(f"{len(remap)-len(matched)} of {len(remap)} LoRA tensors did not "
                     "match the model. Not loading a partial adapter.")
            return
        B["llm"].load_state_dict(matched, strict=False)
    st.session_state["_loaded"] = str(ckpt)


@torch.no_grad()
def answer(B, img: Image.Image, question: str, cfg: dict,
           blind: bool, max_new: int, temperature: float) -> str:
    tf = T.Compose([
        T.Resize((cfg["image_size"], cfg["image_size"])),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    emb = B["llm"].get_input_embeddings()
    ed = next(emb.parameters()).device

    if blind:
        vis = torch.zeros(1, 0, B["llm_dim"], device=ed, dtype=B["dt"])
    else:
        pv = tf(img.convert("RGB")).unsqueeze(0).to(B["dev"], dtype=B["dt"])
        vis = B["proj"](B["vision"](pv, token_budget=cfg["token_budget"])).to(ed)

    ids = B["tok"].apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt").to(ed)
    inp = torch.cat([vis, emb(ids).to(B["dt"])], dim=1)
    out = B["llm"].generate(
        inputs_embeds=inp,
        attention_mask=torch.ones(inp.shape[:2], dtype=torch.long, device=ed),
        max_new_tokens=max_new, do_sample=temperature > 0,
        temperature=max(temperature, 1e-5), top_p=0.9,
        pad_token_id=B["tok"].pad_token_id)
    return B["tok"].decode(out[0], skip_special_tokens=True).strip()


# ==============================================================================
st.title("🔍 Keural Mid — VLM Studio")

ckpts = find_checkpoints()
if not ckpts:
    st.error("No checkpoints found under outputs/projector_*/checkpoint-*")
    st.stop()

with st.sidebar:
    st.header("Model")
    sel = st.selectbox("Checkpoint", ckpts,
                       format_func=lambda p: f"{p.parent.name} / {p.name}")
    cfg = ckpt_config(sel)
    val = cfg.get("val") or {}
    st.caption(
        f"image {cfg['image_size']}px · budget {cfg['token_budget']} tokens"
        + (f" · val ppl " + ", ".join(f"{k} {v['ppl']:.2f}" for k, v in val.items())
           if val else ""))
    if cfg.get("_fallback"):
        st.warning("No config.json — assuming 512/512 (what Stage 2 trained at).")

    st.divider()
    st.header("Generation")
    preset = st.radio("Answer length", ["Detailed", "One sentence", "Single word"],
                      help="Long answers hallucinate far more than short ones — "
                           "VQAv2 scored 1.73% strict vs 48.60% lenient because the "
                           "model would not answer briefly.")
    max_new = {"Detailed": 256, "One sentence": 48, "Single word": 12}[preset]
    suffix = {"Detailed": "",
              "One sentence": " Answer in one short sentence.",
              "Single word": " Answer the question using a single word or phrase."}[preset]
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    st.divider()
    if torch.cuda.is_available():
        free, tot = torch.cuda.mem_get_info(0)
        st.caption(f"GPU0 free {free/2**30:.0f} / {tot/2**30:.0f} GiB")

tab_single, tab_batch = st.tabs(["Single image", f"Batch · test_images"])

with tab_single:
    c1, c2 = st.columns([1, 1.35])
    with c1:
        srcs = sorted(TEST_IMAGES.glob("*.jpg")) if TEST_IMAGES.is_dir() else []
        up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
        pick = st.selectbox("…or pick a test image", ["—"] + [p.name for p in srcs])
        img = None
        if up is not None:
            img = Image.open(up)
        elif pick != "—":
            img = Image.open(TEST_IMAGES / pick)
        if img is not None:
            st.image(img, use_container_width=True)

    with c2:
        q = st.text_area("Question", "Describe this image.", height=90)
        compare = st.checkbox(
            "Also run BLIND (no image)", value=True,
            help="Re-asks with the visual tokens removed. Whatever the blind "
                 "column still says came from language priors, not from your "
                 "encoder. The clearest hallucination check available.")
        go = st.button("Run", type="primary", use_container_width=True)

        if go:
            if img is None:
                st.warning("Choose an image first.")
            else:
                B = load_base(DEFAULT_VISION, "Qwen/Qwen2.5-7B-Instruct",
                              cfg.get("hidden_dim", 2048))
                load_weights(B, sel)
                prompt = q + suffix
                with st.spinner("Generating…"):
                    seen = answer(B, img, prompt, cfg, False, max_new, temperature)
                    blind = (answer(B, img, prompt, cfg, True, max_new, temperature)
                             if compare else None)
                if compare:
                    a, b = st.columns(2)
                    a.markdown("**👁 With image**")
                    a.info(seen)
                    b.markdown("**🚫 Blind (no image)**")
                    b.warning(blind)
                    st.caption("Content appearing in BOTH columns came from the "
                               "language model, not from your vision encoder.")
                else:
                    st.info(seen)

with tab_batch:
    st.caption("Runs every image in test_images/ through one model load — the same "
               "sweep used to compare checkpoints by hand.")
    bq = st.text_input("Question for all images", "Describe this image.")
    bblind = st.checkbox("Blind mode", value=False, key="bblind")
    if st.button("Run batch", type="primary"):
        srcs = sorted(TEST_IMAGES.glob("*.jpg"))
        if not srcs:
            st.warning(f"No .jpg files in {TEST_IMAGES}")
        else:
            B = load_base(DEFAULT_VISION, "Qwen/Qwen2.5-7B-Instruct",
                          cfg.get("hidden_dim", 2048))
            load_weights(B, sel)
            bar = st.progress(0.0)
            for i, p in enumerate(srcs):
                out = answer(B, Image.open(p), bq + suffix, cfg,
                             bblind, max_new, temperature)
                cc1, cc2 = st.columns([1, 3])
                cc1.image(str(p), caption=p.name, use_container_width=True)
                cc2.write(out)
                bar.progress((i + 1) / len(srcs))
            bar.empty()
