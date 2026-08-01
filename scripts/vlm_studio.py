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
from collections import Counter
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


def sample_video_frames(path: str, n: int):
    """Uniformly sample n frames. Uniform rather than keyframe-based on purpose:
    keyframe selection biases toward scene cuts and skips steady action."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return [], {}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if total <= 0:
        cap.release()
        return [], {}
    idx = sorted({round(i * (total - 1) / max(n - 1, 1)) for i in range(n)})
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames, {"total": total, "fps": fps,
                    "duration": total / fps if fps else 0}


@torch.no_grad()
def answer_video(B, frames, question: str, cfg: dict, blind: bool,
                 max_new: int, temperature: float, frame_budget: int,
                 markers: bool = True):
    """All frames in ONE batched encoder pass, tokens concatenated.

    There is no temporal signal anywhere in this stack -- the projector and LoRA
    saw only single images and the encoder's RoPE is spatial -- so this
    aggregates per-frame understanding rather than perceiving motion. The
    "Frame k:" markers are the cheapest available substitute for a temporal
    encoding: they give the LLM explicit ordering without any training.
    """
    tf = T.Compose([
        T.Resize((cfg["image_size"], cfg["image_size"])),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    emb = B["llm"].get_input_embeddings()
    ed = next(emb.parameters()).device

    def txt(s):
        ids = B["tok"](s, return_tensors="pt",
                       add_special_tokens=False).input_ids.to(ed)
        return emb(ids).to(B["dt"])

    if blind:
        vis, n_tok = torch.zeros(1, 0, B["llm_dim"], device=ed, dtype=B["dt"]), 0
    else:
        px = torch.stack([tf(im.convert("RGB")) for im in frames]).to(
            B["dev"], dtype=B["dt"])
        out = B["vision"](px, token_budget=frame_budget)   # (F, N, D)
        projected = B["proj"](out).to(ed)
        mask = out.attention_mask.to(ed).bool()
        pieces = []
        for f in range(projected.shape[0]):
            if markers:
                pieces.append(txt(f"\nFrame {f + 1}: "))
            pieces.append(projected[f][mask[f]].unsqueeze(0))   # drop padding
        vis = torch.cat(pieces, dim=1)
        n_tok = int(mask.sum().item())

    preamble = ("" if blind else
                f"The following are {len(frames)} frames sampled in order "
                "from a video.\n")
    ids = B["tok"].apply_chat_template(
        [{"role": "user", "content": preamble + question}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt").to(ed)
    inp = torch.cat([vis, emb(ids).to(B["dt"])], dim=1)
    out_ids = B["llm"].generate(
        inputs_embeds=inp,
        attention_mask=torch.ones(inp.shape[:2], dtype=torch.long, device=ed),
        max_new_tokens=max_new, do_sample=temperature > 0,
        temperature=max(temperature, 1e-5), top_p=0.9,
        pad_token_id=B["tok"].pad_token_id)
    return B["tok"].decode(out_ids[0], skip_special_tokens=True).strip(), n_tok


@torch.no_grad()
def visual_tokens(B, img: Image.Image, cfg: dict):
    """Project one image to LLM-space tokens. Cached per image in the chat tab so
    a follow-up question does not re-run the encoder."""
    tf = T.Compose([
        T.Resize((cfg["image_size"], cfg["image_size"])),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ed = next(B["llm"].get_input_embeddings().parameters()).device
    pv = tf(img.convert("RGB")).unsqueeze(0).to(B["dev"], dtype=B["dt"])
    return B["proj"](B["vision"](pv, token_budget=cfg["token_budget"])).to(ed)


@torch.no_grad()
def answer_chat(B, vis, history, cfg: dict, max_new: int, temperature: float) -> str:
    """Multi-turn answer.

    The visual tokens are prepended ONCE and the entire conversation is then
    rendered through the decoder's own chat template, so turn 3 can see turns 1
    and 2. This matches how Stage 2b was trained: LLaVA-Instruct conversations
    are multi-turn with assistant-only label masking, so the model has seen this
    exact layout.
    """
    emb = B["llm"].get_input_embeddings()
    ed = next(emb.parameters()).device
    ids = B["tok"].apply_chat_template(
        history, tokenize=True, add_generation_prompt=True,
        return_tensors="pt").to(ed)
    inp = torch.cat([vis.to(ed), emb(ids).to(B["dt"])], dim=1)
    out = B["llm"].generate(
        inputs_embeds=inp,
        attention_mask=torch.ones(inp.shape[:2], dtype=torch.long, device=ed),
        max_new_tokens=max_new, do_sample=temperature > 0,
        temperature=max(temperature, 1e-5), top_p=0.9,
        pad_token_id=B["tok"].pad_token_id)
    return B["tok"].decode(out[0], skip_special_tokens=True).strip()


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

(tab_chat, tab_single, tab_batch, tab_video,
 tab_compare, tab_look, tab_prompt, tab_results) = st.tabs(
    ["💬 Chat", "Single image", "Batch · test_images", "Video",
     "⚖️ Compare models", "👁 Where it looks", "🎛 Prompt lab", "📊 Results"])

with tab_chat:
    cc1, cc2 = st.columns([1, 2])
    with cc1:
        srcs = sorted(TEST_IMAGES.glob("*.jpg")) if TEST_IMAGES.is_dir() else []
        cup = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp"], key="cup")
        cpick = st.selectbox("…or a test image", ["—"] + [p.name for p in srcs],
                             key="cpick")
        chat_img, img_key = None, None
        if cup is not None:
            chat_img, img_key = Image.open(cup), f"upload:{cup.name}:{cup.size}"
        elif cpick != "—":
            chat_img, img_key = Image.open(TEST_IMAGES / cpick), f"test:{cpick}"
        if chat_img is not None:
            st.image(chat_img, use_container_width=True)
        if st.button("Clear conversation", use_container_width=True):
            st.session_state["chat_history"] = []
            st.session_state.pop("chat_vis", None)
            st.rerun()
        st.caption(f"{len(st.session_state.get('chat_history', []))} messages · "
                   f"{sel.parent.name}/{sel.name}")

    with cc2:
        # Changing the image (or the checkpoint) invalidates both the cached
        # visual tokens and the conversation -- a follow-up like "what colour is
        # it?" would otherwise silently refer to the previous picture.
        ctx_key = f"{img_key}|{sel}"
        if st.session_state.get("chat_ctx") != ctx_key:
            st.session_state["chat_ctx"] = ctx_key
            st.session_state["chat_history"] = []
            st.session_state.pop("chat_vis", None)

        history = st.session_state.setdefault("chat_history", [])
        for m in history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if chat_img is None:
            st.info("Choose an image on the left, then ask about it. "
                    "Follow-up questions keep the whole conversation in context.")
        else:
            prompt = st.chat_input("Ask about this image…")
            if prompt:
                history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                B = load_base(DEFAULT_VISION, "Qwen/Qwen2.5-7B-Instruct",
                              cfg.get("hidden_dim", 2048))
                load_weights(B, sel)
                if "chat_vis" not in st.session_state:
                    with st.spinner("Encoding image…"):
                        st.session_state["chat_vis"] = visual_tokens(B, chat_img, cfg)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        reply = answer_chat(B, st.session_state["chat_vis"],
                                            history, cfg, max_new, temperature)
                    st.markdown(reply)
                history.append({"role": "assistant", "content": reply})
                st.rerun()

        with st.expander("How multi-turn works here"):
            st.markdown(
                "The image is encoded **once** and its visual tokens are prepended to "
                "the whole conversation, which is then rendered through Qwen's own "
                "chat template. Turn 3 therefore sees turns 1 and 2 — this is the "
                "same layout Stage 2b trained on, since LLaVA-Instruct conversations "
                "are multi-turn with assistant-only label masking.\n\n"
                "Two caveats worth knowing while judging replies: long answers "
                "hallucinate substantially more than short ones, so the sidebar's "
                "**Answer length** control changes reliability, not just verbosity; "
                "and the model is sensitive to phrasing, so a question worded like "
                "the training data (\"Answer using a single word or phrase\") behaves "
                "differently from one that is not.")

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

TEST_VIDEOS = _ROOT / "test_videos"

with tab_video:
    st.caption(
        "Samples frames, encodes them in ONE batched pass, and concatenates the "
        "visual tokens. This aggregates per-frame understanding — it does not "
        "perceive motion. The projector and LoRA saw only single images and the "
        "encoder's positional encoding is spatial, so questions about order, "
        "direction or causality will be answered from language priors.")

    vc1, vc2 = st.columns([1, 1.35])
    with vc1:
        vids = sorted(TEST_VIDEOS.glob("*.mp4")) if TEST_VIDEOS.is_dir() else []
        vup = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "webm"])
        vpick = st.selectbox("…or pick a test video", ["—"] + [p.name for p in vids])
        vpath = None
        if vup is not None:
            tmp = _ROOT / ".upload_video"
            tmp.write_bytes(vup.getbuffer())
            vpath = str(tmp)
        elif vpick != "—":
            vpath = str(TEST_VIDEOS / vpick)
        if vpath:
            st.video(vpath)

    with vc2:
        nf = st.slider("Frames sampled", 2, 16, 8)
        # The single most consequential knob: at 128 tokens/frame a validated
        # test clip lost BOTH people in shot and misread rainbow headwear as
        # flags; at 384 it described them correctly.
        fb = st.select_slider("Vision tokens per frame",
                              options=[64, 128, 192, 256, 384, 512], value=384,
                              help="Total visual context = frames x this. Below "
                                   "~256 the primary subject starts disappearing "
                                   "— measured, not guessed.")
        st.caption(f"→ {nf} × {fb} = **{nf*fb} visual tokens**"
                   f"  (a single image uses {cfg['token_budget']})")
        vq = st.text_area("Question", "What is happening in this video?", height=80,
                          key="vq")
        vmark = st.checkbox("Interleave 'Frame k:' markers", value=True,
                            help="Cheapest substitute for a temporal encoding. "
                                 "Untick to see how much it contributes.")
        vblind = st.checkbox("Also run BLIND (no video)", value=True, key="vblind")
        vgo = st.button("Run", type="primary", use_container_width=True, key="vgo")

    if vgo:
        if not vpath:
            st.warning("Choose a video first.")
        else:
            frames, meta = sample_video_frames(vpath, nf)
            if not frames:
                st.error("Could not decode any frames from that file.")
            else:
                if meta:
                    st.caption(f"{len(frames)} frames sampled from {meta['total']} "
                               f"({meta['duration']:.1f}s @ {meta['fps']:.0f}fps)")
                # Show what the model actually saw. Inspecting the sampled frames
                # is how a bad crop gets caught before the output is misread.
                st.write("**Sampled frames**")
                for row in range(0, len(frames), 8):
                    for col, fr in zip(st.columns(min(8, len(frames) - row)),
                                       frames[row:row + 8]):
                        col.image(fr, use_container_width=True)
                B = load_base(DEFAULT_VISION, "Qwen/Qwen2.5-7B-Instruct",
                              cfg.get("hidden_dim", 2048))
                load_weights(B, sel)
                with st.spinner(f"Encoding {len(frames)} frames…"):
                    seen, ntok = answer_video(B, frames, vq, cfg, False, max_new,
                                              temperature, fb, vmark)
                    blind_a = (answer_video(B, frames, vq, cfg, True, max_new,
                                            temperature, fb, vmark)[0]
                               if vblind else None)
                st.caption(f"{ntok} real visual tokens after padding removal")
                if vblind:
                    a, b = st.columns(2)
                    a.markdown("**🎬 With video**")
                    a.info(seen)
                    b.markdown("**🚫 Blind (no video)**")
                    b.warning(blind_a)
                    st.caption("Content in BOTH columns came from the language "
                               "model, not from your encoder.")
                else:
                    st.info(seen)


# =============================================================================
# Compare models — the same question through several checkpoints
# =============================================================================
with tab_compare:
    st.caption(
        "Runs one image and one question through several checkpoints back to back. "
        "The base LLM stays resident and only the projector (18 MB) and LoRA "
        "adapter (20 MB) swap, so each extra model costs an inference, not a load.")
    kc1, kc2 = st.columns([1, 2])
    with kc1:
        srcs = sorted(TEST_IMAGES.glob("*.jpg")) if TEST_IMAGES.is_dir() else []
        mup = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp"], key="mup")
        mpick = st.selectbox("…or a test image", ["—"] + [p.name for p in srcs], key="mpick")
        m_img = (Image.open(mup) if mup is not None
                 else (Image.open(TEST_IMAGES / mpick) if mpick != "—" else None))
        if m_img is not None:
            st.image(m_img, use_container_width=True)
    with kc2:
        picks = st.multiselect(
            "Checkpoints to compare", ckpts,
            default=ckpts[:3],
            format_func=lambda p: f"{p.parent.name}/{p.name}")
        mq = st.text_area("Question", "Describe this image.", height=80, key="mq")
        mblind = st.checkbox("Add a blind row (no image)", value=True, key="mblind")
        if st.button("Compare", type="primary", use_container_width=True, key="mgo"):
            if m_img is None or not picks:
                st.warning("Pick an image and at least one checkpoint.")
            else:
                B = load_base(DEFAULT_VISION, "Qwen/Qwen2.5-7B-Instruct",
                              cfg.get("hidden_dim", 2048))
                bar = st.progress(0.0)
                rows = []
                for i, c in enumerate(picks):
                    ccfg = ckpt_config(c)
                    load_weights(B, c)
                    a = answer(B, m_img, mq + suffix, ccfg, False, max_new, temperature)
                    rows.append((f"{c.parent.name}/{c.name}", ccfg, a))
                    bar.progress((i + 1) / (len(picks) + (1 if mblind else 0)))
                blind_txt = None
                if mblind:
                    blind_txt = answer(B, m_img, mq + suffix, cfg, True, max_new, temperature)
                    bar.progress(1.0)
                bar.empty()
                for name, ccfg, a in rows:
                    v = ccfg.get("val") or {}
                    meta = " · ".join(f"{k} ppl {x['ppl']:.2f}" for k, x in v.items())
                    st.markdown(f"**{name}**" + (f"  ·  {meta}" if meta else ""))
                    st.info(a)
                if blind_txt is not None:
                    st.markdown("**🚫 Blind — no image (any model agreeing with this "
                                "row is reciting language priors)**")
                    st.warning(blind_txt)


# =============================================================================
# Where it looks — saliency map and ATB token placement
# =============================================================================
with tab_look:
    st.caption(
        "The saliency predictor decides WHERE the ATB tokenizer spends its budget. "
        "This view is specific to this architecture: a fixed-grid encoder has "
        "nothing equivalent to show, because its token positions never move.")
    lc1, lc2 = st.columns([1, 1.4])
    with lc1:
        srcs = sorted(TEST_IMAGES.glob("*.jpg")) if TEST_IMAGES.is_dir() else []
        lup = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp"], key="lup")
        lpick = st.selectbox("…or a test image", ["—"] + [p.name for p in srcs], key="lpick")
        l_img = (Image.open(lup) if lup is not None
                 else (Image.open(TEST_IMAGES / lpick) if lpick != "—" else None))
        lbudget = st.select_slider("Token budget", options=[128, 256, 512, 1024, 2048],
                                   value=cfg["token_budget"], key="lbud")
        lgo = st.button("Analyse", type="primary", use_container_width=True, key="lgo")
    with lc2:
        if lgo and l_img is not None:
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            B = load_base(DEFAULT_VISION, "Qwen/Qwen2.5-7B-Instruct",
                          cfg.get("hidden_dim", 2048))
            load_weights(B, sel)
            tf = T.Compose([T.Resize((cfg["image_size"], cfg["image_size"])), T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with torch.no_grad():
                pv = tf(l_img.convert("RGB")).unsqueeze(0).to(B["dev"], dtype=B["dt"])
                out, sal = B["vision"](pv, token_budget=int(lbudget),
                                       return_saliency_map=True)
            sal_np = sal[0, 0].float().cpu().numpy()
            base = np.asarray(l_img.convert("RGB").resize(
                (cfg["image_size"], cfg["image_size"])))

            fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
            axes[0].imshow(base); axes[0].set_title("input", fontsize=9)
            axes[1].imshow(sal_np, cmap="inferno")
            axes[1].set_title("saliency map", fontsize=9)
            axes[2].imshow(base)
            md = out.spatial_metadata[0].float().cpu().numpy()
            lv = out.level_ids[0].cpu().numpy()
            S = cfg["image_size"]
            COL = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}
            NAME = {0: "global", 1: "region", 2: "detail"}
            for level in (0, 1, 2):
                m = lv == level
                if m.any():
                    axes[2].scatter(md[m, 0] * S, md[m, 1] * S, s=9 if level == 2 else 26,
                                    c=COL[level], alpha=0.75, edgecolors="none",
                                    label=f"{NAME[level]} ({int(m.sum())})")
            axes[2].legend(fontsize=7, loc="lower right", framealpha=0.85)
            axes[2].set_title("token placement by level", fontsize=9)
            for a in axes:
                a.set_xticks([]); a.set_yticks([])
            fig.tight_layout()
            st.pyplot(fig)

            c = Counter(int(v) for v in lv)
            st.caption(
                f"{len(lv)} tokens at budget {lbudget} — "
                + ", ".join(f"{NAME[k]} {c.get(k,0)}" for k in (0, 1, 2))
                + f" · saliency range {sal_np.min():.3f}–{sal_np.max():.3f}, "
                f"mean {sal_np.mean():.3f}")
            st.caption(
                "A saliency map that is flat (min≈max) means the predictor has "
                "collapsed and tokens are effectively being placed on a grid — the "
                "anti-collapse term in the Stage 1 loss exists to prevent exactly that.")
        elif lgo:
            st.warning("Choose an image first.")


# =============================================================================
# Prompt lab — the same question, worded several ways
# =============================================================================
with tab_prompt:
    st.caption(
        "This model is measurably sensitive to phrasing. On POPE the instruction "
        "“Answer yes or no only” collapsed the yes-ratio to 0.050, while the "
        "phrasing it was trained on gave 0.492 against a gold ratio of 0.500 — same "
        "model, same images, same questions. This tab makes that visible on your own "
        "inputs before it surprises you in a demo.")
    pc1, pc2 = st.columns([1, 1.6])
    with pc1:
        srcs = sorted(TEST_IMAGES.glob("*.jpg")) if TEST_IMAGES.is_dir() else []
        pup = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp"], key="pup")
        ppick = st.selectbox("…or a test image", ["—"] + [p.name for p in srcs], key="ppick")
        p_img = (Image.open(pup) if pup is not None
                 else (Image.open(TEST_IMAGES / ppick) if ppick != "—" else None))
        if p_img is not None:
            st.image(p_img, use_container_width=True)
    with pc2:
        core = st.text_input("Core question", "Is there a person in this image?", key="pcore")
        DEFAULT_VARIANTS = "\n".join([
            "",
            " Answer the question using a single word or phrase.",
            " Answer yes or no only.",
            " Answer in one short sentence.",
            " Please describe what you see, then answer.",
        ])
        variants = st.text_area(
            "Instruction suffixes (one per line, blank line = no instruction)",
            DEFAULT_VARIANTS, height=140, key="pvars")
        if st.button("Run all phrasings", type="primary", use_container_width=True, key="pgo"):
            if p_img is None:
                st.warning("Choose an image first.")
            else:
                B = load_base(DEFAULT_VISION, "Qwen/Qwen2.5-7B-Instruct",
                              cfg.get("hidden_dim", 2048))
                load_weights(B, sel)
                lines = variants.split("\n")
                bar = st.progress(0.0)
                for i, sfx in enumerate(lines):
                    a = answer(B, p_img, core + sfx, cfg, False, max_new, temperature)
                    label = sfx.strip() or "(no instruction)"
                    st.markdown(f"**{label}**")
                    st.info(a)
                    bar.progress((i + 1) / max(len(lines), 1))
                bar.empty()
                st.caption("Large differences between rows mean the model is keying on "
                           "instruction format, not only on the image.")


# =============================================================================
# Results — every evaluation on disk, in one table
# =============================================================================
with tab_results:
    st.caption("Reads the evaluation JSON written by scripts/eval_vlm.py. Nothing here "
               "is typed by hand, so it cannot drift from the measurements.")
    import glob as _glob
    files = sorted(_glob.glob(str(_ROOT / "outputs/eval_2b/*.json"))
                   + _glob.glob(str(_ROOT / "outputs/queue/*.json")))
    rows = []
    for f in files:
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        if not isinstance(d, dict) or "benchmark" not in d:
            continue
        r = {"file": Path(f).name, "benchmark": d.get("benchmark"),
             "blind": bool(d.get("blind_baseline")), "n": d.get("n")}
        if "categories" in d:
            r["n"] = sum(v.get("n", 0) for v in d["categories"].values())
            r["score"] = round(sum(v.get("score", 0) for v in d["categories"].values()), 1)
            r["categories"] = len(d["categories"])
        for k in ("accuracy", "lenient_accuracy", "f1", "yes_ratio"):
            if k in d:
                r[k] = round(d[k] * (1 if k == "yes_ratio" else 100), 3)
        if "languages" in d:
            for lang, v in d["languages"].items():
                r[f"ppl_{lang}"] = round(v["perplexity"], 3)
        rows.append(r)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(f"{len(rows)} result files. A benchmark row with blind = True is the "
                   "no-vision baseline for the row above it; the DIFFERENCE between the "
                   "two is what the vision encoder contributes.")
    else:
        st.info("No evaluation results found yet. Run scripts/run_eval_2b.sh.")
