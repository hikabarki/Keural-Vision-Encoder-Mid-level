"""
Keural Mid — Stage 2 VLM data streams.

Two streaming generators for projector/VLM training, plus batch builders that
handle chat-template formatting and correct label masking:

  * caption_tar_stream  — Stage 2a: WebDataset tar shards (.jpg + .txt).
                          Label = the caption (train on caption generation).
  * llava_json_stream   — Stage 2b: LLaVA-Instruct-style JSONL (image + a
                          multi-turn conversation). Label = ASSISTANT turns
                          only; image tokens and user turns are masked (-100).

Both yield (PIL.Image, target_spec) where target_spec is either a caption
string (2a) or a list of {role, content} messages (2b). The batch builders
turn a pulled list into tokenized tensors ready to splice with visual embeds.

Label-masking is the part naive VLM training gets wrong: if user turns aren't
masked, the model trains to predict the *question*, not the answer. This module
masks everything except assistant content.
"""

from __future__ import annotations

import glob
import io
import json
import os
import random
import tarfile
from pathlib import Path

import torch
from PIL import Image

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Stage 2a — caption tar shards
# ---------------------------------------------------------------------------

def caption_tar_stream(tar_dirs):
    """Infinite shuffled generator of (PIL.Image, caption_str) from tar shards."""
    dirs = [tar_dirs] if isinstance(tar_dirs, str) else list(tar_dirs)
    tar_files = []
    for d in dirs:
        found = sorted(glob.glob(os.path.join(d, "*.tar")))
        tar_files.extend(found)
        print(f"[data] caption: {len(found)} tar shards in {d}")
    if not tar_files:
        raise RuntimeError(f"No .tar files found in {dirs}")
    while True:
        random.shuffle(tar_files)
        for tar_path in tar_files:
            try:
                with tarfile.open(tar_path) as tar:
                    members = {m.name: m for m in tar.getmembers()}
                    stems = [n[:-4] for n in members if n.endswith(".jpg")]
                    random.shuffle(stems)
                    for stem in stems:
                        jk, tk = stem + ".jpg", stem + ".txt"
                        if jk not in members or tk not in members:
                            continue
                        try:
                            img = Image.open(io.BytesIO(
                                tar.extractfile(members[jk]).read())).convert("RGB")
                            cap = tar.extractfile(members[tk]).read().decode(
                                "utf-8", errors="replace").strip()
                            if cap:
                                yield img, cap
                        except Exception:
                            continue
            except Exception as e:
                print(f"[data] skip {os.path.basename(tar_path)}: {e}")
                continue


# ---------------------------------------------------------------------------
# Stage 2b — LLaVA-Instruct JSONL
# ---------------------------------------------------------------------------

def _normalize_conversation(record):
    """Accept both LLaVA's native {'conversations':[{'from','value'}]} schema
    and the already-normalized {'messages':[{'role','content'}]} schema.
    Returns a list of {'role','content'} with roles in {'user','assistant'}.
    Strips the LLaVA <image> placeholder token from text (the image is spliced
    in as visual embeddings upstream, not as a text token)."""
    msgs = []
    if "messages" in record:
        for m in record["messages"]:
            role = m.get("role")
            content = (m.get("content") or "").replace("<image>", "").strip()
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})
    elif "conversations" in record:
        role_map = {"human": "user", "gpt": "assistant"}
        for m in record["conversations"]:
            role = role_map.get(m.get("from"))
            content = (m.get("value") or "").replace("<image>", "").strip()
            if role and content:
                msgs.append({"role": role, "content": content})
    return msgs


def _resolve_image_path(record, image_root):
    """LLaVA records carry an 'image' field relative to an image root."""
    img_rel = record.get("image") or record.get("image_path")
    if img_rel is None:
        return None
    p = Path(image_root) / img_rel
    return p if p.exists() else None


def llava_json_stream(jsonl_path, image_root, skip_missing=True):
    """Infinite shuffled generator of (PIL.Image, messages) from a LLaVA JSONL.

    Records missing an image (or whose image file is absent on disk) are
    skipped when skip_missing=True — logged once as a running count so silent
    data loss is visible.
    """
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"[data] llava: {len(records)} records in {jsonl_path}")

    missing = 0
    while True:
        random.shuffle(records)
        for rec in records:
            msgs = _normalize_conversation(rec)
            if not msgs or msgs[-1]["role"] != "assistant":
                continue  # need at least one assistant turn to train on
            img_path = _resolve_image_path(rec, image_root)
            if img_path is None:
                missing += 1
                if skip_missing:
                    if missing % 500 == 1:
                        print(f"[data] WARNING: {missing} records skipped (image missing)")
                    continue
                raise FileNotFoundError(f"Image not found for record: {rec.get('image')}")
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            yield img, msgs


def pull_batch(stream, n):
    """Pull n (image, target) pairs from a stream generator."""
    imgs, targets = [], []
    for img, tgt in stream:
        imgs.append(img)
        targets.append(tgt)
        if len(imgs) >= n:
            break
    return imgs, targets


# ---------------------------------------------------------------------------
# Batch builders — tokenize + build labels
# ---------------------------------------------------------------------------

def build_caption_batch(captions, tokenizer, max_length, device):
    """Stage 2a: label = full caption (with 'Caption: ' prefix). Padding masked."""
    enc = tokenizer(["Caption: " + c for c in captions], padding=True,
                    truncation=True, max_length=max_length, return_tensors="pt",
                    add_special_tokens=True)
    input_ids = enc.input_ids.to(device)
    attn = enc.attention_mask.to(device)
    labels = input_ids.clone()
    labels[attn == 0] = IGNORE_INDEX
    return input_ids, attn, labels


def build_llava_batch(conversations, tokenizer, max_length, device):
    """Stage 2b: apply the chat template, then mask everything except assistant
    tokens. Returns (input_ids, attention_mask, labels), right-padded.

    Masking strategy: for each sample, tokenize the conversation incrementally
    turn-by-turn with the chat template so we know the exact token span of each
    assistant response, and set labels only on those spans.
    """
    all_ids, all_labels = [], []
    for msgs in conversations:
        ids, labels = [], []
        # Rebuild prefix-by-prefix so each assistant span is locatable.
        prev_len = 0
        for i in range(len(msgs)):
            partial = msgs[: i + 1]
            add_gen = msgs[i]["role"] == "user"  # add gen prompt after a user turn
            toks = tokenizer.apply_chat_template(
                partial, tokenize=True, add_generation_prompt=add_gen)
            new = toks[prev_len:]
            ids.extend(new)
            if msgs[i]["role"] == "assistant":
                labels.extend(new)            # train on assistant tokens
            else:
                labels.extend([IGNORE_INDEX] * len(new))  # mask user/system
            prev_len = len(toks)
        # truncate
        ids = ids[:max_length]
        labels = labels[:max_length]
        all_ids.append(ids)
        all_labels.append(labels)

    # right-pad to the batch max
    maxlen = max(len(x) for x in all_ids)
    pad_id = tokenizer.pad_token_id
    input_ids, attn, labels = [], [], []
    for ids, labs in zip(all_ids, all_labels):
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attn.append([1] * len(ids) + [0] * pad)
        labels.append(labs + [IGNORE_INDEX] * pad)
    return (torch.tensor(input_ids, device=device),
            torch.tensor(attn, device=device),
            torch.tensor(labels, device=device))
