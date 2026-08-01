"""
Pass 2 of the technical report rewrite.

Promotes Stage 2 out of "Future Roadmap" into a completed-results section with
measured numbers and figures, corrects the remaining audited claims, and expands
the reference list to cover every work the document now cites.

Order matters: all in-place text edits happen first (indices stable), then
section renumbering, then insertions from the BOTTOM UP so earlier indices are
never invalidated.
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "docs/reports/Keural_Mid_Technical_Documentation_Report_v2.docx"
FIGS = ROOT / "docs/figures"
WIDTH = Inches(6.0)

doc = Document(DST)
P = doc.paragraphs


def settext(idx, text):
    p = P[idx]
    if p.runs:
        p.runs[0].text = text
        for r in p.runs[1:]:
            r.text = ""
    else:
        p.add_run(text)


# =============================================================================
# A. In-place corrections to the roadmap  (C104 critical, C105-C107, C114-C115, C125)
# =============================================================================
# C104: the COYO corpus was deleted from disk; the index was never confirmed built.
settext(125,
    "Stage 2 (multimodal alignment) has now been completed and is documented in "
    "Section 5. The stages below are the remaining strategic path from an aligned "
    "vision-language model to a preference-optimized one. A separate, previously "
    "scoped option — continued pre-training of the vision encoder itself on the "
    "reserved COYO-700M Filtered subset — is no longer immediately available: those "
    "shards (~32M pairs, ~2.0 TB) were downloaded during Stage 1 data collection but "
    "were never confirmed to have been indexed into a training TSV, and the entire "
    "training corpus was deleted from disk after the Stage 1 run completed. Reviving "
    "that option would require re-downloading and re-indexing the data.")

# C114/C115: the trained mixture is 603,255 records at 60/40, not 518,665 at 70/30.
settext(130,
    "Bilingual scope (English and Korean). Because the Korean instruction set "
    "references the same COCO images as the English one, bilingual training required "
    "no additional image data and no code changes — the tokenizer and label-masking "
    "logic are language-agnostic. The corpus actually trained on comprises 603,255 "
    "merged and shuffled records: 241,302 English and 361,953 Korean, a 40/60 split. "
    "An earlier 518,665-record mixture (156,712 English, 361,953 Korean, a 30/70 "
    "split) was rebuilt before the run by adding 84,590 English short-answer records "
    "drawn from the LLaVA-1.5 mixture on COCO images. That rebalancing was corrective "
    "rather than cosmetic: it is documented in Section 5.4.")

# C125: Stage 2 checkpoints now exist, and the port is more than a straight port.
settext(135,
    "Primary dependency: a multimodal preference dataset. One is publicly available — "
    "openbmb/RLHF-V-Dataset provides human-annotated chosen/rejected response pairs "
    "targeting hallucination reduction, and openbmb/RLAIF-V-Dataset offers a larger "
    "machine-generated alternative. No bespoke annotation effort is therefore required "
    "to begin Stage 3. The Stage 2 dependency is also cleared: usable reference "
    "checkpoints now exist. The remaining work is larger than a straight port of the "
    "proof-of-concept script, which performs text-only DPO: the loss must be computed "
    "over sequences that begin with projected visual tokens, and the reference policy "
    "is obtained by disabling the LoRA adapter rather than by holding a second copy of "
    "the 7B decoder in memory.")

# =============================================================================
# B. Renumber sections 5/6/7 -> 6/7/8 to make room for the new Section 5
# =============================================================================
RENUM = {
    104: "6. Mathematical Foundations",
    106: "6.1 Stage 1 — Contrastive & Auxiliary Losses (summary)",
    108: "6.2 Reward Modeling (Bradley-Terry)",
    112: "6.3 Direct Preference Optimization (DPO)",
    116: "6.4 Reinforcement Learning from Human Feedback (RLHF)",
    124: "7. Remaining Roadmap & Next Steps",
    126: "7.1 Stage 2 — Vision-Language Alignment (completed; see Section 5)",
    132: "7.2 Stage 3: Direct Preference Optimization (DPO)",
    136: "7.3 Stage 4: Reinforcement Learning from Human Feedback (RLHF)",
    141: "8. References",
}
for i, t in RENUM.items():
    settext(i, t)

# Cross-references inside the maths section point at "Section 5.x"; bump to 6.x.
for i, p in enumerate(P):
    if "Section 5." in p.text and p.style.name == "Normal":
        settext(i, p.text.replace("Section 5.2", "Section 6.2")
                          .replace("Section 5.3", "Section 6.3")
                          .replace("Section 5.4", "Section 6.4")
                          .replace("Section 5.1", "Section 6.1"))

# =============================================================================
# C. Build the new Section 5 and insert it before old Section 5 (index 104)
# =============================================================================
anchor = P[104]._p          # "6. Mathematical Foundations"
_tail = []                  # built in order, then inserted as a block


def add(text, style="Normal"):
    q = doc.add_paragraph(text, style=style)
    _tail.append(q._p)
    return q


def add_fig(name, caption):
    path = FIGS / name
    if not path.exists():
        return
    q = doc.add_paragraph()
    q.alignment = WD_ALIGN_PARAGRAPH.CENTER
    q.add_run().add_picture(str(path), width=WIDTH)
    _tail.append(q._p)
    c = doc.add_paragraph(caption, style="Normal")
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in c.runs:
        r.font.size = Pt(9)
        r.italic = True
    _tail.append(c._p)


def add_table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    for c, h in zip(t.rows[0].cells, headers):
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for c, v in zip(cells, row):
            c.text = ""
            r = c.paragraphs[0].add_run(str(v))
            r.font.size = Pt(9)
    _tail.append(t._element)
    add("")


add("5. Stage 2: Vision-Language Alignment (Completed)", "Heading 1")
add("Stage 2 attaches the frozen Keural Mid encoder to a language-model decoder, "
    "producing an instruction-following vision-language model. The encoder is frozen "
    "throughout — it is never fine-tuned at any learning rate — so every Stage 2 "
    "result below is attributable to the Stage 1 representation plus a trained "
    "projector and decoder adapters.")

add("5.1 Configuration", "Heading 2")
add("The decoder is Qwen2.5-7B-Instruct (hidden dimension 3,584), loaded in bfloat16 "
    "and frozen except for LoRA adapters (rank 16, alpha 32, applied to q_proj and "
    "v_proj). The bridge is a LevelAwareProjector: a two-layer MLP (768 → 2,048 → "
    "3,584) plus a learned additive bias per semantic level, indexed by the level_ids "
    "the encoder already emits. That per-level bias is the mechanism by which HCT "
    "reaches the decoder — it gives the language model an explicit global/region/"
    "detail signal that a fixed-grid encoder cannot supply. Trainable parameters "
    "total 13,979,136: 8,932,864 in the projector and 5,046,272 in the LoRA adapters, "
    "against 7.62 billion frozen decoder parameters and 183.6 million frozen encoder "
    "parameters.")
add("Stage 2 runs at 512×512 input while holding the token budget at 512. This is "
    "possible without retraining positional parameters because the continuous 2D-RoPE "
    "encodes position from real normalized coordinates rather than a grid index. All "
    "runs used bfloat16 on 2× NVIDIA H200 under DistributedDataParallel at an "
    "effective batch of 64, with seed 42 fixed for reproducibility.")

add("5.2 Training phases", "Heading 2")
add("Training followed the standard two-phase LLaVA-style recipe, extended with a "
    "bilingual phase. Each phase warm-starts the projector and LoRA weights from the "
    "previous checkpoint with a fresh optimizer and learning-rate schedule.")
add_table(
    ["Phase", "Data", "Records", "Steps", "LR", "Held-out perplexity"],
    [["2a — caption alignment", "LLaVA-Pretrain", "557,128", "5,000", "5e-4", "31.81 → 17.30 (EN)"],
     ["2b — instruction tuning", "LLaVA-Instruct-150K", "156,712", "8,000", "2e-5", "3.22 → 2.90 (EN)"],
     ["2b — bilingual EN+KO", "Merged EN/KO mixture", "603,255", "12,000", "2e-5", "2.90 → 2.14 (KO)"]])
add("Perplexity values are comparable only within a phase. Phase 2a is measured on "
    "caption data; phase 2b on instruction data with assistant-only label masking. "
    "The two are different quantities and must not be compared across rows.")
add_fig("stage2_val_perplexity.png",
        "Figure 7. Stage 2b held-out perplexity for the English-only run (8,000 steps) "
        "and the Korean split of the bilingual run (12,000 steps). Each curve is "
        "comparable only with itself — the runs use different data and different "
        "label masking.")

add("5.3 Training stability", "Heading 2")
add("Stage 2a initially failed to converge reliably. An early run suffered a gradient "
    "instability between steps 450 and 650 — raw gradient norms reached roughly 40× "
    "the clipping threshold — and held-out perplexity spiked from 25.74 to 124.50, "
    "never recovering past 53. The cause was not the learning rate alone: the trainer "
    "set no random seed, so projector and LoRA weights initialized differently on "
    "every launch. Two runs on identical data and identical hyperparameters reached "
    "perplexity 23.90 and 124.50 respectively, which made success a matter of chance "
    "and failures impossible to reproduce. Fixing the seed at 42, halving the peak "
    "learning rate from 1e-3 to 5e-4, and lengthening warmup from 200 to 500 steps "
    "removed the instability entirely and produced the best result for the stage.")
add_fig("stage2a_val_perplexity.png",
        "Figure 8. Stage 2a held-out perplexity. The stabilized run (seed 42, peak LR "
        "5e-4, warmup 500) reaches 17.30, against 53.0 for the unseeded run at LR 1e-3 "
        "whose gradients diverged at steps 450–650. The dashed line marks the previous "
        "best result of 23.90.")
add_fig("stage2a_grad_norm.png",
        "Figure 9. Gradient norm before clipping, log scale. The unstable run exceeds "
        "the clipping threshold throughout steps 450–650; the stabilized run remains "
        "below it for the entire run after step 60.")

add("5.4 Evaluation", "Heading 2")
add("A vision-language model can score respectably on visual benchmarks while "
    "ignoring the image entirely, because questions such as “what colour is the "
    "banana?” are substantially answerable from language priors. Every benchmark below "
    "is therefore reported twice: once normally, and once with the visual tokens "
    "removed from the input while everything else is held constant. The difference "
    "between the two is the measured contribution of the vision pipeline, and it is "
    "the only figure here that a language model cannot produce on its own.")
add_table(
    ["Benchmark", "Metric", "With image", "Blind", "Contribution"],
    [["POPE (n = 500)", "Accuracy", "68.00%", "43.40%", "+24.60 points"],
     ["POPE (n = 500)", "F1", "67.74%", "21.17%", "+46.57 points"],
     ["POPE (n = 500)", "Yes-ratio", "0.492", "0.218", "gold ratio 0.500"],
     ["VQAv2 (n = 1,000)", "Strict exact match", "53.87%", "42.10%", "+11.77 points"],
     ["VQAv2 (n = 1,000)", "Lenient accuracy", "54.40%", "42.60%", "+11.80 points"],
     ["MMBench (n = 500)", "Accuracy", "63.80%", "—", "—"],
     ["Held-out (n = 500)", "Perplexity, English", "2.99", "—", "—"],
     ["Held-out (n = 500)", "Perplexity, Korean", "2.15", "—", "—"]])
add_fig("stage2_vision_contribution.png",
        "Figure 10. Measured vision contribution on the bilingual checkpoint. The bold "
        "figure above each pair is the sighted-minus-blind gap — what the encoder "
        "contributes beyond the decoder's language priors.")
add("Two properties of these numbers are worth stating explicitly. The POPE yes-ratio "
    "of 0.492 against a gold ratio of 0.500 indicates the model is well calibrated on "
    "object-existence questions rather than defaulting to one answer; a model that "
    "replied “yes” indiscriminately would score near 50% accuracy while being useless, "
    "so accuracy alone is not interpretable without it. And the blind VQAv2 score of "
    "42.10% falls where language-prior performance is expected to fall, which is "
    "corroborating evidence that the measurement itself is sound.")
add("The bilingual mixture was rebuilt before the final run in response to a measured "
    "failure. The English-only Stage 2b model scored 1.73% strict against 48.60% "
    "lenient on VQAv2 — a 47-point divergence caused not by faulty perception but by "
    "answer format: asked “Is this rice noodle soup?” with gold answer “yes”, the model "
    "replied “Yes, this is a bowl of rice noodle soup”, which exact-match scoring "
    "credits as zero. The mixture contained only verbose conversational data and no "
    "short-answer supervision in English. Adding 84,590 English short-answer records "
    "raised strict accuracy from 1.73% to 53.87%, and strict and lenient scores now "
    "agree to within 0.53 points, which is the signature of the format problem having "
    "been resolved.")

add("5.5 Known limitations", "Heading 2")
add("Long-form generation fabricates detail. Asked for a detailed description, the "
    "model invents plausible objects that are not present — cutlery on a table, "
    "additional animals in a field. Short-answer responses are markedly more reliable, "
    "which is consistent with the benchmark results above and is why answer length is "
    "an explicit control in the evaluation tooling.")
add("Text recognition is partial. The model reads large, high-contrast signage "
    "correctly but no OCR-specific data was present in Stage 1 or Stage 2 training. A "
    "Stage 2c experiment adding scene-text supervision at higher input resolution was "
    "in progress at the time of writing and is not reported here.")
add("Output is sensitive to prompt phrasing. On POPE, the instruction “Answer yes or "
    "no only” — a phrasing absent from training — collapsed the yes-ratio to 0.050, "
    "while the trained phrasing “Answer the question using a single word or phrase” "
    "gave 0.492. All figures in Section 5.4 use the trained phrasing. This sensitivity "
    "is a genuine property of the model and a candidate target for preference "
    "optimization in Stage 3.")
add("Sample sizes are 500 to 1,000 items, which carries roughly ±4 percentage points "
    "of sampling error at n = 500. Reported differences smaller than that should not "
    "be treated as meaningful. An MME evaluation was attempted but the run covered "
    "only 3 of the benchmark's 14 categories and is therefore not reported; a "
    "full-coverage MME run is outstanding.")

for el in _tail:
    anchor.addprevious(el)

# =============================================================================
# D. References — add every work the document now cites  (audit: refs)
# =============================================================================
last = doc.paragraphs[-1]
NEW_REFS = [
    '[13] H. Liu, C. Li, Y. Li, and Y. J. Lee, "Improved Baselines with Visual Instruction Tuning," in Proc. CVPR, 2024.',
    '[14] Qwen Team, "Qwen2.5 Technical Report," arXiv:2412.15115, 2024.',
    '[15] E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," in Proc. ICLR, 2022.',
    '[16] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient Finetuning of Quantized LLMs," in Proc. NeurIPS, 2023.',
    '[17] Z. Chen et al., "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks," in Proc. CVPR, 2024.',
    '[18] Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao, and J.-R. Wen, "Evaluating Object Hallucination in Large Vision-Language Models," in Proc. EMNLP, 2023.',
    '[19] Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh, "Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering," in Proc. CVPR, 2017.',
    '[20] C. Fu et al., "MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models," arXiv:2306.13394, 2023.',
    '[21] Y. Liu et al., "MMBench: Is Your Multi-modal Model an All-around Player?," in Proc. ECCV, 2024.',
    '[22] A. Singh et al., "Towards VQA Models That Can Read," in Proc. CVPR, 2019.',
    '[23] E. Jang, S. Gu, and B. Poole, "Categorical Reparameterization with Gumbel-Softmax," in Proc. ICLR, 2017.',
    '[24] C. J. Maddison, A. Mnih, and Y. W. Teh, "The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables," in Proc. ICLR, 2017.',
    '[25] R. A. Bradley and M. E. Terry, "Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons," Biometrika, vol. 39, no. 3/4, pp. 324-345, 1952.',
    '[26] T. Yu et al., "RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback," in Proc. CVPR, 2024.',
    '[27] S. Gadre et al., "DataComp: In Search of the Next Generation of Multimodal Datasets," in Proc. NeurIPS, 2023.',
    '[28] M. Byeon, B. Park, H. Kim, S. Lee, W. Baek, and S. Kim, "COYO-700M: Image-Text Pair Dataset," 2022.',
]
for r in NEW_REFS:
    q = doc.add_paragraph(r, style="Normal")
    last._p.addnext(q._p)
    last = q

doc.save(DST)
d2 = Document(DST)
ns = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
print(f"saved {DST.name}")
print(f"  paragraphs {len(d2.paragraphs)}  tables {len(d2.tables)}  "
      f"images {sum(1 for p in d2.paragraphs if p._p.findall('.//' + ns + 'blip'))}")
print(f"  styles {sorted({p.style.name for p in d2.paragraphs})}")
