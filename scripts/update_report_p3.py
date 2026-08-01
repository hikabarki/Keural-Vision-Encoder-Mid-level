"""
Pass 3 — record Stage 2c (OCR) and Stage 3 (DPO) in the technical report.

Both were run after pass 2 was written. Stage 2c was previously recorded as a
flat failure; that was wrong and is corrected here — v1 gained +27.5 MME OCR and
v2 reached the best MME total of any checkpoint. Stage 3 DPO is recorded as a
negative result with its diagnosis, because a documented failure with a cause is
worth more to a future reader than an omission.
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "docs/reports/Keural_Mid_Technical_Documentation_Report.docx"
FIGS = ROOT / "docs/figures"
WIDTH = Inches(6.0)

doc = Document(DST)
P = doc.paragraphs


def settext(i, t):
    p = P[i]
    if p.runs:
        p.runs[0].text = t
        for r in p.runs[1:]:
            r.text = ""
    else:
        p.add_run(t)


# -- retitle Section 5 to cover both alignment and preference optimization ----
settext(104, "5. Stage 2 and Stage 3: Alignment, Specialization and "
             "Preference Optimization (Completed)")

# -- 5.5: the OCR paragraph moves into its own section, so shorten it here ----
settext(131,
    "Text recognition is partial in the Stage 2b model. No OCR-specific data was "
    "present in Stage 1 or Stage 2b training, yet it scores 110.0 on the MME OCR "
    "category against a blind baseline of 50.0 and reads large, high-contrast "
    "signage correctly, so some text sensitivity is already present in the "
    "distilled representation. Two dedicated OCR runs and their outcomes are "
    "reported in Section 5.6.")

# =============================================================================
# Insert 5.6, 5.7, 5.8 before "6. Mathematical Foundations"
# =============================================================================
anchor = P[134]._p
tail = []


def add(text, style="Normal"):
    q = doc.add_paragraph(text, style=style)
    tail.append(q._p)
    return q


def add_fig(name, caption):
    p = FIGS / name
    if not p.exists():
        return
    q = doc.add_paragraph()
    q.alignment = WD_ALIGN_PARAGRAPH.CENTER
    q.add_run().add_picture(str(p), width=WIDTH)
    tail.append(q._p)
    c = doc.add_paragraph(caption, style="Normal")
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in c.runs:
        r.font.size = Pt(9)
        r.italic = True
    tail.append(c._p)


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
    tail.append(t._element)
    add("")


# ----------------------------------------------------------------- 5.6 OCR --
add("5.6 Stage 2c — OCR specialization", "Heading 2")
add(
    "Two runs tested whether text-reading could be strengthened without retraining "
    "the frozen encoder. Both warm-started from the Stage 2b bilingual checkpoint "
    "and trained on 30,000 TextVQA scene-text examples mixed with general "
    "instruction data replayed to prevent catastrophic forgetting.")
add(
    "Run v1 used a 1:1 replay ratio, 4,000 steps at 1,024×1,024 input with a "
    "1,024-token budget. It succeeded at its stated goal: MME OCR rose from 110.0 "
    "to 137.5 and the MME total rose 40.8 points, with large gains in landmark "
    "(+59.5), celebrity (+21.2) and count (+15.0). It also caused a regression that "
    "the OCR metric alone would not have revealed: POPE accuracy fell from 70.50% to "
    "59.20% and the yes-ratio moved from 0.503 to 0.886, meaning the model began "
    "answering “yes” to 89% of object-existence questions. POPE F1 was almost "
    "unchanged (70.59% to 70.56%) because F1 rewards recall and a model that always "
    "answers yes has near-perfect recall — only accuracy and the yes-ratio exposed "
    "the damage, which is why both are reported throughout this document.")
add(
    "The cause was measured rather than assumed. Binary-question supervision had "
    "been diluted: the v1 mixture carried yes/no answers in 22.4% of records against "
    "the Stage 2b mixture's 39.3%, because TextVQA answers are almost never binary. "
    "Separately, re-evaluating v1 at 512×512 rather than its training resolution "
    "recovered 2.6 of its 5.2-point VQAv2 drop, showing that part of the apparent "
    "regression was the resolution change rather than the OCR data.")
add(
    "Run v2 corrected one variable: the replay ratio was raised to 2:1 and "
    "stratified across yes/no and non-yes/no records to restore 39.0% binary "
    "coverage, with 2,000 steps at 512×512. It produced the highest MME total of any "
    "checkpoint (1337.8) and recovered part of the calibration loss (yes-ratio 0.886 "
    "to 0.773, POPE accuracy 59.20% to 64.40%), but OCR fell back from 137.5 to "
    "115.0. Raising replay restored general ability and diluted the OCR signal "
    "simultaneously; the two move in opposition under the same control.")
add_table(
    ["Metric", "Stage 2b", "2c v1", "2c v2"],
    [["MME total (14 categories)", "1258.9", "1299.7", "1337.8"],
     ["MME OCR category", "110.0", "137.5", "115.0"],
     ["POPE accuracy (n = 2,000)", "70.50%", "59.20%", "64.40%"],
     ["POPE yes-ratio", "0.503", "0.886", "0.773"],
     ["VQAv2 strict (n = 1,000)", "53.87%", "51.27%", "52.70%"]])
add(
    "The conclusion is that OCR capability and binary-question calibration are in "
    "genuine tension for this model at this data scale, not that OCR training "
    "failed. 30,000 scene-text examples over 2,000 steps is a small intervention; "
    "obtaining both would require more OCR data and more replay simultaneously, "
    "which is a larger run than the available compute allowed.")

# ----------------------------------------------------------------- 5.7 DPO --
add("5.7 Stage 3 — Direct Preference Optimization (negative result)", "Heading 2")
add(
    "Stage 3 applied Direct Preference Optimization to reduce hallucination, using "
    "openbmb/RLHF-V-Dataset: 5,516 training pairs and 200 held out, each a (image, "
    "question, chosen, rejected) tuple in which a human corrected a hallucinated "
    "response. Only the LoRA adapters trained; the encoder and projector stayed "
    "frozen. Reference log-probabilities were precomputed once with the Stage 2b "
    "adapter active, so pi_ref is the SFT policy the run starts from rather than the "
    "base decoder — a distinction the objective depends on, and one confirmed "
    "empirically by a step-0 implicit reward margin of 0.0000. Hyperparameters were "
    "beta = 0.1, learning rate 5e-6, 2,000 steps at an effective batch of 8.")
add(
    "By its own objective the run succeeded. Held-out reward accuracy — the fraction "
    "of pairs where the human-corrected answer scores above the hallucinated one — "
    "rose from chance to 0.890, peaking at 0.910, and held-out loss fell from 0.497 "
    "to 0.337.")
add(
    "Every downstream measure nevertheless deteriorated. POPE accuracy fell from "
    "70.50% to 61.00% and the yes-ratio collapsed from 0.503 to 0.116: the model "
    "began denying the presence of objects in 88% of cases. An earlier checkpoint at "
    "step 500 was equally affected (60.55%, yes-ratio 0.112), so this is a property "
    "of the objective on this data rather than of over-training. Long-form output "
    "degraded in the same direction — asked to describe a rider on a horse, the "
    "Stage 2b model answers “a man riding a horse on a grassy field” while the DPO "
    "model answers “a man standing on a grassy field with a horse… holding a rope… "
    "holding a stick”, replacing one correct claim with several incorrect ones.")
add(
    "The diagnosis is that RLHF-V's chosen responses are human edits which REMOVE "
    "hallucinated objects. Optimizing that preference taught the surface pattern "
    "“claim fewer things”, which the model generalized into “deny that things "
    "exist”. The failure was visible in the training signal before evaluation: the "
    "implicit reward margin grew from +0.22 to +14.58, roughly sixty-fold, while "
    "held-out reward accuracy plateaued at 0.90 by step 600. The model spent the "
    "remaining 1,400 steps becoming confident about pairs it already ranked "
    "correctly — the standard signature of preference over-optimization under too "
    "weak a KL constraint.")
add(
    "The implementation is correct and reusable; what failed is the choice of "
    "beta and, more fundamentally, the assumption that preference data built for "
    "long-form correction transfers to short binary questions. A retry would use a "
    "substantially stronger KL constraint (beta ≈ 0.5) and stop near step 400, and "
    "should be evaluated on POPE before any other metric.")

# ---------------------------------------------------- 5.8 model selection ----
add("5.8 Model selection", "Heading 2")
add(
    "Four checkpoints exist after Stage 2a. They do not form a ranking: each stage "
    "after supervised fine-tuning traded one capability for another, and the "
    "recommended deliverable is therefore the Stage 2b bilingual model rather than "
    "the most recently trained one.")
add_fig("model_selection.png",
        "Figure 11. POPE accuracy with MME total annotated (upper) and POPE "
        "yes-ratio against the gold ratio of 0.500 (lower). Distance from 0.500 is "
        "the calibration failure: only Stage 2b sits on the line. Stage 2c drifts "
        "toward answering “yes”, Stage 3 DPO collapses toward “no”. Accuracy alone "
        "conceals this, since a model answering one way consistently still scores "
        "near 50% on a balanced benchmark.")
add(
    "Recommendation: Stage 2b bilingual, checkpoint-12000, is the general-purpose "
    "model. It holds the best POPE accuracy (70.50%) and the only calibrated "
    "yes-ratio (0.503 against a gold 0.500), the best VQAv2 strict accuracy "
    "(53.04% at n = 5,000), and bilingual capability with Korean held-out perplexity "
    "2.14. Stage 2c v2 is a defensible alternative where MME total is the headline "
    "metric (1337.8, the best of the four) and where binary-question calibration "
    "matters less. Stage 2c v1 is the OCR specialist. The Stage 3 DPO checkpoint is "
    "not recommended for use and is retained as a documented result.")

for el in tail:
    anchor.addprevious(el)

# ------------------------------------------- roadmap: DPO has been attempted --
for i, p in enumerate(doc.paragraphs):
    if p.style.name == "Heading 2" and p.text.strip().startswith("7.2 Stage 3"):
        q = doc.add_paragraph(
            "Status: attempted and documented in Section 5.7. The run optimized its "
            "own objective successfully (held-out reward accuracy 0.890) but "
            "collapsed binary-question calibration, and the checkpoint is not "
            "recommended for use. The material below remains the plan for a retry "
            "under a stronger KL constraint.", style="Normal")
        p._p.addnext(q._p)
        break

doc.save(DST)
d = Document(DST)
ns = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
print(f"saved {DST.name}")
print(f"  paragraphs {len(d.paragraphs)}  tables {len(d.tables)}  "
      f"figures {sum(1 for p in d.paragraphs if p._p.findall('.//' + ns + 'blip'))}")
print("  sections:")
for p in d.paragraphs:
    if p.style.name in ("Heading 1", "Heading 2") and p.text.strip():
        pre = "" if p.style.name == "Heading 1" else "    "
        print(f"    {pre}{p.text.strip()[:66]}")
