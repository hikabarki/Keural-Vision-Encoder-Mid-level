"""
Build weekly_report_Hika_Barki_07-31.docx from the 07-24 template.

Template fidelity is the requirement: this COPIES the 07-24 file, strips only its
body content, and rewrites it using the same styles the template already
defines (Section Title / Heading 1 / Heading 2 / Normal / List Paragraph /
바탕글) and the same Table Grid tables. The header, footer, approval table,
logo, page geometry and theme are inherited untouched, so nothing about the
format is re-invented.
"""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "weekly_report_Hika_Barki_07-24.docx"
DST = ROOT / "weekly_report_Hika_Barki_07-31.docx"
FIGS = ROOT / "docs" / "figures"
GUI = ROOT / "figures.docx"
WIDTH = Inches(6.20)          # the width the template uses for full-bleed figures

shutil.copy(SRC, DST)
doc = Document(DST)

# ---------------------------------------------------------------- keep the head
# Paragraphs 0..15 are the letterhead, logo, approval table anchor and date.
# Everything from the first "Section Title" onward is this week's content.
first_section = next(i for i, p in enumerate(doc.paragraphs)
                     if p.style.name == "Section Title")
head = doc.paragraphs[:first_section]

for p in doc.paragraphs[first_section:]:
    p._element.getparent().remove(p._element)
# The template's second table is last week's format-readiness table; drop it and
# keep table 0 (the 결재 approval block).
for t in doc.tables[1:]:
    t._element.getparent().remove(t._element)

# Update the title line and the date, in place, preserving their run formatting.
for p in head:
    if p.style.name == "바탕글" and p.text.strip():
        if p.runs:
            p.runs[0].text = ("Keural Mid Vision Encoder — Stage 2 VLM Alignment, "
                              "Instruction Tuning and Evaluation")
            for r in p.runs[1:]:
                r.text = ""
    if p.text.strip() == "2026.07.24":
        for r in p.runs:
            r.text = r.text.replace("2026.07.24", "2026.07.31")

body = doc.element.body


def para(text="", style="Normal"):
    p = doc.add_paragraph(text, style=style)
    body.append(p._element)
    return p


def h1(t):
    return para(t, "Heading 1")


def h2(t):
    return para(t, "Heading 2")


def bullet(t):
    return para(t, "List Paragraph")


def section(t):
    return para(t, "Section Title")


def figure(path: Path, caption: str, width=WIDTH):
    p = doc.add_paragraph()
    body.append(p._element)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=width)
    c = para(caption, "Normal")
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in c.runs:
        r.font.size = Pt(9)
        r.italic = True
    return c


def table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    body.append(t._element)
    for c, htxt in zip(t.rows[0].cells, headers):
        c.text = ""
        run = c.paragraphs[0].add_run(htxt)
        run.bold = True
        run.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for c, v in zip(cells, row):
            c.text = ""
            run = c.paragraphs[0].add_run(str(v))
            run.font.size = Pt(9)
    para("")
    return t


# Extract the GUI screenshots the user prepared in figures.docx.
gui_imgs = []
if GUI.exists():
    import zipfile
    tmp = ROOT / ".gui_figs"
    tmp.mkdir(exist_ok=True)
    with zipfile.ZipFile(GUI) as z:
        for n in sorted(x for x in z.namelist() if x.startswith("word/media/")):
            out = tmp / Path(n).name
            out.write_bytes(z.read(n))
            gui_imgs.append(out)

# ==============================================================================
section("Keural Mid Vision Encoder — Stage 2 VLM Alignment, Instruction Tuning and Evaluation")

h1("개요")
h2("이번 주 개발 내용")
para(
    "This week the Keural Mid vision encoder (183.5M parameters) was connected to a "
    "large language model and turned into a working vision-language model. Stage 2a "
    "trained the LevelAwareProjector that maps the encoder's 768-dim visual tokens "
    "into Qwen2.5-7B-Instruct's embedding space; Stage 2b then instruction-tuned the "
    "result so the model answers questions about images rather than only captioning "
    "them. A bilingual (English + Korean) Stage 2b run is currently training. "
    "Alongside the training work, the evaluation stack was completed and a Streamlit "
    "GUI was built so model quality can be inspected directly rather than inferred "
    "from loss curves.")
para("Key deliverables this week:")
bullet("Completed Stage 2a projector alignment: held-out perplexity 31.81 → 17.30 over "
       "5,000 steps on 2×H200 with DDP, the best result achieved for this stage.")
bullet("Diagnosed and fixed a training instability that had capped an earlier run at "
       "perplexity 53: the trainer had no random seed, so projector and LoRA "
       "initialisation differed on every launch and success was a matter of luck. "
       "Adding seed 42, halving peak learning rate to 5e-4 and lengthening warmup to "
       "500 steps removed the failure entirely.")
bullet("Completed Stage 2b instruction tuning (8,000 steps, LLaVA-Instruct-150K), "
       "producing a model that follows instructions and answers open-ended questions.")
bullet("Built the full evaluation stack and ran it: POPE hallucination benchmark, "
       "VQAv2, and the blind (no-vision) baseline that isolates what the vision "
       "encoder actually contributes.")
bullet("Measured a +9.37 point sighted-versus-blind gap on VQAv2, the first direct "
       "quantitative evidence that the vision encoder contributes real information "
       "rather than the language model answering from priors.")
bullet("Fixed six defects in the training and evaluation code, four of which failed "
       "silently — including one that would have made Stage 2b discard all of Stage "
       "2a and train from random weights for six hours with a plausible-looking loss "
       "curve.")
bullet("Built VLM Studio, a Streamlit GUI with side-by-side sighted/blind comparison, "
       "checkpoint switching without reloading the 7B model, and answer-length presets.")
bullet("Launched the bilingual (English + Korean) Stage 2b run on a rebuilt 603,255-record "
       "mix; Korean held-out perplexity is 2.42 at step 1,250 of 12,000.")

# ------------------------------------------------------------------------------
h1("상세 개발 내용")

h2("Stage 2a — 프로젝터 정렬 (Projector Alignment)")
para(
    "Stage 2a trains only the LevelAwareProjector (8.9M parameters) and LoRA adapters "
    "(5.0M) while the vision encoder and Qwen2.5-7B remain frozen. Training ran on "
    "2×H200 under DistributedDataParallel at an effective batch of 64, 512×512 input "
    "resolution and a 512-token ATB budget, over the LLaVA-Pretrain 558K caption set. "
    "Held-out validation perplexity fell from 31.81 to 17.30 across 5,000 steps and "
    "was still descending at the end, with the improvement rate flattening from "
    "-4.54% to -0.95% per evaluation — the signal that the stage had converged.")
if (FIGS / "stage2a_val_perplexity.png").exists():
    figure(FIGS / "stage2a_val_perplexity.png",
           "Figure 1 — Stage 2a held-out validation perplexity. The stabilised run "
           "(seed 42, peak LR 5e-4, warmup 500) reaches 17.3, against 53.0 for the "
           "earlier unseeded run at LR 1e-3 whose gradients blew up at steps 450–650. "
           "The dashed line marks the previous best result of 23.9.")

h2("학습 불안정성 진단 및 해결")
para(
    "An earlier Stage 2a run degraded sharply mid-training: validation perplexity "
    "spiked from 25.74 to 124.50 between steps 500 and 750 and never recovered past "
    "53. Gradient norms recorded before clipping reached roughly 40× the clip "
    "threshold across steps 450–650, precisely where the learning-rate schedule sat "
    "at its 1e-3 peak. Gradient clipping prevented divergence but could not prevent "
    "the damage. The root cause was not the learning rate alone: the trainer set no "
    "random seed, so projector and LoRA weights initialised differently on every "
    "launch — two runs on identical data and identical hyper-parameters reached 23.9 "
    "and 124.5 respectively. Whether training succeeded was therefore chance, and "
    "failures could not be reproduced or diagnosed.")
if (FIGS / "stage2a_grad_norm.png").exists():
    figure(FIGS / "stage2a_grad_norm.png",
           "Figure 2 — Gradient norm before clipping, log scale. The unstable run "
           "spikes far above the clip threshold through steps 450–650; the stabilised "
           "run stays below the threshold for the entire run after step 60.")

h2("Stage 2b — 명령어 튜닝 (Instruction Tuning)")
para(
    "Stage 2b warm-starts from Stage 2a's checkpoint-5000 and fine-tunes on "
    "LLaVA-Instruct-150K (156,712 samples) at learning rate 2e-5 with assistant-only "
    "label masking, so the model is trained to produce answers rather than to echo "
    "questions. Training loss fell to 1.24 and held-out perplexity converged to 2.90 "
    "by roughly step 6,500 of 8,000. The resulting model follows instructions and "
    "produces structured multi-paragraph descriptions, a qualitative change from "
    "Stage 2a's caption-only behaviour. Note that Stage 2b perplexity is not "
    "comparable to Stage 2a's: the data and the label-masking scheme both differ.")
if (FIGS / "stage2b_training.png").exists():
    figure(FIGS / "stage2b_training.png",
           "Figure 3 — Stage 2b instruction tuning: training loss (upper) and "
           "held-out perplexity (lower), shown as separate panels because the two "
           "measures are on different scales. Perplexity converges near step 6,500.")

h2("평가 도구 및 VLM Studio GUI")
para(
    "A Streamlit application was built so model quality can be judged directly. Its "
    "central feature is a side-by-side comparison in which the same question is "
    "answered twice — once with the image and once with the visual tokens removed. "
    "Any content that appears in both columns originates in the language model, not "
    "in the vision encoder, which makes hallucination visible at a glance rather "
    "than by careful reading. The base LLM loads once and only the projector (18 MB) "
    "and LoRA adapter (20 MB) are swapped when changing checkpoints, so comparing "
    "two checkpoints takes seconds instead of two full model loads.")
if len(gui_imgs) > 0:
    figure(gui_imgs[0],
           "Figure 4 — VLM Studio, sighted versus blind comparison. With the image, "
           "the model correctly reports two skiers, one in a red jacket, holding "
           "their skis on a snowy slope. Blind, it invents a close-up of a face. The "
           "two answers share nothing, which is direct evidence that the vision "
           "encoder is driving the description.")
if len(gui_imgs) > 1:
    figure(gui_imgs[1],
           "Figure 5 — Short-answer mode on the tennis image: “A man is playing "
           "tennis on a court, holding a tennis racket and swinging it to hit the "
           "ball,” against a blind answer describing a man in a suit beside a "
           "sculpture. The answer-length presets exist because long-form output "
           "hallucinates substantially more than short output.")
if len(gui_imgs) > 2:
    figure(gui_imgs[2],
           "Figure 6 — The bilingual Stage 2b checkpoint (Korean held-out perplexity "
           "2.48) describing a rider on a horse in a grassy field with a black jacket "
           "and brown hat. The earlier English-only checkpoint invented additional "
           "horses on this same image; that fabrication is absent here.")

# ------------------------------------------------------------------------------
h1("개발 문제 및 해결 방법")

h2("Stage 2b 웜스타트 누락 — 조용한 실패")
para(
    "Problem: both Stage 2b launch scripts would have started from a randomly "
    "initialised projector, discarding all of Stage 2a. Stage 2b writes to a "
    "different output directory than Stage 2a, and the --resume flag only searches "
    "its own output directory; no mechanism existed to load weights across stages. "
    "The script's own header comment stated that it resumed from the Stage 2a "
    "projector, so the intent was documented while the implementation was absent. "
    "Because Stage 2b runs at 2e-5 — a fine-tuning rate, 25× too low to train a "
    "projector from scratch — the run would have completed normally with a plausible "
    "loss curve and produced a badly undertrained model, with nothing in the logs to "
    "indicate the problem.")
para(
    "Solution: added an --init-from flag that loads projector and LoRA weights from "
    "any checkpoint directory with a fresh optimiser and learning-rate schedule, "
    "which is the correct behaviour across stages. Both launchers now default to "
    "Stage 2a's checkpoint-5000, and the startup log prints the warm-start source "
    "and the tensor count so the operator can verify it took effect.")

h2("--resume 기능의 3중 결함")
para(
    "Problem: the resume path contained three independent defects. The projector "
    "state was loaded into the DistributedDataParallel wrapper rather than the "
    "underlying module, which would raise a key-mismatch error on any multi-GPU "
    "resume; the LoRA adapter was saved but never reloaded, silently discarding "
    "roughly 5M of the 14M trainable parameters; and checkpoint rotation called "
    "int() on every checkpoint-* directory name, so a single manually copied backup "
    "would terminate the run at its first save.")
para(
    "Solution: weight loading was unified into one function shared by --resume and "
    "--init-from, so the fixes apply to both paths. Checkpoint discovery now skips "
    "and logs non-numeric directories instead of crashing, and resume fails loudly "
    "when the optimiser state or adapter weights are missing rather than continuing "
    "from cold state.")

h2("체크포인트 메타데이터 부재")
para(
    "Problem: a checkpoint stored weights but no record of how to run them. The "
    "token budget, image size and projector hidden dimension must match training "
    "exactly or output degrades with no error, and nothing in the checkpoint "
    "recorded them — the inference script defaulted to 384px against a projector "
    "trained at 512px. Checkpoint rotation also kept only the last three "
    "checkpoints unconditionally, so a run whose validation worsened near the end "
    "would delete its own best model.")
para(
    "Solution: each checkpoint now writes a config.json carrying the three "
    "inference-critical values plus provenance (seed, learning rate, schedule, "
    "effective batch, data path, warm-start source) and the validation scores at "
    "that checkpoint. The inference tools read and adopt it automatically. Rotation "
    "now protects the best-scoring checkpoint, and RNG state is saved so a resumed "
    "run continues the same stochastic stream.")

h2("VQAv2 평가 오류 — 형식과 정확도의 혼동")
para(
    "Problem: VQAv2 initially crashed outright because the benchmark stores each "
    "annotation as a dictionary rather than a string. Once fixed, the model scored "
    "1.73% sighted against 5.07% blind — a negative gap, and far below the ~40% a "
    "text-only 7B model reaches from language priors. Inspecting predictions showed "
    "the cause: asked “Is this rice noodle soup?” with gold answer “yes”, the model "
    "replies “Yes, this is a bowl of rice noodle soup.” Exact-match scoring returned "
    "zero for answers that were correct. The metric was measuring output format, not "
    "perception.")
para(
    "Solution: the evaluator now reports both the standard strict score, which "
    "remains comparable to published numbers, and a lenient score that credits an "
    "answer when the gold string appears in it. Genuinely wrong answers score zero "
    "under both. Sample predictions are saved with each run so a suspicious score "
    "can be diagnosed without re-running the benchmark. The underlying cause is a "
    "data gap: Stage 2b trained only on verbose conversational data, whereas "
    "LLaVA-1.5 adds short-answer VQA splits precisely to teach terseness.")

h2("이중언어 검증에서 영어 지표 누락")
para(
    "Problem: the bilingual run was configured to validate in both English and "
    "Korean, and its log confirmed both validation sets loading. Only Korean results "
    "were ever reported. The English validation records store bare image filenames "
    "while the Korean records store paths relative to the dataset root; since a "
    "single --val-image-root serves both, every English record failed its existence "
    "check and the language was dropped from the report with no message.")
para(
    "Solution: a corrected English validation file with root-relative paths was "
    "generated and verified (1,000 of 1,000 images resolve), and validation now "
    "prints an explicit warning when a language yields no usable records instead of "
    "omitting it silently. The currently running job continues to report Korean only; "
    "English quality for it will be measured post-hoc with the evaluation scripts.")

# ------------------------------------------------------------------------------
h1("이번 주 연구 결과 분석 및 평가")

h2("학습 결과")
table(["Stage", "Data", "Steps", "Held-out perplexity", "Status"],
      [["2a — projector alignment", "LLaVA-Pretrain 558K", "5,000", "31.81 → 17.30", "Complete"],
       ["2b — instruction tuning (EN)", "LLaVA-Instruct 150K", "8,000", "3.22 → 2.90", "Complete"],
       ["2b — bilingual (EN + KO)", "Mixed EN/KO 603K", "12,000", "2.90 → 2.42 (KO)", "In progress"]])
para(
    "Stage 2b perplexity values are not comparable with Stage 2a's: the two stages "
    "use different data and different label masking. Each stage should be compared "
    "only against itself.")

h2("벤치마크 평가 결과")
table(["Benchmark", "Metric", "Sighted", "Blind (no vision)", "Gap"],
      [["VQAv2 (n=1000)", "Lenient accuracy", "48.60%", "39.23%", "+9.37 points"],
       ["VQAv2 (n=1000)", "Strict exact match", "1.73%", "5.07%", "−3.33 points"],
       ["POPE (n=500)", "Accuracy", "68.00%", "—", "—"],
       ["POPE (n=500)", "F1", "70.70%", "—", "—"],
       ["POPE (n=500)", "Yes-ratio", "0.592", "—", "—"]])
para(
    "The +9.37 point lenient gap is the most important number produced this week. A "
    "blind model answers from language priors alone, and its 39.23% score matches the "
    "~40% expected from question statistics, which confirms the measurement is sound. "
    "The gap is therefore a direct measure of what the vision encoder contributes. It "
    "is real but modest, consistent with an encoder scoring 37.2% zero-shot on "
    "ImageNet. The strict row is reported for completeness but measures answer format, "
    "not perception. POPE's yes-ratio of 0.592 is healthy — a model answering “yes” "
    "indiscriminately would sit near 1.0 — indicating the model genuinely "
    "discriminates when asked direct questions.")

h2("정성 평가 및 한계")
para(
    "A fixed set of ten test images spanning food, people, outdoor scenes, sport, "
    "signage, cluttered objects, animals and vehicles was used to track quality "
    "across checkpoints. Object-level recognition is solid: the model correctly "
    "identifies skiers on a mountain, a baby in a crib with stuffed animals, a tennis "
    "player mid-action, a bird on a branch and a train on tracks. Two limitations are "
    "structural rather than a matter of further training. First, the model cannot "
    "read text: shown a NO PARKING sign it produced fabricated text at every "
    "checkpoint, which is expected because no OCR data is present in training. "
    "Second, long-form output fabricates detail — asked for a detailed description it "
    "invents plausible objects such as cutlery on a table or additional animals in a "
    "field. Short-answer mode is markedly more reliable, which is why the GUI exposes "
    "answer length directly and why short-answer training data was added to the "
    "bilingual mix.")

h2("종합 평가")
para(
    "The Keural Mid vision encoder is now a functioning vision-language model, and "
    "its contribution has been measured rather than assumed. The +9.37 point gap over "
    "a blind baseline, a healthy POPE yes-ratio and clearly image-dependent outputs "
    "in the GUI together establish that the encoder carries real visual information "
    "into the language model. The principal weakness is that generated text currently "
    "outruns what the model can actually see, producing fluent but partly fabricated "
    "long-form descriptions. This is a data and output-length problem with identified "
    "remedies rather than a defect in the encoder.")

# ------------------------------------------------------------------------------
h1("결론 및 향후 연구 방향")
h2("이번 주 결론")
para(
    "Stage 2a and Stage 2b were completed and the encoder became a working VLM. A "
    "training instability that had capped an earlier run at perplexity 53 was traced "
    "to unseeded initialisation and eliminated, taking the stage to its best result "
    "of 17.30. The evaluation stack was completed and executed, producing the first "
    "quantitative evidence that the vision pipeline contributes (+9.37 points over "
    "blind). Six code defects were fixed, four of which failed silently and one of "
    "which would have wasted a six-hour training run. A GUI now makes quality "
    "inspectable directly, and a bilingual run is training on a rebuilt data mix.")
h2("향후 연구 방향")
bullet("Complete the bilingual Stage 2b run (12,000 steps) and evaluate English and "
       "Korean separately; no public Korean VLM benchmark exists, so held-out "
       "perplexity plus qualitative review through the GUI is the available measure.")
bullet("Re-run POPE and VQAv2 on the bilingual checkpoint to determine whether the "
       "added short-answer data closes the strict/lenient gap, which is the direct "
       "test of the format hypothesis.")
bullet("Address long-form hallucination with denser grounded caption data such as "
       "ShareGPT4V, and with output-length control at inference.")
bullet("Add OCR capability, currently absent entirely: staged datasets such as "
       "SynthDoG, DocVQA and TextVQA would create a capability the model does not "
       "presently have.")
bullet("Evaluate unfreezing the vision encoder during late Stage 2b, standard "
       "practice in LLaVA-1.5, to let visual features adapt to the language model.")
bullet("Consider additional Stage 1 encoder training: at 37.2% zero-shot ImageNet the "
       "encoder is the ceiling on everything downstream, and improving it raises "
       "every subsequent metric.")

h1("첨부 자료")
para("Repository:")
bullet("github.com/mkd-hika/keural-mid-vision-encoder")
para("Model:")
bullet("huggingface.co/mkd-hika/keural-mid-vision-encoder")

doc.save(DST)
print(f"wrote {DST}")
d2 = Document(DST)
print(f"  paragraphs: {len(d2.paragraphs)}  tables: {len(d2.tables)}")
print(f"  figures embedded: {sum(1 for p in d2.paragraphs if p._p.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}blip'))}")
print(f"  styles used: {sorted({p.style.name for p in d2.paragraphs})}")
