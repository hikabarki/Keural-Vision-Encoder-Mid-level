# Test images

Six COCO train2017 images chosen for **diversity**, so a single batch run
exercises different failure modes of the Stage 2 VLM.

| File | Contents | What it probes |
|------|----------|----------------|
| `01_food_bento.jpg` | Bento box: broccoli, meatball, bread+butter, almonds, fruit, cookies | Fine-grained multi-object; easy to hallucinate ("rice") |
| `02_baby_crib.jpg` | Smiling baby in a crib with tiger + teddy plush toys | People + objects, indoor |
| `03_skiers_mountain.jpg` | Two skiers on a snowy slope, pines and peaks behind | Outdoor scene + landscape context |
| `04_tennis_player.jpg` | Player mid-serve on a blue/green court | Single-subject action / pose |
| `05_street_sign_text.jpg` | "NO PARKING SIDE OF STREET" sign, dim lighting | **OCR** — expected to fail; no OCR data in training yet (see docs/OCR_ROADMAP.md) |
| `06_objects_flatlay.jpg` | ~12 small objects on wood: watch, revolver, keys, penknife, wallet | Many small objects — stresses the 512-token ATB budget |

Source: COCO train2017 (`/home/work/keural-model/vlm-datasets/coco/train2017`),
original IDs 9, 132516, 221794, 310411, 88517, 442543.
Not committed to git — regenerate by copying those IDs.

## Run all six on one model load

```bash
python3 scripts/demo_vlm.py \
  --projector-ckpt outputs/projector_2a/checkpoint-1000/projector.pt \
  --lora-ckpt      outputs/projector_2a/checkpoint-1000/lora_adapter \
  --image-dir test_images/ \
  --question "Describe this image."
```
