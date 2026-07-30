# Test images

Ten COCO train2017 images chosen for **diversity**, so a single batch run
exercises different failure modes of the Stage 2 VLM rather than ten variations
of one test.

| File | Contents | What it probes |
|------|----------|----------------|
| `01_food_bento.jpg` | Bento box: broccoli, meatball, bread+butter, almonds, fruit, cookies | Fine-grained multi-object; invites the "rice"/"seafood" hallucination |
| `02_baby_crib.jpg` | Smiling baby in a crib with tiger + teddy plush toys | People + objects, indoor |
| `03_skiers_mountain.jpg` | Two skiers on a snowy slope, pines and peaks behind | Outdoor scene + landscape context |
| `04_tennis_player.jpg` | Player mid-serve on a blue/green court | Single-subject action / pose |
| `05_street_sign_text.jpg` | "NO PARKING SIDE OF STREET" sign, dim lighting | **OCR** — expected to fail; no OCR data in training (see docs/OCR_ROADMAP.md) |
| `06_objects_flatlay.jpg` | ~12 small objects on wood: watch, revolver, keys, penknife, wallet | Many small objects — stresses the 512-token ATB budget |
| `07_floating_market.jpg` | Vendor in a boat beside boats of mangosteens, rambutans, bananas | Cluttered scene, water, unusual composition |
| `08_bird_branch.jpg` | Yellow-breasted bird perched among green leaves | Animal close-up, shallow depth of field |
| `09_horse_rider.jpg` | Rider on a dark horse in a grass field with trees | Animal + person + landscape together |
| `10_train_tracks.jpg` | Yellow/white commuter train, tracks, overhead catenary | Vehicle / transport; small text on the carriage |

Source: COCO train2017 (`/home/work/keural-model/vlm-datasets/coco/train2017`),
original IDs 9, 132516, 221794, 310411, 88517, 442543, 24609, 202129, 290881,
334513. Not committed to git — regenerate by copying those IDs.

## Run all ten on one model load

```bash
CKPT=outputs/projector_2a/checkpoint-2000
CUDA_VISIBLE_DEVICES=1 python3 scripts/demo_vlm.py \
  --projector-ckpt $CKPT/projector.pt \
  --lora-ckpt      $CKPT/lora_adapter \
  --image-dir test_images/ \
  --question "Describe this image."
```
