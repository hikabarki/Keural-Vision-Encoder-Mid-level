# Dataset Download Guide

**Server:** 2× H200 141GB
**Storage:** `/home/work/keural-model/vision-encoder-dataset/` (7.1 TB free on NFS)
**Venv:** `source /home/work/Hika/vision-encoder/mid/venv_mid/bin/activate`

---

## Storage Plan

| Dataset | Pairs | Est. Size |
|---|---|---|
| DataComp-1B Medium (full) | ~128M | ~4.2 TB |
| COYO-700M filtered (≥0.28, capped) | ~60M | ~2.5 TB |
| **Total** | **~188M** | **~6.7 TB** |

> Start DataComp first. Only start COYO Step 2 (images) **after** DataComp finishes — storage is tight.

---

## Step 0 — Setup

```bash
source /home/work/Hika/vision-encoder/mid/venv_mid/bin/activate

mkdir -p /home/work/keural-model/vision-encoder-dataset/datacomp_meta \
         /home/work/keural-model/vision-encoder-dataset/datacomp_medium \
         /home/work/keural-model/vision-encoder-dataset/coyo_meta \
         /home/work/keural-model/vision-encoder-dataset/coyo_filtered
```

---

## Dataset 1 — DataComp-1B Medium

### Step 1 — Download metadata

```bash
nohup python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='mlfoundations/datacomp_1b',
    repo_type='dataset',
    local_dir='/home/work/keural-model/vision-encoder-dataset/datacomp_meta',
    ignore_patterns=['*.tar', '*.zip', '*.npz']
)
print('DataComp metadata done.')
" > /home/work/keural-model/vision-encoder-dataset/datacomp_download.log 2>&1 &
echo "DataComp metadata PID: $!"
```

Monitor:
```bash
tail -f /home/work/keural-model/vision-encoder-dataset/datacomp_download.log
```

### Step 2 — Download images
> Run only after Step 1 finishes (check log for "DataComp metadata done.")

```bash
nohup img2dataset \
    --url_list /home/work/keural-model/vision-encoder-dataset/datacomp_meta \
    --input_format parquet \
    --url_col url \
    --caption_col text \
    --output_format webdataset \
    --output_folder /home/work/keural-model/vision-encoder-dataset/datacomp_medium \
    --processes_count 16 \
    --thread_count 64 \
    --image_size 512 \
    --resize_mode keep_ratio_largest \
    --resize_only_if_bigger True \
    --encode_format jpg \
    --encode_quality 95 \
    --number_sample_per_shard 10000 \
>> /home/work/keural-model/vision-encoder-dataset/datacomp_download.log 2>&1 &
echo "DataComp images PID: $!"
```

---

## Dataset 2 — COYO-700M Filtered

> Step 1 (filter) can run **in parallel with DataComp**.
> Step 2 (images) must wait until DataComp finishes downloading.

### Step 1 — Filter metadata (capped at 60M pairs)

```bash
nohup python3 -c "
from datasets import load_dataset
import pandas as pd

ds = load_dataset('kakaobrain/coyo-700m', split='train', streaming=True)
rows = []
for i, sample in enumerate(ds):
    if sample.get('clip_similarity_vitb32', 0) >= 0.28:
        rows.append({'url': sample['url'], 'text': sample['text'],
                     'similarity': sample['clip_similarity_vitb32']})
    if i % 5_000_000 == 0:
        print(f'{i:,} scanned, {len(rows):,} kept', flush=True)
    if len(rows) >= 60_000_000:
        print('Reached 60M cap, stopping.')
        break
df = pd.DataFrame(rows)
df.to_parquet('/home/work/keural-model/vision-encoder-dataset/coyo_meta/coyo_filtered.parquet', index=False)
print(f'Done: {len(df):,} pairs saved')
" > /home/work/keural-model/vision-encoder-dataset/coyo_download.log 2>&1 &
echo "COYO filter PID: $!"
```

Monitor:
```bash
tail -f /home/work/keural-model/vision-encoder-dataset/coyo_download.log
```

### Step 2 — Download images
> Run only after:
> - COYO Step 1 finishes (check log for "Done: X pairs saved")
> - DataComp Step 2 finishes (to free up disk space)

```bash
nohup img2dataset \
    --url_list /home/work/keural-model/vision-encoder-dataset/coyo_meta/coyo_filtered.parquet \
    --input_format parquet \
    --url_col url \
    --caption_col text \
    --output_format webdataset \
    --output_folder /home/work/keural-model/vision-encoder-dataset/coyo_filtered \
    --processes_count 16 \
    --thread_count 64 \
    --image_size 512 \
    --resize_mode keep_ratio_largest \
    --encode_format jpg \
    --encode_quality 95 \
    --number_sample_per_shard 10000 \
>> /home/work/keural-model/vision-encoder-dataset/coyo_download.log 2>&1 &
echo "COYO images PID: $!"
```

---

## Step 3 — Build TSV Indexes
> Run after both datasets are fully downloaded.

```bash
nohup python3 -c "
import glob, tarfile, pandas as pd
from tqdm import tqdm

def build_index(dataset_path, output_tsv):
    rows = []
    tars = sorted(glob.glob(f'{dataset_path}/**/*.tar', recursive=True))
    for tar_path in tqdm(tars, desc=output_tsv):
        try:
            with tarfile.open(tar_path) as tf:
                members = {m.name: m for m in tf.getmembers()}
                for m in members.values():
                    if not m.name.endswith('.txt'):
                        continue
                    key = m.name.replace('.txt', '')
                    if f'{key}.jpg' not in members:
                        continue
                    caption = tf.extractfile(m).read().decode('utf-8', 'ignore').strip()
                    if len(caption) < 5:
                        continue
                    rows.append({'tar': tar_path, 'key': key, 'caption': caption})
        except:
            continue
    pd.DataFrame(rows).to_csv(output_tsv, sep='\t', index=False)
    print(f'Wrote {len(rows):,} rows to {output_tsv}')

import os
os.makedirs('/home/work/keural-model/vision-encoder-dataset/indexes', exist_ok=True)
build_index('/home/work/keural-model/vision-encoder-dataset/datacomp_medium',
            '/home/work/keural-model/vision-encoder-dataset/indexes/datacomp_train.tsv')
build_index('/home/work/keural-model/vision-encoder-dataset/coyo_filtered',
            '/home/work/keural-model/vision-encoder-dataset/indexes/coyo_train.tsv')
" > /home/work/keural-model/vision-encoder-dataset/index_build.log 2>&1 &
echo "Index builder PID: $!"
```

---

## Step 4 — Verify

```bash
wc -l /home/work/keural-model/vision-encoder-dataset/indexes/datacomp_train.tsv
wc -l /home/work/keural-model/vision-encoder-dataset/indexes/coyo_train.tsv
df -h /home/work/keural-model
```

---

## Useful Commands

```bash
# Check disk space
df -h /home/work/keural-model

# Check folder sizes
du -sh /home/work/keural-model/vision-encoder-dataset/*/

# Check if a nohup process is still running
ps aux | grep img2dataset
ps aux | grep python3

# Kill a process by PID
kill <PID>
```

---

## Final Dataset Paths for Training Config

```
datacomp_images_root:  /home/work/keural-model/vision-encoder-dataset/datacomp_medium
datacomp_captions:     /home/work/keural-model/vision-encoder-dataset/indexes/datacomp_train.tsv
coyo_images_root:      /home/work/keural-model/vision-encoder-dataset/coyo_filtered
coyo_captions:         /home/work/keural-model/vision-encoder-dataset/indexes/coyo_train.tsv
```
