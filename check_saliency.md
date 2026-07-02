
CUDA_VISIBLE_DEVICES="" python scripts/viz_saliency.py \
  --checkpoint outputs/run/checkpoint-3000 \
  --images <your_folder> --out outputs/saliency_test --size 384
