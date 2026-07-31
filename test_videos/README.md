# Test videos

No real video files exist on this machine, so these are generated from
`test_images/` with ffmpeg. Regenerate with the commands in this file.

| File | Motion | What it probes |
|------|--------|----------------|
| `01_pan_skiers.mp4` | Camera pan, static subjects | Scene + subject recognition across frames; the subjects do NOT move |
| `02_zoom_food.mp4` | Slow zoom in | Whether scale change alters what the model reports |
| `03_scene_sequence.mp4` | **4 different scenes**, 1.5 s each: bird → horse → train → tennis | **The only clip where frame ORDER carries information** — the real test of whether temporal understanding exists |

## Important limitation

Clips 1 and 2 contain **camera** motion but no **subject** motion — nothing in
the scene actually moves. They cannot test motion understanding; they test
whether frame aggregation is stable. Clip 3 is the meaningful temporal probe:
ask "what order do the scenes appear in?" and compare the answer against
bird → horse → train → tennis.

Expect clip 3 to fail on ordering. The projector and LoRA were trained only on
single images and the encoder's RoPE is spatial, so no temporal signal exists
anywhere in the stack — a confident answer about order is the language model
guessing. That failure is the point of having the clip.

## Regenerate

```bash
ffmpeg -y -loop 1 -i test_images/03_skiers_mountain.jpg \
  -vf "scale=1100:-1,crop=640:480:'(iw-640)*t/5':'(ih-480)/2',fps=12" -t 5 \
  -pix_fmt yuv420p test_videos/01_pan_skiers.mp4
```
