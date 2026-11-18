# Bathroom Revival Weekend - programmatic edit

Python + ffmpeg pipeline that cuts the DJI clips from May 23-25, 2025 into a 72-second, beat-locked
before-and-after story, in 16:9 and 9:16.

## Run it

Put the `.MP4` masters and `.LRF` proxies in one folder (the Drive footage folder, downloaded as-is), then:

```bash
python3 edit.py all --media /path/to/footage --aspect both --proxy   # fast 540p preview from the LRFs
python3 edit.py render --media /path/to/footage --aspect both        # final, from the MP4 masters
```

Needs `ffmpeg`/`ffprobe`, `numpy` and `Pillow`. Output lands in `out/`.

| Flag | Effect |
|---|---|
| `--aspect 16:9` / `9:16` / `both` | widescreen, vertical, or both |
| `--height 2160` | 4K export (default 1080) |
| `--proxy` | preview render from the LRFs instead of the masters |
| `--music track.wav --bpm 92 --music-offset 0.4` | use a licensed track; re-run `edl` so cuts follow its tempo |
| `--exclude 112633_0196,0198` | leave clips out (matched against the filename); applies at `scan`, then re-run `edl` |
| `--input-lut dlogm_to_709.cube` | conversion LUT for D-Log M / HDR footage, applied before the grade |

## Stages

1. `scan` pairs each LRF with its MP4 by filename (Drive's `-003` and `(1)` download suffixes are tolerated) and
   assigns acts by timestamp: Friday midday, Friday night + Saturday, Sunday morning.
2. `analyze` reads only the LRFs: motion energy (cut on action, steady glides), where the action sits
   in frame (9:16 crop position), midtone colour per session (white-balance matching), speech-like
   audio (reactions that duck the music) and sharp transients (pop SFX).
3. `edl` builds the cut on the beat grid and writes `edl.json` (what `render` reads - hand-edit it
   freely), `EDL.md` and `bathroom_revival.edl` (CMX3600).
4. `render` conforms against the MP4 masters: speed ramps, crop, white balance, warm LUT, transitions,
   titles, lower-thirds, the clock pill, and the audio mix.

Act titles, the story window and beat counts are constants at the top of `edit.py`.

## What is generated rather than sourced

- **Music**: a synthesised lo-fi beat at 90 BPM, as a rights-free placeholder. Swap in a real track with `--music`.
- **Whoosh / pop SFX** and the **warm interior LUT** (`build/warm_interior.cube`) are generated in code.

## Status of the committed EDL

`EDL.md` / `edl.json` in this folder were built from Drive metadata only (clip start times and
durations). The footage itself was never opened, so in-points are evenly spaced, not content-aware.
Running `all` against the real files replaces them.
