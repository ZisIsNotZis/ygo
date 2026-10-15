### ygo — Yu-Gi-Oh! deck/art playground (personal experiments)

**What:** A private scratch collection of scripts and notebooks around Yu-Gi-Oh!
decklists and card art: a hand-written GGML/GGUF inference wrapper used to run
DINOv2 in Python, PCA-based card-art cropping, DINOv2-embedding video shot
selection, n-gram-based deck naming/archiving, and small torch models that
generate or score decks.

**Why it exists:** To answer hobby questions with real tooling rather than manual
work: which frame of a replay is representative, how to auto-crop card artwork,
which cards cluster by archetype, what a plausible deck looks like, and how to
name/archive ~146k decklists so they are findable. It doubles as a from-scratch
exercise in driving `ggml`/GGUF from Python (`gg.py`) instead of using a
framework.

**How to run:** No single entrypoint. Each artifact is standalone and expects data
that is *not* in this directory:

- `python3 stat.py` — needs `cards.cdb` (YGOPRO card database) in the CWD;
  prints per-race/attribute archetype statistics as YAML.
- Notebooks (`model*.ipynb`, `deck*.ipynb`, `tok.ipynb`, `name.ipynb`,
  `nomodel.ipynb`) — need `cards.cdb`, `lflist.conf`, a CUDA GPU, and for the
  training variants a `deck/**/*.ydk` corpus; the `*inf.ipynb` variants are
  inference-only.
- `python3 dino2.py <images...>` — needs a local DINOv2 GGUF plus the external
  `ggml` Python module (`from ggml import *`); it also resolves the model from a
  hard-coded absolute path (see sanitisation).
- `python3 cut.py` / `cutvid.py` — need the `dino2` outputs and `ffmpeg`/`ffprobe`
  (QSV hardware accel paths in `cutvid.py`).
- `ydkrename.py` — needs `cards.cdb`, `lflist.conf`, `strings.conf`, and a
  `../pack` directory.
- `clipdown.sh` / `ypicdown.sh` — X11 clipboard capture and card-image download
  helpers (need `clipnotify`, `xclip`, `sqlite3`, `curl`, ImageMagick).

**Status:** current (2026-era: Python 3.12 / PEP 695 generics, torch 2.x, modern
GGML) — but unmaintained as a project: it is a personal experiment pile, not a
packaged tool. It is an archive of hobby work, not a library anyone else can run
without the private `cards.cdb` and a DINOv2 GGUF. Do not describe it as a
publishable product.

**License:** See `LICENSE` for this repository's AGPL-3.0 license. External
data it depends on (`cards.cdb`, card images from a third-party CDN, DINOv2
weights) has separate licensing and is not covered by this license.
