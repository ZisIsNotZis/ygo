<p align="center">
  <img src="assets/logo.svg" alt="ygo" width="200" height="40">
</p>

**English** | [简体中文](README.zh-CN.md)

# ygo

### Yu-Gi-Oh! deck and card-art playground — a personal experiment archive

**It runs DINOv2 directly on `ggml` from Python — a hand-written GGUF loader and
tensor graph, no framework — and spends the resulting embeddings on hobby
problems: auto-cropping card artwork, picking one representative frame per shot
from a duel video, and naming/archiving a pile of `.ydk` decklists so they can be
found again.**

Everything here was written to answer a question about actual cards, decks or
replays. The data it expects (`cards.cdb`, a DINOv2 GGUF, a `deck/` corpus) is
**not** in this repo; nothing here is packaged, versioned or published. Treat it
as readable source, not as a tool you install.

---

## Quick start

There is no package metadata, no install target and no single entrypoint. Each
file is standalone; the fastest one that runs is the statistics script, which
needs only `cards.cdb` (the YGOPRO sqlite card database) in the working
directory:

```bash
git clone https://github.com/ZisIsNotZis/ygo.git
cd ygo
# put a YGOPRO cards.cdb here
pip install pyyaml        # stat.py: sqlite3 (stdlib) + yaml
python3 stat.py           # prints race / attribute archetype statistics as YAML
```

Requirements by artifact (imports, read off the files — nothing is pinned):

| Artifact | Needs |
| --- | --- |
| `stat.py` | Python 3.8+, `pyyaml`, `cards.cdb` |
| `gg.py` | Python **3.12+** (`def dbg[**P, T]`, PEP 695 generics), the external `ggml` Python module, `numpy` |
| `dino2.py`, `cut.py`, `cutvid.py` | the above + `pillow`, `scikit-learn`; `cutvid.py` also needs `ffmpeg`/`ffprobe` with QSV (Intel hardware decode) |
| `ydkrename.py` | `xxhash`, plus `../cards.cdb`, `../strings.conf` and `../pack/` |
| notebooks | `numpy`, `scipy`, `torch` (2.x, CUDA for the training ones), `cards.cdb`, `lflist.conf`, a `deck/**/*.ydk` corpus |
| `clipdown.sh` / `ypicdown.sh` | `clipnotify`, `xclip`, `sqlite3`, `curl`, ImageMagick (`mogrify`) |

The `ggml` module is imported as a top-level package and is **not** vendored
here; `dino2.init()` also defaults to an absolute GGUF path
(`/home/z/gguf/dinov2-with-registers-small-imagenet1k-1-layer-f16.gguf`) — point
it at your own DINOv2 ViT-with-registers export before running anything below
`dino2.py`.

## What it can do now

- **`gg.py` — GGML/GGUF from scratch.** A `Tensor` class over `ggml_tensor_p`
  with shape/stride introspection, `view`-based `flatten`/`permute`/`T`, `cont`,
  `cat`, numpy round-trip, and the ops a ViT needs: `conv`, `lin`, `mlp`,
  `swiglu`, `norm`, `softmax`, `gelu`/`silu`, `rope`. Plus graph construction,
  `ggml_gallocr` allocation, backend compute, and a `@dbg` decorator that traces
  every op's shapes.
- **`dino2.py` — DINOv2 inference in pure ggml.** Patch-embedding conv, class
  and register tokens, bicubically interpolated position embeddings, per-layer
  attention/norm/layer-scale, SwiGLU for the 40-layer variant. Returns the CLS
  vector *and* the patch grid; `PCA=1` also writes a PCA-coloured JPEG of the
  patch features next to each input.
- **`cut.py` — card-art autocrop.** Embeds each non-square image, projects the
  patch grid to one channel with PCA, sums it along the long axis and cuts where
  the running sum peaks — i.e. it finds the artwork and drops the frame.
- **`cutvid.py` — shot selection for duel videos.** Streams 16×16 greyscale
  frames through `ffmpeg` (QSV) at 3 fps, drops runs that barely change, embeds
  the survivors with DINOv2, keeps one frame per k-means cluster, then crops the
  horizontal band the PCA projection says carries the action — one `.webp` per
  shot. Env knobs: `X0 Y0 X1 Y1 K0 K1 R ENC VIS`.
- **`ydkrename.py` — self-naming deck archive.** Caps copies at 3 and decks at
  75 cards, dates the deck from the `../pack/` set lists, then greedily appends
  the longest frequent n-gram shared by the cards' names, descriptions and
  archetype set names until the count drops below `CNT`; result is
  `YYMMDD<archetype>/<xxhash64>.ydk`.
- **`name.ipynb`** — the earlier n-gram-only version of the same naming idea.
- **`tok.ipynb`** — numpy-only feature selection over card names: pairwise
  bit-packed co-occurrence, entropy-guided merging, no tokenizer library.
- **`model*.ipynb` — deck models in torch.** Causal-transformer encoders over
  card-id sequences (`Embedding`/`EmbeddingBag`, `SwiGLU`, `MultiheadAttention`,
  residual `Sequential`, custom `triu`/`tril` index builders); `deck1inf.ipynb`
  completes a deck autoregressively from three copies of a seed card while
  respecting the 40/15 split and per-card banlist counts.
- **`deck*.ipynb`** — dataset prep: alias resolution, extra-deck type mask
  (`0x4802040`), copy limits from `lflist.conf`, card co-occurrence matrices.
- **`stat.py`** — race/attribute distributions and cross-tabulations as YAML.
- **`cdb2.sql`** — rebuilds a YGOPRO-compatible `cards.cdb` (`datas` + `texts`,
  type/race/attribute bitmasks, link arrows) from a legacy text dump.
- **`clipdown.sh` / `ypicdown.sh`** — clipboard capture of `.ydk` text into
  timestamped files, and a CDN sweep that downloads the card images a database
  references but `pics/` lacks.

## Architecture

Two independent pipelines that share one sqlite database and one embedding model:

```
cards.cdb (YGOPRO sqlite)          DINOv2 f16 GGUF
        |                                  |
        |                    gg.py  (ctypes -> libggml/gguf, graph + gallocr)
        |                                  |
        |                           dino2.py (CLS + patch grid)
        |                             /          \
  stat.py, name/                cut.py        cutvid.py
  tok/deck*/model*            (card art)     (video frames)
  notebooks                       |               |
        |                    cropped images   shot .webp
  ydkrename.py -> YYMMDD<archetype>/<hash>.ydk
```

Nothing imports the notebooks; they are snapshots of experiments, each loading a
`cards.*` / `deck*` pickle from the previous step. The only shared code is
`gg.py`, which `dino2.py` builds the model in and `cutvid.py` imports from it.

```
ygo/
├── gg.py            # ggml/gguf ctypes layer: Tensor, ops, graph, allocator
├── dino2.py         # DINOv2 ViT forward pass in ggml; writes *.dino2.*f tensors
├── cut.py           # PCA autocrop of card art
├── cutvid.py        # stream -> dedupe -> embed -> k-means -> one crop per shot
├── ydkrename.py     # n-gram deck naming and archival
├── stat.py          # race/attribute statistics as YAML
├── cdb2.sql         # rebuild a cards.cdb from a text dump
├── clipdown.sh
├── ypicdown.sh
├── deck.7z          # compressed deck corpus (the notebooks expect it unpacked as deck/)
├── deck*.ipynb      # dataset prep and the first deck models
├── model*.ipynb     # transformer deck models; *inf = inference only
├── name.ipynb, tok.ipynb, nomodel.ipynb
├── docs/project-status.md
└── LICENSE
```

## How to develop

There is no build, no test suite, no linter config and no CI — by design: this is
an archive, so all you can run is what you see. The one automated check that
exists is syntax:

```bash
python3 -m py_compile gg.py dino2.py cut.py cutvid.py ydkrename.py stat.py
```

Verification is manual: run an artifact against your own `cards.cdb`, GGUF and
images and look at the output (`stat.py` prints YAML; `cut.py` rewrites the image
in place — work on copies; `PCA=1` makes `dino2.py` emit a visualisation).

Python here is written with PEP 695 generics and heavy expression-level
unpacking (`gg.py`, `ydkrename.py`, `cutvid.py`), so 3.12+ is required for those
files; the dependencies are whatever the imports say, unpinned and untracked.

## How to contribute

This is a personal archive — see [CONTRIBUTING.md](CONTRIBUTING.md). Issues that
point at a real bug with a reproducing input are welcome; feature requests are
not, because nothing here is maintained.

## Roadmap

**Implemented**

- GGML/GGUF tensor layer and DINOv2 inference from Python (`gg.py`, `dino2.py`)
- Card-art autocrop and video shot selection on DINOv2 embeddings (`cut.py`, `cutvid.py`)
- Deck naming/archival from card text n-grams (`ydkrename.py`, `name.ipynb`)
- Deck dataset prep, statistics and `cards.cdb` rebuild (`deck*.ipynb`, `stat.py`, `cdb2.sql`)
- Transformer deck models and autoregressive deck completion (`model*.ipynb`, `deck1inf.ipynb`)

**Planned: nothing.** The repository is an unmaintained dump of finished
experiments, kept because the questions and the answers are still interesting.
If it were ever picked up again, the obvious next steps would be to remove the
hard-coded GGUF path from `dino2.init()` and to give `deck1inf.ipynb` a real
deck-legality check instead of the main/extra counter it has now — but that is a
sketch, not a promise.

## License

[AGPL-3.0](LICENSE) — see `LICENSE`. The external data it depends on
(`cards.cdb`, card images from a third-party CDN, DINOv2 weights) carries its own
licensing and is not covered by this one.
