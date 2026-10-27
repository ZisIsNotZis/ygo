# Contributing

This repository is a personal archive of Yu-Gi-Oh! deck and card-art
experiments. It is **not maintained**: there is no release, no support window and
no plan to accept features. Read the [README](README.md) first — especially the
note that the data these scripts need (`cards.cdb`, a DINOv2 GGUF, a `deck/`
corpus) is not in the repository.

## Scope

In scope: a real bug in one of the scripts, reported with the input that triggers
it and the traceback.

Out of scope: packaging, dependency pinning, refactors, new models, API design,
"make it a library", and anything that assumes a test suite exists where none
does.

## Running checks

There is no test suite, no linter config and no CI. The only automated check is
syntax:

```bash
python3 -m py_compile gg.py dino2.py cut.py cutvid.py ydkrename.py stat.py
```

Beyond that, verification is manual and uses your own data. `cut.py` rewrites
images in place and `ydkrename.py` renames files, so run both on copies.

## Pull requests

- Small, single-purpose commits with a Conventional Commit subject
  (`fix:`, `docs:`, `chore:`).
- Do not add CI workflows, lockfiles or generated artifacts.
- If your change needs a new dependency, say so in the description; nothing here
  is pinned, so a new import is a real cost.
- Expect a slow reply, or none.

## Issues

Include: the script or notebook and cell, the Python version, the traceback, and
the smallest input that reproduces it. Report packaging/licensing problems with
the upstream projects that own the data (`cards.cdb`, card images, DINOv2
weights) to those projects; this repository only reads them.
