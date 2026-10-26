# CHAMELEON

An offline design-resizing engine that preserves every element, original asset,
text string, font family and brand style while adapting layout to a new canvas.

## Run

Use Python 3.11 to match the submission container.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m engine.main
```

Each task in `input/` produces a schema-checked JSON document and an actual
renderer PNG in `output/`. Invalid tasks or renderer failures produce a nonzero
exit status; remaining tasks continue. Missing image assets use the official
renderer placeholder and produce a warning. The missing Level 3 holiday badge
is intentional: do not add or generate it.

## Layout approach

The engine infers roles and relationships from geometry, typography, stacking
and explicit groups; it does not use element names as semantic labels. Nearby
canvas shapes preserve the composition when it fits cleanly. Larger changes
search 28 initial candidates: the original composition, 18 original reflow
variants, and 9 flexible variants. Flexible variants allocate space to images,
copy and grouped controls, and use dynamic programming to arrange groups in rows.
Text is measured with the same fonts and wrapping code as the renderer.

Selection first rejects broken structure and minimizes collisions/overflow.
Among valid candidates, it scores readability, hierarchy, alignment, occupied
space and semantic relationships. Readability has absolute pixel-size targets
and a penalty for the least readable text, so a 2 px paragraph is not treated
as successful just because it fits. Contrast samples multiple positions against
composited image/shape layers, respecting image fit, rotation and opacity.
This remains a sampled heuristic, not a complete pixel-level aesthetic judge.

The schema and independent output validator enforce exact ID multiplicity,
requested dimensions, finite numbers, rotated bounds, immutable properties,
and image proportions. Background `cover` boxes can change shape while the
renderer preserves asset pixels through proportional scaling and cropping.

Diagnostics, candidate previews and unresolved readability warnings are saved
under `output/level2_diagnostics/`. Diagnostics are not official scores.

## Offline container

```bash
docker build --platform linux/amd64 -t my-chameleon-engine .
docker run --rm --platform linux/amd64 --network none \
  -v "$(pwd)/input:/app/input:ro" \
  -v "$(pwd)/output:/app/output" my-chameleon-engine
```

The default configuration uses no model and makes no runtime HTTP calls. The
base image digest and renderer Git revision are pinned; Docker installs the
Python dependency lock in `requirements.lock.txt`. Regenerate that lock after
intentional dependency updates. OS package repositories and Python build tools
are not snapshot-locked; preserve the tested container digest for submission.

## Optional local VLM

`CHAMELEON_VLM=off` is the default. `auto` attempts a local VLM and falls back on
failure; `required` reports model failures rather than masking them. Configure
`CHAMELEON_VLM_MODEL` to an already-installed vision model and optionally set
`CHAMELEON_VLM_BUDGET` (seconds, default 600). The client only contacts
`127.0.0.1:11434`; it never pulls models. It checks installed model metadata and
vision capability before generation.

The VLM may interpret roles and relationships, rank rendered candidates, and
request bounded revisions. Reading-order hints affect flexible packing;
alignment and existing layering hints affect selection. Cyclic reading-order
hints are ignored. Revisions can try another layout family. Explicit groups and
immutable content always take precedence. Candidate ordering is reversed for
confirmation to catch positional bias, with at most seven generation calls.

The current default model name in the code is experimental, not a claim of
organizer approval. The approved model list and size limit are still pending.
The standard Docker image does **not** contain Ollama or weights. Before enabling
VLMs in a submission, package the approved runtime and all weights at build time,
start the server inside the same container, then test with `--network none`.
A host-side Ollama service is not that container's loopback server. Publicly
accessible adapted weights are required for reproducibility. See the
[Ollama server configuration documentation](https://docs.ollama.com/faq).

## Verification and visual benchmark

```bash
python -m unittest discover -s tests -v
python -m engine.resize --check-level2-tests output/regression
python -m engine.benchmark --input input --output output/benchmark
python -m engine.benchmark --input tests/fixtures \
  --output output/independent --formats 1200x300 480x960 900x900 728x90
```

The regression suite covers 38 generated fixtures, 100 seeded geometry cases,
ID/permutation invariance and mock VLM protocol failures. Additional tests cover
an independently authored museum poster, the extreme supplied sample, model
relationship hints, immutable content and batch error handling. Mock tests do
not establish real VLM quality.

The benchmark writes source, composition-preserving baseline and selected PNGs,
a comparison table, timings, structural results, readability warnings and model
usage. Inspect images at their actual size. After an approved vision model is
available, repeat the same benchmark with `--vlm required` and a different output
directory; compare visual quality, failure rate, runtime and memory before
choosing a submission mode.

## Current limitations

The 728x90 sample contains far more copy and badges than can comfortably fit.
The updated layout improves headline/body sizes and reports remaining small
text, but does not make that sample fully readable. No text is deleted or
rewritten to hide the problem. Human comparison across more independent designs
and real VLM evaluation remain necessary. There is no verified official score.

The challenge PDF places the Express add-on in the finale, while the standalone
FAQ mentions a stage-one viewer. Confirm that discrepancy with organizers.
Asset regeneration is prohibited by the explicit rules. See `RULES.md` and
`AUDIT.md` for the full review and implementation status.
