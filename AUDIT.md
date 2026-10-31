# CHAMELEON implementation review

## Submission preparation follow-up

Added a compact wide-banner candidate family with a full-width fine-print footer,
reduced control padding and additional body-copy allocation. On the supplied
Linux Level 3 render, body text is about 8.33 px, legal text 8 px and badge/CTA
labels about 9.02 px. The headline trades down to 11.86 px; the ribbon remains
7.57 px. Residual readability warnings remain. No content or assets are replaced.

Added a static Express viewer with original PNG previews, actual-size review,
file pairing, text-size inspection, downloads and SDK image insertion. Adobe CLI
build/package and browser upload/error-state checks pass. SDK insertion was
mock-tested; a real Adobe host/account test and organizer acceptance remain open.
The viewer has no npm runtime dependencies; Adobe CLI development dependencies
currently have upstream security advisories. The viewer is excluded from the
Python container. Docker explicitly defaults to VLM off. `engine.preflight`
checks existing output pairs and reports quality warnings separately.

Direct PDF review confirms that pages 8 and 14 require the engine, Dockerfile
and README for stage one; the full engine-connected add-on, pitch and video are
finale deliverables. The preview viewer is optional for stage one under that PDF.
The PDF explicitly leaves the submission destination/form to a later announcement.

The historical findings below describe earlier implementations. Use the current
README and SUBMISSION.md for run instructions and remaining submission questions.

## Implementation follow-up

The original findings below describe commit `26d9e43`. The working tree now
implements absolute readability targets and warnings, nine additional flexible
layout candidates, row packing for grouped controls, source-prominence image
allocation, sampled composited contrast, VLM relationship scoring and reading-order
placement, and revisions that can explore other layout families.

Independent schema/contract validation now gates outputs. Duplicate IDs,
nonfinite geometry, altered content, image distortion and rotated overflow are
rejected. The real renderer is required; task failures return a nonzero batch
status, remove stale results and allow other inputs to continue.

The README now documents the actual approach. The renderer revision, Python
dependencies and base-image digest are pinned. A visual benchmark CLI and an
independently authored museum design were added alongside eight new regression
tests. Those tests, the existing 38 fixtures and 100 seeded cases, mock VLM
protocol checks, and offline Docker processing of all four inputs passed.

For the supplied Level 3 target, headline size improved from 6.83 to 12.79 px,
body copy from 3.76 to 7.81 px, and legal copy from 2.05 to 5.68 px. The ribbon
remains approximately 7.34 px; badges and CTA are approximately 8.01 px. These
are measured improvements, not a claim of full readability or a winning score.
The independent benchmark completed four formats without failures, with minimum
font sizes of 16.50, 9.60, 13.50 and 8.00 px respectively.

Still pending: organizer model eligibility/size limits; real-VLM quality and
memory measurements; a model-bearing container with its runtime started inside
the container; broader human aesthetic evaluation; and any required Express
viewer/finale experience. The installed local models do not include the configured
vision model, so only the mock VLM was tested. Default execution remains offline
and model-free. The contrast metric is sampled rather than a full glyph-pixel
occlusion detector. Intentional layering remains conservative.

Reviewed commit `26d9e43` against the 14-page challenge PDF, RULES.md, FAQ.md,
the schema, engine, renderer adapter, and container configuration.

## Conclusion

There is a useful deterministic layout engine already implemented. Level 2 is
reasonably readable on the supplied example. Level 3 remains the main quality
gap: fitting every element inside the canvas currently counts as success even
when the text is too small to read. A VLM is optional under the brief; the
existing VLM integration is experimental and is not packaged for offline Docker.
Neither the available rubric nor these tests can establish a winning score.

## Requirements and ambiguities

- Preserve every input element ID exactly once, its text, font family, image
  asset, and brand styling. Reposition and resize; preserve image proportions.
- Return valid JSON for the requested canvas and a legible official-renderer PNG.
- Run offline in the standard container; install dependencies and any model
  weights during the build. Public adapted weights are required if fine-tuning.
- Do not regenerate asset pixels. PDF page 7 mentions regeneration as a finale
  idea, but page 12 explicitly prohibits it. Follow the explicit prohibition.
- VLMs are permitted, not required. The approved model list, model-size limit,
  and exact hardware budget are unresolved in the supplied documents. The
  configured Qwen model must not be assumed approved.
- PDF pages 8 and 14 require the engine and approach README at stage one, and
  the Express add-on at the finale. The standalone FAQ instead mentions a required
  stage-one viewer. Confirm this discrepancy with organizers before submission.
- No numeric official rubric weights are supplied. The weights in `_score()`
  are the implementation's own heuristic, not Adobe's scoring formula.

## Already implemented

| Capability | Evidence in engine/resize.py |
| --- | --- |
| Composition-preserving resize, group movement, rotated bounds | `_preserve_composition`, `_contain` |
| Font measurement and text fitting using renderer fonts | `_glyph_widths`, `_text_fits`, `_fit_text` |
| Geometry-based roles and explicit/inferred groups | `_graph` |
| Three reflow families, six parameter settings each, plus baseline | `_compose`, `_search` |
| Metadata, ID multiplicity, bounds and image-ratio checks | `_assess` |
| Collision, text-fit, contrast and hierarchy heuristics | `_assess`, `_score` |
| Optional local VLM interpretation, ranking and bounded revisions | `_LocalVLM`, `_interpret`, `_judge`, `_revisions` |
| Deterministic fallback, call/time budgets, ranking-order confirmation | `_run_layout` |
| Generated cases, permutation checks, geometry fuzzing, mock VLM tests | `_check_level2_tests` |

## Findings, in priority order

### 1. Tiny text is accepted as a good layout

The actual 728x90 Level 3 output has headline 6.83 px, subheading 3.76 px,
legal copy 2.05 px, and CTA 9.46 px. Both the orchestrator and internal assessment
report no defects. The rendered output confirms poor readability.

`_assess` checks whether text fits, but not whether a person can read it.
`_score` derives desired text sizes from the short canvas dimension; at 90 px,
supporting text's reference size is only 4.5 px. `_stack` shrinks whole stacks.

Implement role-sensitive readability penalties with absolute pixel-size checks,
and explicitly report unavoidable quality compromises. These thresholds are
engineering choices, not new competition rules. Search alternative arrangements
before reducing type sizes; never delete or rewrite text to satisfy a threshold.

### 2. Search cannot express enough arrangements

The 19 initial candidates come from only three related packing families.
Within each region, nodes are vertically stacked. Badge-heavy layouts and
long copy can consume height while other horizontal space remains underused.

Add row packing, flexible region boundaries, mixed rows/columns, grouping-aware
partitioning and local movement/size refinement. Allocate space by measured copy
requirements and visual importance. Preserve intended image/text relationships
and allow intentional overlap. Measure improvement on unseen layouts, not only
the supplied banner.

### 3. VLM support is not ready for submission

The default mode is `off`. The client expects Ollama at `127.0.0.1:11434`, while
the Dockerfile includes neither Ollama nor model weights nor server startup.
Host Ollama is not the container's loopback service under `--network none`.

`_interpret` stores relationships as `vlm_hint` edges, but placement and scoring
do not consume those hints. Role changes do influence packing. `_revisions`
adjusts only visual fraction, margin, gap and text scale; it cannot switch to a
new structural layout family during refinement or apply element-specific edits.

Use a permitted local VLM to interpret roles and critique rendered candidates,
while deterministic code enforces constraints. Connect relation hints to actual
placement objectives. Test real-model quality against VLM-off on the same held-out
cases, including reversed candidate order, runtime, memory and failure rates.
Mock protocol tests do not establish visual judgment quality. Package a model only
after that comparison demonstrates a benefit and organizers confirm eligibility.

### 4. Contrast and overlap assessment are incomplete

`_contrast` samples the background near the text center and handles containing
shapes. It does not evaluate every glyph region against the actual composited
backdrop, image texture, cover-crop mapping or partial overlays. `_assess` uses
bounding boxes and group exceptions rather than a complete model of intentional
layering. Strengthen raster-based readability checks and source relationship
preservation without changing original asset pixels.

### 5. Validation and robustness need broader coverage

The generated test set is mostly variants of one synthetic source. Its
`held_out` flag does not make those cases an independent visual benchmark.
The 100 random cases change target dimensions and font sizes but do not cover
arbitrary new design structures. Add independent compositions, many text boxes,
multiple products, dense copy, varied groups, rotated objects, and extreme formats.
Track schema validity separately from aesthetic/readability outcomes.

`main.py` uses sets for ID checking, can fall back to placeholder rendering on
ImportError, and returns success despite structural warnings. An uncaught task
error aborts the batch. Add submission preflight checks and meaningful error
reporting; do not accept a fallback PNG as evidence of visual correctness.

### 6. Submission documentation and reproducibility are unfinished

README.md still describes the starter uniform-stretch baseline. Document the
actual search, invariants, test commands, VLM modes, known Level 3 limits, and
offline execution. Pin the renderer Git revision and dependency versions after
validation so future builds reproduce the tested environment.

## Verification evidence

- Installed dependencies in a temporary environment; no application code changed.
- Ran `python -m engine.main` on all four input files using the real renderer.
- All four output documents passed Draft 2020-12 schema validation and exact ID
  multiplicity comparison; internal assessments reported no structural defects.
- Existing test suite passed all 38 generated fixtures, 100 seeded geometry
  cases, and mock VLM protocol/integration checks. The first sandboxed attempt
  could not bind the mock localhost server; the authorized rerun completed.
  Deliberately impossible fixtures retain expected text-overflow limitations.
  No real VLM was exercised.
- Built the linux/amd64 Python 3.11 container and ran all four inputs with
  `--network none`. All completed successfully. Docker output documents passed
  schema validation. Level 1 JSON matched local outputs; Levels 2 and 3 differed
  across the local macOS and Linux rendering environments. A nonfatal Fontconfig
  configuration warning appeared; the Level 3 PNG still rendered actual assets
  and text. This verifies the default deterministic path, not VLM packaging.
- Inspected all three official sample PNGs. Level 1 preserves a sparse composition;
  Level 2 is readable but could balance the product and badge better; Level 3 is
  difficult to read at its actual size.
- The missing Level 3 holiday badge is intentional. Preserve that fixture; do not
  invent a replacement asset.
- Local timings: first task 19.48 seconds including initial imports; subsequent
  tasks 0.01-0.57 seconds for resize calls. These are not reference-hardware results
  or complete end-to-end timings.
- Detailed generated artifacts are under ignored `output/audit/` and
  `output/level2_diagnostics/`.

## Recommended implementation sequence

1. Add a regression benchmark that exposes tiny text, image visibility and
   whitespace imbalance, including independently authored designs.
2. Improve extreme-aspect layout generation and readability-aware selection.
3. Wire semantic relationships into placement and compare real VLM-on/off outputs.
4. Finalize reproducible offline packaging, README and organizer clarifications.
5. For the finale, build Express integration with layout alternatives, locks,
   explanations and batch resize, while retaining the compliant stage-one engine.
