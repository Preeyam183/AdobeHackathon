# Submission preparation

The submission consists of the Python engine, Dockerfile, schema, dependency lock,
renderer adapter, brand assets/fonts and approach README. The optional Express
viewer lives in `viewer/`; build its ZIP for previewing or an additional deliverable. Keep generated
outputs, local environments and node_modules out of source control.

## Reproduce the checks

```sh
docker build --platform linux/amd64 -t my-chameleon-engine .
docker run --rm --platform linux/amd64 --network none \
  -v "$PWD/input:/app/input:ro" -v "$PWD/output:/app/output" my-chameleon-engine
docker run --rm --platform linux/amd64 --network none \
  -v "$PWD/input:/app/input:ro" -v "$PWD/output:/app/output:ro" \
  --entrypoint python my-chameleon-engine -m engine.preflight
```

Default `CHAMELEON_VLM=off` is explicit in Docker. No model, server or runtime
HTTP access is required. Do not enable VLM mode in this submission image.
A successful preflight verifies local artifacts, not an official score. Readability
warnings remain visible even when structural checks pass. Inspect each PNG at
actual size in the viewer. The Level 3 missing holiday badge is intentional.

## Requirements verified against the challenge PDF

Reviewed `Final%20Chameleon%20Challenge.pdf` directly, including rendered pages 8
and 14. Section 5.1 (page 8) requires the engine and an approach/run README for
stage one. The PDF's own FAQ Q5 (page 14) specifies code, Dockerfile and README
before code freeze. The full engine-connected Express add-on, short pitch and
brief demo video are finale deliverables (pages 7-8).

Accordingly, the viewer is an optional stage-one extra, not a blocker under this
PDF. The older standalone FAQ mentions a stage-one viewer; the PDF's own FAQ
omits that requirement. Follow the supplied PDF for this checklist unless a newer
organizer announcement says otherwise. This viewer previews already-generated
outputs; it is not yet the full engine-connected finale experience.

Page 11 requires a standard linux/amd64 container, build-time dependencies and
network-disabled runtime. It evaluates both JSON and rendered PNG. Page 12 fixes
text content/font families and prohibits asset-pixel regeneration. Page 13 points
to the repository schema for exact JSON fields. All four supplied tasks pass the
local contract and PNG checks; hidden-design quality is not an official score.

Page 14 explicitly says the destination and submission form will be announced
separately. No portal URL, exact form fields, deadline date or upload limit is
specified in this PDF; it has no interactive form fields. Those details therefore
remain pending, rather than missing implementation work.

Before pressing Submit, verify the actual portal's requested fields:

- Team name, members, contact and eligibility details.
- Repository URL, reviewer access, frozen commit and any required image/archive URL.
- Required deliverables and whether the viewer ZIP or a demonstration link is needed.
- Deadline and timezone, upload limits and exact submission format.
- Any required licensing, attribution, originality or model-use declarations.

These are preparation prompts, not a claim that these fields appear in the form.
Use the portal's exact wording and supply real team details. No form values have
been invented or submitted. The engine uses no model by default and does not
regenerate asset pixels. Dependencies retain their own licenses.

## Final viewer acceptance check

Build/package instructions are in `viewer/README.md`. In a real Adobe Express
session, load one output JSON/PNG pair, switch to actual size, check warnings,
insert the PNG, and confirm its appearance on the page. Browser automation tests
the SDK boundary with a mock; a real Adobe account is needed for this final check.
