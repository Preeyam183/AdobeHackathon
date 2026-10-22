# Repository Guidelines

## Project Structure & Module Organization

CHAMELEON is a Python design-resizing engine. Implement layout logic in `engine/resize.py`; `engine/main.py` loads tasks, runs structural checks, and writes JSON and PNG outputs. `renderer/converter/` contains the vendored schema-to-renderer adapter. `brand_kit/assets/` and `brand_kit/fonts/` supply images and fonts. Three difficulty-level fixtures live in `input/`; generated artifacts go in ignored `output/`.

Consult `chameleon_design_schema_v0.9.json` for the data contract and `RULES.md` for scoring constraints. Prefer the current README and implementation when older FAQ setup instructions differ.

## Build, Test, and Development Commands

Run from the repository root; paths are relative to the working directory. Use Python 3.11 to match Docker.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m engine.main
```

The final command processes every `input/*.json`, producing `<task>_output.json` and `<task>_output.png` in `output/`.

```bash
docker build --platform linux/amd64 -t my-chameleon-engine .
docker run --rm --network none \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" my-chameleon-engine
```

These commands build and exercise the offline submission environment. Add dependencies to `requirements.txt`; install dependencies and any model weights during image build.

## Coding Style & Naming Conventions

Follow existing Python conventions: four-space indentation, `snake_case` functions and variables, `UPPER_SNAKE_CASE` constants, type annotations, and concise docstrings. Keep resizing changes focused on `resize(source, target_canvas)` and avoid mutating source data. No formatter or linter configuration is currently provided.

## Testing Guidelines

There is no automated test suite or coverage threshold. Run all three sample tasks and inspect both JSON and rendered PNGs. Structural warnings do not fail the process and are not full schema validation; investigate them, accounting for the Level 2 decorative bleed. Confirm previews use real fonts and assets rather than fallback boxes. Preserve the intentionally missing Level 3 badge fixture.

## Commit & Pull Request Guidelines

The short Git history uses imperative, sentence-case subjects, such as “Flatten repo to starter root, drop dummy-submission.” Follow that style. PRs should describe the layout change, list validation commands, link relevant issues, and include before/after previews for visual changes. Keep generated outputs uncommitted.

## Design Constraints

Preserve every element ID exactly once, image proportions, text content, and font families. Reposition and resize existing assets; never regenerate or replace their pixel content. Keep runtime execution offline.
