"""
Orchestrator — you shouldn't need to edit this file. It:
  1. Reads every *.json task file from /app/input (source + target_canvas)
  2. Calls your resize() from resize.py
  3. Validates the submission contract (NOT the official scoring;
     see RULES.md for what the judge actually checks)
  4. Writes <task>_output.json and <task>_output.png to /app/output

Mirrors the same input/ -> output/ convention as the earlier India-program
challenges: drop task files in, get one JSON + one PNG out per task, no
manual steps in between.
"""

import json
import sys
import time
from pathlib import Path

from engine.resize import resize

# Relative to the current working directory, which is /app both inside the
# Docker image (WORKDIR /app) and when run locally from this repo's root -
# so this works identically in both "python -m engine.main" (no Docker) and
# the containerized entrypoint.
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
ASSETS_DIR = Path("brand_kit/assets")

def self_check(source: dict, output: dict, target_canvas: dict | None = None) -> list[str]:
    from engine.validation import validate_output
    return validate_output(source, output, target_canvas)


def render_to_png(design: dict, assets_dir: Path, png_path: Path) -> list[str]:
    """Require real rendering; a placeholder preview must not pass submission checks."""
    from renderer.converter.convert import render_design_to_png
    return render_design_to_png(design, assets_dir, str(png_path),
                                max_size=max(design["canvas"].values()))


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    task_files = sorted(INPUT_DIR.glob("*.json"))
    if not task_files:
        print(f"No task files found in {INPUT_DIR}", file=sys.stderr)
        return 1

    exit_code = 0
    from engine.validation import schema_problems
    for task_path in task_files:
        stem = task_path.stem
        json_path = OUTPUT_DIR / f"{stem}_output.json"
        png_path = OUTPUT_DIR / f"{stem}_output.png"
        # Remove stale generated results so a failed rerun cannot look successful.
        json_path.unlink(missing_ok=True)
        png_path.unlink(missing_ok=True)
        try:
            task = json.loads(task_path.read_text())
            source, target_canvas = task["source"], task["target_canvas"]
            problems = schema_problems(source)
            if problems:
                raise ValueError("; ".join(problems))
            t0 = time.monotonic()
            output = resize(source, target_canvas)
            problems = self_check(source, output, target_canvas)
            if problems:
                raise ValueError("; ".join(problems))
            warnings = render_to_png(output, ASSETS_DIR, png_path)
            json_path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
            elapsed = time.monotonic() - t0
            print(f"[{task_path.name}] OK in {elapsed:.2f}s -> {json_path.name}, {png_path.name}")
            for warning in warnings:
                print(f"    - Renderer: {warning}")
        except Exception as exc:
            json_path.unlink(missing_ok=True)
            png_path.unlink(missing_ok=True)
            exit_code = 1
            print(f"[{task_path.name}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
