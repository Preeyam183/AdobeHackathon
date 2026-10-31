"""Verify existing engine artifacts before packaging a submission."""

import argparse
import json
from pathlib import Path

from PIL import Image

from engine.resize import _assess, _graph
from engine.validation import schema_problems, validate_output


def check_task(task_path: Path, output_dir: Path) -> dict:
    """Check the schema, preservation contract, real PNG dimensions and quality."""
    result = dict(task=task_path.name, errors=[], warnings=[])
    try:
        task = json.loads(task_path.read_text())
        source, target = task["source"], task["target_canvas"]
        problems = schema_problems(source)
        if problems:
            raise ValueError("; ".join(problems))
        stem = task_path.stem + "_output"
        design = json.loads((output_dir / (stem + ".json")).read_text())
        result["errors"].extend(validate_output(source, design, target))
        with Image.open(output_dir / (stem + ".png")) as png:
            if png.format != "PNG" or png.size != (target["width"], target["height"]):
                result["errors"].append("PNG format or dimensions do not match target")
            png.verify()
        if not result["errors"]:
            assessment = _assess(source, design, _graph(source))
            result["warnings"].extend(assessment["defects"] + assessment["readability"])
            result["font_sizes"] = {e["id"]: round(e["font_size"], 3)
                                    for e in design["elements"] if e["type"] == "text"}
            for e in design["elements"]:
                if e["type"] == "image" and not (Path("brand_kit/assets") / e["asset_id"]).is_file():
                    result["warnings"].append("missing asset: " + e["asset_id"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["errors"].append(str(exc))
    result["status"] = "FAIL" if result["errors"] else "PASS"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("input"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    tasks = sorted(args.input.glob("*.json"))
    results = [check_task(path, args.output) for path in tasks]
    passed = bool(results) and all(r["status"] == "PASS" for r in results)
    report = dict(status="PASS" if passed else "FAIL", tasks=results,
                  scope="Local artifact checks only; visual review and organizer submission requirements remain separate.")
    serialized = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized)
    print(serialized, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
