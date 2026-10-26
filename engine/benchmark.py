"""Reproducible visual benchmark; local metrics are not the official scorer."""

import argparse
import json
import os
from pathlib import Path
import time

from engine.resize import _assess, _graph, _preserve_composition, _render, _run_layout, _score
from engine.validation import schema_problems, validate_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("input"))
    parser.add_argument("--output", type=Path, default=Path("output/benchmark"))
    parser.add_argument("--vlm", choices=("off", "required"), default="off")
    parser.add_argument("--formats", nargs="*", help="Optional target sizes, e.g. 1600x400 728x90")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((Path.cwd() / "output").resolve()):
        parser.error("Benchmark artifacts must be below output/")
    formats = []
    for value in args.formats or []:
        try:
            w, h = map(int, value.lower().split("x"))
            if min(w, h) <= 0:
                raise ValueError()
            formats.append(dict(width=w, height=h))
        except ValueError:
            parser.error(f"Invalid format: {value}")
    paths = sorted(args.input.glob("*.json"))
    if not paths:
        parser.error("No input task JSON files found")
    os.environ["CHAMELEON_VLM"] = args.vlm
    args.output.mkdir(parents=True, exist_ok=True)
    results, failures = [], []
    for path in paths:
        try:
            task = json.loads(path.read_text())
            source = task["source"]
            problems = schema_problems(source)
            if problems:
                raise ValueError("; ".join(problems))
            targets = formats or [task["target_canvas"]]
        except Exception as exc:
            failures.append(dict(case=path.name, error=str(exc)))
            continue
        for target in targets:
            name = f"{path.stem}_{target['width']}x{target['height']}"
            folder = args.output / name
            folder.mkdir(parents=True, exist_ok=True)
            try:
                started = time.monotonic()
                selected, report = _run_layout(source, target, folder)
                problems = validate_output(source, selected, target)
                if problems:
                    raise ValueError("; ".join(problems))
                graph = _graph(source)
                baseline = _preserve_composition(source, target)
                _render(source, folder / "source.png")
                _render(baseline, folder / "baseline.png")
                _render(selected, folder / "selected.png")
                fonts = [e["font_size"] for e in selected["elements"] if e["type"] == "text" and e["content"]]
                results.append(dict(case=name, selected=report["selected"], mode=args.vlm,
                    vlm_used=report["vlm_used"], model_digest=report["model_digest"],
                    model_calls=report["model_calls"], minimum_font_size=min(fonts) if fonts else None,
                    assessment=report["assessment"], baseline_assessment=_assess(source, baseline, graph),
                    baseline_score=_score(source, baseline, graph), selected_score=_score(source, selected, graph),
                    seconds=round(time.monotonic() - started, 3)))
            except Exception as exc:
                failures.append(dict(case=name, error=f"{type(exc).__name__}: {exc}"))
    summary = dict(mode=args.vlm, cases=results, failures=failures,
                   note="Heuristic metrics only; compare the rendered PNGs at actual size. No official score is inferred.")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = ["# Visual benchmark", "", summary["note"], "",
             "| Case | Layout | Minimum font | Small text warnings | Seconds |", "| --- | --- | --- | --- | --- |"]
    for row in results:
        font = "n/a" if row["minimum_font_size"] is None else f"{row['minimum_font_size']:.2f} px"
        lines.append(f"| [{row['case']}]({row['case']}/selected.png) | {row['selected']} | {font} | {len(row['assessment']['readability'])} | {row['seconds']} |")
    lines += ["", "Each case contains source.png, baseline.png, selected.png, selected.json and report.json."]
    if failures:
        lines += ["", "Failures:"] + [f"- {r['case']}: {r['error']}" for r in failures]
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    print(f"{len(results)} cases completed, {len(failures)} failures; {args.output / 'README.md'}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
