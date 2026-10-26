"""Submission contract checks, independent of the layout search heuristics."""

from collections import Counter
from functools import lru_cache
import json
from math import cos, isfinite, radians, sin
from pathlib import Path

from jsonschema import Draft202012Validator


GEOMETRY = {"x", "y", "width", "height", "font_size", "letter_spacing", "corner_radius", "stroke_width"}


@lru_cache(maxsize=1)
def schema_validator() -> Draft202012Validator:
    path = Path(__file__).resolve().parents[1] / "chameleon_design_schema_v0.9.json"
    return Draft202012Validator(json.loads(path.read_text()))


def schema_problems(design: dict) -> list[str]:
    problems = [f"schema {'.'.join(map(str, e.absolute_path))}: {e.message}"
                for e in schema_validator().iter_errors(design)]
    if problems:
        return problems
    ids = [e["id"] for e in design["elements"]]
    if len(ids) != len(set(ids)):
        problems.append("duplicate element IDs")
    for e in design["elements"]:
        for key, value in e.items():
            if isinstance(value, (int, float)) and not isfinite(value):
                problems.append(f"nonfinite {key}: {e['id']}")
    return problems


def validate_output(source: dict, output: dict, target: dict | None = None) -> list[str]:
    problems = schema_problems(output)
    if problems:
        return problems
    if target is not None and output["canvas"] != target:
        problems.append("output canvas does not match target")
    original = {e["id"]: e for e in source["elements"]}
    if Counter(e["id"] for e in source["elements"]) != Counter(e["id"] for e in output["elements"]):
        problems.append("element ID multiplicity mismatch")
    w, h = output["canvas"]["width"], output["canvas"]["height"]
    sw, sh = source["canvas"]["width"], source["canvas"]["height"]
    for e in output["elements"]:
        old = original.get(e["id"])
        if old is None:
            continue
        if {k: v for k, v in old.items() if k not in GEOMETRY} != {k: v for k, v in e.items() if k not in GEOMETRY}:
            problems.append("changed immutable properties: " + e["id"])
        angle = radians(e.get("rotation", 0))
        cx, cy = e["x"] + e["width"] / 2, e["y"] + e["height"] / 2
        for dx in (-e["width"] / 2, e["width"] / 2):
            for dy in (-e["height"] / 2, e["height"] / 2):
                x, y = cx + cos(angle) * dx - sin(angle) * dy, cy + sin(angle) * dx + cos(angle) * dy
                if not (-1e-5 <= x <= w + 1e-5 and -1e-5 <= y <= h + 1e-5):
                    problems.append("off-canvas rotated bounds: " + e["id"])
                    break
        if e["type"] in {"text", "image"} and min(e["width"], e["height"]) <= 0:
            problems.append("nonpositive rendered dimensions: " + e["id"])
        background = (old["type"] == "image" and old.get("fit") == "cover" and
                      old["x"] <= 1 and old["y"] <= 1 and old["x"] + old["width"] >= sw - 1 and
                      old["y"] + old["height"] >= sh - 1 and not old.get("rotation", 0) % 360 and
                      all(old.get("z_index", 0) <= a.get("z_index", 0) for a in source["elements"]))
        if e["type"] == "image" and not background:
            delta = abs(e["width"] * old["height"] - e["height"] * old["width"])
            if delta > 1e-5 * max(1, old["width"] * old["height"]):
                problems.append("changed image proportions: " + e["id"])
    return list(dict.fromkeys(problems))
