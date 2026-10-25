"""Composition-preserving resizing for nearby canvas shapes, without ML."""

from copy import deepcopy
from functools import lru_cache
from math import cos, radians, sin
from pathlib import Path
from typing import Any


def _scale_element(element: dict[str, Any], scale: float) -> None:
    """Scale geometry and dimensional styling together on a copied element."""
    for field in ("x", "y", "width", "height", "letter_spacing",
                  "corner_radius", "stroke_width"):
        if field in element:
            element[field] *= scale
    if element["type"] == "text":
        element["font_size"] = max(1.0, element["font_size"] * scale)


def _bounds(elements: list[dict[str, Any]], rotated: bool = False) -> tuple[float, ...]:
    """Return bounds, optionally including both nominal and rotated boxes."""
    left = top = float("inf")
    right = bottom = float("-inf")
    for element in elements:
        x, y = element["x"], element["y"]
        width, height = element["width"], element["height"]
        half_width, half_height = width / 2, height / 2
        if rotated:
            angle = radians(element.get("rotation", 0))
            half_width = max(half_width, (abs(cos(angle)) * width +
                                         abs(sin(angle)) * height) / 2)
            half_height = max(half_height, (abs(sin(angle)) * width +
                                           abs(cos(angle)) * height) / 2)
        cx, cy = x + width / 2, y + height / 2
        left, top = min(left, cx - half_width), min(top, cy - half_height)
        right, bottom = max(right, cx + half_width), max(bottom, cy + half_height)
    return left, top, right, bottom


def _anchor(position: float, length: float, canvas_length: float) -> float:
    travel = canvas_length - length
    if abs(travel) < 1e-9:
        return 0.5
    return min(1.0, max(0.0, position / travel))


def _aligned_offsets(
    bounds: list[tuple[float, ...]], occupied_bounds: list[tuple[float, ...]],
    source_length: float,
    target_length: float, scale: float, axis: int,
) -> list[float]:
    """Give units sharing an edge or center the same axis displacement.

    Match like features only, within one source pixel. Connected units use
    their combined bounds as an anchor, retaining existing small offsets
    rather than snapping them. Group membership is handled before this step.
    """
    parents = list(range(len(bounds)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for fraction in (0.0, 0.5, 1.0):
        features = sorted(
            (box[axis] + fraction * (box[axis + 2] - box[axis]), index)
            for index, box in enumerate(bounds)
        )
        start = 0
        for end in range(1, len(features)):
            if features[end][0] - features[start][0] <= 1.0:
                parents[root(features[end][1])] = root(features[start][1])
            else:
                start = end

    components: dict[int, list[int]] = {}
    for index in range(len(bounds)):
        components.setdefault(root(index), []).append(index)
    offsets = [0.0] * len(bounds)
    spare = max(0.0, target_length - scale * source_length)
    for members in components.values():
        first = min(bounds[index][axis] for index in members)
        last = max(bounds[index][axis + 2] for index in members)
        displacement = _anchor(first, last - first, source_length) * spare
        # Shift the whole alignment component when rotation or source bleed
        # needs extra clearance. Fall back to per-unit containment only if
        # no shared translation can fit all members at the current scale.
        minimum = max(-scale * occupied_bounds[index][axis] for index in members)
        maximum = min(target_length - scale * occupied_bounds[index][axis + 2]
                      for index in members)
        if minimum <= maximum:
            displacement = min(maximum, max(minimum, displacement))
        for index in members:
            offsets[index] = displacement
    return offsets


def _contain(elements: list[dict[str, Any]], width: float, height: float) -> None:
    """Fit a unit without changing its internal proportions or rotation."""
    left, top, right, bottom = _bounds(elements, rotated=True)
    factor = min(1.0, width / max(right - left, 1e-9),
                 height / max(bottom - top, 1e-9))
    if factor < 1.0:
        for element in elements:
            _scale_element(element, factor)
        left, top, right, bottom = _bounds(elements, rotated=True)
    dx = max(-left, min(0.0, width - right))
    dy = max(-top, min(0.0, height - bottom))
    for element in elements:
        element["x"] += dx
        element["y"] += dy


@lru_cache(maxsize=1)
def _font_manager() -> Any:
    """Use the renderer's brand fonts without downloading font data."""
    from renderer.converter.brand_fonts import load_brand_font_manager

    fonts = Path(__file__).resolve().parents[1] / "brand_kit" / "fonts"
    return load_brand_font_manager(str(fonts))


def _text_fits(element: dict[str, Any], size: float) -> bool:
    """Match renderer wrapping, then check line widths and estimated height."""
    from renderer.converter.wrap import char_widths, wrap_text

    content = element["content"]
    if not content:
        return True
    manager = _font_manager()
    family = element["font_family"]
    weight = element.get("font_weight", "regular")
    italic = element.get("font_style", "normal") == "italic"
    style = "bolditalic" if weight == "bold" and italic else (
        "bold" if weight == "bold" else "italic" if italic else "regular")
    widths = char_widths(manager, family, weight, style, size, content)
    wrapped = wrap_text(content, widths, element["width"])

    # Wrapping uses the base face in the adapter; rendered runs may be wider.
    rendered_widths = list(widths)
    for run in element.get("style_runs") or []:
        start = max(0, int(run["start"]))
        # The existing adapter applies run end indices inclusively.
        end = min(len(content), int(run["end"]) + 1)
        run_weight = run.get("font_weight", weight)
        run_italic = run.get("font_style", "italic" if italic else "normal") == "italic"
        run_style = "bolditalic" if run_weight == "bold" and run_italic else (
            "bold" if run_weight == "bold" else "italic" if run_italic else "regular")
        if start < end and ("font_weight" in run or "font_style" in run):
            rendered_widths[start:end] = char_widths(
                manager, family, run_weight, run_style, size, content[start:end])

    lines = wrapped.split("\n")
    max_lines = element.get("max_lines")
    if max_lines is not None and len(lines) > max_lines:
        return False
    required_height = len(lines) * size * element.get("line_height", 1.2) * 1.1
    if required_height > element["height"] + 1e-6:
        return False
    offset = 0
    spacing = max(0.0, element.get("letter_spacing", 0.0))
    for line in lines:
        line_width = sum(rendered_widths[offset:offset + len(line)])
        line_width += spacing * max(0, len(line) - 1)
        if line_width > element["width"] + 1e-6:
            return False
        offset += len(line) + 1
    return True


def _fit_text(element: dict[str, Any]) -> None:
    upper = element["font_size"]
    if _text_fits(element, upper):
        return
    lower = 1.0
    if not _text_fits(element, lower):
        element["font_size"] = lower
        return
    for _ in range(20):
        middle = (lower + upper) / 2
        if _text_fits(element, middle):
            lower = middle
        else:
            upper = middle
    element["font_size"] = lower


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    """Adapt a composition while retaining every element and original asset.

    Foreground units scale uniformly and retain shared edges and centers.
    This deliberately preserves composition; extreme aspect-ratio changes
    need a future reflow strategy. Text that cannot fit even at 1 px keeps
    its content rather than being truncated or hidden.
    """
    design = deepcopy(source)
    design["canvas"] = deepcopy(target_canvas)
    source_width, source_height = source["canvas"]["width"], source["canvas"]["height"]
    width, height = target_canvas["width"], target_canvas["height"]
    scale = min(width / source_width, height / source_height)
    elements = design["elements"]
    units: dict[tuple[str, Any], list[dict[str, Any]]] = {}

    for index, element in enumerate(elements):
        covers = (element["x"] <= 1 and element["y"] <= 1 and
                  element["x"] + element["width"] >= source_width - 1 and
                  element["y"] + element["height"] >= source_height - 1)
        below = all(element.get("z_index", 0) < other.get("z_index", 0)
                    for other in elements if other is not element)
        if (element["type"] == "image" and element.get("fit") == "cover"
                and covers and below and element.get("rotation", 0) % 360 == 0):
            element.update(x=0, y=0, width=width, height=height)
            continue
        group = element.get("group_id")
        key = ("group", group) if group is not None else ("element", index)
        units.setdefault(key, []).append(element)

    placement_units = list(units.values())
    bounds = [_bounds(unit) for unit in placement_units]
    occupied_bounds = [_bounds(unit, rotated=True) for unit in placement_units]
    x_offsets = _aligned_offsets(bounds, occupied_bounds, source_width, width, scale, axis=0)
    y_offsets = _aligned_offsets(bounds, occupied_bounds, source_height, height, scale, axis=1)
    for unit, dx, dy in zip(placement_units, x_offsets, y_offsets):
        for element in unit:
            _scale_element(element, scale)
            element["x"] += dx
            element["y"] += dy
        # Bounds take precedence if source bleed or rotation makes the shared
        # alignment infeasible. Never move an entire component off-canvas.
        _contain(unit, width, height)
        for element in unit:
            if element["type"] == "text":
                _fit_text(element)
    return design
