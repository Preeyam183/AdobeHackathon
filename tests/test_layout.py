"""Contract, readability and semantic-hint regressions using real font metrics."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from engine import main
from engine.resize import _candidate, _graph, _hint_order, _preserve_composition, _relationship_score, _revisions, _run_layout
from engine.validation import schema_problems, validate_output

ROOT = Path(__file__).resolve().parents[1]


def independent_poster() -> dict:
    return json.loads((ROOT / "tests/fixtures/museum.json").read_text())["source"]


class LayoutTests(unittest.TestCase):
    def test_extreme_sample_readability_and_contract(self):
        task = json.loads((ROOT / "input/chameleon_sample_level3_advanced.json").read_text())
        saved = deepcopy(task["source"])
        result, report = _run_layout(task["source"], task["target_canvas"], render=False)
        self.assertEqual(validate_output(saved, result, task["target_canvas"]), [])
        self.assertEqual(task["source"], saved)
        fonts = {e["id"]: e["font_size"] for e in result["elements"] if e["type"] == "text"}
        self.assertGreaterEqual(fonts["headline"], 11.5)
        self.assertGreaterEqual(fonts["subhead"], 8)
        self.assertGreaterEqual(fonts["legal_text"], 8)
        self.assertGreaterEqual(min(fonts[k] for k in ("badge1_label", "badge2_label", "badge3_label", "cta_label")), 8)
        self.assertEqual(report["assessment"]["defects"], [])
        self.assertTrue(report["assessment"]["readability"], "Residual tiny text must be reported")

    def test_banner_layout_does_not_depend_on_element_names(self):
        task = json.loads((ROOT / "input/chameleon_sample_level3_advanced.json").read_text())
        source = deepcopy(task["source"])
        expected, _ = _run_layout(source, task["target_canvas"], render=False)
        mapping = {e["id"]: f"item-{i}" for i, e in enumerate(source["elements"])}
        for e in source["elements"]:
            e["id"] = mapping[e["id"]]
        source["elements"].reverse()
        actual, _ = _run_layout(source, task["target_canvas"], render=False)
        self.assertEqual(validate_output(source, actual, task["target_canvas"]), [])
        indexed = {e["id"]: e for e in actual["elements"]}
        for e in expected["elements"]:
            self.assertEqual(dict(e, id=mapping[e["id"]]), indexed[mapping[e["id"]]])

    def test_independent_composition_across_formats(self):
        source = independent_poster()
        for target in (dict(width=1200, height=300), dict(width=480, height=960), dict(width=900, height=900)):
            with self.subTest(target=target):
                result, report = _run_layout(source, target, render=False)
                self.assertEqual(validate_output(source, result, target), [])
                self.assertEqual(report["assessment"]["defects"], [])
                self.assertGreaterEqual(min(e["font_size"] for e in result["elements"] if e["type"] == "text"), 8)

    def test_duplicate_ids_and_metadata_are_rejected(self):
        source = independent_poster()
        duplicate = deepcopy(source)
        duplicate["elements"].append(deepcopy(duplicate["elements"][1]))
        self.assertIn("duplicate element IDs", schema_problems(duplicate))
        changed = deepcopy(source)
        changed["elements"][1]["content"] = "Rewritten copy"
        self.assertTrue(any("immutable" in p for p in validate_output(source, changed)))
        changed["elements"][1]["x"] = float("nan")
        self.assertTrue(any("nonfinite" in p for p in validate_output(source, changed)))

    def test_reading_order_hints_affect_placement_order_and_ignore_cycles(self):
        graph = _graph(independent_poster())
        nodes = graph["nodes"][:3]
        graph["edges"].extend([
            dict(kind="vlm_hint", first=2, second=0, relation="reading_order"),
            dict(kind="vlm_hint", first=0, second=2, relation="reading_order")])
        ordered = _hint_order(nodes, graph)
        self.assertLess(ordered.index(nodes[2]), ordered.index(nodes[0]))
        self.assertEqual(len(ordered), len(nodes))

    def test_relationship_alignment_changes_candidate_score(self):
        source = independent_poster()
        graph = _graph(source)
        graph["edges"] = [dict(kind="vlm_hint", first=0, second=1, relation="alignment")]
        aligned = _relationship_score(source, graph)
        changed = deepcopy(source)
        node_id = graph["nodes"][1]["members"][0]["id"]
        next(e for e in changed["elements"] if e["id"] == node_id)["x"] += 150
        self.assertGreater(aligned, _relationship_score(changed, graph))

    def test_vlm_can_reflow_a_preserved_incumbent(self):
        source, target = independent_poster(), dict(width=1200, height=300)
        graph = _graph(source)
        incumbent = _candidate(source, _preserve_composition(source, target), graph, "preserved", -1, {})
        issues = [dict(candidate="preserved", parameter="text", delta=.1)]
        proposals = _revisions(source, target, graph, incumbent, issues, 0, set())
        self.assertTrue(proposals)
        self.assertTrue(all(p["family"] >= 3 for p in proposals))
        self.assertTrue(all(not validate_output(source, p["design"], target) for p in proposals))

    def test_batch_continues_after_bad_input_and_clears_stale_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs, outputs = root / "input", root / "output"
            inputs.mkdir(); outputs.mkdir()
            (inputs / "a.json").write_text("invalid json")
            (inputs / "b.json").write_text(json.dumps(dict(source=independent_poster(), target_canvas=dict(width=900, height=1200))))
            (outputs / "a_output.json").write_text("stale")
            with patch.object(main, "INPUT_DIR", inputs), patch.object(main, "OUTPUT_DIR", outputs), \
                    patch.object(main, "resize", side_effect=lambda source, target: deepcopy(source)), \
                    patch.object(main, "render_to_png", return_value=[]):
                self.assertEqual(main.main(), 1)
            self.assertFalse((outputs / "a_output.json").exists())
            self.assertTrue((outputs / "b_output.json").exists())

    def test_renderer_failure_is_not_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "task.json").write_text(json.dumps(dict(source=independent_poster(), target_canvas=dict(width=900, height=1200))))
            with patch.object(main, "INPUT_DIR", root), patch.object(main, "OUTPUT_DIR", root / "out"), \
                    patch.object(main, "resize", side_effect=lambda source, target: deepcopy(source)), \
                    patch.object(main, "render_to_png", side_effect=ImportError("renderer missing")):
                self.assertEqual(main.main(), 1)
            self.assertFalse((root / "out/task_output.json").exists())


if __name__ == "__main__":
    unittest.main()
