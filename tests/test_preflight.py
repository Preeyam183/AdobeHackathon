"""Submission preflight must reject stale or mismatched deliverables."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from engine.preflight import check_task


class PreflightTests(unittest.TestCase):
    def test_valid_pair_and_bad_artifacts(self):
        source = dict(canvas=dict(width=64, height=64), elements=[dict(
            id="copy", type="text", x=0, y=0, width=64, height=30,
            content="Hello", font_family="Solstice Sans", font_size=12,
            color="rgba(255,255,255,1)")])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_path = root / "sample.json"
            task_path.write_text(json.dumps(dict(source=source, target_canvas=source["canvas"])))
            output_path = root / "sample_output.json"
            output_path.write_text(json.dumps(source))
            png_path = root / "sample_output.png"
            Image.new("RGB", (64, 64)).save(png_path)
            self.assertEqual(check_task(task_path, root)["status"], "PASS")
            changed = deepcopy(source)
            changed["elements"][0]["content"] = "Changed"
            output_path.write_text(json.dumps(changed))
            self.assertEqual(check_task(task_path, root)["status"], "FAIL")
            output_path.write_text(json.dumps(source))
            Image.new("RGB", (32, 64)).save(png_path)
            self.assertEqual(check_task(task_path, root)["status"], "FAIL")
            png_path.write_bytes(b"not a PNG")
            self.assertEqual(check_task(task_path, root)["status"], "FAIL")
            png_path.unlink()
            self.assertEqual(check_task(task_path, root)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
