"""CPU-only release staging integrity and artifact rejection checks."""
import json
from pathlib import Path
import tempfile
import unittest

from reproducibility import release
from reproducibility.common import sha256


class ReleaseTests(unittest.TestCase):
    def test_forbidden_paths_and_assets(self):
        for value in ("../outside.py", "/absolute.py", ".git/config", "weights.pth", "data.mdb", "crop.png", "run.log", "test_predictions.csv"):
            with self.assertRaises(ValueError):
                release.validate_name(value)
        release.validate_name("reproducibility/reference/TABLE1_EXPECTED.csv")

    def test_staged_manifest_detects_change_omission_and_extra(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); source = p / "source"; source.mkdir()
            a = source / "a.py"; a.write_text("pass\n")
            (p / "EXPORT_FILES.json").write_text(json.dumps([dict(path="a.py",bytes=a.stat().st_size,sha256=sha256(a))]))
            self.assertEqual(release.verify(p),1)
            a.write_text("fail\n")
            with self.assertRaises(ValueError):release.verify(p)
            a.write_text("pass\n"); extra = source / "extra.py";extra.write_text("pass\n")
            with self.assertRaises(ValueError):release.verify(p)
            extra.unlink();a.unlink()
            with self.assertRaises(ValueError):release.verify(p)

    def test_no_history_or_weight_rewriter(self):
        self.assertIn("misc4torch/translate_weights.py", release.EXCLUDED)
        self.assertIn("PUBLICATION_DECISIONS.md", release.EXCLUDED)


if __name__ == "__main__":
    unittest.main()
