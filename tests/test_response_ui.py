import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ResponseUITests(unittest.TestCase):
    def test_response_enhancement_asset_exists(self):
        asset = ROOT / "static" / "response_enhancements.js"
        self.assertTrue(asset.is_file())
        self.assertGreater(asset.stat().st_size, 1000)

    def test_response_enhancement_contains_required_artifacts(self):
        source = (ROOT / "static" / "response_enhancements.js").read_text(encoding="utf-8")
        for marker in (
            "lumina-artifact",
            "lumina-copy",
            "lumina-save",
            "lumina-run",
            "lumina-table-copy",
            "window.fetch",
        ):
            self.assertIn(marker, source)

    def test_html_loader_is_injected_by_app(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("response_enhancements.js?v=1", source)
        self.assertIn("@app.after_request", source)


if __name__ == "__main__":
    unittest.main()
