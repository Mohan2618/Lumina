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


    def test_system_prompt_requires_structured_complete_answers(self):
        prompt = (ROOT / "lumina" / "prompts" / "system_prompt.py").read_text(encoding="utf-8")
        for requirement in (
            "Answer the user's actual question directly",
            "Never stop in the middle of a sentence, code block, list item, or explanation",
            "Avoid duplicated examples, repeated headings, malformed Markdown",
            "one complete runnable code block",
            "Time and space complexity",
            "state meaning, transition derivation, base cases",
        ):
            self.assertIn(requirement, prompt)

    def test_frontend_markdown_renderer_handles_fenced_code_and_lists(self):
        source = (ROOT / "frontend" / "src" / "App.jsx").read_text(encoding="utf-8")
        self.assertIn(r"const fence = /", source)
        self.assertIn(r"const numbered = raw.match", source)
        self.assertIn("function StructuredResponse", source)
        self.assertIn("function MarkdownText", source)

    def test_html_loader_is_injected_by_app(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("response_enhancements.js?v=1", source)
        self.assertIn("@app.after_request", source)


if __name__ == "__main__":
    unittest.main()
