import unittest

from lumina.services.ai_service import detect_op


class AIRoutingTests(unittest.TestCase):
    def assert_intent(self, prompt, expected):
        result = detect_op(prompt)
        self.assertIsNotNone(result, prompt)
        self.assertIn(f'"intent":"{expected}"', result, prompt)

    def test_existing_image_edits_do_not_route_to_generation(self):
        edit_prompts = [
            "Make the image brighter",
            "Brighten this image",
            "Make this image darker",
            "Increase the brightness",
            "Reduce the brightness",
            "Make the image sharper",
            "Increase the contrast",
            "Make the colors more vibrant",
            "Convert this image to grayscale",
            "Blur the image",
            "Turn the image into a cartoon",
            "Make this image a 4K desktop wallpaper",
        ]
        for prompt in edit_prompts:
            with self.subTest(prompt=prompt):
                result = detect_op(prompt, has_existing_image=True)
                self.assertIsNotNone(result)
                self.assertNotIn('"intent":"generate_image"', result)

    def test_generation_requests_still_route_to_generation(self):
        generation_prompts = [
            "Generate an image of a mountain at sunset",
            "Create an image of a futuristic city",
            "Make an image of a golden retriever on the beach",
            "Draw a fantasy castle",
            "Show me a picture of a red sports car",
        ]
        for prompt in generation_prompts:
            with self.subTest(prompt=prompt):
                self.assert_intent(prompt, "generate_image")

    def test_common_edit_intents(self):
        cases = {
            "Make the image brighter": "brightness",
            "Make the image darker": "brightness",
            "Increase the contrast": "contrast",
            "Make the image sharper": "sharpen",
            "Make the colors more vibrant": "saturation",
            "Convert this image to grayscale": "grayscale",
            "Blur the image": "blur",
            "Make it brighter": "brightness",
            "Darken it": "brightness",
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                result = detect_op(prompt, has_existing_image=True)
                self.assertIsNotNone(result, prompt)
                self.assertIn(f'\"intent\":\"{expected}\"', result, prompt)
