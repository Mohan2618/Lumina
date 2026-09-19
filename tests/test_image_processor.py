import unittest

from PIL import Image

from lumina.image_processing.processor import OPERATIONS, process_image


EXPECTED_OPERATIONS = {
    "rotate", "flip", "resize", "resize_pct", "crop", "thumbnail", "wallpaper_4k",
    "grayscale", "invert", "sepia", "blur", "sharpen", "edge", "emboss",
    "contrast", "brightness", "saturation", "hue", "white_balance",
    "shadows_highlights", "hdr", "sketch", "cartoon", "watercolor",
    "oil_painting", "neon_glow", "glitch", "halftone", "lomo",
    "cross_process", "duotone", "pop_art", "stained_glass", "pointillism",
    "ascii_art", "thermal_vision", "double_exposure", "tilt_shift", "bokeh",
    "fisheye", "mosaic", "pixelate", "noise", "vignette", "clahe", "denoise",
    "xray_enhance", "mri_enhance", "ct_enhance", "fundus_enhance", "skin_analyze",
    "wound_analyze", "segment", "morphology", "sobel", "canny", "face_detect",
    "color_palette", "histogram_eq", "quality_check", "super_resolution", "deblur",
    "colorize_bw", "restore_old",
}


class ImageProcessorTests(unittest.TestCase):
    def test_all_existing_operations_are_registered(self):
        self.assertTrue(EXPECTED_OPERATIONS.issubset(OPERATIONS.keys()))

    def test_basic_operation(self):
        image = Image.new("RGB", (64, 32), (120, 80, 40))
        result = process_image(image, "grayscale", {})
        self.assertIsNotNone(result)
        self.assertEqual(result.size, image.size)

    def test_every_registered_operation_executes(self):
        image = Image.new("RGB", (128, 96), (120, 80, 40))
        for intent in sorted(EXPECTED_OPERATIONS):
            with self.subTest(intent=intent):
                result = process_image(image, intent, {})
                self.assertIsNotNone(result, f"{intent} returned None")
                self.assertIsInstance(result, Image.Image, f"{intent} did not return a PIL image")
                self.assertGreater(result.width, 0)
                self.assertGreater(result.height, 0)

    def test_wallpaper_keeps_4k_canvas(self):
        image = Image.new("RGB", (800, 600), (120, 80, 40))
        result = process_image(image, "wallpaper_4k", {})
        self.assertEqual(result.size, (3840, 2160))


if __name__ == "__main__":
    unittest.main()
