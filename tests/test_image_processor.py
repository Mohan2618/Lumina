import unittest

from PIL import Image

from lumina.image_processing.processor import OPERATIONS, process_image


EXPECTED_OPERATIONS = {
    "rotate", "flip", "resize", "resize_pct", "crop", "thumbnail",
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


if __name__ == "__main__":
    unittest.main()
