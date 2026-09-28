import io
import struct
import unittest
import zlib

from PIL import Image

from lumina.utils.image import ImageValidationError, bytes_to_pil, file_to_pil


class ImageValidationTests(unittest.TestCase):
    def make_png(self, size=(32, 32)):
        buf = io.BytesIO()
        Image.new("RGB", size, (20, 40, 60)).save(buf, format="PNG")
        return buf.getvalue()

    def test_valid_image_is_decoded(self):
        result = bytes_to_pil(self.make_png())
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (32, 32))

    def test_invalid_bytes_are_rejected(self):
        with self.assertRaises(ImageValidationError):
            bytes_to_pil(b"not an image")

    def test_oversized_pixel_count_is_rejected(self):
        width, height = 7000, 6000
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        chunk = struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF)
        png = b"\x89PNG\r\n\x1a\n" + chunk + b"\x00\x00\x00\x00IEND\xaeB\x60\x82"
        with self.assertRaises(ImageValidationError):
            bytes_to_pil(png)

    def test_file_upload_uses_same_validation(self):
        from werkzeug.datastructures import FileStorage
        result = file_to_pil(FileStorage(stream=io.BytesIO(self.make_png()), filename="photo.png"))
        self.assertEqual(result.size, (32, 32))


if __name__ == "__main__":
    unittest.main()
