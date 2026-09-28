"""Unit tests for agent utilities and image handling in main.py."""

import os
import tempfile

import pytest

from main import encode_image


class TestAgentUtilities:
    """Test suite for agent multimodal input handling."""

    def test_encode_image_existing_file(self):
        # Test with the repository's sample image
        if os.path.isfile("images.jpg"):
            b64_str, mime_type = encode_image("images.jpg")
            assert isinstance(b64_str, str)
            assert len(b64_str) > 0
            assert mime_type in ["image/jpeg", "image/jpg"]

    def test_encode_image_with_temp_png(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(b"\x89PNG\r\n\x1a\nfake png header")
            tmp_path = tmp.name

        try:
            b64_str, mime_type = encode_image(tmp_path)
            assert isinstance(b64_str, str)
            assert mime_type == "image/png"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_encode_image_non_existent_file_raises_error(self):
        with pytest.raises(FileNotFoundError):
            encode_image("this_file_does_not_exist_12345.jpg")

    def test_encode_image_strips_quotes(self):
        if os.path.isfile("images.jpg"):
            quoted_path = '"images.jpg"'
            b64_str, mime_type = encode_image(quoted_path)
            assert len(b64_str) > 0
