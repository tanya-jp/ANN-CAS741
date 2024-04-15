"""
This module tests the Input class functionality from the input_image module. 
It uses the pytest framework along with unittest.mock's patch decorator to 
simulate various file input scenarios. The test cases
verify if the Input class correctly handles image files according 
to specified constraints on file existence,
file format, and image dimensions.

Attributes:
    TestInput (class): A class used to group together methods that test 
    the Input class's functionality.

Methods:
    test_valid_input_jpg: Tests that a valid JPEG image meets the required 
                        format and size constraints.
    test_valid_input_png: Tests that a valid PNG image meets the required 
                        format and size constraints.
    test_invalid_format: Tests the response of the system when an image 
                        of an unsupported format is processed.
    test_invalid_format_and_size: Tests the system's response when an 
                                image fails both format and size checks.
    test_nonexistent_file: Verifies that a FileNotFoundError is 
                        raised for a non-existent file path.
    test_invalid_size_png: Checks how the system handles a PNG image 
                        that does not meet the size requirements.
    test_invalid_size_jpg: Checks how the system handles a JPEG image 
                        that does not meet the size requirements.
"""
import pytest
import os
import numpy as np
import sys
sys.path.insert(0, '../src')  # Add src directory to Python's search path

from input_image import Input  # Import the Input class
from unittest.mock import patch

class TestInput:
    @patch("builtins.input", return_value="image_test_cases/valid_image.jpg")
    @patch("os.path.isfile", return_value=True)
    @patch("matplotlib.pyplot.imread", return_value=np.random.rand(32, 32, 3))  # Mocking a valid image size
    def test_valid_input_jpg(self, mock_input, mock_isfile, mock_imread):
        """
        Tests that a valid JPEG image is correctly identified and passes the format and size checks.
        """
        input_instance = Input()
        result = input_instance.set_image()
        assert result.shape == (32, 32, 3)  # Test returns correct shape

    @patch("builtins.input", return_value="image_test_cases/valid_image.png")
    @patch("os.path.isfile", return_value=True)
    @patch("matplotlib.pyplot.imread", return_value=np.random.rand(32, 32, 3))  # Mocking a valid image size
    def test_valid_input_png(self, mock_input, mock_isfile, mock_imread):
        """
        Tests that a valid PNG image is correctly identified and passes the format and size checks.
        """
        input_instance = Input()
        result = input_instance.set_image()
        assert result.shape == (32, 32, 3)  # Test returns correct shape

    @patch("builtins.input", return_value="image_test_cases/invalid_image.gif")
    @patch("os.path.isfile", return_value=True)
    def test_invalid_format(self, mock_input, mock_isfile):
        """
        Tests the system's response when an unsupported image format (GIF) is used.
        """
        input_instance = Input()
        with pytest.raises(Exception) as excinfo:
            result = input_instance.set_image()
        assert "file format" in str(excinfo.value)  # Check the correct exception message
    
    
    @patch("builtins.input", return_value="image_test_cases/invalid_image_size.gif")
    @patch("os.path.isfile", return_value=True)
    def test_invalid_format_and_size(self, mock_input, mock_isfile):
        """
        Tests how the system handles an image failing both format and size validations.
        """
        input_instance = Input()
        with pytest.raises(Exception) as excinfo:
            result = input_instance.set_image()
        assert "file format" in str(excinfo.value)  # Check the correct exception message

    @patch("builtins.input", return_value="image_test_cases/nonexistent_image.jpg")
    @patch("os.path.isfile", return_value=False)
    def test_nonexistent_file(self, mock_input, mock_isfile):
        """
        Verifies that the system correctly handles the case of a non-existent file path.
        """
        input_instance = Input()
        with pytest.raises(FileNotFoundError) as excinfo:
            result = input_instance.set_image()
        assert "was not found" in str(excinfo.value)  # Check the correct exception message

    @patch("builtins.input", return_value="image_test_cases/wrong_size_image.png")
    @patch("os.path.isfile", return_value=True)
    @patch("matplotlib.pyplot.imread", return_value=np.random.rand(64, 64, 3))  # Mocking an incorrect image size
    def test_invalid_size_png(self, mock_input, mock_isfile, mock_imread):
        """
        Tests how the system handles an image failing the size validation on pngs.
        """
        input_instance = Input()
        with pytest.raises(Exception) as excinfo:
            result = input_instance.set_image()
        assert "size of the image" in str(excinfo.value)  # Check the correct exception message

    @patch("builtins.input", return_value="image_test_cases/wrong_size_image.jpg")
    @patch("os.path.isfile", return_value=True)
    @patch("matplotlib.pyplot.imread", return_value=np.random.rand(64, 64, 3))  # Mocking an incorrect image size
    def test_invalid_size_jpg(self, mock_input, mock_isfile, mock_imread):
        """
        Tests how the system handles an image failing the size validation on jpgs.
        """
        input_instance = Input()
        with pytest.raises(Exception) as excinfo:
            result = input_instance.set_image()
        assert "size of the image" in str(excinfo.value)  # Check the correct exception message


