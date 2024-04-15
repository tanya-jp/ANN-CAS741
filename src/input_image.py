"""
Module for image input handling.

This module provides the Input class, designed for loading and validating
 image data from a user-provided file path. 
It ensures that the image meets certain criteria such as file format and dimensions. 
The module is geared towards  preprocessing images for further analysis
 or processing in image-based applications.

Classes:
    Input: Handles the loading and validation of images based
         on predefined specifications.
"""
import os
import matplotlib.pyplot as plt

class Input():
    # pylint: disable=too-few-public-methods
    """
    A class to handle image input for processing.

    This class allows for loading an image from a specified path. It checks the existence,
    format, and size of the image to ensure it meets predefined criteria.

    Attributes:
        HEIGHT (int): Expected height of the image in pixels.
        WIDTH (int): Expected width of the image in pixels.
        IMAGE_FORMAT (list): List of acceptable image file formats.

    Methods:
        set_image: Loads an image from a given file path and performs checks on its properties.
    """

    HEIGHT = 32
    WIDTH = 32
    IMAGE_FORMAT = [".PNG", ".JPEG", ".JPG"]

    def set_image(self):
        """
        Loads an image from the user-provided path and validates its format and size.

        This method prompts the user to enter an image file path, then checks the file's
        existence, format, and dimensions against the class's defined standards. It reads
        the image and performs validation checks, raising exceptions for any discrepancies.

        Returns:
            array: An array representation of the image data.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            Exception: If the file format or image size does not match the expected standards.
        """
        file_path = input("Enter directory: ")

        # Check if file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} was not found.")

        # Check file format
        if not any(str(file_path).upper().endswith(ext) for ext in self.IMAGE_FORMAT):
            raise Exception(f"The file format of {file_path} is not valid. Expected formats: {self.IMAGE_FORMAT}") # pylint: disable=C0301, W0719

        # Check image size
        image_data = plt.imread(file_path)
        if (image_data.shape[:2] != (self.HEIGHT, self.WIDTH)
            or image_data.shape[:2] != (self.WIDTH, self.HEIGHT)):
            raise Exception(f"The size of the image {file_path} is invalid. Expected size: {self.HEIGHT} in {self.WIDTH}") # pylint: disable=C0301, W0719

        return image_data
