"""
Module for preparing image input for processing.

This module provides the InputPrep class, which handles the 
preparation of image data for further processing or analysis. 
It involves setting the image, converting it to grayscale, 
normalizing the pixel values, and flattening the image data.

Classes:
    InputPrep: Handles the preprocessing steps for image data.
"""
import numpy as np

from input_image import Input
from data import Data

class InputPrep():
    """
    A class to preprocess image input.

    This class encompasses methods to load an image, convert it to grayscale, 
    normalize its pixels, and flatten the image data. It utilizes functionalities 
    provided by the Input and Data classes.

    Attributes:
        input_image (array): Stores the image data.
        processor (Data): An instance of the Data class for image processing operations.

    Methods:
        set_image_pixel: Loads an image using the Input class.
        rgb2gray: Converts the image to grayscale.
        prep_pixels: Normalizes the pixel values of the image.
        flat_data: Flattens the image data into a 1D vector.
        get_input: Executes the preprocessing steps and returns the processed image.
    """

    def __init__(self):
        self.input_image = []
        self.processor = Data()

    def set_image_pixel(self):
        """
        Load an image using the Input class and store it.
        """
        input_module = Input()
        self.input_image = input_module.set_image()

    def rgb2gray(self):
        """
        Convert the input image to grayscale to reduce its complexity.
        """
        self.input_image = self.processor.rgb2gray(self.input_image, True)

    def prep_pixels(self):
        """
        Normalize the pixel values of the image to a range between 0 and 1.
        """
        self.input_image = self.processor.prep_pixels(self.input_image)

    def flat_data(self):
        """
        Flatten the image data from a 2D matrix to a 1D vector.
        """
        self.input_image = self.input_image.reshape(1024, 1)

    def get_input(self):
        """
        Execute the preprocessing steps and return the processed image.

        This method sequentially calls other methods to load, convert to grayscale, 
        normalize, and flatten the image data.

        Returns:
            array: The preprocessed image data as a 1D vector.
        """
        self.set_image_pixel()
        self.rgb2gray()
        self.prep_pixels()
        self.flat_data()
        return self.input_image

if __name__ == '__main__':
    inp = InputPrep()
    print(np.shape(inp.get_input()))
    