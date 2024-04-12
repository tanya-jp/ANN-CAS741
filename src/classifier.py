"""
Module for classifying images using a pretrained model.

This module integrates image preprocessing and classification by using a 
pretrained model to classify images into predefined categories. It uses the 
InputPrep class for preprocessing and the Model class
for loading the model and performing classifications.

Classes:
    Classifier: Manages the image classification process.
"""
from model import Model
from input_prep import InputPrep

class Classifier():
    """
    A class that manages the process of classifying images into predefined categories.

    This class handles the image preprocessing using the InputPrep class and uses a pretrained
    model for classifying the processed images.

    Attributes:
        input_image (array): The processed image ready for classification.
        class_name (str): The name of the class to which the image is classified.
        classes (dict): A dictionary mapping class indices to their respective names.
    """
    def __init__(self):
        """
        Initializes the Classifier object with a mapping of class indices to class names.
        """
        # self.model = None
        self.input_image = None
        self.class_name = None
        self.classes = {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck"
        }

    def set_image_pixel(self):
        """
        Prepares the image for classification by preprocessing it using the InputPrep class.
        """
        input_prep = InputPrep()
        self.input_image = input_prep.get_input()

    def get_class(self):
        """
        Classifies the preprocessed image using a pretrained model and 
        maps the output to a class name.

        Returns:
            str: The name of the class to which the image has been classified.
        """
        m = Model()
        model_name = 'trained_params.npy'
        predicted_class = m.load_trained_classifier(self.input_image, model_name)
        self.class_name = self.classes[predicted_class]
        return self.class_name
