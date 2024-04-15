"""
Module for handling output operations related to a classification process.

This module provides the Output class which manages the output procedures 
of a classification model.
It includes methods for setting the classified name, appending feedback 
to a file, and saving user feedback.

Classes:
    Output: Manages output operations for the classification model.
"""
from classifier import Classifier

class Output():
    """
    Handles output operations and user feedback for a classified image.

    This class interacts with a Classifier to determine the class of an image and handles
    user feedback regarding the accuracy of the classification. 
    It supports appending feedback to a file.

    Attributes:
        classifier (Classifier): An instance of the Classifier class 
                                to handle image classification.
        class_name (str): The name of the class identified by the classifier.
    """
    def __init__(self):
        """
        Initializes the Output class with a Classifier instance.
        """
        self.classifier = Classifier()
        self.class_name = None

    def set_class_name(self):
        """
        Determines the class name of an image using the classifier.

        Returns:
            str: The class name determined by the classifier.
        """
        self.classifier.set_image_pixel()
        self.class_name = self.classifier.get_class()
        return self.class_name

    def append_to_file(self, file_path, sentence):
        """
        Appends a given sentence to a file.

        Parameters:
            file_path (str): The path to the file where the sentence will be appended.
            sentence (str): The sentence to append to the file.

        Prints a success message or error message depending on 
        the outcome of the file operation.
        """
        try:
            # Open the file in append mode ('a')
            with open(file_path, 'a', encoding='utf-8') as file:
                file.write(sentence + '\n')  # Add a newline after the sentence
            print("Sentence added successfully.")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error: The file was not found. Original exception: {e}")
        except IOError as e:
            raise IOError(f"Error: An I/O error occurred while writing to the file. Original exception: {e}")
        except Exception as e: #pylint: disable=W0718
            print(f"An unexpected error occurred: {e}")

    def save_feedback(self):
        """
        Collects and saves user feedback on the classification result to a text file.

        Prompts the user to agree or disagree with the classification result, and if disagreed,
        allows the user to provide the expected class. The result is appended to a feedback file.
        """
        feedback = None
        expected_class = self.class_name
        acceptable_classes = ["airplane", "automobile", "bird", "cat",
                            "deer", "dog", "frog", "horse", "ship", "truck"]
        while feedback not in ["y", "n"]:
            feedback = input("Do you agree with the predicted class?: (y/n) ")
            feedback = feedback.lower()

        if feedback == "n":
            print("supported classes:")
            print(acceptable_classes)
            expected_class = input("What was the expected class? ")
            expected_class = expected_class.lower()

            while expected_class not in acceptable_classes:
                print("supported classes:")
                print(acceptable_classes)
                expected_class = input("Please type an acceptable class? ")
                expected_class = expected_class.lower()
        self.append_to_file("feedback.txt", str(self.class_name) + " " + str(expected_class))

