"""
Module for managing machine learning models.

This module contains the `Model` class responsible for saving and loading trained model 
parameters. It integrates with the `TrainTest` class from the `train_and_test` module 
for training processes and manages the model data using NumPy array files.

Classes:
    Model: Manages the functionality to save and load trained model parameters using .npy files.
"""
import numpy as np

from train_and_test import TrainTest

class Model():
    """
    Handles the saving and loading of trained model parameters.

    This class provides methods to save the parameters of a trained model into a .npy file
    and to load these parameters from a file. It utilizes the `TrainTest` class for training and
    obtaining the necessary parameters.

    Methods:
        save_model: Saves the trained model parameters to 'model.npy'.
        load_model: Loads model parameters from a specified .npy file.
        load_trained_classifier: Loads a trained classifier and uses it 
                                to predict a class based on the input image.
    """

    def __init__(self):
        self.training = TrainTest()

    def save_model(self):
        """
        Save the parameters based on trained model using `TrainTest`.
        Save the model to a file named 'model.npy'. 
        If an error occurs during file handling, the error is printed and raised.

        Returns:
        bool: True if the model parameters are successfully saved; False otherwise.
        """

        trained_params, total_costs_vectorized, start_time, end_time = self.training.train()
        print(type(trained_params))
        print(f"Training lasted {end_time - start_time} seconds")
        self.training.result(total_costs_vectorized, trained_params)
        # Saving the npy to a file
        try:
            np.save('trained_params.npy', trained_params)
            return True
        except PermissionError as pe:
            print(f"Permission error: {pe}")
            raise
        except IOError as io:
            print(f"I/O error: {io}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise
    def load_model(self, file_name):
        """
        Load model parameters from a specified npy file.

        Attempts to open and read the specified file, returning the model parameters stored in it.
        Raises custom exceptions for file not found and invalid npy format.

        Parameters:
        file_name (str): The name of the file to load the model parameters from.

        Returns:
        dict: The model parameters stored in the file.

        Raises:
        FileReadError: If the file does not exist or an error occurs during file reading.
        """
        try:
            trained_params = np.load(file_name, allow_pickle=True)
            trained_params_dict = trained_params.item()
            return trained_params_dict
        except FileNotFoundError:
            print("File not found. Please check the file path.")
            raise
        except Exception as e: #pylint: disable=W0718
            print("An error occurred while loading the array:")
            print(e)
            raise

    def load_trained_classifier(self, input_image, model_name):
        """
        Loads a trained classifier and uses it to predict a class for the input image.

        Parameters:
            input_image: The input image data.
            model_name (str): The filename of the trained model parameters.

        Returns:
            The predicted class.
        """
        parameters = self.load_model(model_name)
        predicted_class = self.training.calculate_percentage_of_accuracy(
                            input_image, parameters, input_image=True)
        return self.training.calculate_percentage_of_accuracy(
                            input_image, parameters, input_image=True)
