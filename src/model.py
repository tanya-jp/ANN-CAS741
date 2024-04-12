"""
Module for managing machine learning model.

This module contains the `Model` class that handles the saving and loading 
of trained model parameters. It interacts with the `TrainTest` class from 
the `train_and_test` module for the training process and manages the model 
data using np array files.

Classes:
    Model: Provides functionalities to save and load the trained model parameters.
"""
import numpy as np

from train_and_test import TrainTest

class Model():
    """
    A class for handling the saving and loading of trained model parameters.

    This class provides methods to save the parameters of a trained model into a npy file and
    to load these parameters from a file. It utilizes the `TrainTest` class for training and
    obtaining the necessary parameters.

    Methods:
    save_model: Trains a model using `TrainTest` and saves the parameters to 'model.npy'.
    load_model: Loads and returns model parameters from a given noy file.
    """

    def __init__(self):
        self.training = TrainTest()
        return

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
        except Exception as e:
            print(f"An error occurred: {e}")
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
        except Exception as e:
            print("An error occurred while loading the array:")
            print(e)
        
    def load_trained_classifier(self, input_image, model_name):
        parameters = self.load_model(model_name)
        predicted_class = self.training.calculate_percentage_of_accuracy(input_image, parameters, input_image = True)
        return predicted_class

if __name__ == '__main__':
    # model = Model()

    # model.save_model()

    arr = np.load('trained_params.npy', allow_pickle=True)
    # print(type(arr))
    # print(arr)
    # Assuming arr is a 0-dimensional ndarray containing a dictionary
    extracted_dict = arr.item()  # or arr[()]

    # Now, extracted_dict is your dictionary
    print(list(extracted_dict.keys()))  # This should output <class 'dict'>
    
