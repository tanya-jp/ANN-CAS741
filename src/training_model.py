"""
Module for neural network training model representation.

This module defines the TrainingModel class, which is used for constructing and 
managing the structure of a neural network for training purposes. The class includes
definitions for the number of layers in the network, the number of neurons in each layer, 
and functionalities for initializing gradients used in network training.

Classes:
    TrainingModel: Encapsulates the structure and components of a neural network model,
                   including initialization and management of network gradients.
"""
import numpy as np

# pylint: disable=too-few-public-methods
class TrainingModel():
    """
    A class representing a neural network training model.

    This class encapsulates the structure and necessary components of a neural network model 
    for training purposes. It includes functionalities to initialize and manage various 
    aspects of the network such as layer structure and gradient initialization.

    Attributes:
        LAYERS_NUMBER (int): The number of layers in the neural network, including 
        the input, hidden, and output layers.
        LAYERS_NEURONS (list): A list containing the number of neurons in each layer of the network.

    Methods:
        create_gradients_zeros: Initializes gradients for backpropagation as zero arrays.
    """

    LAYERS_NUMBER = 5
    LAYERS_NEURONS = [1024, 256, 128, 32, 10]

    def __init__(self):
        return

    def create_gradients_zeros(self):
        """
        Initialize zero gradients for each layer in the neural network.

        Creates and returns a list of numpy arrays, each initialized to zeros. 
        These arrays represent the gradients for the 'a' activations, 
        the 'W' weights, and the 'b' biases for each layer in 
        the network. The arrays are structured to match the size requirements 
        of each corresponding layer.

        Returns:
            list: A list of numpy zero arrays representing the gradients 
                  for the neural network layers.
                  The order of gradients in the list is as follows: gradients 
                  for 'a' activations of hidden layers
                  (in reverse order), gradients for 'W' weights (in reverse order), 
                  and gradients for 'b' biases (in reverse order).
        """

        grads = []

        # a
        for i in range (self.LAYERS_NUMBER-2, 0, -1):
            grads.append(np.zeros((self.LAYERS_NEURONS[i], 1)))

        # W
        for i in range (self.LAYERS_NUMBER-1, 0, -1):
            grads.append(np.zeros((self.LAYERS_NEURONS[i], self.LAYERS_NEURONS[i-1])))

        # b
        for i in range (self.LAYERS_NUMBER-1, 0, -1):
            grads.append(np.zeros((self.LAYERS_NEURONS[i], 1)))

        # For a network with 4 layers (one input, two hedden layers, one output)
        # it will be grad_a2, grad_a1, grad_W3, grad_W2, grad_W1, grad_b3, grad_b2, grad_b1
        return grads
