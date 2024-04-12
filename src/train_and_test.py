"""
Module for training and testing a neural network.

This module contains the TrainTest class responsible for training a neural network using
stochastic gradient descent and evaluating its performance. It involves initializing network
parameters, performing feedforward operations, backpropagation, and computing accuracy metrics.

Classes:
    TrainTest: Handles the training and testing of a neural network.
"""
import random
import time
import numpy as np

import matplotlib.pyplot as plt

from data import Data
from training_model import TrainingModel

class TrainTest():
    """
    Class responsible for training and testing a neural network model.

    This class encompasses methods for initializing neural network parameters, 
    training the network using stochastic gradient descent, and calculating the 
    accuracy of the model.

    Attributes:
        BATCH_SIZE (int): Number of samples per batch of training.
        LEARNING_RATE (float): Learning rate for the gradient descent optimization.
        EPOCHS (int): Number of training epochs.
        layers (list): List of layers in the neural network.
        gradients (list): Gradients used for training the network.
        training_model (TrainingModel): Instance of the TrainingModel class for 
                                        managing network gradients.
        train_set (list): Training dataset.
        test_data (list): Testing dataset.

    Methods:
        set_layers: Initializes the layers of the network.
        initialize_parameters: Initializes the weights and biases of the network.
        feed_forward: Performs the forward propagation through the network.
        extract_parameters: Extracts parameters from the network for backpropagation.
        backpropagation: Performs backpropagation to update the network parameters.
        sigmoid: Sigmoid activation function.
        sigmoid_deriv: Derivative of the sigmoid function.
        compute_cost: Computes the cost of the network predictions.
        train: Trains the neural network model.
        calculate_percentage_of_accuracy: Calculates the accuracy of the model.
    """

    BATCH_SIZE = 16
    LEARNING_RATE = 0.1
    EPOCHS = 2

    def __init__(self):
        self.layers = []
        self.gradients = []
        self.training_model = TrainingModel()
        self.gradients = self.training_model.create_gradients_zeros()

        self.train_set = None
        self.test_set = None
        self.set_layers()

    def set_layers(self):
        """
        Initializes the layers of the neural network based on the gradients.
        """
        # Number of layers = x + 1
        x = (len(self.gradients) + 1) // 3

        # Find number of neurons
        self.layers.append(len(self.gradients[2*x-2][1]))
        for i in range (2*x-2, x-2, -1):
            self.layers.append(len(self.gradients[i]))

    def initialize_parameters(self):
        """
        Initializes network parameters with random weights and zero biases.
        
        Returns:
            dict: Dictionary containing initialized weights and biases.
        """

        parameters = {}
        center = 0
        margin = 1
        for i in range(1, len(self.layers)):
            # Draw random samples from a normal (Gaussian) distribution
            s = (self.layers[i], self.layers[i-1])
            parameters['W'+str(i)] = np.random.normal(center, margin, size = s)
            # Zero bias vector
            parameters['b' + str(i)] = np.zeros((self.layers[i],1))

        return parameters

    def feed_forward(self, new_a, parameters):
        """
        Performs forward propagation through the network.
        Return output of the network from forward calculations

        Parameters:
            new_a: The input activations for the network.
            parameters: Dictionary of network parameters (weights and biases).

        Returns:
            tuple: Output of the network and cache with intermediate values for backpropagation.
        """
        caches = []

        # Claculate forward process for each layer
        for l in range (1, len(self.layers)):
            prev_a = new_a
            # Extract weight and biase from the list of parameters
            w = parameters['W' + str(l)]
            b = parameters['b' + str(l)]
            # New a is calculated based on the formula, using sigmoid as activation function
            z = np.dot(w,prev_a) + b
            new_a = self.sigmoid(z)

            # Cache -> ((a, W, b), z)
            cache = ((prev_a, w, b), z)
            caches.append(cache)

        return new_a, caches

    def extract_parameters(self, caches):
        """
        Extracts parameters from caches generated during feed forward.

        Parameters:
            caches: List of cached tuples containing network parameters.

        Returns:
            tuple: Extracted activations (a), weights (w), and linear transforms (z).
        """

        # Number of layers = x + 1
        x = (len(self.gradients) + 1) // 3

        # cache -> ((a, W, b), z)
        a = []
        for i in range (x):
            a.append(caches[i][0][0])

        w = []
        for i in range (x):
            w.append(caches[i][0][1])

        z = []
        for i in range (x):
            z.append(caches[i][1])

        return a, w, z

    def backpropagation(self, caches, output, y):
        """
        Performs the backpropagation algorithm to compute gradients.
        Calculate gradients of wights and biases

        Parameters:
            caches: Cached data from the forward pass.
            output: Output from the forward pass.
            y: Actual labels.

        Returns:
            dict: Gradients of network parameters.
        """

        # Number of layers = x + 1
        x = (len(self.gradients) + 1) // 3

        self.gradients = self.training_model.create_gradients_zeros()

        a, w, z = self.extract_parameters(caches)
        a.append(output)

        grad_a = []
        for i in range (x-2, -1, -1):
            grad_a.append(self.gradients[i])

        grad_w = []
        for i in range (2*x-2, x-2, -1):
            grad_w.append(self.gradients[i])

        grad_b = []
        for i in range (len(self.gradients)-1, 2*x-2, -1):
            grad_b.append(self.gradients[i])

        # grad_a2, grad_a1, grad_W3, grad_W2, grad_W1, grad_b2, grad_b1, grad_b0 = self.gradients
        # calculat gradients of out put of layer (a)

        grad_a[-1] += np.transpose(w[-1]) @ (2 * self.sigmoid_deriv(z[-1]) * (a[-1] - y))
        for i in range (len(grad_a)-2, -1, -1):
            grad_a[i] += np.transpose(w[i+1]) @ ( self.sigmoid_deriv(z[i+1]) * grad_a[i+1])

        # calculate gradients of weights
        grad_w[-1] += (2 * self.sigmoid_deriv(z[-1]) * (a[-1] - y)) @ (np.transpose(a[-2]))
        for i in range (len(grad_a)-2, -1, -1):
            grad_w[i] += (self.sigmoid_deriv(z[i]) * grad_a[i]) @ (np.transpose(a[i]))

        # calculate gradients of biases
        grad_b[-1] += (2 * self.sigmoid_deriv(z[-1]) * (a[-1] - y))
        for i in range (len(grad_b)-2, -1, -1):
            grad_b[i] += (self.sigmoid_deriv(z[i]) * grad_a[i])

        # define a dictionare
        # keys -> label of gradients
        # values -> gradients
        new_gradients = {}
        for i, grad in enumerate(grad_b, start=1):
            new_gradients["db" + str(i)] = grad
        for i, grad in enumerate(grad_w, start=1):
            new_gradients["dW" + str(i)] = grad

        return new_gradients

    def sigmoid(self, x):
        """
        Calculate the sigmoid function.
        
        Parameters:
        x (numpy array or float): The input value(s) for which the sigmoid function is calculated.

        Returns:
        numpy array or float: The sigmoid function output for the input value(s).
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, z):
        """
        Calculate the derivative of the sigmoid function.

        Parameters:
        z (numpy array): The input value(s) for which the derivative of 
        the sigmoid function is calculated.

        Returns:
        numpy array: The derivative of the sigmoid function at each element of the input array.
        """
        a = self.sigmoid(z)
        return a * (1 - a)

    def compute_cost(self, predicted, actual):
        """
        Calculate the sum of squared errors (SSE) cost between predicted and actual values.

        Parameters:
        predicted (numpy array): Predicted values outputted by the model.
        actual (numpy array): Actual values/labels corresponding to the data.

        Returns:
        float: The computed SSE cost.
        """
        cost = ((predicted - actual)**2).sum()
        return cost

    def train(self):
        """
        Train the neural network model using stochastic gradient descent.

        The training process involves initializing parameters, performing forward and backward 
        passes for each batch, and updating the parameters using the computed gradients. The 
        process is repeated for a specified number of epochs.

        Returns:
        tuple: The final trained parameters, cost for each epoch, 
               start time, and end time of training.
        """
        # pylint: disable=too-many-locals
        # save start time to caculate training time
        start_time = time.time()

        self.train_set, self.test_set = (Data()).get_dataset()

        # initialize W and b
        parameters = self.initialize_parameters()

        # Number of layers = x + 1
        x = (len(self.gradients) + 1) // 3

        total_costs = []
        for i in range(self.EPOCHS):
            # if i%10==0:
            print("EPOCH ", i)

            # data must be shuffled each epoch time
            random.shuffle(self.train_set)
            cost = 0
            train_set_size = len(self.train_set)
            batch_num = train_set_size//self.BATCH_SIZE

            for n in range(batch_num):
                # # make zero arrays for gradients
                # grads = self.training_model.create_gradients_zeros()
                # which batch of train set is using to train
                first_train_data = n * self.BATCH_SIZE
                last_train_data = (n+1) * self.BATCH_SIZE
                batch = self.train_set[first_train_data: last_train_data]

                all_gradients = {}

                for b in batch:

                    batch_data = b[0]
                    batch_label = b[1]
                    output, caches = self.feed_forward(batch_data, parameters)

                    gradients = self.backpropagation(caches, output, batch_label)

                    # # extract gradients and add them
                    # grad_b3 += gradients["db3"]
                    # grad_b2 += gradients["db2"]
                    # grad_b1 += gradients["db1"]
                    # grad_W3 += gradients["dW3"]
                    # grad_W2 += gradients["dW2"]
                    # grad_W1 += gradients["dW1"]

                    for i in range (x):
                        # Check if the key exists in all_gradients. If not, initialize it to 0.
                        if "db" + str(i + 1) not in all_gradients:
                            all_gradients["db" + str(i + 1)] = 0
                        all_gradients["db"+str(i+1)] += gradients["db"+str(i+1)]
                    for i in range (x):
                        # Check if the key exists in all_gradients. If not, initialize it to 0.
                        if "dW" + str(i + 1) not in all_gradients:
                            all_gradients["db" + str(i + 1)] = 0
                        all_gradients["dW"+str(i+1)] = gradients["dW"+str(i+1)]

                    # cost of this item in batch added to total cost oc this batch
                    cost += self.compute_cost(output, batch_label)

                # update parameters, weights and biases
                for l in range(1, len(self.layers)):
                    layer_index = str(l)
                    grad_w_update = self.LEARNING_RATE * all_gradients["dW" + layer_index]
                    grad_w_update /= self.BATCH_SIZE
                    parameters["W" + layer_index] -= grad_w_update

                    grad_b_update = self.LEARNING_RATE * all_gradients["db" + layer_index]
                    grad_b_update /= self.BATCH_SIZE
                    parameters["b" + layer_index] -= grad_b_update

            total_costs.append(cost/train_set_size)
            # save end time to caculate training time
            end_time = time.time()

        return parameters, total_costs, start_time, end_time

    def calculate_percentage_of_accuracy(self, data, parameters, input_image = False):
        """
        Calculate the accuracy of the neural network model on a given dataset or 
        find the predicted class of an input image.

        Accuracy is determined by comparing the predicted labels against the actual labels and 
        calculating the percentage of correct predictions.

        Parameters:
        data (list): The dataset for evaluation, consisting of 
                    data points and their corresponding labels.
        parameters (dict): The neural network parameters (weights and biases).
        input_image (bool): If True, treat `data` as a single image, else as dataset.

        Returns:
        float or int: Accuracy percentage if input_image is False, or class index if True.
    """
        correct_perediction = 0
        number_of_data = len(data)
        predicted_y = None
        for i in range(number_of_data):
            if not input_image:
                # data
                x = data[i][0]
                # label
                y = np.argmax(data[i][1])
            else:
                x=data

            output, _ = self.feed_forward(x, parameters)
            predicted_y = np.argmax(output)
            # check if the prediction was correct
            if not input_image and y == predicted_y:
                correct_perediction += 1

        # calculate acuuracy based on correct predictions
        accuracy = correct_perediction /number_of_data

        if input_image:
            return predicted_y

        return accuracy*100

    def result(self, epochs_costs, trained_params):
        """
        Displays the accuracy percentages for training and testing 
        datasets and plots the training cost over epochs.

        This method calculates and prints the accuracy on both the 
        training and testing datasets using the trained model parameters. 
        It also generates a plot of the training costs over epochs to 
        visually assess the model's learning progress.

        Parameters:
            epochs_costs (list): A list of cost values recorded at 
                                each epoch during training.
            trained_params (dict): The parameters of the trained 
                                model used to calculate accuracies.

        Outputs:
            This method prints the accuracy percentages for 
            the training and test datasets to the console and
            displays a line plot of the training costs over 
            epochs, showing changes in cost with each epoch.
        """
        print("on train data: " +
                str(self.calculate_percentage_of_accuracy(self.train_set, trained_params)) + " %")
        print("on test data: " +
                str(self.calculate_percentage_of_accuracy(self.test_set, trained_params)) + " %")
        plt.plot(epochs_costs)
        total_costs_size = len(epochs_costs)
        plt.xticks(np.arange(total_costs_size), np.arange(1, total_costs_size+1))
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.show()
