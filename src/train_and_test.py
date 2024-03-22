import numpy as np
from data import Data
from training_model import TrainingModel

class TrainTest():

    BATCH_SIZE = 16
    LEARNING_RATE = 0.3
    EPOCHS = 20

    def __init__(self):
        self.layers = []
        self.gradients = []
        
    # Set layers based on zero gradients
    def set_layers(self):
        training_model = TrainingModel()
        self.gradients = training_model.create_gradients_zeros()

        # Number of layers = x + 1
        x = (len(self.gradients) + 1) // 3

        # Find number of neurons
        self.layers.append(len(self.gradients[2*x-2][1]))
        for i in range (2*x-2, x-2, -1):
            self.layers.append(len(self.gradients[i]))

    # Allocate random normal W matrix and zero b vector for each layer
    def initialize_parameters(self):

        parameters = {}
        center = 0
        margin = 1
        
        for i in range(1, len(self.layers)):
            # Draw random samples from a normal (Gaussian) distribution
            parameters['W'+str(i)] = np.random.normal(center, margin, size = (self.layers[i], self.layers[i-1]))
            # Zero bias vector
            parameters['b' + str(i)] = np.zeros((self.layers[i],1))  

        return parameters 

    # Extract parameters that is saved during forwardfeeding from the cache
    def extract_parameters(self, caches):

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

    # Return output of the network from forward calculations
    def feed_forward(self, predicted, parameters):
        caches = []

        # Claculate forward process for each layer
        for l in range (1, self.layers):
            prev_a = predicted 
            # Extract weight and biase from the list of parameters
            W = parameters['W' + str(l)]
            b = parameters['b' + str(l)]
            # New a is calculated based on the formula, using sigmoid as activation function
            Z = np.dot(W,prev_a) + b
            predicted = self.sigmoid(Z)

            # Cache -> ((a, W, b), z)
            cache = ((prev_a, W, b), Z)
            caches.append(cache)
                
        return predicted, caches

    # Calculate gradients of wights and biases
    # TODO
    def backpropagation(self, caches, output, y):

        # Number of layers = x + 1
        x = (len(self.gradients) + 1) // 3

        a3 = output
        a, w, z = self.extract_parameters(caches)

        grad_a = []
        for i in range (x-2, -1, -1):
            grad_a.append(self.gradients[i])

        grad_w = []
        for i in range (2*x-2, x-2, -1):
            grad_w.append(self.gradients[i])
        
        grad_b = []
        for i in range (len(self.gradients)-1, 2*x-2):
            grad_b.append(self.gradients[i])

        # grad_a2, grad_a1, grad_W3, grad_W2, grad_W1, grad_b2, grad_b1, grad_b0 = self.gradients
        
        # calculat gradients of out put of layer(a)
        grad_a2 += np.transpose(W3) @ (2 * sigmoid_deriv(z3) * (a3 - y))
        grad_a1 += np.transpose(W2) @ ( sigmoid_deriv(z2) * grad_a2)


        # calculate gradients of weights
        grad_W3 += (2 * sigmoid_deriv(z3) * (a3 - y)) @ (np.transpose(a2))
        grad_W2 += (sigmoid_deriv(z2) * grad_a2) @ (np.transpose(a1))
        grad_W1 += (2 * sigmoid_deriv(z1) * grad_a1) @ (np.transpose(a0))

        # calculate gradients of biases
        grad_b2 += (2 * sigmoid_deriv(z3) * (a3 - y))
        grad_b1 += (sigmoid_deriv(z2) * grad_a2)
        grad_b0 += (sigmoid_deriv(z1) * grad_a1)

        # define a dictionare
        # keys -> label of gradients
        # values -> gradients 
        gradients = {}
        gradients["db1"] = grad_b0
        gradients["db2"] = grad_b1
        gradients["db3"] = grad_b2
        gradients["dW1"] = grad_W1
        gradients["dW2"] = grad_W2
        gradients["dW3"] = grad_W3
        
        return gradients

    # Claculate sigmoid function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Calculate derivation of sigmoid
    def sigmoid_deriv(z):
        a = sigmoid(z)
        return a * (1 - a)

    # Calculate SSE cost 
    def compute_cost(predicted, actual):
        cost = ((predicted - actual)**2).sum()
        return cost
    
        

if __name__ == '__main__': 
    d = TrainTest()
    d.set_layers()
    d.initialize_parameters()
