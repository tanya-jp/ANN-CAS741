import numpy as np
import random
import time
import matplotlib.pyplot as plt
from data import Data
from training_model import TrainingModel

class TrainTest():

    BATCH_SIZE = 16
    LEARNING_RATE = 0.1
    EPOCHS = 40

    def __init__(self):
        self.layers = []
        self.gradients = []
        self.training_model = TrainingModel()
        self.gradients = self.training_model.create_gradients_zeros()
        
    # Set layers based on zero gradients
    def set_layers(self):

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

    # Return output of the network from forward calculations
    def feed_forward(self, new_a, parameters):
        caches = []

        # Claculate forward process for each layer
        for l in range (1, len(self.layers)):
            prev_a = new_a 
            # Extract weight and biase from the list of parameters
            W = parameters['W' + str(l)]
            b = parameters['b' + str(l)]
            # New a is calculated based on the formula, using sigmoid as activation function
            Z = np.dot(W,prev_a) + b
            new_a = self.sigmoid(Z)

            # Cache -> ((a, W, b), z)
            cache = ((prev_a, W, b), Z)
            caches.append(cache)
                
        return new_a, caches

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

    # Calculate gradients of wights and biases
    def backpropagation(self, caches, output, y):

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
        for i in range (len(grad_b)):
            new_gradients["db"+str(i+1)] = grad_b[i]     
        for i in range (len(grad_w)):
            new_gradients["dW"+str(i+1)] = grad_w[i]
        
        return new_gradients

    # Claculate sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # Calculate derivation of sigmoid
    def sigmoid_deriv(self, z):
        a = self.sigmoid(z)
        return a * (1 - a)

    # Calculate SSE cost 
    def compute_cost(self, predicted, actual):
        cost = ((predicted - actual)**2).sum()
        return cost

    # apply stochastic gradient descent on input train_set and update weights
    def train(self, train_set):
        # save start time to caculate training time
        start_time = time.time()

        # initialize W and b
        parameters = self.initialize_parameters()

        # Number of layers = x + 1
        x = (len(self.gradients) + 1) // 3
        
        total_costs = []
        layers_len = len(self.layers)
        for i in range(self.EPOCHS):
            # if i%10==0:
            print("EPOCH ", i)

            # data must be shuffled each epoch time
            random.shuffle(train_set)
            cost = 0
            train_set_size = len(train_set)
            batch_num = train_set_size//self.BATCH_SIZE

            for n in range(batch_num):
                # # make zero arrays for gradients
                # grads = self.training_model.create_gradients_zeros()
                
                # which batch of train set is using to train
                first_train_data = n * self.BATCH_SIZE
                last_train_data = (n+1) * self.BATCH_SIZE
                batch = train_set[first_train_data: last_train_data]

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
                for l in range(layers_len-1):
                    parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - self.LEARNING_RATE * (all_gradients["dW" + str(l+1)]/ self.BATCH_SIZE)
                    parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - self.LEARNING_RATE * (all_gradients["db" + str(l+1)]/ self.BATCH_SIZE)
            
            total_costs.append(cost/train_set_size)
            # save end time to caculate training time
            end_time = time.time()
                    
        return parameters, total_costs, start_time, end_time 

    # calculate accuracy of network's output
    def calculate_percentage_of_accuracy(self, data, parameters):
        correct_perediction = 0
        number_of_data = len(data)
        for i in range(number_of_data):
            # data
            X = data[i][0]
            # label
            Y = np.argmax(data[i][1])
                
            output, caches = self.feed_forward(X, parameters)
            predictedY = np.argmax(output)
            # check if the prediction was correct
            if Y == predictedY:
                correct_perediction += 1

        # calculate acuuracy based on correct predictions
        accuracy = correct_perediction /number_of_data
            
        return accuracy*100
            
#plot epoch cost
def plot_cost_epoch(epochs_costs):
  plt.plot(epochs_costs)
  total_costs_size = len(epochs_costs)
  plt.xticks(np.arange(total_costs_size), np.arange(1, total_costs_size+1))
  plt.xlabel("Epoch")
  plt.ylabel("Cost")
  plt.show()

        

if __name__ == '__main__': 
    training = TrainTest()
    training.set_layers()

    dataset = Data()
    train_data, test_data = dataset.get_dataset()
    trained_params, total_costs_vectorized, start_time, end_time = training.train(train_data)
    print(f"Training lasted {end_time - start_time} seconds")

    print("Accuracy after training process:")
    print("on train data: " + str(training.calculate_percentage_of_accuracy(train_data, trained_params)) + " %")
    print("on test data: " + str(training.calculate_percentage_of_accuracy(test_data, trained_params)) + " %")

    plot_cost_epoch(total_costs_vectorized)


    # costs = []
    # train_accuracy = []
    # test_accuracy = []
    # # train 10 times and cost and acc of each train and test will be added to their list
    # for i in range(10):
    #     trained_params_10, epochs_costs_10, start_time, end_time = train(train_data)
    #     train_acc = calculate_percentage_of_accuracy(train_set, trained_params_10)
    #     test_acc = calculate_percentage_of_accuracy(test_set, trained_params_10)
    #     train_accuracy.append(train_acc)
    #     test_accuracy.append(test_acc)
    #     costs.append(epochs_costs_10)
