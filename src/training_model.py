import numpy as np

class TrainingModel():

    LAYERS_NUMBER = 4
    LAYERS_NEURONS = [1024, 16, 16, 10]

    def __init__(self):
        return

    # Create np zeros for all needed gradients based on size of layers
    def create_gradients_zeros(self):

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

if __name__ == '__main__': 
    d = TrainingModel()
    d.create_gradients_zeros()