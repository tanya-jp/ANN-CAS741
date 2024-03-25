import numpy as np 

from input_image import Input

class InputPrep():

    def __init__(self):
        self.input_image = []

    def set_image_pixel(self):
        input_module = Input()
        self.input_image = input_module.get_image()

    # Convert RGB data into grayscale to reduce complexity
    def rgb2gray(self):
        r, g, b = self.input_image[:, :, 0], self.input_image[:, :,1], self.input_image[:, :, 2]
        self.input_image = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    # Scale pixels to change the range of data between 0 and 1
    def prep_pixels(self):
        # Convert from integers to floats
        self.input_image = self.input_image.astype('float32')

        # Normalize to range 0-1
        self.input_image = self.input_image / 255.0
    
    # Flat the 2D matrices to an 1D vector
    def flat_data(self):
        self.input_image = self.input_image.reshape(1024, 1)

    def get_input(self):
        self.set_image_pixel()
        self.rgb2gray()
        self.prep_pixels()
        self.flat_data()
        return self.input_image

if __name__ == '__main__': 
    inp = InputPrep()
    print(np.shape(inp.get_input()))