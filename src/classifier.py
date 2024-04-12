import numpy as np

from model import Model
from input_prep import InputPrep

class Classifier():
    def __init__(self):
        # self.model = None
        self.input_image = None
        self.calss_name = None
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

    # def load_model(self):
    #     self.model = Model()
    #     model_name = 'trained_params.npy'
    #     self.model = self.model.load_model(model_name)
    
    def set_image_pixel(self):
        input_prep = InputPrep()
        self.input_image = input_prep.get_input()

    def get_class(self):
        m = Model()
        model_name = 'trained_params.npy'
        predicted_class = m.load_trained_classifier(self.input_image, model_name)
        self.class_name = self.classes[predicted_class]
        return self.class_name
    
if __name__ == '__main__':
    cl = Classifier()
    cl.set_image_pixel()
    print(cl.get_class())



