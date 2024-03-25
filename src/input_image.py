import numpy as np
import os
import matplotlib.pyplot as plt

class Input():

    HEIGHT = 32
    WIDTH = 32
    IMAGE_FORMAT = [".PNG", ".JPEG", ".JPG"]

    def __init__(self):
        return


    def get_image(self):
        file_path = input("Enter directory: ")

        # Check if file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} was not found.")
        
        # Check file format
        if not any(str(file_path).upper().endswith(ext) for ext in self.IMAGE_FORMAT):
            raise Exception(f"The file format of {file_path} is not valid. Expected formats: {self.IMAGE_FORMAT}")

        # Check image size
        image_data = plt.imread(file_path)
        if image_data.shape[0] != self.HEIGHT or image_data.shape[1] != self.WIDTH or image_data.shape[1] != self.HEIGHT or image_data.shape[0] != self.WIDTH:
            raise Exception(f"The size of the image {file_path} is invalid. Expected size: {self.HEIGHT} in {self.WIDTH}")

        return image_data



if __name__ == '__main__': 
    inp = Input()
    inp.set_image()