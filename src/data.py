
import gdown
import tarfile
import os
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot
import numpy as np
import random

class Data():

    def __init__(self):

        self.train_data = []
        self.train_images = []
        self.test_data = []
        self.test_images = []

    def save_data(self):
        # Download the file from Google Drive
        url = 'https://drive.google.com/uc?id=1Y1vgzPvMeVcXSxDfOlCVia7wsU7p8M6g'
        output = 'CIFAR10.tar.gz'
        unzipped = 'CIFAR10'

        if not os.path.isfile(output):
            gdown.download(url, output, quiet=False)

        # Check if the file was downloaded successfully
        if not os.path.isfile(output):
            raise Exception("UnableToDownload")
        else:
            print("Dataset downloaded")

        # Extract the downloaded tar.gz file
        if not os.path.isfile(unzipped):
            with tarfile.open(output) as file:
                file.extractall()

    def load_data(self): 
        # Download dataset
        self.train_data, self.test_data = cifar10.load_data()
        (self.train_images, trainY) = self.train_data
        (self.test_images, testY) = self.test_data

        # one hot encode target values
        trainY = to_categorical(trainY)
        testY = to_categorical(testY)

        # Check if the file was downloaded successfully
        if self.train_images.shape != (50000, 32, 32, 3) or trainY.shape != (50000, 10) or self.test_images.shape != (10000, 32, 32, 3) or testY.shape != (10000, 10):
            raise Exception("UnableToDownload")
        
        else:
            print("Downloaded")

    # Convert RGB data into grayscale in order to reduce complexity
    def rgb2gray(self):
        # Train images
        r, g, b = self.train_images[:, :, :, 0], self.train_images[:, :, :,1], self.train_images[:, :, :, 2]
        self.train_images = 0.2989 * r + 0.5870 * g + 0.1140 * b

        # Test images
        r, g, b = self.test_images[:, :, :, 0], self.test_images[:, :, :,1], self.test_images[:, :, :, 2]
        self.test_images = 0.2989 * r + 0.5870 * g + 0.1140 * b

        print("Grayscaled")

    # Scale pixels to change the range of data between 0 and 1
    def prep_pixels(self):
        # Convert from integers to floats
        self.train_images = self.train_images.astype('float32')
        self.test_images = self.test_images.astype('float32')

        # Normalize to range 0-1
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0
        print("Normalized")

    # Flats the 2D matrices to an 1D vector
    def flat_data(self):
        self.train_images = self.train_images.reshape(-1, 1024)
        self.test_images = self.test_images.reshape(-1, 1024)

        print("flatten")

    def shuffle_data(self):
        # make data sets by appending data and labels
        (raw_train_images, train_labels) = self.train_data
        (raw_test_images, test_labels) = self.test_data

        self.train_data = []
        self.test_data = []

        # one hot encode target values
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)

        for i in range(len(self.train_images)):
            self.train_data.append((self.train_images[i].reshape(1024, 1), train_labels[i].reshape(10,1)))

        for i in range(len(self.test_images)):
            self.test_data.append((self.test_images[i].reshape(1024, 1), test_labels[i].reshape(10,1)))

        # shuffle data matrix
        random.shuffle(self.train_data)
        random.shuffle(self.test_data)

        print("shuffled")

    def get_dataset(self):
        self.load_data()
        self.rgb2gray()
        self.prep_pixels()
        self.flat_data()
        self.shuffle_data()
        return self.train_data, self.test_data

if __name__ == '__main__': 
    d = Data()
    d.get_dataset()

