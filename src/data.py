"""Module for loading and processing data.

This module includes functions for loading and processing CIFAR 10 dataset
used in our application. It provides utilities for loading,
grayscaling, normalizing, flattiening, and shuffling train and test dataset.
"""
import random
import os
import tarfile
import gdown
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras

class Data():

    """Class for loading and processing dataset.

    This class provides functionalities to download, process, and prepare 
    CIFAR 10 for the image classification tasks. 
    It handles tasks such as downloading data, converting RGB 
    images to grayscale, normalizing pixel values,
    flattening image data, and shuffling the data for randomness.

    Attributes:
        train_data (list): List to store training data.
        train_images (list): List to store training images.
        test_data (list): List to store testing data.
        test_images (list): List to store testing images.

    Methods:
        save_data: Downloads and extracts dataset from a given URL.
        load_data: Loads the dataset into the class attributes.
        rgb2gray: Converts RGB image data to grayscale.
        prep_pixels: Normalizes pixel values in the dataset.
        flat_data: Flattens the image data from 2D to 1D.
        shuffle_data: Randomizes the order of the dataset.
        get_dataset: Processes the dataset and returns the processed training and testing data.
    """


    def __init__(self):

        self.train_data = []
        self.train_images = []
        self.test_data = []
        self.test_images = []

    def save_data(self):
        """
        Download the file from Google Drive and save it locally if it 
        isn't already saved
        """
        url = 'https://drive.google.com/uc?id=1Y1vgzPvMeVcXSxDfOlCVia7wsU7p8M6g'
        output = 'CIFAR10.tar.gz'
        unzipped = 'CIFAR10'

        if not os.path.isfile(output):
            gdown.download(url, output, quiet=False)

        # Check if the file was downloaded successfully
        if not os.path.isfile(output):
            raise Exception("UnableToDownload")

        print("Dataset downloaded")

        # Extract the downloaded tar.gz file
        if not os.path.isfile(unzipped):
            with tarfile.open(output) as file:
                file.extractall()

    def load_data(self):
        """
        Load train and test dataset, ecnod the labels 
        and check if it is loaded completely.
        """
        # Download dataset
        self.train_data, self.test_data = keras.datasets.cifar10.load_data()
        (self.train_images, train_y) = self.train_data
        (self.test_images, test_y) = self.test_data

        # one hot encode target values
        train_y = keras.utils.to_categorical(train_y)
        test_y = keras.utils.to_categorical(test_y)

        # Check if the file was downloaded successfully
        if (self.train_images.shape != (50000, 32, 32, 3) or train_y.shape != (50000, 10) or
            self.test_images.shape != (10000, 32, 32, 3) or test_y.shape != (10000, 10)):
            raise Exception("UnableToDownload")
        print("Downloaded")

    def rgb2gray(self, images, input_image = False):
        """
        Convert RGB image data into grayscale.

        This method is used to reduce the complexity of image data by converting 
        RGB (Red, Green, Blue) images to grayscale. 
        In a grayscale image, each pixel represents a shade of gray corresponding 
        to the intensity of the original colors.

        Parameters:
        images (array): A numpy array representing one or more images in RGB format. 
                        If a single image, it should be a 3D array 
                        (height, width, color_channels).
                        If multiple images, it should be a 4D array 
                        (number_images, height, width, color_channels).
        input_image (bool): A flag to indicate if the provided 'images' 
                            parameter is a single image (True) or 
                            a batch of images (False). Default is False.

        Returns:
        array: The transformed grayscale images. If 'images' was a 
                single image, the return is a 2D array (height, width).
                If 'images' was a batch of images, the return is a 3D 
                array (number_images, height, width).
        """
        if input_image:
            r, g, b = images[:, :, 0], images[:, :,1], images[:, :, 2]
        else:
            r, g, b = images[:, :, :, 0], images[:, :, :,1], images[:, :, :, 2]
        images = 0.2989 * r + 0.5870 * g + 0.1140 * b
        print("Grayscaled")

        return images

    def prep_pixels(self, images):
        """
        Normalize image pixel values to the [0, 1] range.

        Converts image data to float32 and scales pixel values to range between 0 and 1 
        for neural network compatibility.
        
        Parameters:
        images (array): A numpy array of image data.

        Returns:
        array: Normalized image data.
        """
         # Convert from float64 to float32
        images = images.astype('float32')

        # Normalize to range 0-1
        images = images / 255.0
        print("Normalized")
        return images

    def flat_data(self, images):
        """
        Flatten 2D image matrices into 1D vectors.

        Reshapes each image in the given array from a 2D matrix to a 1D vector.

        Parameters:
        images (array): A numpy array of 2D image matrices.

        Returns:
        array: A numpy array of 1D image vectors.
        """
        images = images.reshape(-1, 1024)

        print("flatten")
        return images

    def shuffle_data(self, data, images):
        """
        Shuffle image data and labels together.

        Combines images with their corresponding labels, one-hot encodes the labels, 
        and then shuffles the combined data for randomness.

        Parameters:
        data (tuple): A tuple containing two elements: images and labels.
        images (array): A numpy array of images.

        Returns:
        list: A list of tuples, where each tuple contains a shuffled image and its label.
        """
        # make data sets by appending data and labels
        ( _, labels) = data

        data = []

        # one hot encode target values
        labels = keras.utils.to_categorical(labels)

        for i, image in enumerate(images):
            data.append((image.reshape(1024, 1), labels[i].reshape(10,1)))


        # shuffle data matrix
        random.shuffle(data)

        print("shuffled")

        return data


    def get_dataset(self):
        """
        Process and retrieve the training and testing datasets.

        This method orchestrates various data processing steps including loading data, 
        converting to grayscale, normalizing pixel values, flattening images, and shuffling data.
        It ensures the dataset is properly formatted and prepared for use in machine learning
        models.

        Returns:
        tuple: A tuple containing processed training and testing datasets.
    """
        self.load_data()
        self.train_images = self.rgb2gray(self.train_images)
        self.test_images = self.rgb2gray(self.test_images)
        self.train_images = self.prep_pixels(self.train_images)
        self.test_images = self.prep_pixels(self.test_images)
        self.train_images = self.flat_data(self.train_images)
        self.test_images = self.flat_data(self.test_images)
        self.train_data = self.shuffle_data(self.train_data, self.train_images)
        self.test_images = self.shuffle_data(self.test_data, self.test_images)
        return self.train_data, self.test_data
if __name__ == '__main__':
    d = Data()
    d.get_dataset()
