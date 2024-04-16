import unittest
import random
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras.utils
from unittest.mock import patch, MagicMock
import numpy as np
import sys
sys.path.insert(0, '../src')  # Add src directory to Python's search path

from data import Data 

class TestData(unittest.TestCase):
    def setUp(self):
        # This setup runs before each test method
        self.data = Data()
        self.test_image = np.array([np.random.rand(32, 32, 3) for _ in range(10)])  # 10 random 32x32 RGB images
        self.test_labels = np.random.randint(0, 10, 10)  # Random labels for the images
        self.processed_test_image = self.test_image.mean(axis=3)  # Simplified processing example

    @patch('data.Data.load_data')
    @patch('data.Data.rgb2gray')
    @patch('data.Data.prep_pixels')
    @patch('data.Data.flat_data')
    @patch('data.Data.shuffle_data')
    def test_get_dataset(self, mock_shuffle, mock_flat, mock_prep, mock_gray, mock_load):
        # Set up mocks
        mock_load.return_value = None
        mock_gray.side_effect = lambda x: x.mean(axis=3)  # Mocking RGB to grayscale conversion
        mock_prep.side_effect = lambda x: x / 255.0  # Mocking normalization
        mock_flat.side_effect = lambda x: x.reshape(x.shape[0], -1)  # Mocking flattening
        mock_shuffle.side_effect = lambda data, images: list(zip(images, data[1]))  # Mocking shuffling

        # Using a specific predefined image for testing
        self.data.train_images = self.test_image
        self.data.train_data = (self.data.train_images, self.test_labels)
        self.data.test_images = self.test_image
        self.data.test_data = (self.data.test_images, self.test_labels)

        # Call the method under test
        train, test = self.data.get_dataset()

        # Assertions to check if the method behaves as expected
        for img, _ in train:
            self.assertEqual(img.shape, (1024,))  # Checking if images are flattened correctly

        # Assertions to check if the method behaves as expected
        for img, _ in test:
            self.assertEqual(img.shape, (1024,))  # Checking if images are flattened correctly

        # Check if preprocessing steps are called correctly
        mock_gray.assert_called()
        mock_prep.assert_called()
        mock_flat.assert_called()
        mock_shuffle.assert_called()
    @patch('keras.datasets.cifar10.load_data')
    def test_load_data_success(self, mock_load_data):
        """Test successful loading of CIFAR-10 data and proper shape checks."""
        # Setup mock
        mock_train_images = np.random.rand(50000, 32, 32, 3)
        mock_test_images = np.random.rand(10000, 32, 32, 3)
        mock_train_labels = np.random.randint(10, size=(50000, 1))
        mock_test_labels = np.random.randint(10, size=(10000, 1))
        mock_load_data.return_value = ((mock_train_images, mock_train_labels), (mock_test_images, mock_test_labels))

        # Call the method
        self.data.load_data()

        # Check shapes to ensure data was loaded correctly (mocked method mimics expected behavior)
        self.assertEqual(self.data.train_images.shape, (50000, 32, 32, 3))
        self.assertEqual(self.data.test_images.shape, (10000, 32, 32, 3))

    @patch('keras.datasets.cifar10.load_data')
    def test_load_data_failure(self, mock_load_data):
        """Test loading of CIFAR-10 data with incorrect shapes to trigger exception."""
        # Incorrect data shapes should trigger the "UnableToDownload" exception
        mock_train_images = np.random.rand(50000, 32, 32)  # Incorrect shape
        mock_test_images = np.random.rand(10000, 32, 32)  # Incorrect shape
        mock_train_labels = np.random.randint(10, size=(50000, 1))
        mock_test_labels = np.random.randint(10, size=(10000, 1))
        mock_load_data.return_value = ((mock_train_images, mock_train_labels), (mock_test_images, mock_test_labels))
        
        with self.assertRaises(Exception) as context:
            self.data.load_data()
        
        # Check that the specific exception is raised
        self.assertIn("UnableToDownload", str(context.exception))

    def test_rgb2gray_single_image(self):
        """Test grayscale conversion for a single RGB image."""
        # Create a single test image (3D array)
        test_image = np.random.rand(32, 32, 3) * 255
        # Convert the image
        gray_image = self.data.rgb2gray(test_image, input_image=True)

        # Check the output shape
        self.assertEqual(gray_image.shape, (32, 32))
        # Check that conversion formula has been applied
        expected_image = 0.2989 * test_image[:, :, 0] + 0.5870 * test_image[:, :, 1] + 0.1140 * test_image[:, :, 2]
        np.testing.assert_array_almost_equal(gray_image, expected_image)

    def test_rgb2gray_multiple_images(self):
        """Test grayscale conversion for a batch of RGB images."""
        # Create a batch of test images (4D array)
        test_images = np.random.rand(10, 32, 32, 3) * 255
        # Convert the images
        gray_images = self.data.rgb2gray(test_images, input_image=False)

        # Check the output shape
        self.assertEqual(gray_images.shape, (10, 32, 32))
        # Check that conversion formula has been applied to all images
        for i in range(10):
            expected_image = 0.2989 * test_images[i, :, :, 0] + 0.5870 * test_images[i, :, :, 1] + 0.1140 * test_images[i, :, :, 2]
            np.testing.assert_array_almost_equal(gray_images[i], expected_image)
    
    def test_prep_pixels(self):
        """Test the normalization of image pixel values."""
        # Generate a test image data array with random integers
        test_images = np.random.randint(0, 256, (10, 24, 24, 3), dtype='uint8')

        # Normalize the images using the method under test
        normalized_images = self.data.prep_pixels(test_images)

        # Check that the output data type is float32
        self.assertTrue(normalized_images.dtype == np.float32, "Data type should be float32")

        # Check that all values are in the range [0, 1]
        self.assertTrue(np.all(normalized_images >= 0) and np.all(normalized_images <= 1), "Pixel values should be in the range [0, 1]")

        # Ensure the shape of the output array is unchanged
        self.assertEqual(test_images.shape, normalized_images.shape, "Output shape should be the same as input shape")

        # Optionally, check the exact transformation of a known pixel value
        # Calculate expected values for validation
        expected_images = test_images.astype('float32') / 255.0
        np.testing.assert_array_almost_equal(normalized_images, expected_images, decimal=5, err_msg="Normalized image data does not match expected values")

    def test_flat_data(self):
        """Test flattening of 2D image matrices into 1D vectors."""
        # Create a batch of test images (3D array)
        # Ensure the pixel values are integers as typically expected in image data
        test_images = np.random.randint(0, 256, (10, 32, 32), dtype=np.uint8)
        
        # Flatten the images
        flat_images = self.data.flat_data(test_images)

        # Check the output shape to make sure it matches the expected flat vector length
        self.assertEqual(flat_images.shape, (10, 1024), "Output shape should be (number_of_images, 1024)")

        # Additional validation to check if all values are still within the correct range (0-255 for uint8 data)
        self.assertTrue((flat_images >= 0).all() and (flat_images <= 255).all(), "All pixel values should be within 0-255")

    def test_shuffle_data(self):
        """Test shuffling of image data with labels."""
        # Generate some test data
        images = np.random.rand(10, 1024)  # 10 flattened images
        labels = np.random.randint(0, 10, (10, 1))  # Ensure labels are correctly shaped as (10, 1)

        # Prepare the data as expected by the shuffle_data function
        # This includes making sure labels have the correct size and reshaping
        one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)  # Using 10 classes for the CIFAR-10 dataset

        # Combine images with their corresponding labels
        combined_data = [(img.reshape(1024, 1), lbl.reshape(10, 1)) for img, lbl in zip(images, one_hot_labels)]

        # Randomly shuffle the combined data to simulate the method's internal shuffle
        random.shuffle(combined_data)  # Shuffle here to mimic internal data shuffle

        # Invoke the shuffle_data method
        shuffled_data = self.data.shuffle_data((images, labels), images)

        # Verify that data is shuffled by checking that not all elements are in the same order
        initial_order = [item[1].argmax() for item in combined_data]
        shuffled_order = [item[1].argmax() for item in shuffled_data]
        self.assertNotEqual(initial_order, shuffled_order, "Data should be shuffled")

        # Check the structure and type of the output
        self.assertIsInstance(shuffled_data, list, "Output should be a list")
        self.assertTrue(all(isinstance(item, tuple) and len(item) == 2 for item in shuffled_data), "Each item should be a tuple of (image, label)")

        # Additional checks to ensure the integrity and shape of shuffled data
        self.assertEqual(len(shuffled_data), 10, "There should be 10 items in the shuffled data")
        self.assertTupleEqual(shuffled_data[0][0].shape, (1024, 1), "Image shape should be (1024, 1)")
        self.assertTupleEqual(shuffled_data[0][1].shape, (10, 1), "Label shape should be (10, 1)")

if __name__ == '__main__':
    unittest.main()