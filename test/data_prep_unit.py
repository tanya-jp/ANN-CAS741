import unittest
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

if __name__ == '__main__':
    unittest.main()