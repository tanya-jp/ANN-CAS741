import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from numpy.testing import assert_array_equal

import sys
sys.path.insert(0, '../src')

from model import Model

class TestModel(unittest.TestCase):

    def setUp(self):
        """ Set up necessary mocks for the tests """
        self.model = Model()
        self.train_test_mock = MagicMock()
        self.model.training = self.train_test_mock
        self.model.training.train.return_value = (np.array([1, 2, 3]), 'costs', 0, 1)
        self.model.training.result = MagicMock()


    def test_save_model_success(self):
        """ Test the save_model method under normal conditions """
        self.train_test_mock.train.return_value = (np.array([1, 2, 3]), 'costs', 0, 1)
        with patch('numpy.save') as mock_save:
            result = self.model.save_model()
            self.assertTrue(result)
            args, kwargs = mock_save.call_args
            assert_array_equal(args[1], np.array([1, 2, 3]))
            self.assertEqual(args[0], 'trained_params.npy')

    def test_save_model_permission_error(self):
        """ Test the save_model method with a PermissionError """
        self.train_test_mock.train.return_value = (np.array([1, 2, 3]), 'costs', 0, 1)
        with patch('numpy.save', side_effect=PermissionError("Permission Denied")):
            with self.assertRaises(PermissionError):
                self.model.save_model()

    def test_load_model_file_not_found(self):
        """ Test the load_model method with a FileNotFoundError """
        with patch('numpy.load', side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):
                self.model.load_model('nonexistent.npy')

    def test_load_model_success(self):
        """ Test the load_model method under normal conditions """
        expected_dict = {'param': 'value'}
        with patch('numpy.load', return_value=np.array(expected_dict, dtype=object)):
            result = self.model.load_model('existent.npy')
            self.assertEqual(result, expected_dict)

    def test_load_trained_classifier(self):
        """ Test the load_trained_classifier method """
        self.train_test_mock.calculate_percentage_of_accuracy.return_value = 'TestClass'
        with patch.object(Model, 'load_model', return_value={'weights': [1, 2, 3]}):
            result = self.model.load_trained_classifier('image_data', 'model_name.npy')
            self.assertEqual(result, 'TestClass')

    def test_load_model_io_error(self):
        """ Test the load_model method for handling IOError, simulating read failure """
        with patch('numpy.load', side_effect=IOError("Could not read file")):
            with self.assertRaises(IOError) as cm:
                self.model.load_model("fakefile.npy")
            the_exception = cm.exception
            self.assertEqual(str(the_exception), "Could not read file")
    
    @patch('numpy.save')
    def test_save_model_io_error_handling(self, mock_save):
        """Test that IOError is correctly raised and formatted in save_model."""
        # Setup the mock to raise an IOError when numpy.save is called
        mock_save.side_effect = IOError("Failed to write to disk")
        
        # Assert that an IOError is raised and check the message format
        with self.assertRaises(IOError) as context:
            self.model.save_model()
        
        # Validate that the error message is correctly formatted
        self.assertEqual(str(context.exception), "I/O error: Failed to write to disk")

    @patch('numpy.save')
    def test_save_model_general_exception_handling(self, mock_save):
        """Test that a general Exception is properly re-raised with a custom message in save_model."""
        # Set up the mock to raise a general Exception when numpy.save is called
        mock_save.side_effect = Exception("A generic error")

        # Assert that an Exception is raised and check the custom message format
        with self.assertRaises(Exception) as context:
            self.model.save_model()

        # Verify that the error message is formatted as expected
        self.assertEqual(str(context.exception), "An unexpected error occurred: A generic error")




if __name__ == '__main__':
    unittest.main()
