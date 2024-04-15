import pytest
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, '../src')  # Add src directory to Python's search path
from output import Output 
import distutils as _distutils


class TestClassifier:
    @patch('model.Model.load_trained_classifier', return_value=3)  # Mocking to return 'cat'
    @patch('input_prep.InputPrep.get_input')
    def test_get_class_cat_with_output_class(self, mock_get_input, mock_load_trained_classifier):
        # Setup the mock to simulate processing 'valid_image.jpg'
        mock_get_input.return_value = MagicMock(name='classifier_test_cases/cat.jpg')
        
        output_instance = Output()  # Create an instance of Output
        class_name = output_instance.set_class_name()  # Get the class name using the Output methods
        
        assert class_name == 'cat', "The image should be classified as 'cat'"

    @patch('model.Model.load_trained_classifier', return_value=9)  # Mocking to return 'horse'
    @patch('input_prep.InputPrep.get_input')
    def test_get_class_truck_with_output_class(self, mock_get_input, mock_load_trained_classifier):
        
        # Setup the mock to simulate processing 'valid_image.jpg'
        mock_get_input.return_value = MagicMock(name='classifier_test_cases/truck.jpg')
            
        output_instance = Output()  # Create an instance of Output
        class_name = output_instance.set_class_name()  # Get the class name using the Output methods
            
        assert class_name == 'truck', "The image should be classified as 'truck'"
