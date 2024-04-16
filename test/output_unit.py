import unittest
from unittest.mock import patch, mock_open, MagicMock

import sys
sys.path.insert(0, '../src')
from output import Output

class TestOutputClass(unittest.TestCase):
    def test_append_to_file_success(self):
        with patch('builtins.open', mock_open()) as mocked_file:
            output = Output()
            output.append_to_file("feedback.txt", "Test sentence")
            mocked_file.assert_called_with("feedback.txt", 'a', encoding='utf-8')
            mocked_file().write.assert_called_once_with("Test sentence\n")

    def test_append_to_file_file_not_found(self):
        with patch('builtins.open', mock_open()) as mocked_file:
            mocked_file.side_effect = FileNotFoundError
            output = Output()
            with self.assertRaises(FileNotFoundError):
                output.append_to_file("dummy_path.txt", "Test sentence")

    def test_append_to_file_io_error(self):
        with patch('builtins.open', mock_open()) as mocked_file:
            mocked_file.side_effect = OSError  # Changed from IOError to OSError for Python 3 compatibility
            output = Output()
            with self.assertRaises(OSError):  # Changed from IOError to OSError
                output.append_to_file("feedback.txt", "Test sentence")

    @patch("builtins.print")
    def test_append_to_file_unexpected_exception(self, mock_print):
        # Use an exception type that you do not already catch specifically
        with patch('builtins.open', mock_open()) as mocked_file:
            mocked_file.side_effect = Exception("Unexpected error")

            output = Output()
            output.append_to_file("dummy_path.txt", "Test sentence")

            # Check that print was called with the correct message
            mock_print.assert_called_with("An unexpected error occurred: Unexpected error")


    def test_save_feedback_agree_with_classification(self):
            with patch('output.Output.append_to_file') as mocked_append, \
                patch('builtins.input', side_effect=['y']):
                output = Output()
                output.class_name = "dog"  # Simulating an existing class name set by previous operations
                output.save_feedback()
                mocked_append.assert_called_once_with("feedback.txt", "dog dog")

    def test_save_feedback_disagree_with_classification(self):
        with patch('output.Output.append_to_file') as mocked_append, \
             patch('builtins.input', side_effect=['n', 'cat']):
            output = Output()
            output.class_name = "dog"
            output.save_feedback()
            mocked_append.assert_called_once_with("feedback.txt", "dog cat")

    def test_save_feedback_disagree_retry_classification(self):
        with patch('output.Output.append_to_file') as mocked_append, \
             patch('builtins.input', side_effect=['n', 'xyz', 'cat']):
            output = Output()
            output.class_name = "dog"
            output.save_feedback()
            mocked_append.assert_called_once_with("feedback.txt", "dog cat")

    @patch('model.Model.load_trained_classifier', return_value=3)  # Mocking to return 'cat'
    @patch('input_prep.InputPrep.get_input')
    def test_get_class_cat_with_output_class(self, mock_get_input, mock_load_trained_classifier):
        # Setup the mock to simulate processing 'valid_image.jpg'
        mock_get_input.return_value = MagicMock(name='classifier_test_cases/cat.jpg')
        
        output_instance = Output()  # Create an instance of Output
        class_name = output_instance.set_class_name()  # Get the class name using the Output methods
        
        assert class_name == 'cat', "The image should be classified as 'cat'"

