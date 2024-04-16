import unittest                    
from unittest.mock import patch, MagicMock, mock_open, ANY  # Tools for mocking
import sys                          
import numpy as np                  
sys.path.insert(0, '../src')  # Add src directory to Python's search path

from control import main 
from output import Output

class TestMainScript(unittest.TestCase):
    @patch('control.Model')
    @patch('control.Output')
    @patch('builtins.input', side_effect=['1'])  # User selects '1' to train the model
    def test_train_model(self, mocked_input, mocked_Output, mocked_Model):
        mocked_model_instance = MagicMock()
        mocked_Model.return_value = mocked_model_instance
        
        main()

        # Check if Model is instantiated and save_model is called
        mocked_Model.assert_called_once()
        mocked_model_instance.save_model.assert_called_once()

    @patch('builtins.input', side_effect=['2'])  # User selects '2' to classify an image and provide feedback
    @patch('control.Output')
    @patch('control.Model')
    def test_classify_image(self, mocked_Model, mocked_Output, mocked_input):
        mocked_output_instance = MagicMock()
        mocked_Output.return_value = mocked_output_instance
        
        main()

        # Check if Output is instantiated, methods are called
        mocked_Output.assert_called_once()
        mocked_output_instance.set_class_name.assert_called_once()
        mocked_output_instance.save_feedback.assert_called_once()
        # Validate the flow of function calls
        mocked_output_instance.set_class_name.assert_called_once()
        mocked_output_instance.save_feedback.assert_called_once()
        with patch('builtins.open', mock_open()) as mocked_file:
            output = Output()
            output.append_to_file("feedback.txt", 'dog dog')
            mocked_file.assert_called_with("feedback.txt", 'a', encoding='utf-8')
            mocked_file().write.assert_called_once_with('dog dog\n')

# If the script is run directly (instead of imported), run the tests.
if __name__ == '__main__':
    unittest.main()