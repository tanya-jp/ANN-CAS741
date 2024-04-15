import unittest
from unittest.mock import patch, MagicMock
import sys
sys.path.insert(0, '../src')  # Add src directory to Python's search path

from control import main 

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

    @patch('control.Model')
    @patch('control.Output')
    @patch('builtins.input', side_effect=['2'])  # User selects '2' to classify an image and provide feedback
    def test_classify_image(self, mocked_input, mocked_Output, mocked_Model):
        mocked_output_instance = MagicMock()
        mocked_Output.return_value = mocked_output_instance
        
        main()

        # Check if Output is instantiated, methods are called
        mocked_Output.assert_called_once()
        mocked_output_instance.set_class_name.assert_called_once()
        mocked_output_instance.save_feedback.assert_called_once()

# If the script is run directly (instead of imported), run the tests.
if __name__ == '__main__':
    unittest.main()