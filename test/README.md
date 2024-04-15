# Automated tests

This folder contains unittests and pytest files of important modules and their results.

The test files are as follows:

  - [control_unit.py](https://github.com/tanya-jp/ANN-CAS741/blob/main/test/control_unit.py): 
      Unit test of the main module that controls the whole project

  - [data_prep_unit.py](https://github.com/tanya-jp/ANN-CAS741/blob/main/test/data_prep_unit.py): 
      Unit test of the data module that prepares the training, test and input data

  - [test_classifier.py](https://github.com/tanya-jp/ANN-CAS741/blob/main/test/test_classifier.py): 
      Pytest of the output module that classifies the input data, using trained model

  - [test_image_properties.py](https://github.com/tanya-jp/ANN-CAS741/blob/main/test/test_image_properties.py): 
      Pytest of the input module to make sure this module checks input data properties correctly
    
  - [control_unit.log](https://github.com/tanya-jp/ANN-CAS741/blob/main/test/control_unit.log): 
      The result of the unit test of main module that controls the whole project
  - [res.txt](https://github.com/tanya-jp/ANN-CAS741/blob/main/test/training_res.txt):
    The results of the training process
    
The results of these files are saved as `.log` files, available here.

The folders are as follows:

  - [classifier_test_cases](https://github.com/tanya-jp/ANN-CAS741/tree/main/test/classifier_test_cases):
    [test_classifier.py](https://github.com/tanya-jp/ANN-CAS741/blob/main/test/test_classifier.py)
    uses these images.

  - [image_test_cases](https://github.com/tanya-jp/ANN-CAS741/tree/main/test/image_test_cases):
    [test_image_properties.py](https://github.com/tanya-jp/ANN-CAS741/blob/main/test/test_image_properties.py)
    uses these images.

