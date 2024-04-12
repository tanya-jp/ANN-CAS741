"""
Main executable script for training a model or classifying an image.

This script provides a command-line interface for the user to 
choose between training a model or classifying an image. It utilizes 
classes from the 'model' and 'output' modules to perform these tasks.

Usage:
    Run the script and follow the on-screen prompts:
    1. To train the model, choose option '1'.
    2. To classify an image and provide feedback, choose option '2'.

Functions:
    None directly defined; functionality is provided via class methods from imported modules.

Modules:
    model: Contains the Model class responsible for model training and saving.
    output: Contains the Output class used for classifying an image and handling user feedback.
"""
from model import  Model
from output import Output

if __name__ == "__main__":
    task = "0" # pylint: disable=C0103
    print("What do you want to do?")
    print("1. Train the model")
    print("2. Classify an image")

    while task not in ["1", "2"]:
        task = input("Please enter 1 or 2 ")

    if task == "1":
        m = Model()
        m.save_model()

    else:
        o = Output()
        class_name = o.set_class_name()
        print(class_name)
        o.save_feedback()
    