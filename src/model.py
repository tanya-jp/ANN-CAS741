import json

from train_and_test import TrainTest

class Model():

    def __init__(self):
        return

    def save_model(self):
        training = TrainTest()
        trained_params, total_costs_vectorized, start_time, end_time = training.train()

        # Saving the dictionary to a file        
        try:
            with open('model.json', 'w') as json_file:
                json.dump(trained_params, json_file)
            return True
        except Exception as e:
            print(f"An error occurred: {e}")
            raise

    def load_model(self, file_name):
        try:
            with open(file_name, 'r') as json_file:
                return json.load(json_file)
        except FileNotFoundError:
            raise FileReadError(f"The file {file_name} was not found.")
        except json.JSONDecodeError:
            raise FileReadError(f"The file {file_name} is not a valid JSON file.")
        except Exception as e:
            raise FileReadError(f"An error occurred while reading the file: {e}")
