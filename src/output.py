from classifier import Classifier

class Output():
    def __init__(self):
        self.classifier = Classifier()
        self.class_name = None
    
    def set_class_name(self):
        self.classifier.set_image_pixel()
        self.class_name = self.classifier.get_class()
        return self.class_name

    def append_to_file(self, file_path, sentence):
        try:
            # Open the file in append mode ('a')
            with open(file_path, 'a') as file:
                file.write(sentence + '\n')  # Add a newline after the sentence
            print("Sentence added successfully.")
        except FileNotFoundError:
            print("Error: The file was not found.")
        except IOError:
            print("Error: An I/O error occurred while writing to the file.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def save_feedback(self):
        feedback = None
        expected_class = self.class_name
        acceptable_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        while feedback not in ["y", "n"]:
            feedback = input("Do you agree with the predicted class?: (y/n) ")
            feedback = feedback.lower()
        
        if feedback == "n":
            print("supported classes:")
            print(acceptable_classes)
            expected_class = input("What was the expected class? ")
            expected_class = expected_class.lower()
            
            while expected_class not in acceptable_classes:
                print("supported classes:")
                print(acceptable_classes)
                expected_class = input("Please type an acceptable class? ")
                expected_class = expected_class.lower()
        self.append_to_file("feedback.txt", str(self.class_name) + " " + str(expected_class))


if __name__ == '__main__':
    o = Output()
    o.save_feedback()
