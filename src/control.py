from model import  Model
from output import Output

if __name__ == "__main__":
    task = "0"
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
    