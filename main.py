import PIL

from scripts.model import Model
from scripts.GUI import run_GUI

def main():
    print("Loading model...")
    model = Model()
    model.load_and_translate('./model/learners.pkl', './model/tta_res.pkl')
    
    print("Starting GUI...")
    run_GUI(model)

if __name__ == "__main__":
    main()