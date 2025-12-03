from fastai.vision.all import *
import cv2

class Model:
    # init
    def __init__(self):
        self.learners = []
        self.tta_res = []
        self.vocab = []
        self.idxs_general = []
        self.targs_general = []
        self.categories = []

    def load_and_translate(self, learners_file_path: str, tta_res_file_path: str) -> None:
        # toggle cpu mode
        torch.cuda.is_available = lambda : False
        
        self.learners = load_pickle(learners_file_path)
        self.tta_res = load_pickle(tta_res_file_path)

        tta_prs = first(zip(*self.tta_res))
        _, targs = self.tta_res[0]

        avg_pr = torch.stack(tta_prs).mean(0)

        idxs = avg_pr.argmax(dim=1)
        idxs.shape

        self.vocab = self.learners[0].dls.vocab

        def translate_num_to_text(y):
            return [self.vocab[i] for i in y]

        def translate(y):
            new_y = []
            for label in y:
                if label in ['disposable_plastic_cutlery', 'plastic_detergent_bottles', 'plastic_food_containers', 
                            'plastic_shopping_bags', 'plastic_soda_bottles', 'plastic_straws', 'plastic_trash_bags', 
                            'plastic_water_bottles', 'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 
                            'steel_food_cans', 'paper_cups', 'plastic_cup_lids', 'glass_beverage_bottles', 
                            'glass_cosmetic_containers', 'glass_food_jars']:
                    new_y.append('containers')
                elif label in ['cardboard_boxes', 'cardboard_packaging', 'magazines', 'newspaper', 'office_paper', ]:
                    new_y.append('papers')
                elif label in ['coffee_grounds', 'eggshells', 'food_waste', 'tea_bags']:
                    new_y.append('green_bin')
                elif label in ['clothing', 'shoes', 'styrofoam_cups', 'styrofoam_food_containers']:
                    new_y.append('waste')

            return new_y

        self.idxs_general = translate(translate_num_to_text(idxs))
        self.targs_general = translate(translate_num_to_text(targs))

        self.categories = ['containers', 'papers', 'green_bin', 'waste']

    def classify(self, image: PILImage) -> str:
        # change to square aspect ratio by cropping out sides
        width, height = image.size
        if width > height:
            left = (width - height) / 2
            right = (width + height) / 2
            top = 0
            bottom = height
            image = image.crop((left, top, right, bottom))
        elif height > width:
            top = (height - width) / 2
            bottom = (height + width) / 2
            left = 0
            right = width
            image = image.crop((left, top, right, bottom))

        # resize for model input
        image = image.resize((256, 256))

        preds = []
        for learner in self.learners:
            pred, _, _ = learner.predict(image)
            preds.append(pred)

        # majority vote
        final_pred = max(set(preds), key=preds.count)
        return final_pred