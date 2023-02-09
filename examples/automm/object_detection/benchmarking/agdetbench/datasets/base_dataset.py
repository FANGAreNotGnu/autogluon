import json
import os
from collections import defaultdict


class AGDetBenchBaseDataset:
    dataset_name = None
    splits = ["train", "test"]

    def __init__(self, data_root):
        self.train_path = os.path.join(data_root, self.dataset_name, "annotations", "train.json")

        if "val" in self.splits:
            self.val_path = os.path.join(data_root, self.dataset_name, "annotations", "val.json")
        else:
            self.val_path = None

        if "trainval" in self.splits:
            self.trainval_path = os.path.join(data_root, self.dataset_name, "annotations", "trainval.json")
        else:
            self.trainval_path = None

        self.test_path = os.path.join(data_root, self.dataset_name, "annotations", "test.json")

        if "traintest" in self.splits:
            self.traintest_path = os.path.join(data_root, self.dataset_name, "annotations", "traintest.json")
        else:
            self.traintest_path = None

        self.paths = {
            "train": self.train_path,
            "val": self.train_path,
            "trainval": self.trainval_path,
            "test": self.test_path,
            "traintest": self.traintest_path,
        }

    def get_split_info(self, split, do_print=True):
        anno_path = self.paths[split]
        if anno_path is None:
            return None
        with open(anno_path, "r") as f:
            anno_dict = json.load(f)

            num_images = len(anno_dict["images"])

            num_categories = len(anno_dict["categories"])

            num_bboxes = len(anno_dict["annotations"])

            bboxes_distribution_per_category = defaultdict(int)
            cat_id_to_name = {int(cat["id"]): cat["name"] for cat in anno_dict["categories"]}
            for bbox in anno_dict["annotations"]:
                bboxes_distribution_per_category[cat_id_to_name[int(bbox["category_id"])]] += 1

            if do_print:
                print(f"#num_images#\n{num_images}")
                print(f"#num_categories#\n{num_categories}")
                print(f"#num_bboxes#\n{num_bboxes}")
                print(f"#bboxes_distribution_per_category#")
                for k,v in bboxes_distribution_per_category.items():
                    print(f"{k}: {v}")

        return {
            "num_images": num_images,
            "num_categories": num_categories,
            "num_bboxes": num_bboxes,
            "bboxes_distribution_per_category": bboxes_distribution_per_category,
        }

    def get_info(self, do_print=True):
        infos = {}
        if do_print:
            print(f"########{self.dataset_name}########")
        for split in self.splits:
            if do_print:
                print(f"###split: {split}###")
            infos[split] = self.get_split_info(split, do_print=do_print)
        return infos

    def print_table(self):
        infos =self.get_info(do_print=False)
        print(f"########{self.dataset_name}########")
        cats = set()
        for split in self.splits:
            cats.update(infos[split]["bboxes_distribution_per_category"].keys())
        cats = list(cats)

        print(f"#splits#\n{self.splits}")

        def print_all(attr):
            to_print = "["
            for split in self.splits:
                to_print += str(infos[split][attr]) + ", "
            to_print = to_print[:-2] + "]"
            print(f"#{attr}#\n{to_print}")

        print_all("num_images")
        print_all("num_categories")
        print_all("num_bboxes")

        print(f"#categories#")
        for cat in cats:
            print(cat)

        print(f"#bboxes_distribution_per_category#")
        for cat in cats:
            to_print = "["
            for split in self.splits:
                to_print += str(infos[split]["bboxes_distribution_per_category"][cat] if cat in infos[split]["bboxes_distribution_per_category"] else 0)
                to_print += ", "
            to_print = to_print[:-2] + "]"
            print(to_print)





if __name__ == "__main__":
    dataset = AGDetBenchBaseDataset("/media/code/datasets/AGDetBench")
    dataset.get_info()
