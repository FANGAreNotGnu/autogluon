import argparse
import time

from autogluon.multimodal import MultiModalPredictor
from dataset_collections import launch_dataset


def presets_finetune(
    dataset_name,
    presets,
    root,
):
    dataset = launch_dataset(key=dataset_name, data_root=root)
    train_path = dataset.train_path
    val_path = dataset.val_path
    test_path = dataset.test_path

    predictor = MultiModalPredictor(
        presets=presets,
        sample_data_path=train_path,
        problem_type="object_detection",
    )

    start = time.time()
    predictor.fit(
        train_path,
        tuning_data=val_path,
    )

    fit_end = time.time()

    predictor.evaluate(test_path)

    eval_end = time.time()

    print("time usage for fit: %.2f" % (fit_end - start))
    print("time usage for eval: %.2f" % (eval_end - fit_end))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str)
    parser.add_argument("-p", "--presets", type=str)
    parser.add_argument("-r", "--root", type=str, default="/media/code/datasets/AGDetBench")
    args = parser.parse_args()

    presets_finetune(
        dataset_name=args.dataset_name,
        presets=args.presets,
        root=args.root,
    )
