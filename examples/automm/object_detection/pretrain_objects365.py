import os
import time

from autogluon.multimodal import MultiModalPredictor


def main():
    data_dir = "/media/code/datasets/object365"
    train_path = os.path.join(data_dir, "train", "annotations", "zhiyuan_objv2_train.json")
    val_path = os.path.join(data_dir, "val", "annotations", "zhiyuan_objv2_val.json")

    checkpoint_name = "yolox_s_8x8_300e_coco"
    num_gpus = -1

    predictor = MultiModalPredictor(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": num_gpus,
            "optimization.val_metric": "map",
        },
        problem_type="object_detection",
        sample_data_path=train_path,
    )

    start = time.time()
    predictor.fit(
        train_path,
        tuning_data=val_path,
        max_num_tuning_data=5000,
        hyperparameters={
            "optimization.learning_rate": 1e-2,
            "optimization.lr_decay": 0.9,
            "optimization.lr_mult": 1,
            "optimization.max_epochs": 15,
            # "optimization.max_steps": 180000,
            "optimization.warmup_steps": 0,
            "optimization.patience": 10,
            "optimization.val_check_interval": 1.0,
            "optimization.check_val_every_n_epoch": 1,
            "optimization.top_k": 20,
            "env.per_gpu_batch_size": 8,  # decrease it when model is large
        },
        clean_ckpts=False,
    )
    end = time.time()

    print("This finetuning takes %.2f seconds." % (end - start))


if __name__ == "__main__":
    main()
