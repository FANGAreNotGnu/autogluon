import argparse
import numpy as np
import os
import pandas as pd
import warnings

from autogluon.multimodal import MultiModalPredictor

warnings.filterwarnings('ignore')
np.random.seed(123)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_root', type=str, default="/media/ag/data/amazonreview/")
    parser.add_argument('-s', '--save_name', type=str, default=None)
    parser.add_argument('-n', '--category_name', type=str, default="All_Beauty")
    parser.add_argument('-r', '--replace_image_with_bool', action="store_true")
    return parser


def replace_image_with_bool(df):
    df["has_image"] = df["image"].isnull()
    df.drop(columns="image")
    return df


def main(args):
    ckpt = os.path.join("/media/ag/autogluon/examples/automm/amazon_review/AutogluonModels", args.save_name)

    train_path = os.path.join(args.data_root, f"{args.category_name}_train.csv")
    val_path = os.path.join(args.data_root, f"{args.category_name}_val.csv")
    test_path = os.path.join(args.data_root, f"{args.category_name}_test.csv")

    train_emb_path = os.path.join(args.data_root, f"{args.category_name}_train_emb_{args.save_name}.npy")
    val_emb_path = os.path.join(args.data_root, f"{args.category_name}_val_emb_{args.save_name}.npy")
    test_emb_path = os.path.join(args.data_root, f"{args.category_name}_test_emb_{args.save_name}.npy")

    train_data = pd.read_csv(train_path, index_col=0)
    val_data = pd.read_csv(val_path, index_col=0)
    test_data = pd.read_csv(test_path, index_col=0)
    if args.replace_image_with_bool:
        train_data = replace_image_with_bool(train_data)
        val_data = replace_image_with_bool(val_data)
        test_data = replace_image_with_bool(test_data)

    label_col = 'overall'

    predictor = MultiModalPredictor.load(ckpt)

    train_embs = predictor.extract_embedding(train_data.drop(columns=label_col))
    np.save(train_emb_path, train_embs)
    print(train_embs.shape)

    val_embs = predictor.extract_embedding(val_data.drop(columns=label_col))
    np.save(val_emb_path, val_embs)
    print(val_embs.shape)

    test_embs = predictor.extract_embedding(test_data.drop(columns=label_col))
    np.save(test_emb_path, test_embs)
    print(test_embs.shape)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
