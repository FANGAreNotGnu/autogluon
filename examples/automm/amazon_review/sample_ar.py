import argparse
import numpy as np
import os
import pandas as pd
import warnings

from autogluon.tabular import TabularPredictor

warnings.filterwarnings('ignore')
np.random.seed(123)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_root', type=str, default="/media/ag/data/amazonreview/")
    parser.add_argument('-n', '--category_name', type=str, default="All_Beauty")
    parser.add_argument('-s', '--save_name', type=str, default=None)
    parser.add_argument('-t', '--num_tests', type=int, default=100)
    parser.add_argument('-a', '--samples_per_test', type=int, default=100)
    return parser


def main(args):
    train_path = os.path.join(args.data_root, f"{args.category_name}_train.csv")
    test_path = os.path.join(args.data_root, f"{args.category_name}_test.csv")

    train_sub_path = os.path.join(args.data_root, f"{args.category_name}_train_{args.num_tests}_{args.samples_per_test}.csv")
    test_sub_path = os.path.join(args.data_root, f"{args.category_name}_test_{args.num_tests}.csv")

    test_I_path = os.path.join(args.data_root, f"{args.category_name}_test_I_{args.save_name}.npy")
    test_D_path = os.path.join(args.data_root, f"{args.category_name}_test_D_{args.save_name}.npy")

    train_data = pd.read_csv(train_path, index_col=0)
    test_sub_data = pd.read_csv(test_path, index_col=0)[:args.num_tests]

    test_I = np.load(test_I_path)
    test_D = np.load(test_D_path)

    training_indices = test_I[:args.num_tests,:args.samples_per_test].flatten()

    train_sub_data = train_data.iloc[training_indices]

    train_sub_data.to_csv(train_sub_path)
    print(len(train_sub_data))
    test_sub_data.to_csv(test_sub_path)
    print(len(test_sub_data))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
