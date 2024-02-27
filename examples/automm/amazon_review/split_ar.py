import argparse
import gzip
import json
import os
import pandas as pd



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_root', type=str, default="/media/ag/data/amazonreview/")
    parser.add_argument('-n', '--category_name', type=str, default="Office_Products")
    parser.add_argument('-t', '--train_ratio', type=float, default=0.8)
    parser.add_argument('-v', '--val_ratio', type=float, default=0.1)
    return parser


def parse_gzip(path):
    g = gzip.open(path, "rb")
    for l in g:
        yield json.loads(l)


def read_df(path):
    i = 0
    df = {}
    for d in parse_gzip(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient="index")


def split_data(
        gz_path,
        train_saved_path,
        val_saved_path,
        trainval_saved_path,
        test_saved_path,
        train_ratio,
        val_ratio,
):
    df = read_df(gz_path)
    df = df.sample(frac=1).reset_index(drop=True)
    N = len(df)
    num_train = int(N * train_ratio)
    num_val = int(N * val_ratio)

    # save train, val, test
    df[:num_train].to_csv(train_saved_path)
    print(len(df[:num_train]))
    df[num_train:num_train+num_val].to_csv(val_saved_path)
    print(len(df[num_train:num_train+num_val]))
    df[:num_train+num_val].to_csv(trainval_saved_path)
    print(len(df[:num_train+num_val]))
    df[num_train+num_val:].to_csv(test_saved_path)
    print(len(df[num_train+num_val:]))

    print("split completed")


def main(args):
    gz_path = os.path.join(args.data_root, f"{args.category_name}.json.gz")
    train_saved_path = os.path.join(args.data_root, f"{args.category_name}_train.csv")
    val_saved_path = os.path.join(args.data_root, f"{args.category_name}_val.csv")
    trainval_saved_path = os.path.join(args.data_root, f"{args.category_name}_trainval.csv")
    test_saved_path = os.path.join(args.data_root, f"{args.category_name}_test.csv")
    split_data(
        gz_path=gz_path,
        train_saved_path=train_saved_path,
        val_saved_path=val_saved_path,
        trainval_saved_path=trainval_saved_path,
        test_saved_path=test_saved_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
