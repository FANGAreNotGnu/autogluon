import argparse
import faiss
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
np.random.seed(123)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_root', type=str, default="/media/ag/data/amazonreview/")
    parser.add_argument('-n', '--category_name', type=str, default="All_Beauty")
    parser.add_argument('-s', '--save_name', type=str)
    parser.add_argument('-m', '--dimension', type=int, default=128) # high_quality:128
    parser.add_argument('-v', '--use_val_as_database', action="store_true")
    parser.add_argument('-k', '--num_neighbors', type=int, default=100) # high_quality:128
    return parser


def main(args):
    train_emb_path = os.path.join(args.data_root, f"{args.category_name}_train_emb_{args.save_name}.npy")
    val_emb_path = os.path.join(args.data_root, f"{args.category_name}_val_emb_{args.save_name}.npy")
    test_emb_path = os.path.join(args.data_root, f"{args.category_name}_test_emb_{args.save_name}.npy")

    test_I_path = os.path.join(args.data_root, f"{args.category_name}_test_I_{args.save_name}.npy")
    test_D_path = os.path.join(args.data_root, f"{args.category_name}_test_D_{args.save_name}.npy")

    index = faiss.IndexFlatL2(args.dimension)  # build the index
    print(index.is_trained)

    database = np.load(train_emb_path)
    index.add(database)
    print(index.ntotal)
    if args.use_val_as_database:
        index.add(np.load(val_emb_path))
        print(index.ntotal)

    D, I = index.search(database[:5], args.num_neighbors)  # sanity check
    print("sanity check:")
    print(I)
    print(D)

    queries = np.load(test_emb_path)
    D, I = index.search(queries, args.num_neighbors)  # actual search
    print(I[:5])  # neighbors of the 5 first queries
    print(D[:5])  # neighbors of the 5 first queries
    print(I[-5:])  # neighbors of the 5 last queries
    print(D[-5:])  # neighbors of the 5 last queries

    np.save(test_I_path,I)
    np.save(test_D_path,D)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
