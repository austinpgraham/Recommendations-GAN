#!/usr/bin/env python3
import os
import sys
import argparse

from surprise import SVD
from surprise import Dataset

from surprise.model_selection import cross_validate

from surprise.prediction_algorithms.knns import KNNWithMeans

def process_args(args=None):
    parser = argparse.ArgumentParser(description="Run and output statistics on SVD recommendations with MovieLens")
    parser.add_argument('-l', '--location', help="Output directory for results")
    args = parser.parse_args()

    return args.location


def main(args=None):
    location = process_args(args)

    out_path = os.path.expanduser(location)
    print('Checking output directory...')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        ans = input("Overwrite output directory?: ").upper()
        if ans == 'N' or ans == 'NO':
            print('Exiting...')
            exit()
    print("Loading dataset...")
    data = Dataset.load_builtin('ml-100k')
    algo = SVD()
    print("Running SVD...")
    sys.stdout = open(os.path.join(location, "svd_out.txt"), 'w')
    result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    from pdb import set_trace; set_trace()
    sys.stdout = sys.__stdout__
    print("Running KNN...")
    algo = KNNWithMeans()
    sys.stdout = open(os.path.join(location, "knn_out.txt"), 'w')
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    print("Done.")


if __name__ == '__main__':
    main(args)
