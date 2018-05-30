#!/usr/bin/env python3
import os
import sys
import argparse

from surprise import SVD
from surprise import Dataset

from surprise.model_selection import cross_validate


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
        os.path.makedirs(out_path)
    print("Loading dataset...")
    data = Dataset.load_builtin('ml-1m')
    algo = SVD()
    print("Running SVD...")
    sys.stdout = open(os.path.join(location, "out.txt"), 'w')
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


if __name__ == '__main__':
    main(args)
