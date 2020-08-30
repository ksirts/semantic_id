#! /usr/bin/env python

# Create training and test data using K-folds
# The expected input is a json lines file and
# the output is K training and test files in json lines format

# Created by: Kairit Sirts
# Created at: 27.08.2020
# Contact: kairit.sirts@gmail.com

from argparse import ArgumentParser
import jsonlines
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os

def get_parsed_args():
    parser = ArgumentParser()
    parser.add_argument("-I", "--input", help="Input file in json lines format. Each item has to have an id attribute")
    parser.add_argument("-S", "--splits", type=int, default=10, help="Number of CV folds")
    parser.add_argument("-O", "--output_path", help="Path to output folder")
    return parser.parse_args()


def main(args):
    # Read the data in
    data = defaultdict(list)
    with jsonlines.open(args.input) as f:
        for item in f:
            data[item["id"]].append(item)

    # Create folds based on id-s
    ids = list(data.keys())
    labels = [data[id_][0]["label"] for id_ in ids]
    skf = StratifiedKFold(n_splits=args.splits)

    # Write training and test files
    basename = os.path.basename(args.input)
    root, ext = os.path.splitext(basename)
    for i, (train, test) in enumerate(skf.split(ids, labels)):
        # Print train items
        with jsonlines.open(os.path.join(args.output_path, root + "_train" + str(i) + ".jl"), 'w') as f: 
            for ind in train:
                for item in data[ids[ind]]:
                    f.write(item)
        # Print test items
        with jsonlines.open(os.path.join(args.output_path, root + "_test" + str(i) + ".jl"), 'w') as f: 
            for ind in test:
                for item in data[ids[ind]]:
                    f.write(item)


if __name__ == "__main__":
    args = get_parsed_args()
    main(args)
