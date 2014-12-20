#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from copy import deepcopy
from random import random
from collections import Counter
import argparse

from util import get_filenames

RESULT_IDX = 0
SEP = ','

TRAIN_PROP = 0.5  # sampling probability of test set
MISSING_SYMBOL = '?'


def parse(data):
    """Parse the data. Return an iterator."""
    for line in data:
        yield line.strip().split(SEP)


def sample(data, train_prop=TRAIN_PROP):
    """Split the data by sampling training set with probability `train_prop`.
       The rest will be put into training set.

       Return
       -------
       training_set, testing_set."""
    training_set = []
    testing_set = []

    for record in data:
        if random() < train_prop:
            training_set.append(record)
        else:
            testing_set.append(record)

    return training_set, testing_set


def main():
    # get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--probability",
                        type=float, default=TRAIN_PROP)
    args = parser.parse_args()

    # parse the file and sample
    files = get_filenames()
    data_parser = parse(file(files.dataset))
    training_set, testing_set = sample(data_parser, args.probability)

    # write to file as json
    with file(files.train, 'w') as train_file:
        json.dump(training_set, train_file)
        print 'Dumped training set to %s,' % files.train,
        print 'size', len(training_set)
    with file(files.test, 'w') as test_file:
        json.dump(testing_set, test_file)
        print 'Dumped test set to %s,' % files.test,
        print 'size', len(testing_set)

    # check the file is written correctly
    with file(files.train, 'r') as train_file:
        assert json.load(train_file) == training_set
        print 'Training file check OK.'
    with file(files.test, 'r') as test_file:
        assert json.load(test_file) == testing_set
        print 'Test file check OK.'


if __name__ == "__main__":
    main()
