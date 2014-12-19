#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from copy import deepcopy
from random import random
from collections import Counter

from util import get_filenames

RESULT_IDX = 0
SEP = ','

TEST_PROP = 0.5  # proportion of test set
MISSING_SYMBOL = '?'


def parse(data):
    for line in data:
        yield line.strip().split(SEP)


def sample(data, test_prop=TEST_PROP):
    training_set = []
    test_set = []

    for record in data:
        if random() < test_prop:
            test_set.append(record)
        else:
            training_set.append(record)

    return training_set, test_set


def split_by_result(data):
    splited = {}
    for record in data:
        result = record[RESULT_IDX]
        splited.setdefault(result, [])
        splited[result].append(record)
    return splited


def main():
    files = get_filenames()
    # parse the file and sample
    training_set, test_set = sample(parse(file(files.dataset)))

    # write to file as json
    with file(files.train, 'w') as train_file:
        json.dump(training_set, train_file)
        print 'Dumped training set to %s,' % files.train,
        print 'size', len(training_set)
    with file(files.test, 'w') as test_file:
        json.dump(test_set, test_file)
        print 'Dumped test set to %s,' % files.test,
        print 'size', len(test_set)

    # write to file as json
    with file(files.train, 'r') as train_file:
        assert json.load(train_file) == training_set
        print 'Training file check OK.'
    with file(files.test, 'r') as test_file:
        assert json.load(test_file) == test_set
        print 'Test file check OK.'


if __name__ == "__main__":
    main()