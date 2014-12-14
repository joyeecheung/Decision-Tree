#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from copy import deepcopy
from random import random
from collections import Counter

RESULT_IDX = 0
SEP = ','

TEST_PROP = 0.5  # proportion of test set
MISSING_SYMBOL = '?'

dataset_filename = 'dataset.txt'
train_filename = 'training.json'
test_filename = 'test.json'


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


def fix_missing_data(data):
    fixed = deepcopy(data)
    attr_list = list(xrange(len(data[0])))
    del attr_list[RESULT_IDX]
    splited = split_by_result(data)

    guess = {}
    for group in splited:
        for attr in attr_list:
            counter = Counter(record[attr] for record in data
                              if record[RESULT_IDX] == group)
            guess.setdefault(group, {})
            guess[group][attr] = counter.most_common()[0][0]

    for record in fixed:
        group = record[RESULT_IDX]
        for attr in attr_list:
            if record[attr] == MISSING_SYMBOL:
                record[attr] = guess[group][attr]

    return fixed


def main():
    # parse the file and sample
    training_set, test_set = sample(parse(file(dataset_filename)))

    # write to file as json
    with file(train_filename, 'w') as train_file:
        json.dump(training_set, train_file)
        print 'Dumped training set to %s,' % train_filename,
        print 'size', len(training_set)
    with file(test_filename, 'w') as test_file:
        json.dump(test_set, test_file)
        print 'Dumped test set to %s,' % test_filename,
        print 'size', len(test_set)

    # write to file as json
    with file(train_filename, 'r') as train_file:
        assert json.load(train_file) == training_set
        print 'Training file check OK.'
    with file(test_filename, 'r') as test_file:
        assert json.load(test_file) == test_set
        print 'Test file check OK.'


if __name__ == "__main__":
    main()
