#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from copy import deepcopy
from random import random
from collections import Counter

RESULT_IDX = 0
SEP = ','
filename = 'dataset.txt'
TEST_PROP = 0.1
MISSING_SYMBOL = '?'
RESULTS = ['democrat', 'republican']
train_file = file('training.json', 'w')
test_file = file('test.json', 'w')


def parse(data):
    for line in data:
        yield line.strip().split(SEP)


def divide(data):
    training_set = []
    test_set = []

    for record in data:
        if random() < TEST_PROP:
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
    training_set, test_set = divide(parse(file(filename)))
    training_set = fix_missing_data(training_set)
    train_file = file('training.json', 'w')
    test_file = file('test.json', 'w')
    json.dump(training_set, train_file)
    json.dump(test_set, test_file)

    train_file.close()
    test_file.close()

    train_file = file('training.json', 'r')
    test_file = file('test.json', 'r')

    print json.load(train_file) == training_set
    print json.load(test_file) == test_set

if __name__ == "__main__":
    main()
