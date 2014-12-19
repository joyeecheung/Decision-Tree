#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter

import matplotlib.pyplot as plt
from numpy import arange
from scipy.interpolate import interp1d

from util import get_filenames
from preprocess import parse, sample
from tree import build_tree, plurality

RESULT_IDX = 0
MIN_GAIN = 0.1


def main():
    files = get_filenames()
    x, y = [], []

    data = list(parse(file(files.dataset)))
    for test_prop in arange(0.1, 0.99, 0.05):
        train_set, test_set = sample(data, test_prop)
        tree = build_tree(train_set).prune(MIN_GAIN)
        check = [record[RESULT_IDX] == plurality(tree.classify(record))
                 for record in test_set]
        counter = Counter(check)
        precision = counter[True] / float(counter[True] + counter[False])
        print 'Test probability = %.2f:' % (test_prop)
        print 'training data size = %d,' % (len(train_set)),
        print 'test data size = %d,' % (len(test_set)),
        print 'precision = %.4f' % (precision)
        x.append(len(train_set))
        y.append(precision)

    xy = sorted(zip(x, y), key=lambda a: a[0])
    x, y = zip(*xy)
    plt.plot(x, y)
    plt.ylim((0.0, 1.0))
    plt.title('Learning Curve')
    plt.xlabel('Sample size')

    plt.ylabel('Precision')
    plt.savefig(files.curve)

if __name__ == "__main__":
    main()
