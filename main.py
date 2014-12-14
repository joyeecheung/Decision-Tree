#!/usr/bin/env python
# -*- coding: utf-8 -*-
from preprocess import parse, sample
from tree import build_tree, plurality
from collections import Counter
import matplotlib.pyplot as plt
from numpy import arange
from scipy.interpolate import interp1d

dataset_filename = 'dataset.txt'
RESULT_IDX = 0


def main():
    x, y = [], []

    data = list(parse(file(dataset_filename)))
    for test_prop in arange(0.1, 0.9, 0.05):
        train_set, test_set = sample(data, test_prop)
        tree = build_tree(train_set)
        check = [record[RESULT_IDX] == plurality(tree.classify(record))
                 for record in test_set]
        counter = Counter(check)
        precision = counter[True] / float(counter[True] + counter[False])
        print len(train_set), len(test_set), precision
        x.append(len(train_set))
        y.append(precision)

    xy = sorted(zip(x, y), key=lambda a: a[0])
    x, y = zip(*xy)
    plt.plot(x, y)
    plt.ylim((0.0, 1.0))
    plt.title('Learning Curve')
    plt.xlabel('Sample size')

    plt.ylabel('Precision')
    plt.savefig('learning-curve.png')

if __name__ == "__main__":
    main()
