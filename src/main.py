#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from util import get_filenames
from preprocess import parse, sample
from tree import build_tree, plurality

RESULT_IDX = 0
MIN_GAIN = 0.1


def main():
    files = get_filenames()
    x, y = [], []

    # generate the learning curve data
    data = list(parse(file(files.dataset)))
    for train_prop in np.arange(0.1, 0.99, 0.05):
        training_set, testing_set = sample(data, train_prop)
        tree = build_tree(training_set).prune(MIN_GAIN)
        check = [record[RESULT_IDX] == plurality(tree.classify(record))
                 for record in testing_set]
        counter = Counter(check)
        precision = counter[True] / float(counter[True] + counter[False])
        print 'Training set sampling probability = %.2f:' % (train_prop)
        print 'training data size = %d,' % (len(training_set)),
        print 'test data size = %d,' % (len(testing_set)),
        print 'precision = %.4f' % (precision)
        x.append(len(training_set))
        y.append(precision)

    # statistics
    ymean, ystd, ymin, ymax = np.mean(y), np.std(y), np.min(y), np.max(y)
    print 'Mean of precision = %.4f' % (ymean)
    print 'Standard deviation of precision = %.4f' % (ystd)
    print 'Min = %.4f, max = %.4f' % (ymin, ymax)
    xy = sorted(zip(x, y), key=lambda a: a[0])
    x, y = zip(*xy)

    # setup decorations
    plt.rc('font', family='serif')
    plt.yticks(np.arange(0.0, 1.0, 0.1))
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.title('Learning Curve')
    plt.xlabel('Training set size')
    plt.ylabel('Precision on test set')

    # plot smoothed learning curve
    xnew = np.linspace(np.min(x), np.max(x), 100)
    ynew = interp1d(x, y)(xnew)
    plt.plot(x, y, '.', xnew, ynew, '--')

    # annotation
    box = dict(boxstyle='square', fc="w", ec="k")
    txt = '$\mu = %.4f$, $\sigma = %.4f$' % (ymean, ystd)
    txt += ', $min = %.4f$, $max = %.4f$' % (ymin, ymax)
    plt.text(170, 0.05, txt, bbox=box)

    plt.savefig(files.curve)
    print 'Save learning curve to', files.curve

if __name__ == "__main__":
    main()
