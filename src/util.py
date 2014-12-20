#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple
import os

DATASET_FILE = 'dataset.txt'
TRAIN_FILE = 'training.json'
TEST_FILE = 'testing.json'
CURVE_FIG = 'learning-curve.png'
TREE_PLOT = 'tree.png'
PRUNED_TREE_PLOT = 'pruned-tree.png'


def get_filenames():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir, _ = os.path.split(file_dir)
    asset_path = os.path.join(parent_dir, 'asset')
    filenames = map(lambda name: os.path.join(asset_path, name),
                    (DATASET_FILE, TRAIN_FILE, TEST_FILE,
                     CURVE_FIG, TREE_PLOT, PRUNED_TREE_PLOT))

    fn = namedtuple('Files', ['dataset', 'train', 'test',
                              'curve', 'tree', 'pruned_tree'])
    return fn(*filenames)
