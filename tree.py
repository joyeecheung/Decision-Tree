#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter, defaultdict
from math import log
from PIL import Image, ImageDraw, ImageFont
import json

from preprocess import parse, sample

############### configuration ##############

RESULT_IDX = 0
SEP = ','
MISSING = '?'

CLASSES = ['democrat', 'republican']
VALUES = ['n', 'y']

TREE_GRID = 60
TREE_SPACING = 10
BLACK = 0
WHITE = 255
LINE_HEIGHT = 18

train_data = json.load(file('training.json'))
test_data = json.load(file('test.json'))

font = ImageFont.truetype('Arial.ttf', LINE_HEIGHT)
abbr = {'republican': 'R', 'democrat': 'D'}

##################################################


def divide_bool(data, attr, value):
    """Divide the dataset into two subset by attr."""
    counter = count(filter(lambda x: x[attr] != MISSING, data), attr)
    if len(counter) == 0:  # all missing
        return data, list(), True

    guess = counter.most_common()[0][0]
    fix_missing = lambda x: guess if x == MISSING else x

    left = lambda x: fix_missing(x[attr]) == value
    right = lambda x: fix_missing(x[attr]) != value

    return filter(left, data), filter(right, data), False


def count(data, attr=RESULT_IDX):
    """Count the attribute distribution."""
    return Counter(record[attr] for record in data)


def entropy(data):
    v = count(data).values()  # count democrats and republicans
    ent, total = 0.0, len(data)

    for vk in v:
        p = float(vk) / total
        ent -= p * log(p, 2)
    return ent


class Node(object):

    """Decision tree node."""

    def __init__(self, attr=RESULT_IDX, value=None,
                 left=None, right=None, leaves=None, count=None):
        self.attr = attr
        self.value = value
        self.leaves = leaves  # the classification or regression
        self.left = left
        self.right = right
        self.count = count

    def __repr__(self):
        return self.to_string()

    def to_string(tree, indent=''):
        result = ''
        if tree.leaves:
            return result + str(tree.leaves) + '\n'
        else:
            # Print the criteria
            result += str(tree.attr) + ':' + str(tree.value) + '? '
            result += str(tree.count) + '\n'
            # Print the branches
            result += indent + 'T->'
            result += tree.left.to_string(indent + '   ')
            result += indent + 'F->'
            result += tree.right.to_string(indent + '   ')
            return result

    def classify(tree, observation):
        """Classify the observation using the decision tree."""
        if tree.leaves:
            return tree.leaves

        value = observation[tree.attr]
        if value == MISSING:
            probe_left = tree.left.classify(observation)
            probe_right = tree.right.classify(observation)

            left_weight = float(tree.left.count) / tree.count
            right_weight = float(tree.right.count) / tree.count
            result = defaultdict(int)
            for k, v in probe_left.items():
                result[k] += v * left_weight
            for k, v in probe_right.items():
                result[k] += v * right_weight

            return dict(result)

        branch = tree.left if value == tree.value else tree.right
        return branch.classify(observation)

    def get_width(tree):
        if not tree.left and not tree.right:
            return 1
        else:
            return tree.left.get_width() + tree.right.get_width()

    def get_height(tree):
        if not tree.left and not tree.right:
            return 0
        else:
            return max(tree.left.get_height(),
                       tree.right.get_height()) + 1

    def to_image(tree):
        """Draw the tree onto an image."""
        width = tree.get_width() * TREE_GRID
        height = (tree.get_height() + 1) * TREE_GRID + TREE_SPACING

        im = Image.new('L', (width, height), WHITE)
        draw = ImageDraw.Draw(im)

        tree.draw_node(draw, width / 2, TREE_SPACING)
        return im

    def draw_node(tree, draw, x, y):
        """Draw the tree onto the draw object."""
        if tree.leaves:
            for idx, key in enumerate(tree.leaves):
                result = abbr[key] + ':' + str(tree.leaves[key])
                draw.text((x, y + idx * LINE_HEIGHT),
                          result, BLACK, font=font)
        else:
            width_left = tree.left.get_width() * TREE_GRID
            width_right = tree.right.get_width() * TREE_GRID

            center_left = x - width_right / 2
            center_right = x + width_left / 2

            draw.text((x - TREE_SPACING, y - TREE_SPACING),
                      str(tree.attr) + ':' + str(tree.value),
                      BLACK, font=font)

            draw.line((x, y, center_left, y + TREE_GRID),
                      fill=BLACK)
            draw.line((x, y, center_right, y + TREE_GRID),
                      fill=BLACK)

            tree.left.draw_node(draw, center_left, y + TREE_GRID)
            tree.right.draw_node(draw, center_right, y + TREE_GRID)


def build_tree(data, attr_list=None, score=entropy):
    """Build the decision tree from data."""
    # empty data
    if len(data) == 0:
        return Node()

    current_score = score(data)
    best_gain, best_criteria, best_sets = 0.0, None, None

    if attr_list is None:
        attr_list = list(xrange(len(data[0])))
        del attr_list[RESULT_IDX]
    value = VALUES[0]

    for attr in attr_list:
        set1, set2, all_missed = divide_bool(data, attr, value)
        if all_missed:
            gain = 0.0
        else:
            p = float(len(set1)) / len(data)
            remainder = p * score(set1) + (1 - p) * score(set2)
            gain = current_score - remainder

        if (gain > best_gain and len(set1) > 0 and len(set2) > 0):
            best_gain, best_criteria = gain, (attr, value)
            best_sets = (set1, set2)

    # create subbranches
    # when there are more attribute to test
    if best_gain > 0 and len(attr_list) != 0 and best_sets:
        attr_list.remove(best_criteria[0])
        left = build_tree(best_sets[0], list(attr_list))
        right = build_tree(best_sets[1], list(attr_list))
        return Node(best_criteria[0], best_criteria[1],
                    left, right, count=len(data))
    else:  # all tested
        return Node(leaves=count(data), count=len(data))


def plurality(counter):
    """Get the decision based on leaves of the decision tree."""
    return max(counter.iterkeys(), key=lambda k: counter[k])


def main():
    tree = build_tree(train_data)
    check = [record[RESULT_IDX] == plurality(tree.classify(record))
             for record in test_data]
    print Counter(check)
    tree.to_image().save('tree.png')


if __name__ == "__main__":
    main()
