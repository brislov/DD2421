import random

import matplotlib.pyplot as plt

import given_files.drawtree_qt5 as drawtree
import given_files.dtree as dtree
import given_files.monkdata as monkdata


"""
Assignment 6:

The depth/complexity of a decision tree determines the variance. By pruning we're simplifying the tree and therefore are
reducing the variance (reduces overfitting). Bias does not seem to be affected by prunning.
"""


"""
Assignment 7:


"""


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


def get_best_tree(currtree):
    found_better_tree = False

    for newtree in dtree.allPruned(currtree):
        if dtree.check(newtree, monkval) > dtree.check(currtree, monkval):
            found_better_tree = True
            currtree = newtree

    if found_better_tree:
        currtree = get_best_tree(currtree)

    return currtree


if __name__ == '__main__':

    n = 100
    datasets = (monkdata.monk1, monkdata.monk3)
    dataset_names = ('MONK-1', 'MONK-3')
    fractions = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

    for dataset, dataset_name in zip(datasets, dataset_names):

        s = '--- {} ---\nFraction\tMean accuracy\n'.format(dataset_name)
        means = list()

        for fraction in fractions:

            mean = 0

            for i in range(n):
                monktrain, monkval = partition(dataset, fraction)

                basetree = dtree.buildTree(monktrain, monkdata.attributes)
                besttree = get_best_tree(basetree)

                mean += (1 - dtree.check(besttree, monkval)) / n * 100

            s += '{}\t\t\t{}\n'.format(fraction, round(mean, 5))
            means.append(mean)

        print(s)

        plt.plot(fractions, means, marker='o', label=dataset_name)

    plt.title('Classification error for each fraction (n = {})'.format(n))
    plt.legend()
    plt.xlabel('fraction')
    plt.ylabel('classification error [%]'.format(n))

    plt.show()
