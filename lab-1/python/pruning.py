import random
import statistics

from tabulate import tabulate
import matplotlib.pyplot as plt

import python.dtree as dtree
import python.monkdata as m


def get_best_tree(currtree):
    found_better_tree = False

    for newtree in dtree.allPruned(currtree):
        if dtree.check(newtree, monkval) > dtree.check(currtree, monkval):
            found_better_tree = True
            currtree = newtree

    if found_better_tree:
        currtree = get_best_tree(currtree)
    return currtree



def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


if __name__ == '__main__':

    dataset_names = ('MONK-1', 'MONK-3')
    datasets = (m.monk1, m.monk3)
    datasets_test = (m.monk1test, m.monk3test)

    fractions = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    decimals = 3
    n = 1000

    header = ['Fraction', 'Mean error (n = {})'.format(n), 'Standard deviation']

    for dataset_name, dataset, dataset_test in zip(dataset_names, datasets, datasets_test):

        data = []
        mean_errors = []
        stdev = []

        for fraction in fractions:

            errors = []

            for i in range(n):
                monktrain, monkval = partition(dataset, fraction)
                built_tree = dtree.buildTree(monktrain, m.attributes)
                best_tree = get_best_tree(built_tree)

                errors.append(1 - dtree.check(best_tree, dataset_test))

            mean_error = round(statistics.mean(errors), decimals)
            mean_errors.append(round(statistics.mean(errors), decimals))

            stdev.append(round(statistics.stdev(errors), decimals))

            data.append([fraction, mean_error, statistics.mean(stdev)])

        print(tabulate(data, header), '\n')

        plt.errorbar(fractions, mean_errors, yerr=stdev, marker='o')

        plt.title('{} (n = {})'.format(dataset_name, n))
        plt.xlabel('fraction')
        plt.ylabel('mean error')
        plt.show()
