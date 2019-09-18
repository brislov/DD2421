from tabulate import tabulate

import given_files.drawtree_qt5 as drawtree
import given_files.dtree as dtree
import given_files.monkdata as monkdata

import p4_information_gain as p4


if __name__ == '__main__':

    # DECISION TREE PART
    # First paragraph
    subsets = list(dtree.select(monkdata.monk1, monkdata.attributes[4], value) for value in monkdata.attributes[4].values)
    subset_names = list('Subset{}'.format(i) for i in range(1, len(subsets) + 1))
    p4.information_gain(subset_names, subsets)  # subsets

    # Second paragraph
    # drawtree.drawTree(dtree.buildTree(monkdata.monk1, monkdata.attributes, maxdepth=2))
    s = '\n# Most Common\n'
    for subset, subset_name in zip(subsets, subset_names):
        s += '{} {}\n'.format(subset_name, dtree.mostCommon(subset))
    print(s)

    # Onwards
    tree = dtree.buildTree(monkdata.monk1, monkdata.attributes, maxdepth=1)  # Two levels
    # tree = dtree.buildTree(monkdata.monk1, monkdata.attributes)  # All levels
    # drawtree.drawTree(tree)  # Show tree

    print('-' * 80, '\n')

    # PERFORMANCE CHECK PART
    name_sets = ('MONK-1', 'MONK-2', 'MONK-3')
    training_sets = (monkdata.monk1, monkdata.monk2, monkdata.monk3)
    test_sets = (monkdata.monk1test, monkdata.monk2test, monkdata.monk3test)

    trees = list(dtree.buildTree(training_set, monkdata.attributes) for training_set in training_sets)


    print('# Performance Check')

    header = ['Dataset', 'Train', 'Test']
    data = []

    for tree, name_set, training_set, test_set in zip(trees, name_sets, training_sets, test_sets):
        data.append([name_set, 1 - round(dtree.check(tree, training_set), 5), 1 - round(dtree.check(tree, test_set), 5)])

    print(tabulate(data, header))
