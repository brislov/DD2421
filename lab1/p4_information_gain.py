import given_files.dtree as dtree
import given_files.monkdata as monkdata


"""
Assignment 3:

Dataset	a1		a2		a3		a4		a5		a6		BestAttr
MONK-1	0.07527	0.00584	0.00471	0.02631	0.28703	0.00076	A5
MONK-2	0.00376	0.00246	0.00106	0.01566	0.01728	0.00625	A5
MONK-3	0.00712	0.29374	0.00083	0.00289	0.25591	0.00708	A2
"""


"""
Assignment 4:

The more information gain an attribute has the smaller the entropy of Sk will be. With an attribute which solely can be
used to find the correct response would have be:
    Gain(S, A) = Entropy(S) - 0 = Entropy(S)
This means that the maximal information gain an attribute can have is the Entropy(S).
"""


def information_gain(datasets, dataset_names):
    """
    Calculates information gain for all attributes in all data sets and prints a table with the result.
    :param datasets: list of datasets.
    :param dataset_names: list of dataset names.
    :return: void
    """
    s = '--- INFORMATION GAIN ---\nDataset\t'

    for i in range(1, 7):
        s += 'a{}\t\t'.format(i)

    s += 'BestAttr'

    for dataset, dataset_name in zip(datasets, dataset_names):

        s += '\n{}\t'.format(dataset_name)

        for gain in list(dtree.averageGain(dataset, monkdata.attributes[i]) for i in range(6)):
            s += '{}\t'.format(round(gain, 5))
            s += '\t' if not gain else ''  # Makes sure that column width is correct if gain = 0

        s += '{}'.format(dtree.bestAttribute(dataset, monkdata.attributes))

    print(s)


if __name__ == '__main__':
    datasets = (monkdata.monk1, monkdata.monk2, monkdata.monk3)
    dataset_names = ('MONK-1', 'MONK-2', 'MONK-3')

    information_gain(datasets, dataset_names)
