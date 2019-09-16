import given_files.dtree as dtree
import given_files.monkdata as monkdata


"""
Assignment 1:

Dataset	Entropy
MONK-1	1.0
MONK-2	0.95712
MONK-3	0.99981
"""


"""
Assignment 2:

Entropy is increases when the uncertainty of outcome increases. If one outcome is heavily biased, e.g. occurs 99%, the 
would be small. Therefore, in non-uniform distributions the entropy will be smaller than in uniform distributions, given
that the same datasets are used.
"""


if __name__ == '__main__':
    s = 'Dataset\tEntropy\n'

    s += 'MONK-1\t{}\n'.format(round(dtree.entropy(monkdata.monk1), 5))
    s += 'MONK-2\t{}\n'.format(round(dtree.entropy(monkdata.monk2), 5))
    s += 'MONK-3\t{}\n'.format(round(dtree.entropy(monkdata.monk3), 5))

    print(s)
