from tabulate import tabulate

import python.dtree as d
import python.monkdata as m


if __name__ == '__main__':

    precision = 5

    header = ['Dataset', 'Entropy']
    data = [
        ['MONK-1', round(d.entropy(m.monk1), precision)],
        ['MONK-2', round(d.entropy(m.monk2), precision)],
        ['MONK-3', round(d.entropy(m.monk3), precision)]
    ]

    print(tabulate(data, header))
