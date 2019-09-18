from tabulate import tabulate

import given_files.dtree as d
import given_files.monkdata as m


def information_gain(dataset_names, datasets):

    header = ['Dataset'] + ['a{}'.format(i) for i in range(1, 7)] + ['Best Attribute']
    data = []

    for dataset_name, dataset in zip(dataset_names, datasets):

        row = [dataset_name]

        for gain in list(d.averageGain(dataset, m.attributes[i]) for i in range(6)):
            row.append(round(gain, 5))

        row.append(d.bestAttribute(dataset, m.attributes))
        data.append(row)

    print('# Information Gain')
    print(tabulate(data, header))


if __name__ == '__main__':

    dataset_names = ('MONK-1', 'MONK-2', 'MONK-3')
    datasets = (m.monk1, m.monk2, m.monk3)

    information_gain(dataset_names, datasets)
