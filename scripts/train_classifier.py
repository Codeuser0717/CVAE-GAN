import context

import pickle

import torch

import src
from src import Classifier, datasets, utils

dataset = 'CAN_HCRL_OTIDS'
# datasets='car_hacking'

if __name__ == '__main__':
    # utils.turn_on_test_mode()

    utils.set_random_state()
    utils.prepare_datasets(dataset)
    utils.set_random_state()
    clf = Classifier('test_0')
    clf.fit(datasets.TrDataset())
    clf.test(datasets.TeDataset())
    print(clf.confusion_matrix)
    clf.print_metrics(4)

