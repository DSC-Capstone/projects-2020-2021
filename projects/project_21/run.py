
from graphsage.utils import load_data
from graphsage.supervised_train import train

import sys


def main(targets):

    train_data = load_data('data/sage_NBA/nba')

    train(train_data)

if __name__ == '__main__':

    targets = sys.argv[1:]

    main(targets)
