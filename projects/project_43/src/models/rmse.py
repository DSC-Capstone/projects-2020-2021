# Copyright (c) 2017 NVIDIA Corporation
from math import sqrt
import sys


def main(configs):
    with open(configs['prediction_location'], 'r') as inpt:
        lines = inpt.readlines()
        n = 0
        denom = 0.0
        for line in lines:
            parts = line.split('\t')
            prediction = float(parts[2]) if not configs['round'] else round(float(parts[2]))
            rating = float(parts[3])
            denom += (prediction - rating)*(prediction - rating)
            n += 1
    print("####################")
    print("RMSE: {}".format(sqrt(denom/n)))
    print("####################")

if __name__ == '__main__':
    main(sys.argv)