#!/usr/bin/env python

import argparse
import json

from src import analysis, data, robot


TARGETS = {
    "cep": analysis.cep,
    "clean_data": data.clean_gps,
    "get_path": analysis.get_path,
    "ground_truth": analysis.plot_ground_truth,
    "robot": robot.Robot,
    "robot_client": robot.RobotClient,
    "test": data.clean_gps,
    "visualize": analysis.visualize_all,
}

CONFIGS = {
    "cep": "config/cep.json",
    "clean_data": "config/clean_gps.json",
    "get_path": "config/get_path.json",
    "ground_truth": "config/ground_truth.json",
    "robot": "config/robot_sim.json",
    "robot_client": "config/robot_client.json",
    "test": "config/test.json",
    "visualize": "config/visualization.json",
}


def main() -> None:
    """Runs project pipeline and calls designated targets"""
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=TARGETS.keys())
    args = parser.parse_args()

    with open(CONFIGS[args.target]) as cf:
        config = json.load(cf)

    # initiate target sequence with designated configuration
    TARGETS[args.target](**config)


if __name__ == '__main__':
    main()
