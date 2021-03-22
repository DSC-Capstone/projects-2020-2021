import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utm


def plot_ground_truth(**config):
    df_gt = pd.read_csv(config["ground_truth"])
    df_log = pd.read_csv(config["logged_wps"])

    x_gt, y_gt = df_gt.x[:], df_gt.y[:]
    x_log, y_log = df_log.x[:], df_log.y[:]

    plot_config = {
        "ground_truth": {"x": x_gt, "y": y_gt},
        "logs": {"x": x_log, "y": y_log}
    }
    plot_multiple(plot_config, show=True)



def plot_multiple(config, show=False):
    fig, ax = plt.subplots(figsize=(10, 10)) # note we must use plt.subplots, not plt.subplot

    # Add circles for CEP and 2DRMS
    # Add points; set size of points in scatter
    point_sizes_gt = np.ones(len(config["ground_truth"]["x"])) * 2
    point_sizes_logs = np.ones(len(config["logs"]["x"])) * 2

    position_gt = plt.scatter(config["ground_truth"]["x"],
                              config["ground_truth"]["y"],
                              point_sizes_gt)
    position_logs = plt.scatter(config["logs"]["x"],
                              config["logs"]["y"],
                               point_sizes_logs)

    plt.legend([position_gt, position_logs],
               ["Ground Truth", "Reported Localization"])
    plt.title("Ground Truth vs Reported GPS Position")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    # zoom in too park points
    # plt.xlim(-scale, scale)
    # plt.ylim(-scale, scale)

    # Save figure
    # fig_file_name = f"zed_f9p_park_vs_street_{scale}m.png"
    # print(f"Saving Figure: {fig_file_name}")

    # plt.savefig(fig_file_name)

    if show:
        plt.show()
