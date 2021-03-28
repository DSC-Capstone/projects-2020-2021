import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utm


def cep(**config):
    """Calculate circular error probable of gps coordinates"""
    df_neo = pd.read_csv(config["neo_fixed_gps"])
    df_zed_park = pd.read_csv(config["zed_fixed_gps_park"])
    df_zed_street = pd.read_csv(config["zed_fixed_gps_street"])

    lats_neo, lons_neo = df_neo.lat[:], df_neo.lon[:]
    lats_zed_park, lons_zed_park = df_zed_park.lat[5000:], df_zed_park.lon[5000:]
    lats_zed_street, lons_zed_street = df_zed_street.lat[:], df_zed_street.lon[:]

    neo_cep, neo_2drms = calculate_cep_2drms(lats_neo, lons_neo)
    zed_park_cep, zed_park_2drms = calculate_cep_2drms(lats_zed_park, lons_zed_park)
    zed_street_cep, zed_street_2drms = calculate_cep_2drms(lats_zed_street, lons_zed_street)

    plot_config = {
        "neo": {"lats": lats_neo,
                "lons": lons_neo,
                "cep": neo_cep,
                "two_drms": neo_2drms},
        "park": {"lats": lats_zed_park,
                 "lons": lons_zed_park,
                 "cep": zed_park_cep,
                 "two_drms": zed_park_2drms},
        "street": {"lats": lats_zed_street,
                 "lons": lons_zed_street,
                 "cep": zed_street_cep,
                 "two_drms": zed_street_2drms},
    }

    plot_results(**plot_config["neo"],
                 title="NEO-M8N CEP and 2DRMS",
                 fig_file_name="neo_m8n_cep_2drms.png")
    plot_results(**plot_config["park"],
                 title="ZED-F9P CEP and 2DRMS at Park",
                 fig_file_name="zed_f9p_park_cep_2drms.png")
    plot_results(**plot_config["street"],
                 title="ZED-F9P CEP and 2DRMS in Neighborhood",
                 fig_file_name="zed_f9p_street_cep_2drms.png")

    for i in range(1, 11):
        plot_multiple(plot_config, scale=i)


def calculate_cep_2drms(lats, lons):
    print("Calculating CEP and 2DRMS")
    utm_x, utm_y = convert_utm(list(zip(lats, lons)))
    sigma_x, sigma_y = np.std(utm_x), np.std(utm_y)

    cep = 0.59 * (sigma_x + sigma_y)
    two_d_rms =  2 * np.sqrt(sigma_x**2 + sigma_y**2)

    return cep, two_d_rms


def calculate_cep(x_utm, y_utm):
    sigma_x, sigma_y = np.std(x_utm), np.std(y_utm)
    cep = 0.59 * (sigma_x + sigma_y)
    return cep


def calculate_two_d_rms(x_utm, y_utm):
    """Calculate 2DRMS for coordinates"""
    sigma_x, sigma_y = np.std(x_utm), np.std(y_utm)
    return 2 * np.sqrt(sigma_x**2 + sigma_y**2)


def convert_utm(coords):
    """Convert latitude and longitude into UTM coordinates"""
    utm_x, utm_y = [], []
    # convert every coordinate pair into utm
    for lat, lon in coords:
        utm_coord = utm.from_latlon(lat, lon)
        utm_x.append(utm_coord[0])
        utm_y.append(utm_coord[1])

    # offset points to be difference from means
    utm_x = (utm_x - np.mean(utm_x))
    utm_y = (utm_y - np.mean(utm_y))

    return utm_x, utm_y


def plot_results(lats, lons, cep, two_drms, title=None, fig_file_name=None, show=False):
    """Plot Coordiantes and respective circles"""
    x, y = convert_utm(list(zip(lats, lons)))
    center = (np.mean(x), np.mean(y))
    fig, ax = plt.subplots(figsize=(10, 10)) # note we must use plt.subplots, not plt.subplot

    # Add circles for CEP and 2DRMS
    circle_cep = plt.Circle(center, radius=cep, color='r', fill=True, alpha=0.15)
    circle_drms = plt.Circle(center, radius=two_drms, color='g', fill=True, alpha=0.15)
    ax.add_patch(circle_cep)
    ax.add_patch(circle_drms)

    # Add points; set size of points in scatter
    point_sizes = np.ones(len(x)) * 2
    position = plt.scatter(x, y, point_sizes)

    # plot formatting
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.title(title)
    plt.xlabel("Delta X (m)")
    plt.ylabel("Delta Y (m)")
    plt.axis('equal')
    plt.legend([position, circle_cep, circle_drms],
               ["Distance From Average", "CEP (50%)", "2DRMS (95-98%)"])

    # Save figure
    if fig_file_name is not None:
        print(f"Saving Figure: {fig_file_name}")
        plt.savefig(fig_file_name)

    # only show plots when called upon
    if show:
        plt.show()


def plot_multiple(config, scale, show=False):
    x_park, y_park = convert_utm(
        list(zip(config["park"]["lats"], config["park"]["lons"])))
    x_street, y_street = convert_utm(
        list(zip(config["street"]["lats"], config["street"]["lons"])))

    center_park = (np.mean(x_park), np.mean(y_park))
    center_street = (np.mean(x_street), np.mean(y_street))

    fig, ax = plt.subplots(figsize=(10, 10)) # note we must use plt.subplots, not plt.subplot

    # Add circles for CEP and 2DRMS
    street_alpha = 0.15
    circle_cep_street = plt.Circle(center_street,
                                   radius=config["street"]["cep"],
                                   color='r', fill=True, alpha=street_alpha)
    circle_drms_street = plt.Circle(center_street,
                                    radius=config["street"]["two_drms"],
                                    color='g', fill=True, alpha=street_alpha)

    circle_cep_park = plt.Circle(center_park,
                                 radius=config["park"]["cep"],
                                 color='r', fill=True, alpha=0.15)
    circle_drms_park = plt.Circle(center_park,
                                  radius=config["park"]["two_drms"],
                                  color='g', fill=True, alpha=0.15)

    # Add circles to plot
    ax.add_patch(circle_cep_street)
    ax.add_patch(circle_drms_street)
    ax.add_patch(circle_cep_park)
    ax.add_patch(circle_drms_park)

    # Add points; set size of points in scatter
    point_sizes_street = np.ones(len(x_street)) * 2
    point_sizes_park = np.ones(len(x_park)) * 2

    position_street = plt.scatter(x_street, y_street, point_sizes_street)
    position_park = plt.scatter(x_park, y_park, point_sizes_park, color="r")

    plt.legend([position_street, position_park],
               ["ZED-F9P Neighborhood", "ZED-F9P Park"])
    plt.title("ZED-F9P Neighborhood vs Park")
    plt.xlabel("Delta X (m)")
    plt.ylabel("Delta Y (m)")

    # zoom in too park points
    plt.xlim(-scale, scale)
    plt.ylim(-scale, scale)

    # Save figure
    fig_file_name = f"zed_f9p_park_vs_street_{scale}m.png"
    print(f"Saving Figure: {fig_file_name}")
    plt.savefig(fig_file_name)

    if show:
        plt.show()
