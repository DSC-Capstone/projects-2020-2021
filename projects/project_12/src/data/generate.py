import datetime
import os
import sys
import warnings
import logging

from src.utils import ensure_path_exists

#python3.8 network_stats.py -i en0 -s -e jeq004-novideo-vpn-20201128.csv

def collect_data(
    username, provider, quality, speed, vpn, platform, clean, date, interface):
    """
    Captures data from network_stats
    """

    logging.info(
        'This functionality is moving to an entirely new repository, and its '
        'development in this repository has ceased being supported.',
    )

    csvmode = '-e'
    date = datetime.date.today().strftime('%Y%m%d')
    network_stats = 'network-stats/network_stats.py'

    output_file = '{}-{}-{}-{}-{}-{}-{}-{}.csv'.format(username, provider, quality, speed, vpn, platform, clean, date)
    ensure_path_exists(output_file)
    command = 'python3.8 {} -i {} -s {} {}'.format(network_stats, interface, csvmode, output_file)
    # os.system(command)
    
    return