import glob
import ipaddress
import itertools
import numpy as np
import os
import pandas as pd
import pathlib

import multiprocessing
import time

import logging

from src.utils import ensure_path_exists

DATA_DIRECTORY = "/teams/DSC180A_FA20_A00/b05vpnxray/group2_data/data"
# network-stats records per-packet timing in milliseconds,
PACKET_TIMESTAMP_UNIT = 'ms'


def clean(df):
    
    """
    Attempts to filter out everything besides the traffic flow between the
    client and VPN service.

    The primary part of cleaning is merely getting rid of irrelevant flows.
    Since we are operating under the assumption that a VPN is in use, our
    clearning can *hopefully* isolate just the client<->VPN flow.

    At the moment, our approach is to remove any communications to link-local
    IPs, any multicast communications, and any communications between two
    private IPs.
    """

    ip1, ip2 = df.IP1.map(ipaddress.ip_address), df.IP2.map(ipaddress.ip_address)

    either_link_local = (
        (ip1.map(lambda x: x.is_link_local))
        | (ip2.map(lambda x: x.is_link_local))
    )

    either_multicast = (
        (ip1.map(lambda x: x.is_multicast))
        | (ip2.map(lambda x: x.is_multicast))
    )

    both_private = (
        (ip1.map(lambda x: x.is_private))
        & (ip2.map(lambda x: x.is_private))
    )

#     print("new shape: ")
#     print(df[~either_link_local
#         & ~either_multicast
#         & ~both_private].shape)
    
   
    return df[~either_link_local
        & ~either_multicast
        & ~both_private]

    

def unbin(df):
    """
    Takes in a DataFrame formatted as a network-stats output and 'unbins' the
    packet-level measurements so that each packet gets its own row.
    """
    
    packet_cols = ['packet_times', 'packet_sizes', 'packet_dirs']
    
    # Convert the strings `val1;val2;` into lists `[val1, val2]`
    df_listed = (
        df[packet_cols]
        .apply(lambda ser: ser.str.split(';').str[:-1])
    )
    
    # 'Explode' the lists so each element gets its own row.
    #
    # Exploding is considerably faster than summing lists and creating a new
    # frame.
    unbinned = (
        df_listed
        # Each list contains integer values, so we'll also cast them now.
        .apply(lambda ser: ser.explode(ignore_index=True).astype(int))
    )
    
    return unbinned   


def _process_file(args):
    """
    Helper to pass multiple arguments during a multiprocessing map.
    """
    
    try:
        return process_file(*args)
    except Exception as e:
        print(args)
        #logging.info(args)
        raise e

def process_file(filepath, out_dir):
    """
    Filters out irrelevant traffic, then extracts packet-level data.
    """
    
    # print(f'Processing {filepath}')
    # logging.info(f'Processing {filepath}')
    df = pd.read_csv(filepath)
    
    # Filter out irrelevant traffic
    
    df = clean(df)

    # Extract packet-level data
   
    df = unbin(df)
    
    # Set the index to timedelta (should be monotonic increasing)
  
    df.columns = ['time', 'size', 'dir'] # NOTE: Renaming to match existing code
    df = df.sort_values('time')
    df['dt_time'] = pd.to_timedelta(df.time - df.time[0], PACKET_TIMESTAMP_UNIT)
    df = df.set_index('dt_time')

    # Finally, save the preprocessed file.

    df.to_csv(pathlib.Path(out_dir, 'preprocessed-'+filepath.name))


    return True


def preprocess_data(source_dir, out_dir):
    """
    Loads data from source, then performs cleaning and preprocessing steps. Each
    preprocessed file is saved to the out directory.
    """
    source_path = pathlib.Path(source_dir)
    out_path = pathlib.Path(out_dir)
    
    # Ensure source exists. If not then we'll create it with symlinking.
    if not source_path.exists():

        # Create the parents. It's important we don't make the final directory
        # otherwise the symlink will fail since it already exists!
        ensure_path_exists(source_path.parent, is_dir=True)

        # Symlink data to make our source directory
        # print(f"Symlinking {source_path} to raw data from {DATA_DIRECTORY}")
        logging.info(f"Symlinking {source_path} to raw data from {DATA_DIRECTORY}")
        source_path.symlink_to(
            pathlib.Path(DATA_DIRECTORY), target_is_directory=True
        )

    # Ensure out directory exists.
    ensure_path_exists(out_path, is_dir=True)
        
    # Clean out existing preprocessed files.
    # print(f"Removing existing files from `{out_path}`")
    logging.info(f"Removing existing files from `{out_path}`")
    for fp in out_path.iterdir():
        fp.unlink()
    
    # We're only working with data which used a VPN, so we can ignore the rest.
    to_process = [
        fp
        # We can pass a glob pattern to further constrain what files we look at.
        for fp in source_path.glob('*.csv')
        if 'novpn' not in fp.name
    ]

    # We'll use a multiprocessing pool to parallelize our preprocessing since
    # it involves computation.
    args = [
        (filepath, out_path)
        for filepath in to_process
    ]

    workers = multiprocessing.cpu_count()
    # print(f'Starting a processing pool of {workers} workers.')
    logging.info(f'Starting a processing pool of {workers} workers.')
    start = time.time()
    pool = multiprocessing.Pool(processes=workers)
    results = pool.map(_process_file, args)
    # print(f'Time elapsed: {round(time.time() - start)} seconds.')
    logging.info(f'Time elapsed: {round(time.time() - start)} seconds.')
    
    results = np.array(list(results))
    # print(f'{sum(results)} input files successfully preprocessed.')
    logging.info(f'{sum(results)} input files successfully preprocessed.')
    # print(f"{sum(~results)} files couldn't be procesed.")
    logging.info(f"{sum(~results)} files couldn't be procesed.")
        
