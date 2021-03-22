import os
import pandas as pd

def make_cts(kallisto_out_dir, cts_dir):
    """
    This method doesn't have an input, but rather takes 50 abundance.tsv in the processed
    Kallisto directory, and makes a matrix that counts different subfeatures.
    """
    abundances_dirs = os.listdir(kallisto_out_dir)
    abundances_dirs.sort()
    # cols_name = pd.read_csv(os.path.join(kallisto_out_dir, abundances_dirs[0], "abundance.tsv"), sep="\t").target_id
    # print(cols_name)
    result = pd.DataFrame()
    for pair in abundances_dirs:
        abundances_dir = os.path.join(kallisto_out_dir, pair, "abundance.tsv")
        print(abundances_dir)
        df = pd.read_csv(abundances_dir, sep="\t")
        df = df.set_index("target_id")
        est_counts = df.est_counts
        result[pair] = est_counts.round(0).astype(int)
    print(result)
    result.to_csv(cts_dir, sep="\t")
    return

def test_make_cts(kallisto_out_dir, cts_dir):
    """
    This method doesn't have an input, but rather takes 50 abundance.tsv in the processed
    Kallisto directory, and makes a matrix that counts different subfeatures.
    """
    abundances_dirs = os.listdir(kallisto_out_dir)
    abundances_dirs.sort()
    # cols_name = pd.read_csv(os.path.join(kallisto_out_dir, abundances_dirs[0], "abundance.tsv"), sep="\t").target_id
    # print(cols_name)
    result = pd.DataFrame()
    for pair in ["SRR7949794"]:
        abundances_dir = os.path.join(kallisto_out_dir, pair, "abundance.tsv")
        print(abundances_dir)
        df = pd.read_csv(abundances_dir, sep="\t")
        df = df.set_index("target_id")
        est_counts = df.est_counts
        result[pair] = est_counts.round(0).astype(int)
    print(result)
    result.to_csv(cts_dir, sep="\t")
    return
