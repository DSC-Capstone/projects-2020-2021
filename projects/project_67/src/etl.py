import os
import shutil

def run_create_data_rscript(data, n_vars, samples_per_cond, rep_id, n_diffexp, upregulated_ratio, dispersion, outlier_type, outlier_ratio, output_file, seq_depth):
    os.system("/opt/conda/envs/r-bio/bin/Rscript src/syntheticDataFunc.R " + data + ' ' + str(n_vars) + ' ' + str(samples_per_cond) + ' ' + str(rep_id) + ' ' + str(n_diffexp) + ' ' + str(upregulated_ratio) + ' ' + str(dispersion) + ' ' + outlier_type + ' ' + str(outlier_ratio) + ' ' + output_file + ' ' + seq_depth)
    return

def run_rscript_test(string):
    os.system("/opt/conda/envs/r-bio/bin/Rscript src/test.R " + string)
    return
