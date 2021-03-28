import os
import shutil

def run_real_rscript(real_data_dir):
    os.system("/opt/conda/envs/r-bio/bin/Rscript src/real.R " + real_data_dir)
    return

def run_real_diff_exp_rscript(in_dir, real_data, tool, tool_rmd, out_dir):
    os.system("/opt/conda/envs/r-bio/bin/Rscript src/real_diff_exp.R " + in_dir + ' ' + real_data + ' ' + tool + ' ' + tool_rmd + ' ' + out_dir)
    return

def run_comparison_real(dataset, tool1, tool2, tool3, tool4, tool5, tool6, out_dir):
    os.system("/opt/conda/envs/r-bio/bin/Rscript src/compareReal.R " + dataset + ' ' + tool1 + ' ' + tool2 + ' ' + tool3 + ' ' + tool4 + ' ' + tool5 + ' ' + tool6 + ' ' + out_dir)
    return