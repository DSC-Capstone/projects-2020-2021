import os
import shutil
#import plotly.express as px

def run_test_rscript(synthetic_name, num_vars, samples_per_cond, rep_id, seq_depth, num_diff_exp, ratio_upregulated, dispersion_num, synData_output_dir, indir, data, tool1, rmdFunc1, tool2, rmdFunc2, tool3, rmdFunc3, tool4, rmdFunc4, tool5, rmdFunc5, outdir):
    if os.path.exists(outdir) and os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.mkdir(outdir)
    
    os.system("/opt/conda/envs/r-bio/bin/Rscript src/test.R " + synthetic_name + ' ' + str(num_vars) + ' ' + str(samples_per_cond) + ' ' + str(rep_id) + ' ' + seq_depth + ' ' + str(num_diff_exp) + ' ' + ratio_upregulated + ' ' + str(dispersion_num) + ' ' + synData_output_dir + ' ' + indir + ' ' + data + ' ' + tool1 + ' ' + rmdFunc1 + ' ' + tool2 + ' ' + rmdFunc2 + ' ' + tool3 + ' ' + rmdFunc3 + ' ' + tool4 + ' ' + rmdFunc4 + ' ' + tool5 + ' ' + rmdFunc5 + ' ' + outdir)
    #df_all = pd.read_csv('~/RNASeqToolComparison/out/test/statistics.csv', sep = ' ')

    #fig = px.box(df, x='Samples per Condition', y='AUC', color='Tool',
    #             title="AUC ")
    #fig.update_xaxes(type='category')
    #fig.write_image("~/RNASeqToolComparison/out/test/auctest.png")
    return
