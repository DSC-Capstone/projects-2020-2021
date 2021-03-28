import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os

    
def visualize_densities(logger, data, dim, outdir):
    '''
    saves KDE plot.
    '''
    pol_flagged = data[data['flagged']][dim]
    pol_unflagged = data[data['flagged'] == False][dim]
    title = 'KDE plot for {}'.format(dim)
    sns.kdeplot(pol_flagged, shade=True).set_title(title)
    sns.kdeplot(pol_unflagged, shade=True)
    plt.legend(title='Tweet Liked', loc='upper left', labels=['Flagged', 'Unflagged'])
    fp = os.path.join(outdir, 'kde_plot_{}.png'.format(dim))
    plt.savefig(fp)
    plt.close()
    logger.info('wrote image to {}'.format(fp))

def visualize_correlation(logger, data, dim1, dim2, outdir):
    '''
    saves correlation plot.
    '''
    data = data[[dim1, dim2, 'flagged']]
    title = 'User {} versus {} Polarity'.format(dim1, dim2)
    sns.scatterplot(x=data[dim1], y=data[dim2],
                    hue=data['flagged']).set_title(title)
    fp = os.path.join(outdir, 'correlation_{}_{}.png'.format(dim1, dim2))
    plt.savefig(fp)
    plt.close()
    logger.info('wrote image to {}'.format(fp))

# Helper functions
def mean_diff(data, dim):
    return np.mean(data[data['flagged'] == True][dim]) - np.mean(data[data['flagged'] == False][dim])

def permutation_test(data, n_reps, dim):
    # Observed statistic
    obs = mean_diff(data, dim)
    
    # Running and permuting n_reps of the data
    trials = []
    for i in range(n_reps):
        shuffled_impres = (
            data['flagged']
            .sample(replace=False, frac=1)
            .reset_index(drop=True)
        )
        shuffled = (
            data
            .assign(**{'flagged': shuffled_impres})
        )
        trials.append(mean_diff(shuffled, dim))
    return np.count_nonzero(np.array(trials) >= obs) / n_reps

def run_permutation_test(logger, data, dim, outdir):
    '''
    Permutation test.
    '''
    outcomes = []
    out = {'Dimension': dim, 
        'observed_mean': mean_diff(data, dim),
        'p-value': permutation_test(data, 1000, dim)}
    fp = os.path.join(outdir, 'perm_test_{}.json'.format(dim))
    with open(fp, 'w') as fout:
        json.dump(out , fout)
    logger.info('wrote results to {}'.format(fp))

def run_two_sided_ttest(logger, data, dim, outdir):
    '''
    Two sided t test.
    '''
    outcomes = []
    flagged = data[data['flagged'] == True][dim]
    unflagged = data[data['flagged'] == False][dim]
    out = {'Dimension': dim, 
        'Results': stats.ttest_ind(flagged, unflagged, equal_var=True)}
    fp = os.path.join(outdir, 'two_sided_ttest_{}.json'.format(dim))
    with open(fp, 'w') as fout:
        json.dump(out , fout)
    logger.info('wrote results to {}'.format(fp))

def run_one_sided_ttest(logger, data, dim, outdir):
    '''
    One sided t test.
    '''
    outcomes = []
    # for dim in dimensions:
    flagged = data[data['flagged'] == True][dim]
    unflagged = data[data['flagged'] == False][dim]
    gen = stats.ttest_ind(flagged, unflagged, equal_var=True)
    out = {'Dimension': dim,
        'p-value': gen[1]/2,
        'Test Statistic': gen[0]
        }
    fp = os.path.join(outdir, 'one_sided_ttest_{}.json'.format(dim))
    with open(fp, 'w') as fout:
        json.dump(out , fout)
    logger.info('wrote results to {}'.format(fp))

def compute_results(logger, data, dims, outdir):
    '''
    Gets all results
    '''
    dim = dims[0]
    for dim in dims:
        visualize_densities(logger, data, dim, outdir)
        run_permutation_test(logger, data, dim, outdir)
        run_two_sided_ttest(logger, data, dim, outdir)
        run_one_sided_ttest(logger, data, dim, outdir)
    for i in range(len(dims)):
        for j in range(i+1, len(dims)):
            visualize_correlation(logger, data, dims[i], dims[j], outdir)
    return 