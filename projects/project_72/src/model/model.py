
from scipy.stats import pearsonr, spearmanr
import numpy as np
def loss_p(x1, x2):
    return (1 / np.abs(pearsonr(x1, x2)[0]))
def loss_s(x1, x2):
    return (1 / np.abs(spearmanr(x1, x2)[0]))
def loss_ps(x1, x2):
    return 1 / (pearsonr(x1, x2)[0] * spearmanr(x1, x2)[0])**2
def pseudo_deriv(h, loss_f, respect):
    """
    https://mathinsight.org/partial_derivative_limit_definition
    """
    return (loss_f(h+respect[:, 0], respect[:, 1]) - loss_f(respect[:, 0], respect[:, 1])) / h

def transform(x, M, b):
    return np.matmul(x, M) + b
