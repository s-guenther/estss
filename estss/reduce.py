#!/usr/bin/env python3
import copy

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform


# ##
# ## Top Level
# ##
# ## ##########################################################################


# ##
# ## Reduce
# ##


# ##
# ## Reduce To
# ##


# ##
# ## Preprocessing
# ##
# ## ##########################################################################


# ##
# ## Feature Space Normalization
# ##

def _normalize_feature_space(df_feat):
    """Transforms a raw feature space to a normalized one. `df_feat` is a
    mxn pandas dataframe where m is the number of points and n the number of
    dimensions. Normalization performed per dimension. Normalized array is
    again a dataframe with same dimensions and row/col index."""
    return df_feat.apply(_outlier_robust_sigmoid, raw=True)


def _outlier_robust_sigmoid(feat_vec):
    """Outlier Robust Sigmoidal Transfrom. Transforms a raw feature vector
    (single dimension) `feat_vec` to normalized output. Taken from
    https://doi.org/10.1098/rsif.2013.0048 Suplementory Material 1 Eq. (2)."""
    med = np.median(feat_vec)
    iqr = stats.iqr(feat_vec)
    inner_term = -(feat_vec - med)/(1.35 * iqr)
    return 1/(1 + np.exp(inner_term))


# ##
# ## Dimensionality Reduction
# ##

def _hierarchical_corr_mat(df_feat, treshold=0.4, method='pearson'):
    # based on https://wil.yegelwel.com/cluster-correlation-matrix/
    # and https://www.kaggle.com/code/sgalella/correlation-heatmaps-with
    # -hierarchical-clustering/notebook
    corr_mat = df_feat.corr(method=method, numeric_only=True)
    # TODO just for testing purposes, remove later
    # sns.heatmap(np.abs(corr_mat),
    #             vmin=0, vmax=1, cmap='Greys')
    # plt.gca().axis('equal')

    dissim = 1 - abs(corr_mat)
    dissim[dissim < 0] = 0
    link = linkage(squareform(dissim), 'complete', optimal_ordering=True)
    link[link < 0] = 0

    clust_labels = fcluster(link, 1 - treshold, criterion='distance')
    idx = np.argsort(clust_labels)

    corr_mat_sort = corr_mat.copy()
    corr_mat_sort = corr_mat_sort.iloc[idx, :].T.iloc[idx, :]

    # TODO just for testing purposes, remove later
    # plt.figure()
    # dendrogram(link, labels=df_feat.columns,
    #            orientation='top', leaf_rotation=90)
    # plt.figure()
    sns.heatmap(
        np.abs(corr_mat_sort),
        vmin=0, vmax=1, cmap=_create_custom_cmap(treshold)
    )
    plt.gca().axis('equal')
    plt.gca().set_title(f'Treshold = {treshold} '
                        f'Cluster = {max(clust_labels)}')


def _create_custom_cmap(treshold=0.5, c_gray='gray', c_col='plasma',
                        n_values=20, cgap=0.2):
    """Creates a custom colormap for intended fro hierarchical correlation
    matrix that combines a greyscale colormap for values below treshold and
    a colored colormap above treshold"""
    # based on https://stackoverflow.com/questions/31051488/
    nval1 = min(n_values - 1, max(int(treshold * n_values), 1))
    nval2 = n_values - nval1
    c1 = plt.cm.get_cmap(c_gray)(
        np.linspace(1, 1 - cgap * treshold, nval1)
    )
    c2 = plt.cm.get_cmap(c_col)(
        np.linspace(1 - (cgap + (1 - cgap) * treshold), 0, nval2)
    )
    colors = np.vstack([c1, c2])
    mymap = mcolors.LinearSegmentedColormap.from_list(
        'mymap', colors, N=n_values
    )
    return mymap


def _modified_pearson(a, b):
    """Modified Pearson Correlation - does not only test linear, but also
    squared, sqr root, and 1/x."""
    c = list()
    c.append(np.corrcoef(a, b)[0, 1])
    c.append(np.corrcoef(a, np.sqrt(b))[0, 1])
    c.append(np.corrcoef(a, b**2)[0, 1])
    c.append(np.corrcoef(a, 1/(b+1e-10))[0, 1])
    return max(np.abs(c))


# ##
# ## Reduction
# ##
# ## ##########################################################################


# ##
# ## Map to Uniform
# ##


# ##
# ## Remove Closest
# ##


# ##
# ## Equilize ND hist
# ##


# ##
# ## Temporary Tests
# ##
# ## ##########################################################################

def _prepare_test_case():
    from timeit import default_timer as timer
    from estss import expand, features

    print(f'Loading time series files (~4GB) ...', end=' ')
    start = timer()
    ts_neg, ts_posneg = expand.get_expanded_ts(
        # ('../data/exp_ts_only_neg.pkl', '../data/exp_ts_only_posneg.pkl')
    )
    ts_neg = ts_neg.iloc[:, :500]
    ts_posneg = ts_posneg.iloc[:, :500]
    print(f'finished in {timer() - start:.2f}s')

    print(f'Calculating features for '
          f'{len(ts_neg.columns) + len(ts_posneg.columns)} '
          f'time series ...', end=' ')
    start = timer()
    feat_neg = features.features(ts_neg)
    feat_posneg = features.features(ts_posneg)
    print(f'finished in {timer() - start:.2f}s')

    return feat_neg, feat_posneg


if __name__ == '__main__':
    FEAT = pd.read_pickle('../data/test_feat.pkl')
    ISZERO = FEAT.apply(lambda col: np.all(col == 0))
    FEAT = FEAT.loc[:, ~ISZERO]
    _hierarchical_corr_mat(FEAT)
    dummybreakpoint = True
