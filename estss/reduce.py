#!/usr/bin/env python3

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
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
    inner_term = -(feat_vec - med) / (1.35 * iqr)
    return 1 / (1 + np.exp(inner_term))


# ##
# ## Dimensionality Reduction / Clustering
# ##


def _modified_pearson(a, b):
    """Modified Pearson Correlation - does not only test linear, but also
    squared, sqr root, and 1/x."""
    c = list()
    c.append(np.corrcoef(a, b)[0, 1])
    c.append(np.corrcoef(a, np.sqrt(b))[0, 1])
    c.append(np.corrcoef(a, b ** 2)[0, 1])
    c.append(np.corrcoef(a, 1 / (b + 1e-10))[0, 1])
    return max(np.abs(c))


def _hierarchical_corr_mat(df_feat, threshold=0.4, method=_modified_pearson,
                           linkmethod='average', clust_crit='distance'):
    """Computes a hierarchical correlation matrix from feature array
    `df_feat`. Features with a correlation above 'threshold' are clustered
    together. Correlation is computed with `method` (default is a modified
    pearson correlation, which also tests squared and 1/x correlation
    besides linear). The linkage method for clustering is provided by
    `linkmethod='average'` and the cluster criterion by
    `clust_crit='distance'`.
    Returns the clustered correlation matrix as dataframe as well as an
    info dict with:
      'cluster' - df that contains inner and outer correlations of clusters
      'threshold', 'linkmethod', 'clustcrit' - Algorithm values set as
          argument of the function
     """
    # based on https://wil.yegelwel.com/cluster-correlation-matrix/
    # and https://www.kaggle.com/code/sgalella/correlation-heatmaps-with
    # -hierarchical-clustering/notebook
    corr_mat = df_feat.corr(method=method, numeric_only=True)
    dissim = 1 - abs(corr_mat)
    dissim[dissim < 0] = 0
    link = linkage(squareform(dissim), method=linkmethod,
                   optimal_ordering=True)
    link[link < 0] = 0
    clust_labels = fcluster(link, 1 - threshold, criterion=clust_crit)
    corr_mat_sort, clust_info = _sort_corr_mat(corr_mat, clust_labels)

    info = dict(cluster=clust_info,
                threshold=threshold,
                linkmethod=linkmethod,
                clust_crit=clust_crit)

    return corr_mat_sort, info


def _plot_hierarchical_corr_mat(corr_mat, info, selected_feat=None):
    """Plots the precomputed correlation matrix `corr_mat`. Additionally
    frames found clusters, highlights cluster representatives and provides
    info on inner and outer cluster correlation as well as algorithm
    information. `info` is a dict returned by `_hierarchical_corr_mat()`.
    `selected_features` is an optional argument that overrides the default
    cluster representatives.
    Returns fig, ax objects of the plot."""
    # Get feature names of cluster representatives, either by taking the first
    # feature of each cluster or through selected_feat argument
    if selected_feat is None:
        selected_feat = _get_first_name_per_cluster(info['cluster'])
    names = [s if s in selected_feat else ""
             for s in list(info['cluster'].index)]

    # Make main plot
    fig, ax = plt.subplots()
    sns.heatmap(
        np.abs(corr_mat),
        vmin=0, vmax=1, cmap=_create_custom_cmap(info['threshold']), ax=ax,
        square=True, xticklabels=False, yticklabels=names
    )

    # Make Cluster Rectangles and highlight Cluster Representatives
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    labels = info["cluster"].label.values
    total = len(labels)
    for ll in set(labels):
        start = np.searchsorted(labels, ll)
        stop = np.searchsorted(labels, ll, side='right')
        ax.add_patch(_cluster_rect(start, stop, total, xlim, ylim))
    for start, nn in enumerate(names):
        if nn == "":
            continue
        ax.add_patch(_cluster_rect(start, start+1, total, xlim, ylim,
                                   facecolor='limegreen', edgecolor='none'))

    # Add cluster inner/outer correlation info
    idict = _cluster_info(info['cluster'], selected_feat)
    iinfo = idict['inner_agg']
    oinfo = idict['outer_agg']
    infostring = (f'Number of Clusters:\n'
                  f'   {max(info["cluster"].label)}\n'
                  f'Avg. Inner-Cluster-Corr.:\n'
                  f'   {iinfo["mean"]:.2f} $\pm$ {iinfo["std"]:.2f}\n'  # noqa
                  f'Avg. Outer-Cluster-Corr.:\n'
                  f'   {oinfo["mean"]:.2f} $\pm$ {oinfo["std"]:.2f}\n\n'
                  f'Computation Info:\n'
                  f'   Threshold: {info["threshold"]}\n'
                  f'   Linkage Method: {info["linkmethod"]}\n'
                  f'   Cluster Criterion: {info["clust_crit"]}')  # noqa
    ax.text((xlim[1] - xlim[0])*1.0 + xlim[0],
            (ylim[1] - ylim[0])*1.0 + ylim[0],
            infostring,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'),
            va='top', ha='right', ma='left')

    return fig, ax


def _cluster_rect(start, end, total, xlim, ylim,
                  edgecolor='limegreen', facecolor='none', linewidth=1.5,
                  **kwargs):
    """Subfuncion of `_plot_hierarchical_corr_mat()`. Plots a rectangle
    framing a cluster. `start` is the start feature number of a cluster,
    `end` the end feature number of this cluster. `total` is the total
    number of features. `xlim` and `ylim` provide the axes limits of the
    axes to draw in. `**kwargs` are passed to `Rectangle` (matplotlib)"""
    xrange = xlim[1] - xlim[0]
    yrange = ylim[1] - ylim[0]
    srel = start / total
    erel = end / total
    xy = (xlim[0] + srel * xrange, ylim[1] - srel * yrange)
    w = (erel - srel) * xrange
    h = -(erel - srel) * yrange
    return Rectangle(xy, w, h, edgecolor=edgecolor, facecolor=facecolor,
                     linewidth=linewidth, **kwargs)


def _sort_corr_mat(mat, labels):
    """Sorts a correlation matrix `mat` based on cluster labels `labels`.
    Performs an outer-cluster sorting, where individual clusters are sorted
    based on `labels` as well as an inner-cluster sorting, where features
    are sorted in ascending order of outer-cluster correlation followed by
    descending order of inner-cluster correlation.

    Parameters
    ----------
    mat: pd.Dataframe
        nxn dataframe containing the correlation matrix, index and columns
        are labeled, where n is the number of features
    labels: list or np.array()
        (n,)-vector containing a cluster-label for each feature

    Returns
    -------
    mat_sort: pd.Dataframe
        nxn correlation matrix, clusters sorted
    clust_info pd.Dataframe
        nx3 dataframe containing the columns
        'label', 'inner_corr', 'outer_corr"""
    idx = np.argsort(labels)
    mat_sort = mat.copy()
    mat_sort = mat_sort.iloc[idx, :].T.iloc[idx, :]
    mat_sort = (
        mat_sort
        .groupby(labels[idx])
        .apply(
            lambda gdf: gdf.assign(
                outer_corr=lambda df: ((df.sum(axis=1) - df.sum()[df.index]) /
                                       (df.shape[1] - df.shape[0])),
                inner_corr=lambda df: df.mean()[df.index]
            )
            .sort_values('outer_corr', ascending=True)
            .sort_values('inner_corr', ascending=False)
        )
        .droplevel(0)
    )

    clust_info = pd.DataFrame(
        dict(label=labels[idx],
             inner_corr=mat_sort['inner_corr'],
             outer_corr=mat_sort['outer_corr']),
        index=mat_sort.index
    )

    mat_sort = mat_sort.drop(columns=['outer_corr', 'inner_corr'])
    mat_sort = mat_sort.loc[:, mat_sort.index]

    return mat_sort, clust_info


def _get_first_name_per_cluster(df_info):
    """Returns the first feature within every cluster (clustered are
    expected to be sorted already). `df_info` contains the column `label`
    and the index holds the feature names. Returned as list."""
    names = (df_info
             .groupby('label')
             .apply(lambda df: pd.Series([df.index[0]])))
    return list(names[0].values)


def _cluster_info(df_info, cnames=None):
    """Takes `df_info` with columns 'label', 'inner_corr' and 'outer_corr'
    and computes statistics per cluster (encoded in 'label'). Returns as
    dict."""
    if cnames is None:
        cnames = _get_first_name_per_cluster(df_info)

    inner = (df_info['inner_corr']
             .groupby(df_info['label'])
             .agg(['mean', 'std', 'min', 'max']))
    inner.rename(dict(zip(inner.index.values, cnames)), inplace=True)
    outer = (df_info['outer_corr']
             .groupby(df_info['label'])
             .agg(['mean', 'std', 'min', 'max']))
    outer.rename(dict(zip(outer.index.values, cnames)), inplace=True)
    inner_agg = (inner['mean']
                 .agg(['mean', 'std', 'min', 'max']))
    outer_agg = (outer['mean']
                 .agg(['mean', 'std', 'min', 'max']))

    return dict(inner=inner, outer=outer,
                inner_agg=inner_agg, outer_agg=outer_agg)


def _create_custom_cmap(threshold=0.4, c_gray='gray', c_col='plasma',
                        n_values=100, cgap=0.2):
    """Creates a custom colormap intended for hierarchical correlation
    matrix that combines a greyscale colormap for values below threshold and
    a colored colormap above threshold"""
    # based on https://stackoverflow.com/questions/31051488/
    nval1 = min(n_values - 1, max(int(threshold * n_values), 1))
    nval2 = n_values - nval1
    c1 = plt.cm.get_cmap(c_gray)(
        np.linspace(1, 1 - cgap * threshold, nval1)
    )
    c2 = plt.cm.get_cmap(c_col)(
        np.linspace(1 - (cgap + (1 - cgap) * threshold), 0, nval2)
    )
    colors = np.vstack([c1, c2])
    mymap = mcolors.LinearSegmentedColormap.from_list(
        'mymap', colors, N=n_values
    )
    return mymap


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
    CORR_MAT, INFO = _hierarchical_corr_mat(FEAT)
    FIG, AX = _plot_hierarchical_corr_mat(CORR_MAT, INFO)
    IDICT = _cluster_info(INFO['cluster'])
    dummybreakpoint = True
