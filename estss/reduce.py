#!/usr/bin/env python3
import copy

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.spatial import KDTree
from scipy.stats import qmc


# ##
# ## Top Level
# ##
# ## ##########################################################################

# ##
# ## Dimensionality Reduction
# ##

# Manually selected cluster representatives. If no inline comment it is the
# automatically selected cluster representative (maximum inner correlation
# with other features). If inline comment, it deviates from the
# automatically determined one and the reason is described.
_REPRESENTATIVES = (
    'temporal_centroid',
    'mean_of_signed_diff',
    'loc_of_last_max',      # lower outer correlation than loc_of_first_max
    'rs_range',
    'dfa',
    'rcp_num',
    'acf_first_zero',
    'median_of_abs_diff',
    'freq_mean',
    'stretch_decreasing',
    'stl_seasonality_strength',
    'mean',
    'iqr',                  # simpler measure than mean_diff_from_mean
    'share_above_mean',     # lower outer correlation than share_below_mean
                            #   and simpler than binarize_mean
    'mode5',
    'mean_2nd_diff',
    'trev'
)


def dimensional_reduced_feature_space(df_feat, choose_dim=_REPRESENTATIVES,
                                      plot=True):
    fspace = _raw_feature_array_to_feature_space(df_feat)
    corr_mat, cinfo = _hierarchical_corr_mat(fspace)
    fspace2 = fspace[list(choose_dim)]
    if plot:
        _plot_hierarchical_corr_mat(corr_mat, cinfo, selected_feat=choose_dim)
        corr_mat2, cinfo2 = _hierarchical_corr_mat(fspace2, threshold=0.5)
        _plot_hierarchical_corr_mat(corr_mat2, cinfo2)
    return fspace2, cinfo


# ##
# ## Reduce Chain
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

def _raw_feature_array_to_feature_space(df_feat):
    """Transforms a raw feature space to a normalized one. `df_feat` is a
    mxn pandas dataframe where m is the number of points and n the number of
    dimensions. Normalization performed per dimension. Normalized array is
    again a dataframe with same dimensions and row/col index."""
    fspace = _prune_feature_space(df_feat)
    fspace = (
        fspace
        .apply(_outlier_robust_sigmoid, raw=True)
        .apply(_curtail_at_whiskers, raw=True)
        .apply(_norm_min_max, raw=True)
    )
    return fspace


# Variable _FEATURES_TO_EXCLUDE:
# Features that are excluded in `_prune_feature_space()`. Only excluded for
# dimensionality reduction and calculation of the reduced time series sets.
# For the final analysis, they are available again. They are removed because
# they are ill-defined in the sense that they are very skewed making it a
# hard dimension do equilize in the reduction algorithm, or clash with
# the time series definition (maxabs=1), or are discrete and not continuous,
# which is viable in general but needs special care which is omitted for
# simplicity.
# Further features that are kept, but may prove to get excluded as well
# because they are quite skewed, too: trev, ami_timescale, rs_range, dfa,
# linearity, stl_spikiness, bocp_conf_max, mean_2nd_diff
_FEATURES_TO_EXCLUDE = (
    'stl_peak',               # discrete dimension
    'stl_trough',             # discrete dimension
    'min',                    # ill-defined
    'max',                    # ill-defined
    'median_of_signed_diff',  # ill-defined
    'peak2peak',              # ill-defined
    'cross_zero',             # ill-defined
    'fund_freq',              # discrete dimension
    'freq_roll_on'            # discrete dimension
)


def _prune_feature_space(df_feat, to_exclude=_FEATURES_TO_EXCLUDE):
    """Removes the columns defined in `to_exclude` from the
    features dataframe `df_feat`. Default excluded features are defined in
    `_FEATURES_TO_EXCLUDE` and are only excluded for dimensionality
    reduction, but are present in the final analysis"""
    return df_feat.drop(list(to_exclude), axis='columns')


def _outlier_robust_sigmoid(feat_vec):
    """Outlier Robust Sigmoidal Transfrom. Transforms a raw feature vector
    (single dimension) `feat_vec` to normalized output. Taken from
    https://doi.org/10.1098/rsif.2013.0048 Suplementory Material 1 Eq. (2)."""
    med = np.median(feat_vec)
    iqr = stats.iqr(feat_vec)
    inner_term = -(feat_vec - med) / (1.35 * iqr + 1e-12)
    return 1 / (1 + np.exp(inner_term))


def _curtail_at_whiskers(feat_vec):
    """Calculates the whiskers of a vector `feat_vec` and curtails all
    outliers above or below the whiskers to the whiskers value. Whiskers
    follow the conventional boxplot definition, which is that they end at a
    data point within 1.5*inter-quartile-range. For more
    details, see e.g. https://en.wikipedia.org/wiki/Box_plot"""
    low_quart = np.percentile(feat_vec, 25)
    up_quart = np.percentile(feat_vec, 75)
    iqr = up_quart - low_quart
    low_whisk = np.min(feat_vec[feat_vec >= (low_quart - 1.5*iqr)])
    up_whisk = np.max(feat_vec[feat_vec <= (up_quart + 1.5*iqr)])
    vec = copy.copy(feat_vec)
    vec[vec < low_whisk] = low_whisk
    vec[vec > up_whisk] = up_whisk
    return vec


def _norm_min_max(feat_vec):
    """Normalizes a vector `feat_vec` to the range of [0, 1]"""
    feat_vec -= np.min(feat_vec)
    feat_vec /= np.max(feat_vec)
    return feat_vec


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


def _plot_corr_mat_scatter(df_feat, samples=200):
    """Plots the correlation matrix of `df_feat` as a scatterplot,
    where each dim is plotted against each other in a matrix, the diagonal
    shows the histogram of each dimension."""
    pg = sns.PairGrid(df_feat.sample(samples))
    pg.map_upper(sns.scatterplot, s=5)
    pg.map_lower(sns.scatterplot, s=5)
    pg.map_diag(sns.histplot, bins=20)
    return pg


# ##
# ## Reduction
# ##
# ## ##########################################################################

# ##
# ## Map to Uniform
# ##

def _map_to_uniform(df_feat, n,
                    distance=2.0, seed=1001, n_tries=3, overrelax=1.05):
    """Will select `n` points from `df_feat` that are roughly uniformely
    distributed within the volume of `df_feat` by generating a
    quasi-random halten sequence and mapping this sequence to the nearest
    points of df_feat. `distance` is the distance metric (manhattan=1,
    euclidean=2, ...). `n_tries` and `overrelax` are parameters of the query
    algorithm."""
    n_dim = df_feat.shape[1]
    n_query = int(n*overrelax)
    # The loop queries successively more points than neccessary as fewer points
    # will be found due to duplications and outside-volume-points
    for _ in range(max(n_tries, 1)):
        uni_set = _uniform_set(n_query, n_dim, seed)
        df_sub = _find_nearest(df_feat, uni_set, distance)
        n_found = df_sub.shape[0]
        n_query = int(n_query + (n - n_found)*overrelax)
        if n_found >= n:
            break
    # In the end, the exact number of queried points will be enforced by
    # random insertion/deletion
    df_sub = _random_add_rm(df_feat, df_sub, n)  # noqa
    return df_sub


def _uniform_set(n, dim, seed=None):
    """Creates a uniformely distributed random set of `n` points,
    each having the dimension `dim` returned as an nxdim array. Each
    dimension will be in the range of [0, 1] (note that the data is within
    the range and not scaled to it, i.e. the bounds are probably not hit as
    the points are randomly generated within the bounds."""
    halton = qmc.Halton(d=dim, seed=seed)
    return halton.random(n)


def _find_nearest(large, small, distance=2.0, rm_outliers=True,
                  rm_duplicates=True, leafsize=None, workers=-1):
    """For each point in `small`, finds the nearest point in the
    dataframe `large`.
    If `rm_outliers=True` removes matches of points in `small` that are far
    away (it is interpreted that these are outside the search space volume
    that will map to the surface of the search space volume. If
    `rm_duplicates=True`, it will remove multiple matched points.
    `leafsize` is a parameter of the kd-tree search. `workers` defines how
    many workers are used for parallel processing"""
    if leafsize is None:
        leafsize = _default_leafsize(large.shape[0])
    tree = KDTree(large, leafsize=leafsize)
    qpoints = [small[i, :] for i in range(small.shape[0])]
    dist, ind = tree.query(qpoints, p=distance, workers=workers)
    if rm_outliers:
        ind = _rm_pts_outside_volume(dist, ind)
    if rm_duplicates:
        ind = _rm_duplicates(ind)
    return large.iloc[ind, :]


def _default_leafsize(arraysize):
    """Determines kd-tree leafsize based on the size of the array."""
    return max(min(arraysize / 100, 50), 10)


def _rm_pts_outside_volume(dist, ind):
    """Looks at the distances `dist` of the queried points `ind`. Looks for
    the most common distances and interpretes distances greater than that as
    points outside the point volume/outliers. These get removed and anly the
    points within the point volume are returned."""
    # TODO refactor, questionable calculation method
    hist, edges = np.histogram(dist, bins=int(len(dist)/2))
    hist[hist <= 2] = 0
    binstep = edges[1] - edges[0]
    binvals = edges[:-1] + binstep
    fakedist = []
    for n_in_bin, bval in zip(hist, binvals):
        fakedist += [bval]*n_in_bin
    fakedist = np.array(fakedist)
    maxbnd = np.percentile(fakedist, 75)
    return ind[dist <= 1.2*maxbnd]


def _rm_duplicates(indexlist):
    """If indices occur multiple times in `indexlist`, they will be unique
    in the end."""
    return list(set(indexlist))


def _random_add_rm(df_large, df_small, n):
    """Creates a dataframe with `n` rows based on the dataframe `df_small`.
    If `df_small` has more elements, the excess elements will be removed,
    if it has fewer elements, random elements from `df_large` will be added."""
    n_small = df_small.shape[0]
    if n_small == n:
        return df_small
    elif n_small > n:
        return df_small.sample(n, axis='rows')
    elif n_small < n:
        n_diff = n - n_small
        diff_index = np.setdiff1d(df_large.index, df_small.index)
        df_diff = df_large.loc[diff_index, :]
        df_enlarged = pd.concat(
            [df_small, df_diff.sample(n_diff, axis='rows')],
            axis='rows'
        )
        return df_enlarged
    else:
        raise RuntimeError('If-...-else reached presumably impossible path')


# ##
# ## Remove Closest
# ##

def _remove_closest(df_feat, n_final, distance=2.0, leafsize=None):
    """Removes points within the dataframe `df_feat` until only `n_final`
    are left. Points are successively removed by removing the closest ones
    to each other. `distance` is the distance metric, `leafsize` an internal
    parameter for kd-tree generation"""
    if leafsize is None:
        leafsize = _default_leafsize(df_feat.shape[0])

    n_feat = df_feat.shape[0]
    n_rm = (n_feat - n_final)
    n_find = n_rm + 2

    # convert to float to be able to replace values with nan
    ids = np.arange(df_feat.shape[0], dtype=float)

    tree = KDTree(df_feat, leafsize=leafsize)
    distances, indexes = tree.query(df_feat, k=n_find, p=distance)
    distances, indexes = distances[:, 1:], indexes[:, 1:]
    for _ in range(n_rm):
        # The algorithm removes rows and replaces values within the
        # distances and indexes array with NaN. At a certain point,
        # the algorithm will fail if it tries to evaluate a NaN value
        rm_ind = _closest_ind(distances, indexes)
        _remove_index_inplace(distances, indexes, ids, rm_ind)

    ids = ids[~np.isnan(ids)]
    # Convert back to integer
    ids = ids.astype('int')

    return df_feat.iloc[ids, :]


def _closest_ind(distances, indexes):
    """Subfunction of `_remove_closest()`
    Find nearest pair specified in the `distances` array and remove the point
    that is closer in general to the others, others is defined by the second
    closest point to each. The index of the closest point within the pair is
    extracted from `indexes`."""
    coord_p1 = np.unravel_index(np.argmin(distances), indexes.shape)
    ind_p1 = coord_p1[0]
    ind_p2 = indexes[coord_p1]
    d_p1 = np.partition(distances[ind_p1, :], 1)[1]
    d_p2 = np.partition(distances[ind_p2, :], 1)[1]
    if d_p1 < d_p2:
        return ind_p1
    elif d_p1 > d_p2:
        return ind_p2
    else:
        raise ValueError(f'There are two points with index {ind_p1} and '
                         f'{ind_p2} that have the same distance to another '
                         f'point')
    # NOTE does not work anymore if rows are removed afterwards. Either do
    #  not remove rows or keep them but set to inf or similar


def _remove_index_inplace(distances, indexes, ids, rm_ind):
    """Subfunction of `_remove_closest()`.
    Remove point row specified by `rm_ind` from 'distances', 'indexes',
    and 'ids' by setting to nan or to inf"""
    ids[rm_ind] = np.nan
    distances[rm_ind, :] = np.inf
    distances[indexes == rm_ind] = np.inf


# ##
# ## Equilize ND hist
# ##


# ##
# ## Temporary Tests
# ##
# ## ##########################################################################

def _test_map_to_uniform():
    df_feat = pd.read_pickle('../data/test_feat.pkl')
    df_feat, cinfo = dimensional_reduced_feature_space(df_feat, plot=False)
    df_feat_uni = _map_to_uniform(df_feat, 500)
    return df_feat_uni


if __name__ == '__main__':
    _test_map_to_uniform()
