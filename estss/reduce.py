#!/usr/bin/env python3
import copy
import pickle
import random
from warnings import warn

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

from estss import expand, features


# ##
# ## Top Level
# ##
# ## ##########################################################################

def get_reduced_sets(file='data/reduced_sets.pkl'):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


def compute_reduced_sets(df_feat_list=None, df_ts_list=None, seed=1340):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    print('# ##\n# ## Load and Prepare Data\n# ##')
    print('##########################################################')
    print('# ## Load Data')
    if df_feat_list is None:
        df_feat_list = features.get_features()
    if df_ts_list is None:
        df_ts_list = expand.get_expanded_ts()

    # reindex second feat and ts dataframe with an offset of +1e6 to be able
    # to differentiate between only_neg/only_posneg in a merged dataframe
    print('# ## Reindex and Merge Input Dataframes')
    feat_posneg = df_feat_list[1]
    feat_posneg.index += int(1e6)
    df_feat_list[1] = feat_posneg
    ts_posneg = df_ts_list[1]
    ts_posneg.columns += int(1e6)
    df_ts_list[1] = ts_posneg

    # merge and normalize combined feature array
    df_feat_merged = pd.concat(df_feat_list, axis=0, ignore_index=False)
    print('# ## Dimensional Reduction')
    df_feat_norm, cinfo = dimensional_reduced_feature_space(
        df_feat_merged, plot=False
    )
    print('# ## Reorder Features')
    df_feat_merged.sort_index(axis=1, inplace=True)

    # split and normalize with minmax again
    df_feat_neg_norm = _norm_min_max(
        df_feat_norm.loc[df_feat_norm.index < 1e6, :]
    )
    df_feat_posneg_norm = _norm_min_max(
        df_feat_norm.loc[df_feat_norm.index >= 1e6, :]
    )

    # call reduce chain on normalized feature arrays
    print('\n# ##\n# ## Reduce Only Neg Set\n# ##')
    print('##########################################################')
    sets_neg = reduce_chain(df_feat_neg_norm)
    print('\n# ##\n# ## Reduce Only PosNeg Set\n# ##')
    print('##########################################################')
    sets_posneg = reduce_chain(df_feat_posneg_norm)

    print('\n# ##\n# ## Merge, Reorganize and Map TS\n# ##')
    print('##########################################################')
    # merge sets
    sets_both = [pd.concat([n, pn]) for n, pn in zip(sets_neg, sets_posneg)]
    df_ts_merged = pd.concat(df_ts_list, axis=1, ignore_index=False)
    # map sets
    sets = _map_sets(sets_both, df_feat_merged, df_ts_merged)

    # return sets_both, df_feat_merged, df_ts_merged
    return sets


def _map_sets(norm_sets, df_feat, df_ts):
    # prepare output dict
    sets = {
        'features': dict(),
        'ts': dict(),
        'norm_space': dict(),
        'info': dict()
    }
    ts_sets = [df_ts.loc[:, nset.index].T for nset in norm_sets]
    feat_sets = [df_feat.loc[nset.index, :] for nset in norm_sets]
    mappings = []
    for nset in norm_sets:
        npoints = nset.shape[0]
        ind_neg = np.arange(npoints/2, dtype=int)
        ind_posneg = np.arange(npoints/2, dtype=int) + 1_000_000
        ind = np.hstack([ind_neg, ind_posneg])
        mapping = pd.Series(data=nset.index, index=ind)
        mappings.append(mapping)

    norm_sets = _reindex_sets(norm_sets, mappings)
    ts_sets = _reindex_sets(ts_sets, mappings)
    ts_sets = [df.T for df in ts_sets]
    feat_sets = _reindex_sets(feat_sets, mappings)

    for feat, ts, nspace, info in zip(feat_sets, ts_sets, norm_sets, mappings):
        npoints = feat.shape[0]
        sets['features'][npoints] = feat
        sets['ts'][npoints] = ts
        sets['norm_space'][npoints] = nspace
        sets['info'][npoints] = info

    return sets


def _reindex_sets(sets, mappings):
    trans_sets = []
    for set_, map_ in zip(sets, mappings):
        tset = set_.rename(index=dict(zip(map_.values, map_.index.values)),
                           errors='raise')
        trans_sets.append(tset)
    return trans_sets


# ##
# ## Reduce Chain
# ##

def reduce_chain(df_feat_norm, set_sizes=(2048, 512, 128, 32), seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Loop through set_sizes to compute sets
    large_set = df_feat_norm
    sets = []
    for n in set_sizes:
        print(f'\n# ## Compute n = {n} Set\n################################')
        small_set = reduce_single(large_set, n, seed)
        sets.append(small_set)
        large_set = small_set

    print('\n# ## Sort Sets')
    sets = _sort_sets(sets)
    return sets


def _sort_sets(in_sets):
    # add cluster column
    sets = copy.deepcopy(in_sets)
    for ii, set_ in enumerate(sets):
        set_['ind'] = set_.index
        set_['set_cluster'] = ii
        smaller_sets = sets[ii:]
        for jj, sset in enumerate(smaller_sets):
            set_.loc[sset.index, 'set_cluster'] = ii + jj + 1
        set_['mean_cluster'] = pd.cut(set_['mean'],
                                      bins=np.linspace(0, 1, 11),
                                      labels=range(10),
                                      include_lowest=True)
        # set_.sort_values(by='iqr', inplace=True)
        # set_.sort_values(by='mean_cluster', inplace=True)
        # set_.sort_values(by='set_cluster', inplace=True, ascending=False)
        set_.sort_values(by=['set_cluster', 'mean_cluster', 'iqr', 'ind'],
                         ascending=[False, False, True, True],
                         inplace=True)
        sets[ii] = set_.drop(columns=['set_cluster', 'mean_cluster', 'ind'])
    return sets


# ##
# ## Reduce Single
# ##

def reduce_single(df_feat, n,
                  seed=None, kws_map=None, kws_rm=None, kws_equi=None):
    if seed is not None:
        np.random.seed(seed)
    if kws_map is None:
        kws_map = dict()
    if kws_rm is None:
        kws_rm = dict()
    if kws_equi is None:
        kws_equi = dict()

    print(f'# ## Map to uniform ...', end=' ')
    df_red = _map_to_uniform(df_feat, int(n*1.2), seed=None, **kws_map)
    print(f'done')
    print(f'# ## Remove closest ...', end=' ')
    df_red = _remove_closest(df_red, n, **kws_rm)
    print(f'done')
    print(f'# ## Equilize nd-hist ...')
    df_red = _equilize_nd_hist(df_red, df_feat, seed=None, **kws_equi)
    print(f'... done')
    return df_red


# ##
# ## Dimensionality Reduction
# ##

# Manually selected cluster representatives. If no inline comment it is the
# automatically selected cluster representative (maximum inner correlation
# with other features). If inline comment, it deviates from the
# automatically determined one and the reason is described.
_REPRESENTATIVES = (
    'temporal_centroid',
    # 'mean_of_signed_diff',  # excluded, high corr with temporal_centroid
    'loc_of_last_max',        # lower outer correlation than loc_of_first_max
    'dfa',
    'rs_range',
    'mode5',
    'share_above_mean',
    'iqr',                    # simpler measure than mean_diff_from_mean
    'mean',
    'rcp_num',
    # 'transition_variance',  # excluded, high corr with acf_first_zero
    'acf_first_zero',
    'median_of_abs_diff',
    'freq_mean',
    # 'stretch_decreasing',        # excluded, high corr with freq_mean
    # 'stl_seasonality_strength',  # excluded, high corr with freq_mean
    'mean_2nd_diff',
    'trev'
)


def dimensional_reduced_feature_space(df_feat, choose_dim=_REPRESENTATIVES,
                                      plot=True):
    fspace = _raw_feature_array_to_feature_space(df_feat)
    corr_mat, cinfo = _hierarchical_corr_mat(fspace)
    if choose_dim is None:
        choose_dim = _get_first_name_per_cluster(cinfo['cluster'])
    fspace2 = fspace[list(choose_dim)]
    if plot:
        _plot_hierarchical_corr_mat(corr_mat, cinfo, selected_feat=choose_dim)
        corr_mat2, cinfo2 = _hierarchical_corr_mat(fspace2, threshold=0.5)
        _plot_hierarchical_corr_mat(corr_mat2, cinfo2)
    return fspace2, cinfo


# ##
# ## Preprocessing
# ##
# ## ##########################################################################

# ##
# ## Feature Space Normalization
# ##

def _raw_feature_array_to_feature_space(df_feat, special_treatment=True):
    """Transforms a raw feature space to a normalized one. `df_feat` is a
    mxn pandas dataframe where m is the number of points and n the number of
    dimensions. Normalization performed per dimension. Normalized array is
    again a dataframe with same dimensions and row/col index."""
    if special_treatment:
        fspace = _prune_feature_space(df_feat)
    else:
        fspace = df_feat
    fspace = (
        fspace
        .apply(_outlier_robust_sigmoid, raw=True)
        .apply(_curtail_at_whiskers, raw=True)
        .apply(_norm_min_max, raw=True)
    )
    if special_treatment:
        fspace = _manually_repair_dfa(fspace)
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
    vec[vec <= low_whisk] = low_whisk
    vec[vec >= up_whisk] = up_whisk
    return vec


def _norm_min_max(feat_vec):
    """Normalizes a vector `feat_vec` to the range of [0, 1]"""
    feat_vec -= np.min(feat_vec)
    feat_vec /= np.max(feat_vec)
    return feat_vec


def _manually_repair_dfa(fspace):
    """The normed expanded set contains only two elements with dfa < 0.1.
    These are removed and the dfa vector normed again as this bin would
    otherwise be unreasonably empty for the equilize_nd_hist algorithm."""
    fspace_mut = fspace.loc[fspace['dfa'] >= 0.1, :]
    fspace_mut.loc[:, 'dfa'] = _norm_min_max(fspace_mut['dfa'])
    return fspace_mut


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
            .sort_values(['inner_corr', 'outer_corr'], ascending=[False, True])
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


def _plot_corr_mat_scatter(df_feat, samples=200, bins=20):
    """Plots the correlation matrix of `df_feat` as a scatterplot,
    where each dim is plotted against each other in a matrix, the diagonal
    shows the histogram of each dimension."""
    pg = sns.PairGrid(df_feat.sample(samples))
    pg.map_upper(sns.scatterplot, s=5)
    pg.map_lower(sns.scatterplot, s=5)
    pg.map_diag(sns.histplot, bins=bins)
    return pg


# ##
# ## Reduction
# ##
# ## ##########################################################################

# ##
# ## Map to Uniform
# ##

def _map_to_uniform(df_feat, n,
                    distance=2.0, n_tries=3, overrelax=1.05, seed=None):
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
        warn(f'There are two points with index {ind_p1} and '
             f'{ind_p2} that have the same distance to another '
             f'point. Choosing {ind_p1}')
        return ind_p1
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

# ## Equilize nd hist

def _equilize_nd_hist(df_feat, df_pool, bins=10, n_addrm=5, n_tries=20,
                      n_max_loops=5000, max_attempts_fail=10, seed=None,
                      return_info=False):
    """Takes the feature dataframe `df_feat` and Computes the nd-hist-array
    with `bins` bins. Consecutively adds and removes elements with
    `_fill_sparse()` and `_remove_empty()` to improve the heterogeneity of
    the set. For more info, see these functions.
    `df_feat` is the original feature dataframe to improve and `df_pool` is
    the extended feature dataframe from which points can be taken. If
    `return_info=True', the function returns a tuple (df_feat_improved,
    df_info) instead of only returning the feature dataframe. `seed`
    initializes the random generator.
    Hyperparameters passed to the subroutines:
    - `bins` defines the number of bins of the nd hist
    - 'n_addrm' how many points to add/ to remove
    - 'n_tries' how many sampled variants are calculated by the subroutines
    Hyperparameters of the outer optimization algorithm:
    - `n_max_loops` how many maximum tries until results are returned if
      premature termination is not hit before
    - `max_attempts_fail´ how many failed attempts are made to improve the
      heterogeneity before adjusting the initial hyperparameters `n_tries`
      and `n_addrm`
    """
    if seed is not None:
        np.random.seed(seed)

    # run empyt dense/fill sparse function n_loops times
    df_before = copy.copy(df_feat)
    failed_attempts = 0
    info = []
    fill_errors = 0
    for i in range(n_max_loops):
        # Show calculation progress by printing loop number
        if (i % 100) == 0:
            print(f'Run {i+1}/{n_max_loops}')

        # Call fill/empty functions
        df_add = _fill_sparse(
            df_before, df_pool,
            bins=bins, n_add=n_addrm, n_tries=n_tries, seed=None, do_warn=False
        )
        n_rm = df_add.shape[0] - df_before.shape[0]
        if n_rm == 0:
            fill_errors += 1
        df_rm = _empty_dense(
            df_add,
            bins=bins, n_rm=n_rm, n_tries=n_tries, seed=None
        )
        df_after = df_rm

        # Evaluate Quality/results of operation
        het_before = _heterogeneity(df_before, bins=bins)
        het_add = _heterogeneity(df_add, bins=bins)
        het_rm = _heterogeneity(df_rm, bins=bins)
        het_after = het_rm

        # update loop var
        if het_after < het_before:
            df_before = df_after
            failed_attempts = 0
        else:
            df_before = df_before
            failed_attempts += 1

        # Update hyperparameter if failed attempts is too high
        if failed_attempts >= max_attempts_fail:
            n_addrm -= 1
            n_tries = int(1.5*n_tries)
            failed_attempts = 0
            if n_addrm != 0:
                print(f'Reached maximum of failed attempts ('
                      f'{max_attempts_fail}). Tuning hyperparameters to '
                      f'n_addrm = {n_addrm} and n_tries = {n_tries} '
                      f'and continue')

        # Gather state information and progress of algorithm
        info.append([i, failed_attempts, n_addrm, n_tries,
                     het_before, het_rm, het_add, het_after])

        # Break loop and algorithm if hyperparameter tuning gets
        # ill-conditioned (n_addrm < 1)
        if n_addrm < 1:
            break

    # Convert Info list to dataframe
    info = pd.DataFrame(
        np.array(info),
        columns=['Run', 'Failed Attempts', 'n_addrm', 'n_tries',
                 'het_before', 'het_rm', 'het_add', 'het_after'])

    # Report failed fill sparse attempts
    print(f'Fill Sparse: '
          f'{fill_errors}/{i+1} attempts were not possible.')  # noqa

    # Write Output
    df_final = df_before
    if return_info:
        return df_final, info
    else:
        return df_final


# ## Empty dense

def _empty_dense(df_feat, bins=10, n_rm=3, n_tries=20, n_largest=5, seed=None):
    """Takes the feature dataframe `df_feat` and Computes the nd-hist-array
    with `bins` bins. Locates the `n_largest`densest bins within this array.
    Randomly removes `n_rm` elements within a randomly chosen bin of
    `n_largest`. Tries `n_tries` variants of random removals and chooses the
    one that returns the best overall heterogeneity of the whole set.
    `seed` initializes the random generator.
    Note that `df_feat` is expected to be normalized in [0, 1] in each
    dimension.
    Returns a copy of the input dataframe with `n_rm` elements removed."""
    if seed is not None:
        np.random.seed(seed)
    if n_rm == 0:
        return df_feat

    # find n densest bins, randomly choose one with propability based on
    #   how dense it is
    hist = _nd_hist(df_feat, bins)
    edges = np.linspace(0, 1, bins+1)
    idxs, vals = _largest_n(hist.to_numpy(), n_largest)
    vals += - 1/bins
    vals[vals < 0] = 0
    i_idxs = np.random.choice(range(len(vals)), p=vals/np.sum(vals))
    # for that bin: determine feature and bounds
    feat_id, edge_id = idxs[i_idxs, :]
    bnds = edges[edge_id], edges[edge_id+1]

    # search in df_feat for all time series that are within bounds in feature
    tf_array = ((df_feat.iloc[:, feat_id] >= bnds[0]) &
                (df_feat.iloc[:, feat_id] <= bnds[1]))
    df_candidates = df_feat.loc[tf_array, :]

    # randomly choose n_rm signals n_tries times
    rm_indexes = [df_candidates.sample(n_rm).index for _ in range(n_tries)]
    # build n_tries new df_feat_rm
    df_rm_list = [df_feat.drop(rm_idxs) for rm_idxs in rm_indexes]
    # evaluate heterogenity for each df_feat_rm
    het = [_heterogeneity(df_feat_rm, bins=bins) for df_feat_rm in df_rm_list]
    # choose best one, return
    return df_rm_list[np.argmin(het)]


def _smallest_n(array, n, maintain_order=True):
    """Determines the `indexes` and `values` of the `n` smallest elements
    within the numpy array `array`.
    taken from https://stackoverflow.com/questions/45689933"""
    idx = np.argpartition(array.ravel(), n)[:n]
    if maintain_order:
        idx = idx[array.ravel()[idx].argsort()]
    idx = np.stack(np.unravel_index(idx, array.shape)).T
    vals = np.array([array[tuple(idx[i, :])] for i in range(idx.shape[0])])
    return idx, vals


def _largest_n(array, n, maintain_order=True):
    """Determines the `indexes` and `values` of the `n` largest elements
    within the numpy array `array`."""
    idx, vals = _smallest_n(-array, n, maintain_order)
    return idx, -vals


# ## Fill Sparse

def _fill_sparse(df_feat, df_pool, bins=10, n_add=3, n_tries=20,
                 n_smallest=8, seed=None, do_warn=True):
    """Takes the feature dataframe `df_feat` and Computes the nd-hist-array
    with `bins` bins. Locates the `n_smallest` sparsest bins within this array.
    Randomly adds `n_add` elements within a randomly chosen bin of
    `n_smallest`. Tries `n_tries` variants of random insertions and chooses the
    one that returns the best overall heterogeneity of the whole set.
    `seed` initializes the random generator. The points are taken from the
    dataframe `df_pool`.
    Note that `df_feat` and `df_pool` are expected to be normalized in [0,
    1] in each dimension.
    Returns a copy of the input dataframe with `n_add` elements inserted."""
    if seed is not None:
        np.random.seed(seed)

    if n_add == 0:
        return df_feat

    # find n emptiest bins, randomly choose one with propability based on
    #   how emtpy it is
    hist = _nd_hist(df_feat, bins)
    edges = np.linspace(0, 1, bins+1)
    idxs, vals = _smallest_n(hist.to_numpy(), n_smallest)
    vals = max(vals) - vals
    vals += 1/bins
    i_idxs = np.random.choice(range(len(vals)), p=vals/np.sum(vals))
    # for that bin: determine feature and bounds
    feat_id, edge_id = idxs[i_idxs, :]
    bounds = edges[edge_id], edges[edge_id+1]

    # remove the search df_feat from the df_pool reference
    df_pool_cleaned = df_pool.drop(index=df_feat.index)
    # search in df_pool_cleaned for all time series that are within bounds in
    #   feature
    tf_array = df_pool_cleaned[df_feat.columns[feat_id]].between(*bounds)
    add_pool = df_pool_cleaned.loc[tf_array, :]

    if (n_pool := add_pool.shape[0]) < n_add:
        if do_warn:
            warn(f'Only {n_pool} points available to add for chosen bin, '
                 f'instead of {n_add}, returning original input instead.')
        return df_feat

    # randomly choose n_add signals n_tries times
    add_samples = [add_pool.sample(n_add) for _ in range(n_tries)]
    # build n_tries new df_char_add
    df_add_list = [pd.concat([df_feat, add_s]) for add_s in add_samples]

    # evaluate heterogenity for each df_char_new
    het = [_heterogeneity(dfadd) for dfadd in df_add_list]

    # choose best one, return
    return df_add_list[np.argmin(het)]


# ##
# ## Methods to determine nd-hist and heterogeneity
# ##


def _nd_hist(df_feat, bins=10):
    """Creates a multidimensional histogram information similar to a
    heatmap. Input is a number of points as nxm array, where n is number of
    points and m is dimension of point. The input is provided as a pandas
    dataframe `df_feat`. Each dimension m is expected to be labeled as a
    feature (columns have names). Returns an m x `bins` array, where m is again
    the dimension and `bins` is the number of bins. Each row represents a
    histogram for one dimension.

    Parameters
    ----------
    df_feat: pandas.DataFrame
        See numpy.histogram(), nxm, n=number of points, m=number of
        features/dimensions
    bins : int, optional, default: 10
        See numpy.histogram(), number of bins of each histogram

    Returns
    -------
    df_hist : pandas.DataFrame
        m x bins, m=number of features/dimension, bins=number of bins
    """
    dim = df_feat.shape[1]
    histarray = np.zeros((dim, bins))
    for ii in range(dim):
        histarray[ii, :], _ = \
            np.histogram(df_feat.iloc[:, ii], bins, [0, 1])
    sums = np.ones((dim, 1)) * df_feat.shape[0]
    histarray /= sums
    df_hist = pd.DataFrame(data=histarray,
                           index=df_feat.columns,
                           columns=np.linspace(1/bins, 1, bins))
    return df_hist


def _heterogeneity(df_feat, bins=10, as_histarray=False):
    """Computes the heterogeneity of dataset `df_feat` The heterogeneity is
    an indirect measure for the discrepancy/uniformity of a set. `df_feat`
    is a mxn feature dataframe with m being the number of points and n being
    the number of features/dimensions of that point. The feature array is
    expected to be normed in [0, 1] per dimension. The heterogeneity is
    calculated by summing up the squared deviations from an uniformely
    distributed histogram with `bins` bins. `as_histarray` is an optional
    argument, if True, the first argument `df_feat` is treated as an
    nd-hist-array instead of an normalized feature array. If false,
    the nd-hist-array will be computed from the feature array.
    If an `histarray` is provied, it must be normed (sum along 1-axis must be
    one).
    Returns a scalar. """
    if as_histarray:
        histarray = df_feat
    else:
        histarray = _nd_hist(df_feat, bins)

    dims, bins = histarray.shape
    max_per_dim = np.sqrt((1 - 1 / bins) ** 2 + 1 / bins ** 2 * (bins - 1))
    average = 1 / bins
    deviation = histarray - average
    # deviation[deviation > 0] = 0   # only penalize bins that are too empty
    return np.sum((np.sum(deviation ** 2, axis=1))**(1/2)) / dims / max_per_dim


def _plot_nd_hist(df_feat, ax=None, bins=10, title='', colorbar=False,
                  xticks=False, ndigits=3, as_histarray=False,
                  gridlinewidth=3, cmap='Greys', norm=None):
    """Plots a multidimensional nd-hist. See `_nd_hist()` for more
    information. `ax` optionally specifies the axis object where to plot the
    nd-hist. `bins=10` determines the number of bins of each histogram,
    `colorbar=False` specifies if an additional colorbar shall be plotted.
    `ndigits=3` determines how many digits will be shown in each bin and as
    a label of how many percent of the points are within this bin. If
    `as_histarray=True`, the first input `df_feat` is treated as a histarray
    dataframe (see `_nd_hist()`) instead of a feature array dataframe.
    `title` labels additionally the nd-histogram plot in the lower left
    corner."""
    if as_histarray:
        histarray = df_feat
    else:
        histarray = _nd_hist(df_feat, bins)

    dim, bins = histarray.shape
    names = histarray.index

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if norm is None:
        kwargs = dict(vmin=0, vmax=2/bins)
    else:
        kwargs = dict(norm=norm)
    mappable = ax.matshow(histarray, aspect='auto', cmap=cmap, **kwargs)
    # Add colorbar if option is set
    if colorbar:
        fig.colorbar(mappable, ax=ax)
    # Loop over data dimensions and create text annotations.
    for ii in range(dim):
        for jj in range(bins):
            color = 'k' if histarray.iloc[ii, jj] < 0.1 else 'w'
            if ndigits is not None:
                ax.text(jj, ii, f'{histarray.iloc[ii, jj]:.{ndigits}f}',
                        ha="center", va="center", color=color, clip_on=True)
    # rewrite x,y ticks
    ax.set_yticks(range(dim),
                  labels=names)
    if xticks:
        ax.set_xticks(np.arange(bins + 1) - 0.5,
                      labels=[f'{x:.2f}' for x in np.linspace(0, 1, bins + 1)])
    else:
        ax.set_xticks([])
    # Add title
    tstring = (
        (f'{title}, Heterogenity = '
         f'{_heterogeneity(histarray, as_histarray=True):.4f}')
        .strip(', ')
    )
    ax.set_title(tstring, {'va': 'top'}, loc='left', y=-0.07)
    # Add secondary y axis with range labels on it
    # make grid to separate each row
    for _, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_yticks(np.arange(histarray.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", axis='y', linestyle='-',
            linewidth=gridlinewidth)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", bottom=False, left=True)
    return fig, ax


# ##
# ## Generate Info Table for Set of Sets
# ##

def _info_for_set_of_sets(set_of_sets):
    data = []
    for set_ in set_of_sets:
        hlist = []
        col_list = []
        for n in [4096, 1024, 256, 64]:
            hlist.append(_heterogeneity(set_['norm_space'][n]))
            hlist.append(_heterogeneity(set_['norm_space'][n][:int(n/2)]))
            hlist.append(_heterogeneity(set_['norm_space'][n][int(n/2):]))
            col_list += [f'H{n}_both', f'H{n}_neg', f'H{n}_posneg']
        data.append(hlist)

    df = pd.DataFrame(data, columns=col_list)  # noqa
    df.describe()
    return df
