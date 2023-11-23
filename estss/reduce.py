#!/usr/bin/env python3
"""The 'reduce.py' submodule is part of a larger module designed for handling
and processing time series data in feature space. This submodule focuses on
reducing and declustering time series data within a feature space, enabling
more manageable and insightful analyses of large datasets.

Important Top Level Functions:
------------------------------
- get_reduced_sets(file='data/reduced_sets.pkl'):
    Loads precomputed reduced sets of data from a pickle file, typically output
    from feature reduction processes.
- compute_reduced_sets(df_feat_list=None, df_ts_list=None, seed=1340):
    Computes and merges reduced sets of features and time series data, applying
    dimensional reduction.
- reduce_chain(df_feat_norm, set_sizes=(2048, 512, 128, 32), seed=None):
    Sequentially reduces the size of a feature set to smaller sets, maintaining
    data representativeness.
- reduce_single(df_feat, n, seed=None, kws_map=None, kws_rm=None,
  kws_equi=None):
    Systematically reduces the number of features in a dataset to a specified
    count.
- dimensional_reduced_feature_space(df_feat, choose_dim=_REPRESENTATIVES,
  plot=True):
    Reduces the dimensionality of a feature space, selecting representative
    features for analysis.

Important Preprocessing Functions:
----------------------------------
- _raw_feature_array_to_feature_space(df_feat, special_treatment=True):
    Transforms a raw feature array into a normalized feature space, applying
    several normalization steps.

Important Dimensionality Reduction Functions:
---------------------------------------------
- _hierarchical_corr_mat(df_feat, threshold=0.4, method=_modified_pearson,
  linkmethod='average', clust_crit='distance'):
    Computes a hierarchical correlation matrix, clustering features based on
    their correlations.
- _plot_hierarchical_corr_mat(corr_mat, info, selected_feat=None,
  clust_color='limegreen', ax=None, write_clust_names=True):
    Visualizes a hierarchical correlation matrix, highlighting clusters and
    their representatives.
- _plot_corr_mat_scatter(df_feat, samples=200, bins=20):
    Creates a scatterplot matrix to visualize pairwise correlations between
    each dimension in a feature dataframe.

Important Reduction Functions:
------------------------------
- _map_to_uniform(df_feat, n, distance=2.0, n_tries=3, overrelax=1.05,
  seed=None):
    Selects points from a feature set to create a uniformly distributed subset.
- _remove_closest(df_feat, n_final, distance=2.0, leafsize=None):
    Removes points from a dataframe until only a specified number of points are
    left.
- _equilize_nd_hist(df_feat, df_pool, bins=10, n_addrm=5, n_tries=20,
  n_max_loops=5000, max_attempts_fail=10, seed=None):
    Optimizes the heterogeneity of a feature set by adding and removing
    elements.

Important Misc Functions:
-------------------------
- _nd_hist(df_feat, bins=10):
    Creates a multidimensional histogram of features for analysis of feature
    distribution.
- _plot_nd_hist(df_feat, ax=None, bins=10, title='', colorbar=False,
  xticks=False, ndigits=3, as_histarray=False):
    Plots a multidimensional histogram, visualizing the distribution of
    features in multiple dimensions.
- _heterogeneity(df_feat, bins=10, as_histarray=False):
    Calculates the heterogeneity of a dataset, reflecting the uniformity or
    discrepancy in feature distribution.

Refer to individual function docstrings for more detailed information and usage
instructions.
"""

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
    """Loads reduced sets of data from a specified pickle file.

    This function reads a file containing precomputed reduced sets of data,
    typically the output from a feature reduction process, and returns the
    data. The file is expected to be in pickle format. By default, it looks for
    a file named 'reduced_sets.pkl' in the 'data' directory, but a different
    file path can be specified.

    Parameters
    ----------
    file : str, default: 'data/reduced_sets.pkl'
        Path to the pickle file containing the reduced data sets.

    Returns
    -------
    data : object
        dict of dicts of data frames. The upper level dict contains the keys
        'features', 'ts', 'norm_space' and 'info'. Each holds a lower level
        dict with the keys 4096, 1024, 256 and 64. Each lower level value is a
        pandas dataframe holding the information defined by the two dict keys.
    """
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


def compute_reduced_sets(df_feat_list=None, df_ts_list=None, seed=1340):
    """Computes reduced sets of features and time series data frames

    This function processes and merges given feature and time series datasets,
    applies dimensional reduction, and then generates reduced sets of features.
    It handles both negative and positive/negative feature sets, merges them,
    and then maps these sets to the corresponding time series data. The
    function allows for seeding the random number generator to ensure
    reproducibility. If no arguments are not provided, default ones are loaded.

    Parameters
    ----------
    df_feat_list : list of pandas.DataFrame, optional
        A list of feature dataframes. If None, default feature sets are loaded.
    df_ts_list : list of pandas.DataFrame, optional
        A list of time series dataframes. If None, default time series sets are
        loaded.
    seed : int, default: 1340
        Seed for the random number generator.

    Returns
    -------
    sets : list of pandas.DataFrame
        A list of merged and mapped feature sets corresponding to the reduced
        time series data.
    """

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
    """Maps normalized sets to corresponding feature and time series data.

    This function creates mappings between normalized sets, features, and time
    series data.  It organizes these mappings into a dictionary containing the
    features, time series, normalized space, and additional information for
    each set. The function processes each normalized set, reindexes them
    according to a specified scheme, and then associates each set with its
    corresponding features and time series data. The result is a comprehensive
    mapping that facilitates easy access to related data across different sets.

    Parameters
    ----------
    norm_sets : list of pandas.DataFrame
        A list of normalized data sets.
    df_feat : pandas.DataFrame
        The original features dataframe.
    df_ts : pandas.DataFrame
        The original time series dataframe.

    Returns
    -------
    sets : dict
        A dictionary containing the mapped sets. Keys include 'features', 'ts'
        (time series), 'norm_space' (normalized space), and 'info' (additional
        information) for each set size.
    """
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
    """Reindexes a list of dataframes based on provided mappings.

    This function takes a list of dataframes ('sets') and a corresponding list
    of mappings, and applies these mappings to reindex each dataframe in the
    list. The reindexing process involves changing the index of each dataframe
    in 'sets' according to the mapping provided.  This is typically used to
    align dataframes with a common indexing scheme for consistency and easier
    comparison or merging.

    Subroutine of `_map_sets()`

    Parameters
    ----------
    sets : list of pandas.DataFrame
        A list of dataframes to be reindexed.
    mappings : list of pandas.Series
        A list of mappings, where each mapping corresponds to a dataframe in
        'sets'. The mapping is a pandas Series where the values are the current
        index and the index of the Series represents the new index.

    Returns
    -------
    trans_sets : list of pandas.DataFrame
        The list of transformed dataframes with updated indices.
    """
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
    """Creates a chain of reduced feature sets from a normalized feature
    dataframe.

    This function sequentially reduces the size of a given feature set to a
    series of smaller sets, as specified by `set_sizes`. Each subsequent set is
    a reduced version of the previous one, with the reduction process aiming to
    maintain the representativeness of the original set. The function also
    sorts the resulting sets after the reduction process. It allows for seeding
    the random number generator to ensure reproducibility.

    Parameters
    ----------
    df_feat_norm : pandas.DataFrame
        The normalized feature dataframe to be reduced.
    set_sizes : tuple of int, default: (2048, 512, 128, 32)
        The sizes of the reduced feature sets to be created.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    sets : list of pandas.DataFrame
        A list of reduced feature dataframes, each corresponding to the sizes
        specified in `set_sizes`.

    Examples
    --------
    >>> df_normalized = pd.DataFrame(np.random.rand(10000, 10))
    >>> reduced_sets = reduce_chain(df_normalized)
    >>> for reduced_set in reduced_sets:
    >>>     print(reduced_set.shape)
    """

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
    """Sorts a list of dataframes based on clustering and feature statistics.

    This function takes a list of dataframes, each representing a set of
    features dataframes , and sorts them based on clustering information and
    statistical measures. It adds a cluster identifier to each feature,
    categorizes features based on the mean value, and sorts them within each
    set. The sorting is performed based on the cluster identifier, mean cluster
    category, interquartile range (IQR), and original index. This process
    organizes the features in each set for better interpretability and
    analysis.

    Parameters
    ----------
    in_sets : list of pandas.DataFrame
        A list of dataframes, each containing a set of features to be sorted.

    Returns
    -------
    sets : list of pandas.DataFrame
        The sorted list of dataframes, with features organized based on
        clustering and statistical measures.
    """
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
    """Reduces the number of elements in a dataset to a specified count.

    This function systematically reduces the number of elements in `df_feat` to
    `n` by applying a series of steps. It first maps the elements to a uniform
    distribution, then removes the closest elements to reduce redundancy, and
    finally equilizes the n-dimensional histogram. This process aims to extract
    a representative and declustered subset of elements. The function allows
    for customizing each step with additional keyword arguments.

    Note: Although not neccessary for a proper functionioning of the
    algorithm, the feature vectors within `df_feat` are expected to be already
    normalized to a range of [0, 1]

    Parameters
    ----------
    df_feat : pandas.DataFrame
        The original feature dataframe from which to select a subset.
    n : int
        The desired number of features in the reduced set.
    seed : int, optional
        Seed for the random number generator.
    kws_map : dict, optional
        Additional keyword arguments for the mapping step.
    kws_rm : dict, optional
        Additional keyword arguments for the feature removal step.
    kws_equi : dict, optional
        Additional keyword arguments for the equilizing step.

    Returns
    -------
    df_red : pandas.DataFrame
        The reduced feature dataframe with `n` features.

    Examples
    --------
    >>> df = pd.DataFrame(np.random.rand(100, 10))
    >>> reduced_df = reduce_single(df, 30)
    >>> print(reduced_df.shape)
    """
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
    """Reduces the dimensionality of a feature space and optionally plots the
    hierarchical correlation matrix.

    This function reduces the dimensionality of the given feature space
    `df_feat` by selecting representative features. It first normalizes the
    feature space and then performs hierarchical clustering to identify
    clusters of correlated features. The user can specify custom dimensions to
    choose or let the function select the first feature in each cluster by
    default. Optionally, it plots the hierarchical correlation matrices before
    and after dimensionality reduction to visualize the clustering and
    correlation structure of features.

    Parameters
    ----------
    df_feat : pandas.DataFrame
        The original high-dimensional feature dataframe.
    choose_dim : list, optional
        List of dimensions to retain after dimensionality reduction. Defaults
        to a preset list of representative features.
    plot : bool, default: True
        If True, plots the hierarchical correlation matrices before and after
        the dimensionality reduction.

    Returns
    -------
    fspace2 : pandas.DataFrame
        The reduced feature space dataframe.
    cinfo : dict
        Information dictionary containing details about the hierarchical
        clustering and the selected features.

    Examples
    --------
    >>> df = pd.DataFrame(np.random.rand(100, 20))
    >>> reduced_df, cluster_info = \
            dimensional_reduced_feature_space(df, choose_dim=None)
    >>> print(reduced_df.head())
    >>> print(cluster_info)
    """

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
    """Transforms a raw feature array into a normalized feature space.

    This function normalizes a raw feature dataframe on a per-dimension basis,
    converting it into a normalized feature space. It applies several steps of
    normalization, including an outlier-robust sigmoid transformation,
    curtailing values at whiskers, and normalizing each feature to the [0, 1]
    range. The function allows for optional special treatment, which includes
    additional pruning and repairing of the feature space.

    Parameters
    ----------
    df_feat : pandas.DataFrame
        A mxn dataframe representing the raw feature space, where 'm' is the
        number of data points and 'n' is the number of dimensions/features.
    special_treatment : bool, default: True
        If True, performs additional processing steps like pruning and manual
        repairs.

    Returns
    -------
    fspace : pandas.DataFrame
        The normalized feature space, maintaining the same dimensions and index
        as `df_feat`.
    """
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
    """Computes and clusters a hierarchical correlation matrix from a feature
    array.

    This function calculates a correlation matrix from the given feature array
    `df_feat` and performs hierarchical clustering on the features. The
    clustering is based on a specified threshold, and a modified Pearson
    correlation method is used by default. This method considers linear,
    squared, square root, and reciprocal correlations. The function returns the
    sorted correlation matrix along with cluster information, including inner
    and outer correlations of clusters and algorithm settings used.

    Parameters
    ----------
    df_feat : pandas.DataFrame or numpy.ndarray
        Feature array for which the hierarchical correlation matrix is
        computed.
    threshold : float, default: 0.4
        Threshold for clustering, with features above this correlation level
        clustered together.
    method : function, default: _modified_pearson
        Correlation computation method. Default is a modified Pearson
        correlation.
    linkmethod : str, default: 'average'
        Linkage method used for clustering.
    clust_crit : str, default: 'distance'
        Criterion used to form clusters.

    Returns
    -------
    corr_mat_sort : pandas.DataFrame
        The sorted correlation matrix after clustering.
    info : dict
        Dictionary containing information about the clustering, including
        cluster details, threshold, linkage method, and clustering criterion.

    Examples
    --------
    >>> df = pd.DataFrame(np.random.rand(10, 5))
    >>> corr_mat, info = _hierarchical_corr_mat(df)
    >>> print(corr_mat)
    >>> print(info)
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


def _plot_hierarchical_corr_mat(corr_mat, info, selected_feat=None,
                                clust_color='limegreen', ax=None,
                                write_clust_names=True, **kwargs):
    """Visualizes a hierarchical correlation matrix with cluster delineation.

    This function plots a precomputed correlation matrix, highlighting the
    identified clusters and their representatives. It frames each cluster and
    can optionally label the y-axis with cluster names.  The plot includes
    additional information about the inner and outer correlations of the
    clusters and allows customization of the cluster representation color and
    other heatmap properties.

    Parameters
    ----------
    corr_mat : pandas.DataFrame or numpy.ndarray
        The precomputed correlation matrix to be visualized.
    info : dict
        Information dictionary returned by `_hierarchical_corr_mat()`,
        containing details about the clusters.
    selected_feat : list, optional
        List of features to highlight as cluster representatives. If None,
        defaults to the first feature of each cluster.
    clust_color : str, default: 'limegreen'
        Color used for highlighting clusters and their representatives.
    ax : matplotlib.axes.Axes, optional
        The axes object on which to plot the heatmap. If None, a new figure and
        axes are created.
    write_clust_names : bool, default: True
        Whether to label the y-axis with cluster names.
    **kwargs
        Additional keyword arguments passed to `sns.heatmap()`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.

    Examples
    --------
    >>> corr_matrix = pd.DataFrame(np.random.rand(10, 10))
    >>> info = {'cluster': pd.DataFrame({'label': np.random.randint(0, 3, 10)})}
    >>> fig, ax = _plot_hierarchical_corr_mat(corr_matrix, info)
    >>> plt.show()
    """
    # Get feature names of cluster representatives, either by taking the first
    # feature of each cluster or through selected_feat argument
    if selected_feat is None:
        selected_feat = _get_first_name_per_cluster(info['cluster'])
    names = [s if s in selected_feat else ""
             for s in list(info['cluster'].index)]

    # Make main plot
    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots()
    if write_clust_names:
        yt_labels = names
    else:
        yt_labels = False
    sns.heatmap(
        np.abs(corr_mat),
        vmin=0, vmax=1, cmap=_create_custom_cmap(info['threshold']), ax=ax,
        square=True, xticklabels=False, yticklabels=yt_labels, **kwargs
    )

    # Make Cluster Rectangles and highlight Cluster Representatives
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    labels = info["cluster"].label.values
    total = len(labels)
    for ll in set(labels):
        start = np.searchsorted(labels, ll)
        stop = np.searchsorted(labels, ll, side='right')
        ax.add_patch(_cluster_rect(start, stop, total, xlim, ylim,
                                   edgecolor=clust_color))
    for start, nn in enumerate(names):
        if nn == "":
            continue
        ax.add_patch(_cluster_rect(start, start+1, total, xlim, ylim,
                                   facecolor=clust_color, edgecolor='none'))

    # Add cluster inner/outer correlation info
    idict = _cluster_info(info['cluster'], selected_feat)
    iinfo = idict['inner_agg']
    oinfo = idict['outer_agg']
    infostring = (f'Number of Clusters:\n'
                  f'   {len(selected_feat)}\n'
                  f'Avg. Inner-Cluster-Corr.:\n'
                  f'   {iinfo["mean"]:.2f} $\pm$ {iinfo["std"]:.2f}\n'  # noqa
                  f'Avg. Outer-Cluster-Corr.:\n'
                  f'   {oinfo["mean"]:.2f} $\pm$ {oinfo["std"]:.2f}\n\n'
                  f'Computation Info:\n'
                  f'   Threshold: {info["threshold"]}\n'
                  f'   Linkage Method: {info["linkmethod"]}\n'
                  f'   Cluster Criterion: {info["clust_crit"]}')  # noqa
    ax.text((xlim[1] - xlim[0])*0.98 + xlim[0],
            (ylim[1] - ylim[0])*0.98 + ylim[0],
            infostring,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'),
            va='top', ha='right', ma='left', fontsize=8)

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


def _create_custom_cmap(threshold=0.4, c_gray='gray', c_col='Blues',
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
        np.linspace(1, (cgap + (1 - cgap) * threshold), nval2)
    )
    colors = np.vstack([c1, c2[::-1, :]])
    mymap = mcolors.LinearSegmentedColormap.from_list(
        'mymap', colors, N=n_values
    )
    return mymap


def _plot_corr_mat_scatter(df_feat, samples=200, bins=20):
    """Creates a scatterplot matrix of the correlation matrix from a feature
    dataframe.

    This function generates a scatterplot matrix to visualize the pairwise
    correlations between each dimension (feature) in the provided dataframe
    `df_feat`. The scatterplots are displayed for each pair of dimensions, with
    histograms along the diagonal showing the distribution of each individual
    dimension. A subset of the data can be sampled for visualization to enhance
    performance and clarity.

    Parameters
    ----------
    df_feat : pandas.DataFrame
        The feature dataframe to be visualized.
    samples : int, default: 200
        The number of data points to sample from `df_feat` for the scatterplot
        matrix.
    bins : int, default: 20
        The number of bins to use for the histograms along the diagonal.

    Returns
    -------
    pg : seaborn.axisgrid.PairGrid
        The PairGrid object containing the scatterplot matrix.

    Examples
    --------
    >>> df = pd.DataFrame(np.random.rand(1000, 5), columns=['A', 'B', 'C', 'D', 'E'])
    >>> pg = _plot_corr_mat_scatter(df)
    >>> plt.show()
    """
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
    """Selects points from a feature set to create a roughly uniformly
    distributed subset.

    This function selects `n` points from the input feature dataframe `df_feat`
    such that they are uniformly distributed within the feature space. It
    employs a quasi-random Halton sequence to generate points in the feature
    space and then finds the nearest points in `df_feat`. The function allows
    for control over the distance metric, the number of attempts, and
    relaxation parameters to ensure an even distribution.

    Parameters
    ----------
    df_feat : pandas.DataFrame
        The input feature dataframe from which to select points.
    n : int
        The number of points to select.
    distance : float, default: 2.0
        The distance metric to use (e.g., 1 for Manhattan, 2 for Euclidean).
    n_tries : int, default: 3
        The number of attempts to generate a uniform set.
    overrelax : float, default: 1.05
        A relaxation parameter to control the density of the queried points.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    pandas.DataFrame
        A subset of `df_feat` that approximates a uniform distribution within
        the feature space.

    Examples
    --------
    >>> df = pd.DataFrame(np.random.rand(1000, 5))
    >>> uniform_df = _map_to_uniform(df, 100)
    """
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
    """Finds the nearest points in one dataframe to each point in another dataframe.

    This function identifies the nearest point in the 'large' dataframe for
    each point in the 'small' dataframe.  It uses a k-d tree for efficient
    nearest neighbor search. The function can also remove outlier matches
    (points in 'small' that are far from any point in 'large') and duplicate
    matches. It supports parallel processing for improved performance.

    Parameters
    ----------
    large : numpy.ndarray or pandas.DataFrame
        The dataframe containing the set of points in which nearest neighbors
        are to be found.
    small : numpy.ndarray or pandas.DataFrame
        The dataframe containing the set of points for which nearest neighbors
        are sought.
    distance : float, default: 2.0
        The distance metric to use (e.g., 1 for Manhattan, 2 for Euclidean).
    rm_outliers : bool, default: True
        If True, removes matches that are considered outliers.
    rm_duplicates : bool, default: True
        If True, removes duplicate matches.
    leafsize : int, optional
        The leaf size parameter of the k-d tree. If None, a default size is
        calculated.
    workers : int, default: -1
        The number of worker threads for parallel processing. -1 means using
        all processors.

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        A subset of 'large' containing the nearest point to each point in 'small'.

    Examples
    --------
    >>> large_df = pd.DataFrame(np.random.rand(1000, 3))
    >>> small_df = pd.DataFrame(np.random.rand(50, 3))
    >>> nearest_points = _find_nearest(large_df, small_df)
    >>> print(nearest_points.shape)
    """
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
    """Systematically removes points from a dataframe to reach a specified
    count.

    This function reduces the number of points in a dataframe `df_feat` to
    `n_final` by iteratively removing the closest pairs of points. It uses a
    k-d tree for efficient distance calculations. The points to be removed are
    determined based on the specified distance metric. This function is useful
    for thinning a dataset by removing redundant or highly similar data points.

    Parameters
    ----------
    df_feat : pandas.DataFrame or numpy.ndarray
        The dataframe from which points are to be removed.
    n_final : int
        The desired number of points to retain in the dataframe.
    distance : float, default: 2.0
        The distance metric for the k-d tree (e.g., 1 for Manhattan, 2 for
        Euclidean).
    leafsize : int, optional
        The leaf size parameter for the k-d tree. If None, a default size is
        used.

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        The reduced dataframe with only `n_final` points remaining.

    Examples
    --------
    >>> df = pd.DataFrame(np.random.rand(1000, 3))
    >>> reduced_df = _remove_closest(df, 500)
    >>> print(reduced_df.shape)
    """
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
    """ Optimizes the heterogeneity of a feature set by adding and removing
    elements.

    This function improves the uniformity of distribution of a feature
    dataframe `df_feat` by iteratively adding and removing elements. It uses
    the `_fill_sparse()` and `_remove_empty()` functions to add elements to
    sparse areas and remove them from dense areas of the feature space,
    respectively. The process is repeated in loops until either the maximum
    number of loops is reached or the heterogeneity can no longer be improved.
    The function allows for tracking and tuning the hyperparameters during the
    optimization process.

    Parameters
    ----------
    df_feat : pandas.DataFrame
        The original feature dataframe to be optimized.
    df_pool : pandas.DataFrame
        The extended feature dataframe from which points can be added to
        `df_feat`.
    bins : int, default: 10
        The number of bins for the nd-hist used in the optimization process.
    n_addrm : int, default: 5
        Number of points to add or remove in each loop iteration.
    n_tries : int, default: 20
        Number of sampled variants calculated by the subroutines in each
        iteration.
    n_max_loops : int, default: 5000
        Maximum number of optimization loops to run.
    max_attempts_fail : int, default: 10
        Number of failed attempts allowed before adjusting the hyperparameters
        `n_tries` and `n_addrm`.
    seed : int, optional
        Seed for the random number generator.
    return_info : bool, default: False
        If True, returns a tuple of the optimized dataframe and a dataframe
        with optimization info.

    Returns
    -------
    pandas.DataFrame
        The optimized feature dataframe if `return_info` is False. Otherwise, a
        tuple of the optimized dataframe and a dataframe containing detailed
        information about each optimization loop.

    Examples
    --------
    >>> df_original = pd.DataFrame(np.random.rand(100, 5))
    >>> df_pool = pd.DataFrame(np.random.rand(200, 5))
    >>> df_optimized = _equilize_nd_hist(df_original, df_pool)
    """
    if seed is not None:
        np.random.seed(seed)

    # run empty dense/fill sparse function n_loops times
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
    """ Calculates the heterogeneity of a dataset represented by a feature
    array or histogram.

    This function measures the heterogeneity of a dataset, which reflects the
    uniformity or discrepancy in the distribution of features. It computes this
    by analyzing the squared deviations from an uniformly distributed
    histogram. The function can operate directly on a feature array or on a
    precomputed histogram array. For a feature array, each feature dimension
    should be normalized in the range [0, 1].

    Parameters
    ----------
    df_feat : pandas.DataFrame or numpy.ndarray
        A mxn feature array, with m points and n dimensions, or a precomputed
        histogram array if `as_histarray` is set to True. Each dimension should
        be normalized in the range [0, 1].
    bins : int, default: 10
        The number of bins to be used in histogram computation.
    as_histarray : bool, default: False
        If True, treats `df_feat` as a precomputed histogram array.

    Returns
    -------
    heterogeneity : float
        A scalar value representing the heterogeneity of the dataset. A higher
        value indicates greater discrepancy in feature distribution.

    Notes
    -----
    The heterogeneity is a measure of how evenly distributed the data is across
    the histogram bins.  This function can be particularly useful in
    understanding the diversity of feature distributions in a dataset.

    Examples
    --------
    >>> df = pd.DataFrame(np.random.rand(100, 4), columns=['A', 'B', 'C', 'D'])
    >>> heterogeneity = _heterogeneity(df)
    >>> print(f'Heterogeneity of the dataset: {heterogeneity}')
    """
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
    """Generates a plot of a multidimensional histogram (nd-hist) from a
    feature array.

    This function creates a visual representation of the distribution of
    features in a multidimensional space. It accepts either a feature array or
    a precomputed histogram array. It provides options to customize the
    appearance of the plot, including the number of bins, color mapping, grid
    appearance, and more.  The plot can be enhanced with a colorbar, custom
    tick labels, and a title.  Each cell's value can be annotated, and the
    overall heterogeneity of the data can be displayed in the title.

    Parameters
    ----------
    df_feat : pandas.DataFrame or numpy.ndarray
        A feature array (dataframe or array) to plot, or a precomputed
        histogram array if `as_histarray` is True.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the histogram. If None, a new figure and axes
        are created.
    bins : int, default: 10
        The number of bins for the histogram in each dimension.
    title : str, default: ''
        Title for the plot, displayed in the lower left corner.
    colorbar : bool, default: False
        Whether to add a colorbar to the plot.
    xticks : bool, default: False
        Whether to display x-axis tick labels.
    ndigits : int, default: 3
        Number of decimal places for labels in each bin.
    as_histarray : bool, default: False
        If True, treats `df_feat` as a precomputed histogram array.
    gridlinewidth : int, default: 3
        Line width for the grid separating histogram bins.
    cmap : str, default: 'Greys'
        Colormap for the histogram plot.
    norm : matplotlib.colors.Normalize, optional
        Normalization for the colormap.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.

    Examples
    --------
    >>> df = pd.DataFrame(np.random.rand(100, 4), columns=['A', 'B', 'C', 'D'])
    >>> fig, ax = _plot_nd_hist(df)
    >>> plt.show()
    """

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
    """ Calculates and summarizes the heterogeneity of sets in a collection.

    For each set in `set_of_sets`, this function evaluates the heterogeneity
    of its 'norm_space' at different sizes (4096, 1024, 256, 64) and their
    halves. The results are aggregated into a DataFrame, providing a
    statistical description for the heterogeneity of each set.

    Parameters:
    - set_of_sets (iterable): A collection of sets with 'norm_space' data.

    Returns:
    - pd.DataFrame: A summary DataFrame with heterogeneity values for
      each set size and its divisions.
    """
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
