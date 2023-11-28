#!/usr/bin/env python3
"""The 'decluster.py' submodule, part of a larger module for handling and
processing time series data in feature space, focuses on declustering and
optimizing the distribution of time series data within a feature space. It
facilitates more manageable and insightful analyzes of large datasets by
reducing redundancy and enhancing the uniformity of data representation.

Key Features:
-------------
- Efficiently reduces and declusters feature sets to facilitate easier analysis
  and processing for subsequent studies.
- Utilizes a variety of statistical and spatial methods to optimize feature set
  heterogeneity.

Top Level Functions:
--------------------
- get_declustered_sets(file='data/declustered_sets.pkl'):
    Loads precomputed declustered sets of data.
- compute_declustered_sets(df_feat_list=None, df_ts_list=None, seed=1340):
    Merges declustered sets of features and time series data, applying
    declustering and reduction techniques.
- decluster_chain(df_feat_norm, set_sizes=(2048, 512, 128, 32), seed=None):
    Sequentially declusteres the size of a feature set to smaller sets,
    ensuring
    data representativeness.
- decluster_single(df_feat, n, seed=None, kws_map=None, kws_rm=None,
  kws_equi=None):
    Systematically declusteres the number of features in a dataset to a
    specified
    count.

Decluster Functions:
--------------------
- map_to_uniform(df_feat, n, distance=2.0, n_tries=3, overrelax=1.05,
  seed=None):
    Creates a uniformly distributed subset from a feature set.
- remove_closest(df_feat, n_final, distance=2.0, leafsize=None):
    Systematically removes points from a dataframe to a specified count.
- equilize_nd_hist(df_feat, df_pool, bins=10, n_addrm=5, n_tries=20,
  n_max_loops=5000, max_attempts_fail=10, seed=None):
    Optimizes the heterogeneity of a feature set through iterative addition and
    removal of elements.

For detailed usage and parameters of each function, refer to the respective
function's docstring within this submodule.

Examples of Usage:
------------------
- To load precomputed declustered sets:
  get_declustered_sets('data/declustered_sets.pkl')
- To compute and merge declustered sets:
  compute_declustered_sets()
- To decluster a normalized feature set:
  decluster_chain(normalized_df)

Note:
-----
- The module interlinks with other submodules in the estss package,
  reflecting a cohesive design approach.
- While it is feasible to reconfigure this submodule as a standalone
  'decluster' package, such an adaptation was not prioritized in the
  development of the estss module. The focus remained on simplicity and
  integration within the broader system.
"""


import copy
import pickle
import random
from warnings import warn

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.stats import qmc

from estss import manifold, features, dimred, util, analyze


# ##
# ## Top Level
# ##
# ## ##########################################################################

def get_declustered_sets(file='data/declustered_sets.pkl'):
    """Loads declustered sets of data from a specified pickle file.

    This function reads a file containing precomputed declustered sets of data,
    typically the output from a feature decluster process, and returns the
    data. The file is expected to be in pickle format. By default, it looks for
    a file named 'declustered_sets.pkl' in the 'data' directory.

    Parameters
    ----------
    file : str, default: 'data/declustered_sets.pkl'
        Path to the pickle file containing the declustered data sets.

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


def compute_declustered_sets(df_feat_list=None, df_ts_list=None, seed=1340):
    """Computes declustered sets of features and time series data frames

    This function processes and merges given feature and time series datasets,
    applies dimensional reduction, and then generates declustered sets of
    features. It handles both negative and positive/negative feature sets,
    merges them, and then maps these sets to the corresponding time series
    data. The function allows for seeding the random number generator to ensure
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
        A list of merged and mapped feature sets corresponding to the
        declustered time series data.
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
        df_ts_list = manifold.get_manifold_ts()

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
    df_feat_norm, cinfo = \
        dimred.dimensional_reduced_feature_space(df_feat_merged)
    print('# ## Reorder Features')
    df_feat_merged.sort_index(axis=1, inplace=True)

    # split and normalize with minmax again
    df_feat_neg_norm = util.norm_min_max(
        df_feat_norm.loc[df_feat_norm.index < 1e6, :]
    )
    df_feat_posneg_norm = util.norm_min_max(
        df_feat_norm.loc[df_feat_norm.index >= 1e6, :]
    )

    # call decluster chain on normalized feature arrays
    print('\n# ##\n# ## Decluster Only Neg Set\n# ##')
    print('##########################################################')
    sets_neg = decluster_chain(df_feat_neg_norm)
    print('\n# ##\n# ## decluster Only PosNeg Set\n# ##')
    print('##########################################################')
    sets_posneg = decluster_chain(df_feat_posneg_norm)

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

def decluster_chain(df_feat_norm, set_sizes=(2048, 512, 128, 32), seed=None):
    """Creates a chain of declusterd feature sets from a normalized feature
    dataframe.

    This function sequentially declusters the size of a given feature set to a
    series of smaller sets, as specified by `set_sizes`. Each subsequent set is
    a declusterd version of the previous one, with the decluster process aiming
    to maintain the representativeness of the original set. The function also
    sorts the resulting sets after the decluster process. It allows for seeding
    the random number generator to ensure reproducibility.

    Parameters
    ----------
    df_feat_norm : pandas.DataFrame
        The normalized feature dataframe to be declustered.
    set_sizes : tuple of int, default: (2048, 512, 128, 32)
        The sizes of the declustered feature sets to be created.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    sets : list of pandas.DataFrame
        A list of declustered feature dataframes, each corresponding to the
        sizes specified in `set_sizes`.

    Examples
    --------
    >>> df_normalized = pd.DataFrame(np.random.rand(10000, 10))
    >>> declustered_sets = decluster_chain(df_normalized)
    >>> for decluster_set in declustered_sets:
    >>>     print(decluster_set.shape)
    """

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Loop through set_sizes to compute sets
    large_set = df_feat_norm
    sets = []
    for n in set_sizes:
        print(f'\n# ## Compute n = {n} Set\n################################')
        small_set = decluster_single(large_set, n, seed)
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
# ## Decluster Single
# ##

def decluster_single(df_feat, n,
                     seed=None, kws_map=None, kws_rm=None, kws_equi=None):
    """Declusters the number of elements in a dataset to a specified count.

    This function systematically declusteres the number of elements in
    `df_feat` to `n` by applying a series of steps. It first maps the elements
    to a uniform distribution, then removes the closest elements to reduce
    redundancy, and finally equilizes the n-dimensional histogram. This process
    aims to extract a representative and declustered subset of elements.
    The function allows for customizing each step with additional keyword
    arguments.

    Note: Although not neccessary for a proper functionioning of the
    algorithm, the feature vectors within `df_feat` are expected to be already
    normalized to a range of [0, 1]

    Parameters
    ----------
    df_feat : pandas.DataFrame
        The original feature dataframe from which to select a subset.
    n : int
        The desired number of features in the declustered set.
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
        The declustered feature dataframe with `n` features.

    Examples
    --------
    >>> df = pd.DataFrame(np.random.rand(100, 10))
    >>> declustered_df = decluster_single(df, 30)
    >>> print(declustered_df.shape)
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
    df_red = map_to_uniform(df_feat, int(n * 1.2), seed=None, **kws_map)
    print(f'done')
    print(f'# ## Remove closest ...', end=' ')
    df_red = remove_closest(df_red, n, **kws_rm)
    print(f'done')
    print(f'# ## Equilize nd-hist ...')
    df_red = equilize_nd_hist(df_red, df_feat, seed=None, **kws_equi)
    print(f'... done')
    return df_red


# ##
# ## Decluster
# ##
# ## ##########################################################################

# ##
# ## Map to Uniform
# ##

def map_to_uniform(df_feat, n,
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
    >>> uniform_df = map_to_uniform(df, 100)
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
    """Finds the nearest points in one dataframe to each point in another
    dataframe.

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
        A subset of 'large' containing the nearest point to each point in
        'small'.

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

def remove_closest(df_feat, n_final, distance=2.0, leafsize=None):
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
    >>> reduced_df = remove_closest(df, 500)
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

def equilize_nd_hist(df_feat, df_pool, bins=10, n_addrm=5, n_tries=20,
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
    >>> df_pool = pd.DataFrame(np.random.rand(200, 5))  # noqa
    >>> df_optimized = equilize_nd_hist(df_original, df_pool)
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
        het_before = analyze.heterogeneity(df_before, bins=bins)
        het_add = analyze.heterogeneity(df_add, bins=bins)
        het_rm = analyze.heterogeneity(df_rm, bins=bins)
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
    hist = analyze.nd_hist(df_feat, bins)
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
    het = [analyze.heterogeneity(df_feat_rm, bins=bins)
           for df_feat_rm in df_rm_list]
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
    hist = analyze.nd_hist(df_feat, bins)
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
    het = [analyze.heterogeneity(dfadd) for dfadd in df_add_list]

    # choose best one, return
    return df_add_list[np.argmin(het)]
