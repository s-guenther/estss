"""
The `dimred.py` submodule is an integral component of a larger framework
dedicated to the analysis and manipulation of time series data within feature
spaces. Focused on dimensionality reduction, this submodule provides tools for
reducing the complexity of datasets while retaining critical information,
thereby facilitating more efficient and insightful analyses.

Important Top Level Functions:
------------------------------
- dimensional_reduced_feature_space(df_feat, choose_dim=_REPRESENTATIVES)
    Applies dimensionality reduction to a feature space, selecting key
    representative features based on hierarchical clustering and correlation
    analysis.

Dimensionality Reduction and Clustering:
----------------------------------------
- raw_feature_array_to_feature_space(df_feat, special_treatment=False):
    Converts a raw feature array into a normalized feature space, applying
    various normalization techniques to enhance data quality and consistency.
- hierarchical_corr_mat(df_feat, threshold=0.4, method=_modified_pearson,
  linkmethod='average', clust_crit='distance'):
    Calculates and clusters a hierarchical correlation matrix from a feature
    array, identifying and grouping correlated features for a streamlined
    analysis.

This submodule leverages advanced statistical and machine learning techniques
to efficiently reduce high-dimensional datasets to their most informative
features. It plays a pivotal role in transforming complex data into a more
manageable and interpretable format, paving the way for deeper insights and
more effective decision-making.

Refer to individual function docstrings for more detailed information and usage
instructions.
"""

import copy

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from estss import util

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


def dimensional_reduced_feature_space(df_feat, choose_dim=_REPRESENTATIVES):
    """Reduces the dimensionality of a feature space with the help of a
    hierarchical correlation matrix.

    This function reduces the dimensionality of the given feature space
    `df_feat` by selecting representative features. It first normalizes the
    feature space and then performs hierarchical clustering to identify
    clusters of correlated features. The user can specify custom dimensions to
    choose or let the function select the first feature in each cluster by
    default.
    
    Parameters
    ----------
    df_feat : pandas.DataFrame
        The original high-dimensional feature dataframe.
    choose_dim : list, optional
        List of dimensions to retain after dimensionality reduction. Defaults
        to a preset list of representative features.

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
    >>>    dimensional_reduced_feature_space(df, choose_dim=None)
    >>> print(reduced_df.head())
    >>> print(cluster_info)
    """
    fspace = raw_feature_array_to_feature_space(df_feat,
                                                special_treatment=True)
    corr_mat, cinfo = hierarchical_corr_mat(fspace)
    if choose_dim is None:
        choose_dim = get_first_name_per_cluster(cinfo['cluster'])
    fspace2 = fspace[list(choose_dim)]
    return fspace2, cinfo


# ##
# ## Preprocessing
# ##
# ## ##########################################################################

# ##
# ## Feature Space Normalization
# ##

def raw_feature_array_to_feature_space(df_feat, special_treatment=False):
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
    special_treatment : bool, default: False
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
        .apply(util.norm_min_max, raw=True)
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


def _manually_repair_dfa(fspace):
    """The normed expanded set contains only two elements with dfa < 0.1.
    These are removed and the dfa vector normed again as this bin would
    otherwise be unreasonably empty for the equilize_nd_hist algorithm."""
    fspace_mut = fspace.loc[fspace['dfa'] >= 0.1, :]
    fspace_mut.loc[:, 'dfa'] = util.norm_min_max(fspace_mut['dfa'])
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


def hierarchical_corr_mat(df_feat, threshold=0.4, method=_modified_pearson,
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
    >>> corr_mat, info = hierarchical_corr_mat(df)  # noqa
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
    corr_mat_sort, clust_info = sort_corr_mat(corr_mat, clust_labels)

    info = dict(cluster=clust_info,
                threshold=threshold,
                linkmethod=linkmethod,
                clust_crit=clust_crit)

    return corr_mat_sort, info


def get_first_name_per_cluster(df_info):
    """Returns the first feature within every cluster (clustered are
    expected to be sorted already). `df_info` contains the column `label`
    and the index holds the feature names. Returned as list."""
    names = (df_info
             .groupby('label')
             .apply(lambda df: pd.Series([df.index[0]])))
    return list(names[0].values)


def sort_corr_mat(mat, labels):
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
