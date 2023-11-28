#!/usr/bin/env python3
"""
The `analyze.py` submodule is a component of a larger framework dedicated to
time series data analysis and manipulation within feature space.
This submodule specializes in analyzing and visualizing the data, particularly
focusing on the distribution of features and correlations within datasets.

Important Misc Functions:
-------------------------
- nd_hist(df_feat, bins=10):
    Generates a multidimensional histogram of features, aiding in the analysis
    of feature distribution across multiple dimensions.
- plot_nd_hist(df_feat, ax=None, bins=10, title='', colorbar=False,
  xticks=False, ndigits=3, as_histarray=False):
    Visualizes a multidimensional histogram, providing an intuitive
    representation of feature distribution in a dataset.
- heterogeneity(df_feat, bins=10, as_histarray=False):
    Computes the heterogeneity of a dataset, a measure that reflects the
    uniformity or variance in the distribution of features.

Visualization and Interpretation Tools:
---------------------------------------
- plot_df_ts(df_ts, n=128, which='head', grid=(8, 4), ylim=(-1, 1), **kwargs):
    Plots a selection of time series from a dataframe, useful for initial data
    exploration and analysis.
- plot_corr_mat_scatter(df_feat, samples=200, bins=20):
    Creates a scatterplot matrix from a feature dataframe, highlighting the
    pairwise correlations and distributions of features.

Hierarchical Correlation Analysis:
----------------------------------
- plot_hierarchical_corr_mat(corr_mat, info, selected_feat=None,
  clust_color='limegreen', ax=None, write_clust_names=True, **kwargs):
    Visualizes a hierarchical correlation matrix, delineating identified
    clusters and providing insights into the correlation structure of the
    dataset.

Refer to individual function docstrings for more detailed information and usage
instructions.
"""

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns

from estss import util, dimred


# ##
# ## Plot a dataframe of time series
# ##

def plot_df_ts(df_ts, n=128, which='head', grid=(8, 4), ylim=(-1, 1),
               **kwargs):
    """Plots a number of time series from a dataframe.

    Dataframe `df_ts` must be an nxm array with m beeing the number of time
    series and n being the number of points of the time series.
    Multiple figures will be created, if `n` exceeds the number of subplots
    spanned by `grid`.

    Parameters
    ----------
    df_ts : pandas.DataFrame
        nxm dataframe containing m time series with n points each
    n : int or None, default: 128
        number of time series to plot, if `n` is None, plot all
    which : str, default: 'random'
        defines whether the first, the last, or a random selection of time
        series shall be plotted, must be 'head', 'tail', 'random'
    grid : 2-element iterable, default: (8, 4)
        2-element vector [rows, cols] Time series are plotted into a grid of
        subplots, defines the dimension of the subplotgrid.
    ylim : 2-element iterable, default: (-1, 1)
        y-limits of the axis, if None, determine automatically
    **kwargs : dict()
        Passed to plot() fcn

    Returns
    -------
    figs : list of matplotlib.pyplot.figure
        list of handles to the figure object of the plot
    axs : list of matplotlib.pyplot.axes
        list of handles to the axes object of the plot"""

    # get a subset of df_ts if n is specified
    if n is not None and n < df_ts.columns.size:
        if which in ['head', 'start']:
            df = df_ts.iloc[:, :n]
        elif which in ['tail', 'end']:
            df = df_ts.iloc[:, -n:]
        elif which == 'random':
            df = df_ts.sample(n, axis='columns')
        else:
            whichlist = ['random', 'head', 'tail', 'start', 'end']
            raise ValueError(f'Unknown argument which = {which}, '
                             f'must be in {whichlist}')
    else:
        df = df_ts

    n_ts = df.columns.size
    n_grid = grid[0]*grid[1]
    n_figs = (n_ts - 1)//n_grid + 1
    figs = []
    axs = []

    for i_fig in range(n_figs):
        fig, ax = plt.subplots(*grid)
        plt.setp(ax, 'frame_on', False)
        for isig, (k, l) in enumerate(np.ndindex(*grid)):
            isig += n_grid*i_fig
            if isig < n_ts:
                ax[k, l].plot([0, 1000], [0, 0], color='gray')
                ax[k, l].plot(df.iloc[:, isig], **kwargs)
                ax[k, l].text(0, 1, str(df.columns[isig]),
                              ha='left', va='top',
                              transform=ax[k, l].transAxes)
            ax[k, l].set_xticks([])
            ax[k, l].set_yticks([])
            if ylim:
                ax[k, l].set_ylim(ylim)
            ax[k, l].grid('off')
        figs.append(fig)
        axs.append(ax)


# ##
# ## Plot init in raw time series
# ##

def _plot_sub_in_ts(ts, start, stop, endpoint=False, samples=1000):
    """Plots a time series `ts` and additionally plots the section
    specified with `start` and `stop` with `samples` samples on top of the
    time series.
    This method is intended to plot a section from the init_ts pool in the
    raw_ts time series, to visualize which section was taken.
    """
    if endpoint:
        stop += 1
    section = ts[start:stop]
    torig = np.arange(len(ts))
    tsec = np.linspace(start, stop, samples)
    section = util.resample_ts(section, samples)
    plt.plot(torig, ts, '-', linewidth=3)
    plt.plot(tsec, section, '-')
    plt.legend(['Original', 'PChip'])


def _plot_sub_in_ts_from_string(selection, datafile='data/ees_ts.pkl'):
    """Same as `_plot_sub_in_ts()`, but takes a selection string in the format
    '<ts_name> <start> - <stop> as input.'
    For this, the raw data as well as the selection file must be available."""
    dset, start, _, stop = selection.split(' ')
    start, stop = int(start), int(stop)
    data = pd.read_pickle(datafile)
    _plot_sub_in_ts(data[dset][0], start, stop)


# ##
# ## Methods to determine nd-hist and heterogeneity
# ##

def nd_hist(df_feat, bins=10):
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


def heterogeneity(df_feat, bins=10, as_histarray=False):
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
    >>> het = heterogeneity(df)
    >>> print(f'Heterogeneity of the dataset: {het}')
    """
    if as_histarray:
        histarray = df_feat
    else:
        histarray = nd_hist(df_feat, bins)

    dims, bins = histarray.shape
    max_per_dim = np.sqrt((1 - 1 / bins) ** 2 + 1 / bins ** 2 * (bins - 1))
    average = 1 / bins
    deviation = histarray - average
    # deviation[deviation > 0] = 0   # only penalize bins that are too empty
    return np.sum((np.sum(deviation ** 2, axis=1))**(1/2)) / dims / max_per_dim


def plot_nd_hist(df_feat, ax=None, bins=10, title='', colorbar=False,
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
    >>> fig, ax = plot_nd_hist(df)  # noqa
    >>> plt.show()
    """

    if as_histarray:
        histarray = df_feat
    else:
        histarray = nd_hist(df_feat, bins)

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
         f'{heterogeneity(histarray, as_histarray=True):.4f}')
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
    """Calculates and summarizes the heterogeneity of sets in a collection.

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
            hlist.append(heterogeneity(set_['norm_space'][n]))
            hlist.append(heterogeneity(set_['norm_space'][n][:int(n / 2)]))
            hlist.append(heterogeneity(set_['norm_space'][n][int(n / 2):]))
            col_list += [f'H{n}_both', f'H{n}_neg', f'H{n}_posneg']
        data.append(hlist)

    df = pd.DataFrame(data, columns=col_list)  # noqa
    df.describe()
    return df


# ##
# ## Hierarchical correlation
# ##

def plot_hierarchical_corr_mat(corr_mat, info, selected_feat=None,
                               clust_color='limegreen', ax=None,
                               write_clust_names=True, **kwargs):
    """Visualizes a hierarchical correlation matrix with cluster delineation.

    This function plots a precomputed correlation matrix, highlighting the
    identified clusters and their representatives. It frames each cluster and
    can optionally label the y-axis with cluster names.  The plot includes
    additional information about the inner and outer correlations of the
    clusters and allows customization of the cluster representation color and
    other heatmap properties.

    Use `dimred.hierarchical_corr_mat()` to provide the first two inputs.

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
    """
    # Get feature names of cluster representatives, either by taking the first
    # feature of each cluster or through selected_feat argument
    if selected_feat is None:
        selected_feat = dimred.get_first_name_per_cluster(info['cluster'])
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


def _cluster_info(df_info, cnames=None):
    """Takes `df_info` with columns 'label', 'inner_corr' and 'outer_corr'
    and computes statistics per cluster (encoded in 'label'). Returns as
    dict."""
    if cnames is None:
        cnames = dimred.get_first_name_per_cluster(df_info)

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


def plot_corr_mat_scatter(df_feat, samples=200, bins=20):
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
    >>> df = pd.DataFrame(np.random.rand(1000, 5),
    >>>                   columns=['A', 'B', 'C', 'D', 'E'])
    >>> pg = plot_corr_mat_scatter(df)  # noqa
    >>> plt.show()
    """
    pg = sns.PairGrid(df_feat.sample(samples))
    pg.map_upper(sns.scatterplot, s=5)
    pg.map_lower(sns.scatterplot, s=5)
    pg.map_diag(sns.histplot, bins=bins)
    return pg
