#!/usr/bin/env python3
"""Module that gathers various non-specific methods and utility functions that
are not associated to a specific sub-module or are used by multiple ones"""

import copy

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import estss.init as init

# ##
# ## Plot a dataframe of time series
# ##


def plot_df_ts(df_ts, n=128, which='head', grid=(8, 4)):
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
                ax[k, l].plot(df.iloc[:, isig])
                ax[k, l].text(0, 0, str(isig), ha='left', va='bottom',
                              transform=ax[k, l].transAxes)
            ax[k, l].set_xticks([])
            ax[k, l].set_yticks([])
            ax[k, l].set_ylim([-1, 1])
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
    raw_ts time series, to visualize which section was taken."""
    if endpoint:
        stop += 1
    orig_section = copy.copy(ts[start:stop])
    mean_ts = np.mean(orig_section)
    max_ts = np.max(np.abs(orig_section - mean_ts))
    torig = np.arange(len(ts))
    tres = np.linspace(start, stop, samples)
    initpchip = init._single_raw_to_init(  # noqa
        ts, start, stop, endpoint, samples)
    plt.plot(torig, ts, '-', linewidth=3)
    plt.plot(tres, initpchip*max_ts + mean_ts, '-')
    plt.legend(['Original', 'PChip'])


def _plot_sub_in_ts_from_string(selection, datafile='data/ees_ts.pkl'):
    """Same as `_plot_sub_in_ts()`, but takes a selection string in the format
    '<ts_name> <start> - <stop> as input.'"""
    dset, start, _, stop = selection.split(' ')
    start, stop = int(start), int(stop)
    data = pd.read_pickle(datafile)
    _plot_sub_in_ts(data[dset][0], start, stop)
