#!/usr/bin/env python3
"""Module that gathers various non-specific methods and utility functions that
are not associated to a specific sub-module or are used by multiple ones"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


# ##
# ## File Input/Output
# ##

def read_df_if_string(df_or_string):
    """Gets a pandas DataFrame as input or an file path to a pandas
    DataFrame as input, return pandas Dataframe"""
    if isinstance(df_or_string, pd.DataFrame):
        return df_or_string
    elif isinstance(df_or_string, str):
        return pd.read_pickle(df_or_string)
    else:
        raise ValueError(f'df_or_string must be a pandas Dataframe object or '
                         f'a string encoding a filepath, got type('
                         f'df_or_string) = {type(df_or_string)}')


# ##
# ## Time Series Manipulation
# ##

def resample_ts(ts, samples=1000):
    """Resamples a timeless time series vector `ts` with an arbitrary
    number of points to `samples` points via pchip Interpolation."""
    origpoints = np.linspace(0, samples-1, len(ts))
    ipoints = np.linspace(0, samples-1, samples)
    interpolator = PchipInterpolator(origpoints, ts)
    return interpolator(ipoints)


def norm_meanmaxabs(ts):
    """Returns a time series `ts` normalized to mean=0 and maxabs = 1. May
    raise ZeroDivisionError"""
    tsmod = ts - np.mean(ts)
    maxval = np.max(np.abs(tsmod))
    return tsmod/maxval


def norm_maxabs(ts):
    """Returns a time series `ts` normalized to maxabs = 1 (but arbitrary
    mean). May raise ZeroDivisionError"""
    return ts/np.max(np.abs(ts))


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
                ax[k, l].text(0, 0, str(df.columns[isig]),
                              ha='left', va='bottom',
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
    section = ts[start:stop]
    torig = np.arange(len(ts))
    tsec = np.linspace(start, stop, samples)
    section = resample_ts(section, samples)
    plt.plot(torig, ts, '-', linewidth=3)
    plt.plot(tsec, section, '-')
    plt.legend(['Original', 'PChip'])


def _plot_sub_in_ts_from_string(selection, datafile='data/ees_ts.pkl'):
    """Same as `_plot_sub_in_ts()`, but takes a selection string in the format
    '<ts_name> <start> - <stop> as input.'"""
    dset, start, _, stop = selection.split(' ')
    start, stop = int(start), int(stop)
    data = pd.read_pickle(datafile)
    _plot_sub_in_ts(data[dset][0], start, stop)
