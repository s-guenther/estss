#!/usr/bin/env python3
"""Generate or load the initial data from raw data. Saves and loads
from disk. This data is the starting point for Superposition + Concatenation
and modification in the Expansion Step.
Relevant method are
    get_init_ts()       to load data
    _ees_ts()            to generate from ifes-ees raw data"""

import random
import warnings

import numpy as np
import pandas as pd

from estss import util


def get_init_ts(df_file='data/init_ts.pkl'):
    """Load initial time series data saved as a pickled pandas dataframe.

    The dataframe is an nxm array with m being the number of time series and n
    the number of datapoints of the time series.

    Parameters
    ----------
    df_file : str, default: 'data/init_ts.pkl'
        File path to the dataframe to load

    Returns
    -------
    df_ts : pandas.DataFrame

    """
    return pd.read_pickle(df_file)


# ##
# ## IfES-EES time series related functions
# ##

def _single_raw_to_init(raw_ts, start, stop, endpoint=False, samples=1000):
    """Takes a single raw time series `raw_ts` as numpy array, extracts a
    subsection defined by [`start`, `stop`], resamples to `samples` points,
    normalizes and returns."""
    if endpoint:
        stop += 1

    init = raw_ts[start:stop]
    init = np.nan_to_num(init)
    if len(init) == samples:
        init -= np.mean(init)
        return init/np.max(np.abs(init))

    init = util.resample_ts(init, samples)

    try:
        init = util.norm_meanmaxabs(init)
    except ZeroDivisionError:
        warnings.warn('Zero time series encountered, returning random time '
                      'series instead.')
        return _single_raw_to_init(np.random.rand(100), 0, 99,
                                   endpoint, samples)
    return init


def _ees_ts(datafile='data/ees_ts.pkl', selectionsfile='data/ees_selections',
            *, _raw_to_init_fcn=_single_raw_to_init):
    """Loads the ifes-ees confidential data, extracts sections and saves
    to data/init.pkl.
    Does only work if raw data and selection file is available."""
    with open(selectionsfile, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]

    ts_array = np.zeros([1000, len(lines)], dtype='float')
    for col, selection in enumerate(lines):
        ts = _raw_to_init_from_string(selection, datafile,
                                      _raw_to_init_fcn=_raw_to_init_fcn)
        ts_array[:, col] = ts
    return pd.DataFrame(ts_array)


def _raw_to_init_from_string(selection, datafile='data/ees_ts.pkl', *,
                             _raw_to_init_fcn=_single_raw_to_init):
    """Wrapper around _single_raw_to_init(). Which time series and start and
    stop are defined in the `selection` string in the format
    '<ts_name> <start> - <stop>'.
    Reads adequate time series from `datafile` and passes everything to
    _single_raw_to_init()."""
    dset, start, _, stop = selection.split(' ')
    start, stop = int(start), int(stop)
    data = pd.read_pickle(datafile)
    return _raw_to_init_fcn(data[dset][0], start, stop)


# ## Random selections

def _gen_and_append_rand_selections(
        datafile='data/ees_ts.pkl',
        selectionsfile='data/ees_selections_manual',
        deviation=0.3, n_final=2048, seed=1,
        save_to_disk='data/ees_selections'):
    """Generates `n_final` selection strings in the format
    '<ts_name> <start> - <stop>'
    and saves them to the file `save_to_disk`
    For generation, takes `selectionsfile` as reference to know which
    lengths are to extract, deviates the found lengths by `deviation` Takes
    `datafile` to know how long the time series are."""

    random.seed(seed)
    with open(selectionsfile, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    data = pd.read_pickle(datafile)

    new_n_per_line = n_final//len(lines) - 1
    newlines = []
    for oldline in lines:
        maxind = len(data[oldline.split(' ')[0]][0])
        for _ in range(new_n_per_line):
            scale = random.uniform(1 - deviation, 1 + deviation)
            newlines += _rand_start_stop_from_string(oldline, maxind, scale)

    if save_to_disk is not None:
        with open(save_to_disk, 'w') as outfile:
            outfile.writelines([line + '\n' for line in lines])
            outfile.writelines(newlines)
    return newlines


def _rand_start_stop_from_string(line, maxind=None, scale_segment=1.0):
    """Subfunction of _gen_and_append_rand_selections() that generates a
    new selection line by taking an old selections line `line` as reference."""
    dset, start, _, stop = line.split(' ')
    if maxind is None:
        maxind = stop
    start, stop = int(start), int(stop)
    dlen = int((stop - start)*scale_segment)
    maxstart = maxind - dlen
    if maxstart < 0:
        maxstart = 1
    newstart = random.randint(0, maxstart)
    newstop = newstart + dlen
    if newstop > maxind:
        newstop = maxind
    return f'{dset} {newstart} - {newstop}\n'
