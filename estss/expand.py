#!/usr/bin/env python3
"""Gathers all functionality to expand or manifold an initial time series set.
This includes:
- recombination:
  (a) concatenation (b) superposition
- signal processing chains:
  chaining the following signal processors: (a) expander (b) compressor
  (c, d, e) curtailment (top, bottom, mid) (f) time distortion
- fix boundary conditions:
  (a) sign (b) initial condition

Relevant methods in this module:
    get_expanded_ts()
    gen_expanded_ts()
    expand()
    recombine()
    modify()
    fix_bounds()"""

# The module is structured as follows
# - get_expanded_ts()
# - gen_expanded_ts() mainly invokes:
#   - expand() mainly invokes:
#     - recombine() mainly invokes:
#       - _concat_strings()
#       - _concat()
#       - _superpos_strings()
#       - _superpos()
#     - modify() mainly invokes:
#       - sig_proc_chain_strings()
#       - sig_proc_chain()
#     - fix_bounds() mainly invokes:
#       - _fix_sign()
#       - _fix_init()

from functools import reduce

import numpy as np
import pandas as pd
import random

from estss import util


# ##
# ## Top Level Functions
# ##
# ## ##########################################################################

def get_expanded_ts():
    pass


def gen_expanded_ts():
    pass


def expand():
    pass


def recombine(df_ts='data/init_ts.pkl', nout_concat=2**13, nout_spos=2**15,
              kwargs_concat=None, kwargs_spos=None):
    """Takes a time series dataframe and recombines the timeseries via
    concatenation and superposition.

    Concatenation chooses a random amount of time series, extracts random
    sections with random length, resamples them to random length and puts
    them one after another (and normalizes and resamples again)

    Superposition chooses a random amount of time series, scales them with a
    random factor, sums them up, and normalizes again.

    Returns a new dataframe containing the initial time series and the
    recombined ones

    Parameters
    ----------
    df_ts : pandas.DataFrame or str, default: 'data/init_ts.pkl'
        mxn Dataframe, where n = number of time series, m = number of points
        in time. If string is passed, a valid path to a dataframe object is
        expected and loaded
    nout_concat : int, default: 2**13
        number of time series after concatenation
    nout_spos : int, default: 2**15
        number of time series after superposition
    kwargs_concat : dict or None, default: None
        kwargs dict that is passed to _concat()
    kwargs_spos : dict or None, default: None
        kwargs dict that is passed to _superpos()

    Returns
    -------
    df_merged : pandas.Dataframe
        mxn dataframe, where m is the number of time steps of the original
        passed dataframe `df_ts`, and n equals `nout_spos`
    merge_strings : list of str
        n - element list, where each element encodes how the according time
        series is created
    """
    if kwargs_concat is None:
        kwargs_concat = dict()
    if kwargs_spos is None:
        kwargs_spos = dict()

    df_init = util.read_df_if_string(df_ts)

    # concat
    n_init = df_init.columns.size
    c_strings = _concat_strings(nout=nout_concat - n_init, nin=n_init,
                                **kwargs_concat)
    df_concat = _concat(c_strings, df_init)
    # merge df
    df_merged = pd.concat([df_init, df_concat], axis=1, ignore_index=True)

    # superpos
    n_merged = df_merged.columns.size
    s_strings = _superpos_strings(nout=nout_spos - n_merged,
                                  nin=n_merged, **kwargs_spos)
    df_spos = _superpos(s_strings, df_merged)
    # merge df again
    df_merged = pd.concat([df_merged, df_spos], axis=1, ignore_index=True)

    # generate and merge stringlist
    i_strings = [str(ii) for ii in range(n_init)]
    strings = \
        i_strings + \
        ['concat ' + c for c in c_strings] + \
        ['superpos ' + s for s in s_strings]
    return df_merged, strings


def modify():
    pass


def fix_bounds():
    pass


# ##
# ## Recombination - Concatenation and Superposition
# ##
# ## ##########################################################################

# ##
# ## Concatenate time series
# ##

def _concat_strings(
        nout=2**13-2**11, nin=2**11, n_ts=(2, 4), ts_len=1000,
        inlenrange=(0.2, 0.5), outlenrange=(0.2, 0.5), seed=2):
    """Creates a `nout`-element list of randomly generated strings that
    define how a `nin`-element set of input time series shall be
    concatenated (number of ts to concat in the range of `n_ts`, section
    length fractions in the range of `inlenrange`, which resampled section
    lenghts in the range of `outlenrange`, final time series length `ts_len`).
    `seed` initializes the random generator.

    The format of the concat strings is:
        <id1>, <id2>, ... | <start1>, <start2>, ... | <end1>, <end2>,
        ... | <newlen1>, <newlen2>, ...
    e.g.:
        957, 61, 742 | 242, 17, 31 | 715, 498, 405 | 401, 225, 429
    which takes the time series 957, 61, 742, extracts the sections (242,
    715), (17, 498), (31, 405) and resamples these sections to 401, 225,
    429 points."""
    random.seed(seed)
    concat_strings = []
    for _ in range(nout):
        nts = random.randint(*n_ts)
        ts_ids = [random.randrange(nin) for _ in range(nts)]
        inlens = [int(random.uniform(*inlenrange) * ts_len)
                  for _ in range(nts)]
        outlens = [int(random.uniform(*outlenrange) * ts_len)
                   for _ in range(nts)]
        starts = [random.randrange(ts_len - inlen) for inlen in inlens]
        stops = [start + inlen for start, inlen in zip(starts, inlens)]
        cstr = ', '.join(map(str, ts_ids)) + ' | ' + \
               ', '.join(map(str, starts)) + ' | ' + \
               ', '.join(map(str, stops)) + ' | ' + \
               ', '.join(map(str, outlens))
        concat_strings.append(cstr)
    return concat_strings


def _concat(concat_strings, df_ts):
    """Takes a list of strings `concat_strings` from the function
    _concat_strings() and a time series dataframe `df_ts` and generates the
    concatenated time series variants specified by the concat strings by
    passing each string to _single_concat_from_string() and merging the
    results to a single data frame"""
    all_concat = [_single_concat_from_string(sel, df_ts)
                  for sel in concat_strings]
    all_concat = np.stack(all_concat, axis=1)
    df_ts_concat = pd.DataFrame(all_concat)
    return df_ts_concat


def _single_concat_from_string(concat_str, df_ts):
    """Parses a single concat_string `concat_str`, extracts the relevant
    time series from `df_ts` and passes them along with the concat
    information to _single_concat().
    Parsed format of the string is defined in _concat_strings()"""
    ts_ids, starts, stops, npoints = concat_str.split(' | ')
    ts_ids = [int(ts_id) for ts_id in ts_ids.split(', ')]

    ts_list = [df_ts[ts_id] for ts_id in ts_ids]
    starts = [int(start) for start in starts.split(', ')]
    stops = [int(stop) for stop in stops.split(', ')]
    npoints = [int(npoint) for npoint in npoints.split(', ')]

    return _single_concat(ts_list, starts, stops, npoints)


def _single_concat(ts_list, starts, stops, npoints):
    """Concatenates a list of time series `ts_list`. Extracts subsections
    defined by `starts` and `stops` (lists of same length as `ts_list`).
    Resamples each section to `npoints` (also a list), concatenates them
    and resamples again."""
    part_ts_list = []
    for ts, start, stop, npoint in zip(ts_list, starts, stops, npoints):
        mod_ts = util.resample_ts(ts[start:stop], samples=npoint)
        part_ts_list.append(mod_ts)
    return _append_ts_with_overlap(part_ts_list)


def _append_ts_with_overlap(ts_list, overlap=9, samples=1000):
    """Takes a list of time series `ts_list` and concatenates them. The
    concatenation does not simply put the time series one after another but
    smoothly transitions linearily from one to the other by `overlap`
    points. Resamples the final concatenated time series to `samples`
    points."""
    n_ts = len(ts_list)
    new_ts_list = []
    for i_ts, ts in enumerate(ts_list):
        if i_ts == 0:
            ts = _smooth_ends(ts, overlap, 'end')
        elif i_ts == n_ts-1:
            ts = _smooth_ends(ts, overlap, 'start')
        else:
            ts = _smooth_ends(ts, overlap, 'both')
        new_ts_list.append(ts)
    merged_ts = \
        reduce(lambda s1, s2: _append_two_ts_with_overlap(s1, s2, overlap),
               new_ts_list)
    return util.norm_meanmaxabs(util.resample_ts(merged_ts, samples))


def _smooth_ends(ts, overlap, whichend='both'):
    """Implements the overlap logic of _append_ts_with_overlap()."""
    len_ts = len(ts)
    multstart = np.hstack([np.linspace(1/(overlap+1), 1-1/(overlap+1),
                                       overlap),
                           np.ones((len_ts - overlap,))])
    multend = np.hstack([np.ones((len_ts - overlap,)),
                         np.linspace(1-1/(overlap+1), 1/(overlap+1), overlap)])
    multboth = multstart*multend
    if whichend == 'start':
        return ts*multstart
    elif whichend == 'end':
        return ts*multend
    elif whichend == 'both':
        return ts*multboth


def _append_two_ts_with_overlap(ts1, ts2, overlap):
    """Concatenates two time series with overlap. Used by
    _append_ts_with_overlap() that concatenates an arbitrary amount of time
    series."""
    ts = np.hstack([ts1[:-overlap],
                     ts1[-overlap:] + ts2[:overlap],
                     ts2[overlap:]])
    return ts


# ##
# ## Superposition
# ##

def _superpos_strings(nout=2**15-2**13, nin=2**13, n_ts=(2, 4),
                      scalerange=(0.2, 1.0), seed=3):
    """Creates a `nout`-element list of randomly generated strings that
    define how a `nin`-element set of input time series shall be
    superpositioned. For each chosen timeseries, a randomly chosen scaling
    factor in `scalerange` is assigned before superpostion. `seed`
    initializes the random generator. `n_ts` defines the range of number of
    timeseries that shall be superpositioned.

    The format of the superposition strings is:
        <id1>, <id2>, ... | <scale1>, <scale2>, ...
    e.g.:
        3357, 5978, 1790 | 0.8, 0.5, 0.2
    which choses signal 3357 with a scaling factor of 0.8, signal 5987 with
    a scaling factor of 0.5 and 1790 with a scaling factor of 0.2. """
    random.seed(seed)
    add_strings = []
    for _ in range(nout):
        nts = random.randint(*n_ts)
        ts_ids = [random.randrange(nin) for _ in range(nts)]
        scales = [int(random.uniform(*scalerange)*10)/10 for _ in range(nts)]
        astr = ', '.join(map(str, ts_ids)) + ' | ' + \
               ', '.join(map(str, scales))
        add_strings.append(astr)
    return add_strings


def _superpos(spos_strings, df_ts):
    """Takes a list of superposition strings `spos_strings` and passes each
    one along with the corresponding time series dataframe `df_ts` to
    _single_superpos_from_string() to generate the individual
    superpositions. Merges the individual results to a single data frame."""
    all_spos = [_single_superpos_from_string(sel, df_ts)
                for sel in spos_strings]
    all_spos = np.stack(all_spos, axis=1)
    df_all_spos = pd.DataFrame(all_spos)
    return df_all_spos


def _single_superpos_from_string(spos_string, df_ts):
    """Parses the superpostion string `spos_string` (defined in
    _superpos_strings()) and passes the information along with the time
    series dataframe `df_ts` to _single_superpos()."""
    ts_ids, scales = spos_string.split(' | ')
    ts_ids = [int(ts_id) for ts_id in ts_ids.split(', ')]

    ts_list = [df_ts[ts_id] for ts_id in ts_ids]
    scales = [float(scale) for scale in scales.split(', ')]

    return _single_superpos(ts_list, scales)


def _single_superpos(ts_list, scales):
    """Takes a list of time series `ts_list` and an equally-sized list of
    `scales`, scales each time series accourdingly, superposes the scaled
    time series and returns the normalized result."""
    scaled_ts_list = [ts * scale for ts, scale in zip(ts_list, scales)]
    spos_ts = reduce(np.add, scaled_ts_list)
    return util.norm_meanmaxabs(spos_ts)


# ##
# ## Modification - Signal Processing Chains
# ##
# ## ##########################################################################


# ##
# ## Fix Boundary Conditions
# ##
# ## ##########################################################################
