#!/usr/bin/env python3
"""Expand or manifold an initial time series set.
This includes:
- recombination:
  (a) concatenation (b) superposition
- signal processing chains:
  chaining the following signal processors: (a) expander (b) compressor
  (c, d, e) curtailment (top, bottom, mid) (f) time distortion
  to modify an input time series
- fix boundary conditions:
  (a) mean (b) sign (c) initial condition

Relevant methods in this module:
    get_expanded_ts()
    gen_expanded_ts()
"""

# The module is structured as follows
# - get_expanded_ts()
# - compute_expanded_ts() mainly invokes:
#   - expand() mainly invokes:
#     - recombine() mainly invokes:
#       - _concat_strings()
#       - _concat()
#       - _superpos_strings()
#       - _superpos()
#     - modify() mainly invokes:
#       - sig_proc_chain_strings()
#       - sig_proc_chain()
#     - fix_constraints() mainly invokes:
#       - _fix_mean()
#       - _fix_sign()
#       - _fix_bounds()
#      - verify_constraints()

from collections import abc
from copy import copy
from functools import reduce
from random import choices, uniform, shuffle

import numpy as np
import pandas as pd
import random
from scipy import interpolate

from estss import util


# ##
# ## Top Level Functions
# ##
# ## ##########################################################################

def get_expanded_ts(df_files=('data/exp_ts_only_neg.pkl',
                              'data/exp_ts_only_posneg.pkl')):
    """Load expanded time series data saved as pickled pandas dataframes.

    Loads 2 dataframes, returns as tuple. The first contains only time
    series that are strictly negative valued, the second contains time
    series that are strictly both positive and negative valued.

    Further constraints that the time series satisfy:
    - mean is negative
    - maxabs is 1
    - init/end boundary condition is fulfilled (energy never exceeds zero)
    (see verfiy_constraints() for more info)

    The dataframe is an nxm array with m being the number of time series and n
    the number of datapoints of the time series.

    The data/ folder contains text files encoding how each time series
    within the dataframes is created

    Parameters
    ----------
    df_files : tuple of str, default: ('data/exp_ts_only_{neg,posneg}.pkl')
        2-tuple with file paths to the dataframes to load

    Returns
    -------
    df_ts_only_neg : pandas.DataFrame
    df_ts_only_posneg : pandas.DataFrame
    """
    return [pd.read_pickle(file) for file in df_files]


def compute_expanded_ts(df_init='data/init_ts.pkl', seed=42):
    """Computes expanded time series dataframes from an initial time
    series dataframe.

    It invokes expand() two times. Once with parameter `kind='only_neg'`,
    once with `kind=only_posnet`. See expand() for more information. The
    generated data is automatically saved to disk by expand().

    Returns a 3-tuple, the first 2 elements are the dataframes, the third is
    a dict with information on how these were created.

    Parameters
    ----------
    df_init : pandas.DataFrame or str, default: 'data/init_ts.pkl'
        mxn Dataframe, where n = number of time series, m = number of points
        in time. If string is passed, a valid path to a dataframe object is
        expected and loaded
    seed : int, default: 42
        The random generator is initialized with this seed, ensures
        reproducability

    Returns
    -------
    df_exp_neg : pandas.DataFrame
        Size of this dataframe depends on the standard parameters of the
        invoked subfunctions and should be nxm with n being the number of
        timesteps (1000) and m being the number of time series (2**18 = 262144)
        This dataframe contains only time series that are strictly negative.
    df_exp_posneg : pandas.DataFrame
        Size of this dataframe depends on the standard parameters of the
        invoked subfunctions and should be nxm with n being the number of
        timesteps (1000) and m being the number of time series (2**18 = 262144)
        This dataframe contains only time series that are strictly both
        positive and negative.
    str_combined : dict of lists of str
        Contains info on how the initial time series dataset was recombined
        with concatenation and superposition and modification.
        Dict keys are
        recombined_neg, recombined_posneg, modified_neg, modified_posneg
    """
    print('\n#\n# Compute Expanded Time Series - only negative\n#')
    df_exp_neg, str_rec_neg, str_mod_neg = \
        expand(df_init, kind='only_neg', seed=seed)

    print('\n#\n# Compute Expanded Time Series - only positive/negative\n#')
    df_exp_posneg, str_rec_posneg, str_mod_posneg = \
        expand(df_init, kind='only_posneg', seed=seed+10)

    str_combined = dict(recombined_neg=str_rec_neg,
                        recombined_posneg=str_mod_posneg,
                        modified_neg=str_mod_neg,
                        modified_posneg=str_mod_posneg)
    return df_exp_neg, df_exp_posneg, str_combined


def expand(df_init='data/init_ts.pkl', kind='only_neg', seed=42,
           save_to_disk='data/exp_ts'):
    """Computes an expanded time series data frame from an initial time
    series dataframe.

    It serially performs the following operations:
    - Recombination (randomly created concatenations and superpositions)
    - Modifications (modified by randomly created signal processing chains)
    - Fix Constraints (ensures mean=0, maxabs=1, integral never exceeds 0,
      ts values only negative/ts values only both pos/neg)

    See recombine(), modify(), fix_constraints() and verify_constraints()
    for more info.

    Parameters
    ----------
    df_init : pandas.DataFrame or str, default: 'data/init_ts.pkl'
        mxn Dataframe, where n = number of time series, m = number of points
        in time. If string is passed, a valid path to a dataframe object is
        expected and loaded
    kind : str, default: 'only_neg'
        must be in ['only_neg', 'only_posneg']
        Depending on the chosen value, it will transform all all time series
        in a way that they are strictly negative valued or in a way that
        they strictly have both negative and positive values
    seed : int, default: 42
        The random generator is initialized with this seed, ensures
        reproducability
    save_to_disk : str, default: 'data/exp_ts'
        saves the dataframe as well as text files that encode how the data
        frame is created to the disk with this leading identifier

    Returns
    -------
    df_exp : pandas.DataFrame
        Size of this dataframe depends on the standard parameters of the
        invoked subfunctions and should be nxm with n being the number of
        timesteps (1000) and m being the number of time series (2**18 = 262144)
    str_rec : list of str
        Contains info on how the initial time series dataset was recombined
        with concatenation and superposition
    str_mod : list of str
        Contains info on how the recombined time series dataset was modified
        by signal processing chains
    """
    print('Recombine ... ')
    df_exp, str_rec = recombine(df_init,
                                kwargs_concat=dict(seed=seed),
                                kwargs_spos=dict(seed=seed+1))

    print(' ... finished\nModify ...')
    df_exp, str_mod = modify(df_exp, kwargs_mod=dict(seed=seed+2))

    print(' ... finished\nFix Constraints ...')
    df_exp = fix_constraints(df_exp, kind=kind)

    print(' ... finished\nVerify Constraints ...')
    df_exp = verify_constraints(df_exp, kind=kind)

    print(' ... finished\nSave to Disk ...')
    if save_to_disk is not None:
        df_exp.to_pickle(f'{save_to_disk}_{kind}.pkl')
        with open(f'{save_to_disk}_recombined_strings_{kind}', 'w') as f:
            f.writelines([f'{line}\n' for line in str_rec])
        with open(f'{save_to_disk}_modified_strings_{kind}', 'w') as f:
            f.writelines([f'{line}\n' for line in str_mod])
    print(' ... finished')

    return df_exp, str_rec, str_mod


def recombine(df_ts='data/init_ts.pkl', nout_concat=2 ** 13, nout_spos=2 ** 15,
              kwargs_concat=None, kwargs_spos=None):
    """Takes a time series dataframe and recombines the timeseries via
    concatenation and superposition.

    Concatenation chooses a random amount of time series, extracts random
    sections with random length, resamples them to random length and puts
    them one after another (and normalizes and resamples again)

    Superposition chooses a random amount of time series, scales them with a
    random factor, sums them up, and normalizes again.

    Returns a new dataframe containing the initial time series and the
    recombined ones.

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
        kwargs dict that is passed to _concat_strings()
    kwargs_spos : dict or None, default: None
        kwargs dict that is passed to _superpos_strings()

    Returns
    -------
    df_merged : pandas.DataFrame
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


def modify(df_ts='data/recombined_ts.pkl', nout_per_nin=8, kwargs_mod=None):
    """Takes a time series dataframe and modifies the time series within by
    randomly generated signal processing chains.

    Available Signal Processors are: Expander, Compressor, Curtail
    Top/Mid/Bottom, Inverter, Shift, Time Distortion. These are chained in
    arbitrary sequence, number and parameterization and applied
    `nout_per_nin` times to each time series in `df_ts`

    Returns a new dataframe containing the modified time series only,
    but with standard parameters (`kwargs_mod` = None) this will include the
    original time series slightly shifted as well.

    Parameters
    ----------
    df_ts : pandas.DataFrame or str or None, default: None
        mxn Dataframe, where n = number of time series, m = number of points
        in time.
        If string is passed, a valid path to a dataframe object is expected
        and loaded.
        If None is passed, recombine() is executed with default parameters
        and the resulting dataframe is used.
    nout_per_nin : int, default: 16
        Number of modifications generated for each input time series in `df_ts`
    kwargs_mod : dict or None, default None
        kwargs dict passed to _sig_proc_chain_strings().

    Returns
    -------
    df_spc : pandas.DataFrame
        mxn dataframe, where m is the number of time steps of the original
        passed dataframe `df_ts`, and n equals `nout_per_nin`*`nin` with
        `nin` beeing the number of time series in `df_ts`.
    spc_strings : list of str
        `n` - element list with n being the number of time series in
        `df_spc`, where each element encodes how the according time
        series is generated by the signal processing chain.
    """
    if df_ts is None:
        print('No df passed, calculating with recombine() first...')
        df_ts = recombine()[0]
        print('...finished')
    if kwargs_mod is None:
        kwargs_mod = dict()
    df_ts = util.read_df_if_string(df_ts)

    n_in = df_ts.columns.size
    spc_strings = _sig_proc_chain_strings(nout_per_nin, n_in, **kwargs_mod)
    df_spc = _sig_proc_chain(spc_strings, df_ts)
    return df_spc, spc_strings


def fix_constraints(df_ts, kind='only_neg'):
    """Takes a time series dataframe `df_ts` and fixes the constraints that
    may be violated.

    I.e., it ensures that:
    - The time series all have a negative mean value
    - The time series are either strictly negative valued or scrictly both
      negative and positive valued
    - The time series shall have its maximum energy at t=0 and minimum at
      t=tend (energy ^= integral of time series)

    Parameters
    ----------
    df_ts : pandas.DataFrame or str
        mxn Dataframe, where n = number of time series, m = number of points
        in time. If string is passed, a valid path to a dataframe object is
        expected and loaded
    kind : str, default: 'only_neg'
        must be in ['only_neg', 'only_posneg']
        Depending on the chosen value, it will transform all all time series
        in a way that they are strictly negative valued or in a way that
        they strictly have both negative and positive values

    Returns
    -------
    df_ts : pandas.DataFrame
        With the same size as the input data frame
    """
    if kind not in (options := ['only_neg', 'only_posneg']):
        raise ValueError(f'kind must be in {options}, found kind = {kind}')

    df_ts = util.read_df_if_string(df_ts)

    df_ts = _fix_mean(df_ts)
    df_ts = _fix_sign(df_ts, kind)
    df_ts = _fix_bounds(df_ts)
    return df_ts


def verify_constraints(df_ts, kind='only_neg'):
    """Takes a time series dataframe `df_ts` and verifies that all
    constraints of all time series are fulfilled. Removes time series that
    do not satisfy the constraints.

    The function checks that all time series satisfy:
    - mean is negative
    - maxabs is 1
    - are strictly negative (`kind='only_neg'`)
      or strictly positive/negative (`kind='only_posneg'`)
    - init/end boundary condition is fulfilled (energy never exceeds zero)

    Although the subroutines of fix_constraints() ensure all this,
    they might damage a constraint by fixing another one (_fix_bounds()
    might damage _fix_sign()).

    Parameters
    ----------
    df_ts : pandas.DataFrame or str, default: 'data/init_ts.pkl'
        mxn Dataframe, where n = number of time series, m = number of points
        in time. If string is passed, a valid path to a dataframe object is
        expected and loaded
    kind : str, default: 'only_neg'
        must be in ['only_neg', 'only_posneg']
        Depending on the chosen value, it will transform all all time series
        in a way that they are strictly negative valued or in a way that
        they strictly have both negative and positive values

    Returns
    -------
    df_ts : pandas.DataFrame
    """
    if kind not in (options := ['only_neg', 'only_posneg']):
        raise ValueError(f'kind must be in {options}, found kind = {kind}')

    df_ts = util.read_df_if_string(df_ts)

    # test all constraints
    df_mean = df_ts.apply(lambda ts: np.mean(ts) < 0)
    df_maxabs = df_ts.apply(lambda ts: np.max(np.abs(ts)) == 1)

    if kind == 'only_neg':
        df_strict = df_ts.apply(lambda ts: np.all(ts <= 0))
    elif kind == 'only_posneg':
        df_strict = df_ts.apply(lambda ts: ~np.all(ts <= 0))
    else:
        raise RuntimeError('If-...-else reached presumably impossible path')

    df_binit = df_ts.apply(lambda ts: np.max(np.cumsum(ts)) <= 0)
    df_bend = df_ts.apply(lambda ts: np.max(np.cumsum(ts[::-1])) <= 0)

    # remove ts in df_ts, where constraints are unfullfilled
    df_all_valid = df_mean & df_maxabs & df_strict & df_binit & df_bend

    return df_ts.loc[:, df_all_valid]


# ##
# ## Recombination - Concatenation and Superposition
# ##
# ## ##########################################################################

# ##
# ## Concatenate time series
# ##

def _concat_strings(
        nout=2 ** 13 - 2 ** 11, nin=2 ** 11, n_ts=(2, 4), ts_len=1000,
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
    return pd.DataFrame(all_concat)


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
        elif i_ts == n_ts - 1:
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
    multstart = np.hstack(
        [np.linspace(1 / (overlap + 1), 1 - 1 / (overlap + 1), overlap),
         np.ones((len_ts - overlap,))])
    multend = np.hstack([np.ones((len_ts - overlap,)),
                         np.linspace(1 - 1 / (overlap + 1), 1 / (overlap + 1),
                                     overlap)])
    multboth = multstart * multend
    if whichend == 'start':
        return ts * multstart
    elif whichend == 'end':
        return ts * multend
    elif whichend == 'both':
        return ts * multboth


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

def _superpos_strings(nout=2 ** 15 - 2 ** 13, nin=2 ** 13, n_ts=(2, 4),
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
    which choses time series 3357 with a scaling factor of 0.8,
    time series 5987 with a scaling factor of 0.5 and
    time series 1790 with a scaling factor of 0.2. """
    random.seed(seed)
    add_strings = []
    for _ in range(nout):
        nts = random.randint(*n_ts)
        ts_ids = [random.randrange(nin) for _ in range(nts)]
        scales = [int(random.uniform(*scalerange) * 10) / 10 for _ in
                  range(nts)]
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

# First, signal processing functions will be defined
# followed by a definition how to apply and chain them
# followed by functions to do so

# ##
# ## Signal Processors
# ##

def _curtail_up(ts, cutoff=0.2):
    """Curtails the top of a time series `ts`; the cutoff value describes how
    much of the top will be cut off as a percentage of the difference between
    maximum and minimum. Will normalise the data afterwards.

    Paramters:
    ----------
    ts : time series array with the relevant data
    cutoff : float, a value between 0 and 1
        0: nothing changes, 1:everything would be cutoff"""
    cutoff_value = ((1 - cutoff) * (np.max(ts) - np.min(ts)) +
                    np.min(ts))
    ts = copy(ts)
    ts[ts > cutoff_value] = cutoff_value
    return util.norm_maxabs(ts)


def _curtail_down(ts, cutoff=0.2):
    """curtails the bottom of a time series `ts`; the cutoff value describes
    how much of the bottom will be cut off as a percentage of the difference
    between maximum and minimum. The data will be normalised afterwards.

    Paramters:
    ----------
    ts : time series array with the relevant data
    cutoff : float, a value between 0 and 1
        0: nothing changes, 1:everything would be cutoff"""
    cutoff_value = cutoff * (np.max(ts) - np.min(ts)) + np.min(ts)
    ts = copy(ts)
    ts[ts < cutoff_value] = cutoff_value
    return util.norm_maxabs(ts)


def _curtail_within(ts, cutoff=0.2):
    """curtails the middle of a time series `ts`; the cutoff value describes
    how much of the middle will be cut off as a percentage of the difference
    between maximum and minimum. The data will be normalised afterwards.

    Paramters:
    ----------
    ts : time series array with the relevant data
    cutoff : float, a value between 0 and 1
        0: nothing changes, 1:everything would be cutoff"""
    cutoff_lower = (0.5 * (1 - cutoff) * (np.max(ts) - np.min(ts)) +
                    np.min(ts))
    cutoff_upper = (0.5 * (1 + cutoff) * (np.max(ts) - np.min(ts)) +
                    np.min(ts))
    cutoff_mid = (cutoff_upper + cutoff_lower) / 2
    ts_upper = ts * (ts > cutoff_upper)
    ts_upper = (ts_upper - cutoff_upper) * (ts_upper > 0)
    ts_lower = ts * (ts < cutoff_lower)
    ts_lower = (ts_lower - cutoff_lower) * (ts_lower < 0)
    ts_merge = ts_upper + ts_lower + cutoff_mid
    return util.norm_maxabs(ts_merge)


def _invert(ts, sign=-1):
    """Inverts time series `ts`, i.e. changes the sign of every time step"""
    if not (sign == 1 or sign == 0 or 1.0):
        raise ValueError(f'Parameter `sign` must be -1 or 1, found {sign}.')
    if sign == 1 or sign == 1.0:
        return ts
    else:
        return -ts


def _gen_random_dist_supports(seed):
    """Generates random support points used for time distortion in
    _distort_time(). The support points define where the time series is
    stretched and where it is compressed."""
    np.random.seed(seed)
    nsupport = np.random.randint(4, 8)
    maxdeviation = 0.5
    randdev = (np.random.rand(nsupport) * 2 - 1) * maxdeviation
    supports = np.cumsum(randdev + 1)
    supports -= supports[0]
    supports /= np.max(supports)
    return supports


def _distort_time(ts, distsupports=None):
    """Distorts a time series `ts` in time domain, i.e. some time regions
    will be stretched, some will be compressed. How this is done is defined
    by `distsupports`. If no argument is passed, it will be randomly
    generated. If a scalar is passed, its also randomly generated with and
    the scalar is used as the seed for the random generator. If a vector is
    passed, it is used as the support points that deviate from a linearily
    spaced sequence between 0 and 1."""
    if distsupports is None:
        distsupports = _gen_random_dist_supports(random.randrange(10))
    elif isinstance(distsupports, abc.Iterable):
        pass
    elif np.isscalar(distsupports):
        distsupports = _gen_random_dist_supports(int(distsupports))
    else:
        msg = f'Unexpected input argument. Input for `distsupport` was:' \
              f' {distsupports}\nShould be int, float or vector.'
        raise ValueError(msg)
    origsupports = np.linspace(0, 1, len(distsupports))

    fun_distort = interpolate.PchipInterpolator(origsupports, distsupports)

    orig_time = np.linspace(0, 1, len(ts))
    dist_time = fun_distort(orig_time)

    fun_dist_ts = interpolate.PchipInterpolator(dist_time, ts)
    return fun_dist_ts(orig_time)


def _expander(ts, exp=2):
    """The expander function will expand (if exp > 0) or compress (if exp < 0)
    a time series `ts`. Expanding will increase the prominence of datapoints
    with a high absolute value; compressing will decrease the prominence of
    single datapoints.

    ts : time series array with the relevant data
    exp : the exponent of the expander function
        exp > 0: expand the time series, exp < 0: compress it"""
    ts = np.abs(ts) ** exp * np.sign(ts)
    return util.norm_maxabs(ts)


def _shift(ts, y=1):
    """Will shift the time series `ts` by a factor `y` of the maximum absolute
    value up along the y-axis. Will be normalised afterwards to max(abs()) = 1
    (but not to mean = 0)

    ts : time series array with the relevant, normalised data
    y : float, amount by which the time series will be shifted upwards.
        as a multiple of the maximum absolute value"""
    ts = ts + y
    return util.norm_maxabs(ts)


# ##
# ## Signal Processing Chain Definition
# ##

# The dict encodes the name of the signal processor as a string (dict key)
# and information about it in the dict val
# The dict val contains a tuple with
# (reference to function, parameter value range, activation probability)
_MODKEYDEF = dict()
_MODKEYDEF['shift'] = (_shift, (-1.5, 1.5), 0.7)
_MODKEYDEF['exp'] = (_expander, (1.3, 2.0), 0.35)
_MODKEYDEF['comp'] = (_expander, (0.4, 0.8), 0.35)
_MODKEYDEF['curt_top'] = (_curtail_up, (0.03, 0.20), 0.12)
_MODKEYDEF['curt_bot'] = (_curtail_down, (0.03, 0.20), 0.12)
_MODKEYDEF['curt_mid'] = (_curtail_within, (0.03, 0.20), 0.12)
_MODKEYDEF['dist'] = (_distort_time, (1, 100), 0.3)
_MODKEYDEF['inv'] = (_invert, (-1, -1), 0.5)


# ##
# ## Signal Processing Chain generation and execution
# ##

def _sig_proc_chain_strings(nout_per_nin=8, nin=2 ** 15, includeorig=True,
                            modkeydef=None, seed=10):
    """Creates a `nout_per_nin`*`n_in`-element list of randomly generated
    strings that define how a `nin`-element set of input time series shall be
    processed by randomly generated signal processing chains. `modkeydef` is a
    dict that defines the signal processors, the parameter range and the
    probability. `includeorig` is bool and defines if one chain is created
    per input time series that returns the original one but slightly shifted.
    `seed` initializes the random generator.

    The format of the signal processing chain strings is:
        <ts_id> - <proc1:val1> | <proc2:val2> | ...
    e.g.:
        1 |> curt_mid:0.09 | shift:0.23 | comp:0.62 | exp:1.58
    which takes the time series with id 1 and applies curt_mid with
    parameter 0.09 followed by shift with parameter 0.23 followed by
    compressor with parameter 0.62 followed by expander with parameter 1.58."""
    random.seed(seed)
    if modkeydef is None:
        modkeydef = copy(_MODKEYDEF)
    chain_strings = []
    for i_in in range(nin):
        spc_string_chunk = [_single_spc_string(i_in, modkeydef)
                            for _ in range(nout_per_nin)]
        if includeorig:
            sval = uniform(0.03, 0.17)
            spc_string_chunk[0] = f'{i_in} |> shift:-{sval:.2f}'
        chain_strings += spc_string_chunk
    return chain_strings


def _single_spc_string(ts_id, modkeydef):
    """Generates a signal processing string specified by the probabilities
    and parameters and signal processors in `modkeydef` for a time series
    with the id `ts_id`. See _sig_proc_chain_strings() for the definition of
    the string."""
    tags = []
    valranges = []
    props = []
    for key, val in modkeydef.items():
        tags += [key]
        valranges += [val[1]]
        props += [val[2]]
    # go through all vars twice and add tag-value pairs to list
    mod_key_list = []
    for _ in range(2):
        for tag, vrange, prop in zip(tags, valranges, props):
            tag = choices([tag, None], weights=[prop, 1 - prop])[0]
            val = uniform(*vrange)
            if tag is not None:
                mod_key_list += [f'{tag}:{val:.2f}']
    # scramble list
    shuffle(mod_key_list)
    # join to one string
    return str(ts_id) + ' |> ' + ' | '.join(mod_key_list)


def _sig_proc_chain(spc_strings, df_ts):
    """Takes a list of strings `spc_strings` from the function
    _sig_proc_chain_strings() and a time series dataframe `df_ts` and
    generates time series modified by the specified signal processing chains
    by passing each string to _single_sig_proc_chain_from_string() and
    merging the results to a single data frame."""
    all_spc = [_single_sig_proc_chain_from_string(spcs, df_ts)
               for spcs in spc_strings]
    all_spc = np.stack(all_spc, axis=1)
    return pd.DataFrame(all_spc)


def _single_sig_proc_chain_from_string(spc_string, df_ts, modkeydef=None):
    """Parses a single signal processing chain string `spc_string`, extracts
    the relavant information: which time series id, which signal processors
    with which parameters with the help of `modkeydef`, gets the time series
    from `df_ts` and passes the results to _single_sig_proc_chain()"""
    if modkeydef is None:
        modkeydef = copy(_MODKEYDEF)
    ts_id, spc_string = spc_string.split(' |> ')
    ts = df_ts[int(ts_id)]
    # split spc_string (remainder) into separate parts
    fcns, paras = [], []
    for part in spc_string.split(' | '):
        if not part:
            break
        tag, valstr = part.split(':')
        fcns.append(modkeydef[tag][0])
        paras.append(float(valstr))
    return _single_sig_proc_chain(ts, fcns, paras)


def _single_sig_proc_chain(ts, fcns, paras):
    """Applies a set of signal processors defined in the list `fcns` with
    the parameters in the list `paras` to a time series `ts`.
    At the end, tests if the mean is negative, if so: shift it so it
    complies this boundary"""
    ts_mod = ts
    for fcn, para in zip(fcns, paras):
        ts_mod = fcn(ts_mod, para)
    return ts_mod


# ##
# ## Fix Constraints
# ##
# ## ##########################################################################


# ##
# ## Fix Mean
# ##

def _fix_mean(df_ts):
    """Applies _single_fix_mean() for every time series in the time series
    dataframe `df_ts`"""
    return df_ts.apply(_single_fix_mean)


def _single_fix_mean(ts, tol=0.01, rand_move_range=0.1):
    """Fixes the mean value of a time series to a negative mean value,
    i.e. if it had a positive mean value of x, it will be inverted to have a
    negative mean value of -x afterwards.
    The logic is a little bit more complex for near-zero-mean-values defined
    by `tol`: If the mean is between -`tol` and +`tol`, the time series will be
    shifted (and then normalized) to have a mean value between -`tol` and
    `rand_move_range`"""
    if (mean_ts := np.mean(ts)) <= -tol:
        return ts
    elif mean_ts >= tol:
        return -ts
    elif -tol < mean_ts < tol:
        ts = util.norm_maxabs(ts - 2 * tol - uniform(tol, rand_move_range))
        if -tol < np.mean(ts) < tol:
            ts = _single_fix_mean(ts)
        return ts
    else:
        raise RuntimeError('If-...-else reached presumably impossible path')


# ##
# ## Fix Sign
# ##

def _fix_sign(df_ts, kind='only_neg'):
    """Applies _single_fix_sign_to_only_neg() if `kind` = 'only_neg'
    and _single_fix_sign_to_only_posneg() if `kind` = 'only_posneg' to
    the whole time series dataframe `df_ts`"""
    if kind not in (options := ['only_neg', 'only_posneg']):
        raise ValueError(f'kind must be in {options}, found kind = {kind}')

    if kind == 'only_neg':
        return df_ts.apply(_single_fix_sign_to_only_neg)
    elif kind == 'only_posneg':
        return df_ts.apply(_single_fix_sign_to_only_posneg)
    else:
        raise RuntimeError('If-...-else reached presumably impossible path')


def _single_fix_sign_to_only_neg(ts):
    """If the time series `ts` does have positive parts, it is shifted and
    normalized afterwards so that the maximum positive value becomes zero"""
    if (max_ts := np.max(ts)) <= 0:
        return ts
    else:
        return util.norm_maxabs(ts - max_ts)


def _single_fix_sign_to_only_posneg(ts):
    """If the time series `ts` does only have negative parts, it is shifted
    in a way that it has both positive and negative parts. The shift
    performed is randomly selected between max(ts) and mean(ts). The result is
    normalized again afterwards."""
    if (max_ts := np.max(ts)) >= 0:
        return ts
    else:
        rand_shift = uniform(max_ts, float(np.mean(ts)))
        return util.norm_maxabs(ts - rand_shift)


# ##
# ## Fix Boundary Conditions
# ##

def _fix_bounds(df_ts):
    """Applies _single_fix_bounds() to every time series in the time series
    data frame 'df_ts'"""
    return df_ts.apply(_single_fix_bounds)


def _single_fix_bounds(ts):
    """ Fixes the initial and end boundary condition of a time series `ts`. The
    boundary condition is that the time series shall have its maximum energy
    (integral) at t=0 and its minimum energy at t=tend. If this condition is
    not fullfilled, the function will overlay a quarter sinus wave at the
    beginning or end with the neccessary energy content to shift the time
    series smoothly down in the neccessary regions.

    The overlay may be applied multiple times from the first peak to the
    last peak, until all are removed.

    Several options were tried to achieve this and were compared empirically.
    Empirically the best results were delivered by an overlay with a
    quarter sine function, other options tried where constant, exponential,
    linear, flipping, loop_checking."""
    ts = np.array(ts)
    while not _bounds_valid(ts):
        ts = _add_quarter_sine(ts)
        flipped = _add_quarter_sine(ts[::-1])
        ts = flipped[::-1]
    return util.norm_maxabs(ts)


def _bounds_valid(ts):
    """Checks that the initial and end boundary condition of a time series
    `ts` are valid, i.e. energy of `ts` never exceeds 0.
    Returns True or False"""
    if np.all(ts <= 0):
        return True
    return _forward_valid(ts) and _backward_valid(ts)


def _forward_valid(ts):
    """Checks that the initial boundary condition of a time series
    `ts` is valid, i.e. energy of `ts` never exceeds 0.
    Returns True or False"""
    return True if np.all(np.cumsum(ts) <= 0) else False


def _backward_valid(ts):
    """Checks that the end boundary condition of a time series
    `ts` is valid, i.e. energy of flipped `ts` never exceeds 0.
    Returns True or False"""
    return _forward_valid(ts[::-1])


def _add_quarter_sine(ts, safety=1.001):
    """Adds a quarter cosine wave at the beginning of a time series `ts`.
    Looks for the maximum energy and position within `ts` constructs a
    quarter cosine wave with this energy amount from the beginning to this
    position and superposes both. The quarter cosine wave is multiplied with
    `safety` to circumvent numerical rounding errors. The resuling time
    series will have an integral that never exceeds zero."""

    # check if time series needs to be fixed, else return prematurely
    if _forward_valid(ts):
        return ts

    # find maximum peak
    energy = np.cumsum(ts)
    ind_max = np.argmax(energy)
    e_max = energy[ind_max]

    # construct quarter sine
    qsin_ind = np.linspace(0, np.pi / 2, ind_max + 1)
    qsin = np.cos(qsin_ind)
    qsin_e = np.sum(qsin)
    qsin *= e_max / qsin_e

    # superpose
    qsin_tail = np.zeros(len(ts) - ind_max - 1)
    qsin = np.concatenate([qsin, qsin_tail])
    return ts - safety * qsin
