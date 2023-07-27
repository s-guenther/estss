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

def get_expanded_ts():
    pass


def gen_expanded_ts():
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


def modify(df_ts=None, nout_per_nin=8, kwargs_mod=None):
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
        series is generated by the signal processing chain."""
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
    which choses time series 3357 with a scaling factor of 0.8,
    time series 5987 with a scaling factor of 0.5 and
    time series 1790 with a scaling factor of 0.2. """
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
    cutoff_value = cutoff*(np.max(ts) - np.min(ts)) + np.min(ts)
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
    cutoff_mid = (cutoff_upper + cutoff_lower)/2
    ts_upper = ts * (ts > cutoff_upper)
    ts_upper = (ts_upper - cutoff_upper)*(ts_upper > 0)
    ts_lower = ts * (ts < cutoff_lower)
    ts_lower = (ts_lower - cutoff_lower)*(ts_lower < 0)
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
    randdev = (np.random.rand(nsupport)*2 - 1)*maxdeviation
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

def _sig_proc_chain_strings(nout_per_nin=8, nin=2**15, includeorig=True,
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
    # check if mean is negative, if not, move it to ensure this
    # TODO FIXME put this into fix bounds
    if (ts_mean := np.mean(ts_mod)) >= -0.01:
        ts_mod = _shift(ts, -2 * np.abs(ts_mean) - 0.01)
    return ts_mod


# ##
# ## Fix Boundary Conditions
# ##
# ## ##########################################################################
