#!/usr/bin/env python3
from copy import copy
import os
import random
from functools import reduce

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

import estss.util


# ##
# ## Top Level Functions
# ##
# ## ##########################################################################

def get_manifold_ts():
    pass


def compute_manifold_ts():
    pass


# ##
# ## High Level Functions
# ##
# ## ##########################################################################

def manifold(df):
    df_merged = copy(df)

    print('Concatenate 3000 --> 16000...')
    df_out = concatenate(df_merged, nout=13046, seed=1)
    # df_out = concatenate(df_merged, nout=13, seed=1)
    df_merged = pd.concat([df_merged, df_out], axis=1, ignore_index=True)

    print('    ... finished\nSuperpose 16000 --> 32000...')
    df_out = superpose(df_merged, nout=16000, seed=2)
    df_merged = pd.concat([df_merged, df_out], axis=1, ignore_index=True)
    del df_out

    print('    ... finished\nConcatenate 32000 --> 128000...')
    base_name = 'examples/redispatch/concat'
    for ii in range(6):
        print(f'    Part {ii+1}/6')
        df_out_concat = concatenate(df_merged, nout=16000, seed=42+ii)
        df_out_concat.to_pickle(f'{base_name}{ii}.pkl')
    print(f'    Merging')
    df_outs = [pd.read_pickle(f'{base_name}{ii}.pkl') for ii in range(6)]
    df_merged = pd.concat([df_merged, *df_outs], axis=1, ignore_index=True)
    del df_outs
    print(f'    Deleting temporary files')
    for ii in range(6):
        # os.remove(f'{base_name}{ii}.pkl')
        pass

    print('    ... finished\nModify 128000 --> 512000...')
    # batch the modify operation to not exceed the ram limits
    # first, store the superpose/concat ts in two separate files
    base_name = 'examples/redispatch/ts_manifold'
    base_save_paths = [f'{base_name}{i}.pkl' for i in (1, 2)]
    ncols = df_merged.columns.size
    ncolshalf = int(ncols/2)
    print('    Processing File 1')
    pt1 = df_merged[range(ncolshalf)]
    pt1.columns = range(ncolshalf)
    pt1.to_pickle(base_save_paths[0])
    del pt1
    print('    Processing File 2')
    pt2 = df_merged[range(ncolshalf, ncols)]
    pt2.columns = range(ncolshalf, ncols)
    pt2.to_pickle(base_save_paths[1])
    del df_merged
    del pt2
    filecounter = 3
    # TODO FIXME in this loop, the calculated df `df_out` does not have the
    #  correct ts ids encoded in the columns. This is fixed manually afterwards
    for _ in range(3):
        for file in base_save_paths:
            print(f'    Processing File {filecounter}')
            df_in = pd.read_pickle(file)
            df_out, _ = modify(df_in, nout_per_nin=1, seed=filecounter + 1337)
            df_out.to_pickle(f'{base_name}{filecounter}.pkl')
            filecounter += 1
    print('    ... finished')
    return None


def concatenate(df_ts, nout=16_000-2988, n_days=(1, 7), seed=4):
    random.seed(seed)
    np.random.seed(seed)

    nts = df_ts.columns.size
    concat_defs = [_make_concat_def(nts, n_days) for _ in range(nout)]
    new_ts_list = [_concat_single(ds, dl, df_ts[ids])
                   for ds, dl, ids in concat_defs]
    ts_array = np.stack(new_ts_list, axis=1)
    return pd.DataFrame(ts_array)


def superpose(df_ts, nout=48_000, n_ts=(2, 2), scalerange=(0.2, 1), seed=3):
    # make id list and scale list
    random.seed(seed)
    nin = df_ts.columns.size
    idlist = []
    scalelist = []
    for _ in range(nout):
        nts = random.randint(*n_ts)
        # ts_ids = [random.randrange(nin) for _ in range(nts)]
        ts_ids = random.sample(list(range(nin)), nts)
        scales = [int(random.uniform(*scalerange) * 10) / 10 for _ in
                  range(nts)]
        idlist.append(ts_ids)
        scalelist.append(scales)

    new_ts_list = [_single_superpos(_df_to_list(df_ts[ids]), scales)
                   for ids, scales in zip(idlist, scalelist)]

    ts_array = np.stack(new_ts_list, axis=1)
    return pd.DataFrame(ts_array)


def modify(df_ts, nout_per_nin=8, seed=5):
    kwargs_mod = dict(
        seed=seed,
        modkeydef=_MODKEYDEF,
        includeorig=False
    )
    return estss.manifold.modify(df_ts, nout_per_nin, kwargs_mod)


# ##
# ## Low Level Functions
# ##
# ## ##########################################################################

# ## Concatenation

def _make_concat_def(nts, n_days=(1, 7)):
    # create a list of how many days per ts. randomly generate 365 numbers
    # between [1, 7], cumsum them and find the index where it exceeds 365
    # correct this index, so the cumsum is exactly 365 and cut the vector
    # daylens at this index
    daylens = np.random.random_integers(*n_days, size=365)
    cum_daylens = np.cumsum(daylens)
    ind = np.searchsorted(cum_daylens, 365, side='right')
    if (vallast := cum_daylens[ind]) > 365:
        daylens[ind] -= (vallast - 365)
    daylens = daylens[:ind+1]

    # generate random start points, in the end, make sure that start point +
    # daylens does not exceed 365 days
    daystarts = np.random.random_integers(0, 364, size=len(daylens))
    dayends = daystarts + daylens
    dayoverlen = (dayends - 365) * ((dayends-365) > 0)
    daystarts -= dayoverlen

    # generate random ts_ids used for concat
    # Hint: mind the difference between np.random.randint and
    # np.random.ranom_integers - the former is [low, high) (exclusive),
    # the latter is [low, high] (inclusive)
    ts_ids = np.random.randint(0, nts, size=len(daylens))

    return daystarts, daylens, ts_ids


def _concat_single(daystarts, daylens, df_ts):
    ts_concat = np.zeros((8760,))
    loop_idx = 0
    for ds, dl, (_, ts) in zip(daystarts, daylens, df_ts.items()):
        hl = dl*24
        hs = dl*24
        s_idx, e_idx = hs, hs + hl
        ts_section = ts[s_idx:e_idx]
        ts_concat[loop_idx:loop_idx+hl] = ts_section
        loop_idx += hl
    return estss.util.norm_maxabs(ts_concat)


# ## Superposition

def _single_superpos(ts_list, scales):
    scaled_ts_list = [ts * scale for ts, scale in zip(ts_list, scales)]
    spos_ts = reduce(np.add, scaled_ts_list)
    return estss.util.norm_maxabs(spos_ts)


# ## Modification

def _curtail_down(ts, cutoff=0.2):
    tscut = copy(ts)
    tscut -= cutoff
    tscut[tscut <= 0] = 0
    if not np.any(tscut > 0):
        print('!!!')
    return estss.util.norm_maxabs(tscut)


def _invert(ts, sign=-1):
    if not (sign == 1 or sign == 0 or 1.0):
        raise ValueError(f'Parameter `sign` must be -1 or 1, found {sign}.')
    if sign == 1 or sign == 1.0:
        return ts
    else:
        return estss.util.norm_maxabs(1 - ts)


def _gen_random_interpolator(nsupports=2, seed=None):
    if seed is not None:
        np.random.rand(seed)
    x = np.array([0, *np.sort(np.random.rand(nsupports)), 1])
    y = np.array([0, *np.sort(np.random.rand(nsupports)), 1])
    return PchipInterpolator(x, y)


def _distort_time(ts, distsupports=2):
    distsupports = int(np.round(distsupports))
    fun_distort = _gen_random_interpolator(distsupports)
    t_orig = np.linspace(0, 1, len(ts))
    t_dist = fun_distort(t_orig)
    fun_dist_ts = PchipInterpolator(t_dist, ts)
    try:
        dist_ts = fun_dist_ts(t_orig)
    except ZeroDivisionError:
        dist_ts = ts
    return dist_ts  # no norm max abs, as many only zero ts are processed here


def _distort_time_24(ts, distsupports=2):
    ts_dist = copy(ts)
    ndays = int(len(ts_dist)/24)
    for day in range(ndays):
        hour_start = day*24
        hour_end = (day + 1)*24
        ts_sub = ts_dist[hour_start:hour_end]
        ts_dist[hour_start:hour_end] = _distort_time(ts_sub, distsupports)
    try:
        ts_dist = estss.util.norm_maxabs(ts_dist)
    except ZeroDivisionError:
        ts_dist = ts
    return ts_dist


def _seasonality(ts, amp=0.5, phase=None):
    if phase is None:
        phase = random.choice([0, np.pi])

    x = np.linspace(0, 2*np.pi, len(ts))
    season = (np.cos(x + phase)/2 + 0.5) * amp + (1 - amp)
    return estss.util.norm_maxabs(ts*season)


_MODKEYDEF = dict()
_MODKEYDEF['exp'] = estss.manifold._MODKEYDEF['exp']  # noqa
_MODKEYDEF['comp'] = estss.manifold._MODKEYDEF['comp']  # noqa
_MODKEYDEF['curt_top'] = (estss.manifold._curtail_up, (0.1, 0.4), 0.15)  # noqa
_MODKEYDEF['curt_bot'] = (_curtail_down, (0.1, 0.4), 0.15)
_MODKEYDEF['inv'] = (_invert, (-1, -1), 0.0005)
_MODKEYDEF['dist'] = (_distort_time_24, (1, 2), 0.3)
_MODKEYDEF['season'] = (_seasonality, (0.1, 0.7), 0.1)


# ##
# ## Helper Level Functions
# ##
# ## ##########################################################################

def _df_to_list(df):
    return [df[col].values for col in df]
