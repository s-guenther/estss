#!/usr/bin/env python3
import random
from functools import reduce

import numpy as np
import pandas as pd

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

def manifold():
    pass


def superpose(df_ts, nout=48_000, n_ts=(2, 4), scalerange=(0.2, 1), seed=3):
    # make id list and scale list
    random.seed(seed)
    nin = df_ts.columns.size
    idlist = []
    scalelist = []
    for _ in range(nout):
        nts = random.randint(*n_ts)
        ts_ids = [random.randrange(nin) for _ in range(nts)]
        scales = [int(random.uniform(*scalerange) * 10) / 10 for _ in
                  range(nts)]
        idlist.append(ts_ids)
        scalelist.append(scales)

    new_ts_list = [_single_superpos(_df_to_list(df_ts[ids]), scales)
                   for ids, scales in zip(idlist, scalelist)]

    ts_array = np.stack(new_ts_list, axis=1)
    return pd.DataFrame(ts_array)


def concatenate(df_ts, nout=16_000-2988, n_days=(1, 7), seed=4):
    random.seed(seed)
    np.random.seed(seed)

    nts = df_ts.columns.size
    concat_defs = [_make_concat_def(nts, n_days) for _ in range(nout)]
    new_ts_list = [_concat_single(ds, dl, df_ts[ids])
                   for ds, dl, ids in concat_defs]
    ts_array = np.stack(new_ts_list, axis=1)
    return pd.DataFrame(ts_array)


def modify():
    pass


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
    pass



# ## Superposition

def _single_superpos(ts_list, scales):
    scaled_ts_list = [ts * scale for ts, scale in zip(ts_list, scales)]
    spos_ts = reduce(np.add, scaled_ts_list)
    return estss.util.norm_maxabs(spos_ts)


# ##
# ## Helper Level Functions
# ##
# ## ##########################################################################

def _df_to_list(df):
    return [df[col].values for col in df]
