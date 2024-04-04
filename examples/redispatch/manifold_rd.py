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


def concatenate():
    pass


def modify():
    pass


# ##
# ## Low Level Functions
# ##
# ## ##########################################################################

def _concat_strings():
    pass


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
