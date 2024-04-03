#!/usr/bin/env python3

import pandas as pd
import scipy.io as sio


def get_init_ts(matfile='examples/redispatch/ts_redispatch.mat'):
    """Loads the data stored in a .mat file in a variable named
    'timeSeries_all' and saves as a pandas df."""
    matdata = sio.loadmat(matfile)['timeSeries_all']
    return pd.DataFrame(matdata)
