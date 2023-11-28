#!/usr/bin/env python3
"""The `util.py` submodule provides various non-specific methods and shared
functions that are not associated to a specific submodule or are used by
multiple ones These utility functions facilitate common data manipulations and
transformations.

Utility Functions Include:
--------------------------
   - read_df_if_string
   - resample_ts
   - norm_meanmaxabs
   - norm_maxabs
   - norm_zscore
   - norm_min_max

Refer to individual function docstrings for more detailed information and usage
instructions.
"""
import numpy as np
import pandas as pd
import scipy.stats
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
    normed = tsmod/maxval
    if np.any(np.isnan(normed)):
        raise ZeroDivisionError
    return normed


def norm_maxabs(ts):
    """Returns a time series `ts` normalized to maxabs = 1 (but arbitrary
    mean). May raise ZeroDivisionError"""
    normed = ts/np.max(np.abs(ts))
    if np.any(np.isnan(normed)):
        raise ZeroDivisionError
    return normed


def norm_zscore(ts):
    """Z-scores the time series `ts`, i.e. mean = 0 and std. deviation = 1."""
    return scipy.stats.zscore(ts)


def norm_min_max(feat_vec):
    """Normalizes a vector `feat_vec` to the range of [0, 1]"""
    feat_vec -= np.min(feat_vec)
    feat_vec /= np.max(feat_vec)
    return feat_vec
