#!/usr/bin/env python3
"""Calculate time series features by mainly invoking the feature calculation
routines of the toolboxes `pycatch22`, `kats`, `tsfel` and `tsfresh` which
are comprehended by a few manually implemented features not present in these
toolboxes.

The features are the starting point for feature engineering and building the
feature space into which the time series are transformed to reduce a large
expanded superset into a final and pruned small subset.

Relevant methods are:
    features()
    single_features()
"""

import numpy as np
import pandas as pd

from kats.tsfeatures.tsfeatures import TsFeatures
from kats.consts import TimeSeriesData
import pycatch22
import tsfel
import tsfresh


# ##
# ## Module Vars, Used by Helper Functions at end of file
# ##
# ## ##########################################################################

# Df has 10 columns:
# ID IDtool orig_name rectified_name use norm_in norm_out comment description
try:
    _DF_FEAT = pd.read_csv('data/feat_tool.csv', delimiter=';')
except FileNotFoundError:
    _DF_FEAT = pd.read_csv('../data/feat_tool.csv', delimiter=';')


# ##
# ## Top Level Functions
# ##
# ## ##########################################################################

def features(df_ts, **kwargs):
    """Calculate all relevant features for all time series of the pandas
    dataframe.

    Gets a mxn time series dataframe, m being the number of time steps,
    n being the number of time series and returns an nxf feature dataframe,
    with n being the number of time series and f being the number of
    features that are calculated for each time series. Applies
    `single_features()` to each time series to do so.

    Note that time series are expected to be normalized by
    `util.normmaxabs()` and are eventually z-scored automatically for some
    features.

    Parameters
    ----------
    df_ts : pandas.dataframe
        mxn pandas time series dataframe, m being the number of time steps,
        n being the number of time series
    **kwargs : dicts
        kwargs-dicts that are passed to the individual subroutines as
        kwargs. The kwargs must be in ['catch22', 'kats', 'tsfel', 'tsfresh']
        and the content of the dicts must be consistent with the
        requirements of the individual subroutines
        See _feat_from_* for individual information.

    Returns
    -------
    df_feat : pandas.dataframe
        nxf pandas feature dataframe, with n being the number of time series
        and f being the number of features
    """
    pass


def single_features(ts, **kwargs):
    """Calculate all relevant features for a single time series.

    Gets a (m,)-np.array() time series, m being the number of time steps and
    returns an 1xf feature dataframe, with f being the number of features that
    are calculated for each time series. Invokes `catch22`, `kats`, `tsfel`,
    `tsfresh` and `extra` to do so.

    Note that time series is expected to be normalized by
    `util.normmaxabs()` and is eventually z-scored automatically for some
    features.

    Parameters
    ----------
    ts : numpy.ndarray
        (m,)-numpy.array time series, m being the number of time steps
    **kwargs : dicts
        kwargs-dicts that are passed to the individual subroutines as
        kwargs. The kwargs must be in ['catch22', 'kats', 'tsfel', 'tsfresh']
        and the content of the dicts must be consistent with the
        requirements of the individual subroutines
        See _feat_from_* for individual information.

    Returns
    -------
    df_feat : pandas.dataframe
        1xf pandas feature dataframe, with f being the number of features
    """
    pass


# ##
# ## Low Level Functions
# ##   Call the separate toolboxes and save the results in a common dataformat
# ##   Each gets a time series as nd-array and returns an 1xf pandas
# ##   dataframe, where f is the number of features specific to the toolboxes
# ##
# ## ##########################################################################


# ##
# ## Catch 22
# ##

def _feat_from_catch22(ts):
    """Invokes `pycatch2.catch22_all()` for the time series `ts` and returns
    result as 1xf feature dataframe."""
    # There is a lot of implicit knowledge in the following lines:
    #   - all vars of catch24 are used
    #   - input time series already normed by norm_maxabs
    #   - all features need this norm or don't care
    feat_dict = pycatch22.catch22_all(ts, catch24=True)
    rect_names = _rectify_names(feat_dict['names'], 'catch22')
    return pd.DataFrame(feat_dict['values'], index=rect_names).T


# ##
# ## Kats
# ##

def _feat_from_kats():
    pass


# ##
# ## Tsfel
# ##

def _feat_from_tsfel():
    pass


# ##
# ## Tsfresh
# ##

def _feat_from_tsfresh():
    pass


# ##
# ## Extra Features manually implemented
# ##

def _feat_from_extra():
    pass


# ##
# ## Common helper functions for catch22, kats, tsfel, tsfresh
# ##
# ## ##########################################################################

def _get_tool_info_tab(src, use=True):
    """Returns info table for oneo toolbox `src`. If `use=True`, return only
    used features, if False, return unused, if None, return all.
    Valid `src` strings are:
        'catch22', 'kats', 'tsfel', 'tsfresh', 'extra'.
    Returned df has 10 columns:
        ID IDtool orig_name rectified_name use
        norm_in norm_out comment description"""
    if use is not None:
        return _DF_FEAT.query(f"src=='{src}' & use=={use}")
    else:
        return _DF_FEAT.query(f"src=='{src}'")


def _rectify_names(name_list, src=None):
    """Changes names in a `name_list` that are returned by a toolbox to the
    rectified names defined in _DF_FEAT. If `src` is provided, use a sublist
    of _DF_FEAT (_get_tool_info_tab() is called with this parameter)."""
    if src is not None:
        df_feat = _get_tool_info_tab(src)
    else:
        df_feat = _DF_FEAT

    name_pairs = {orig: rect for orig, rect
                  in zip(df_feat.orig_name, df_feat.rectified_name)}
    return [name_pairs[name] for name in name_list]


# ##
# ## Test all 4 Toolboxes in Minimal Example
# ##

def __minimal_test():
    """Minimal testing script to see if `kats`, `tsfresh`, `tsfel` and
    `pycatch22` are working."""
    r = np.random.rand(100)

    r_df_kats = pd.DataFrame(data=dict(time=np.arange(r.size), data=r))
    r_kats = TimeSeriesData(r_df_kats)
    feat_kats = TsFeatures().transform(r_kats)

    feat_c24 = pycatch22.catch22_all(r, catch24=True)

    r_df_fresh = pd.DataFrame(data=dict(data=r, id=[1]*len(r)))
    feat_tsfresh = tsfresh.extract_features(r_df_fresh, column_id="id")

    cfg = tsfel.get_features_by_domain()
    r_df_fel = pd.DataFrame(data=dict(r1=r, r2=r))
    feat_tsfel = tsfel.time_series_features_extractor(cfg, r_df_fel)  # noqa

    return feat_kats, feat_c24, feat_tsfel, feat_tsfresh
