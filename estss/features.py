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

import warnings

import numpy as np
import pandas as pd

from kats.tsfeatures.tsfeatures import TsFeatures
from kats.consts import TimeSeriesData
import pycatch22
import tsfel
import tsfresh

from estss import util

# ##
# ## Module Vars, Used by Helper Functions at end of file
# ##
# ## ##########################################################################

# Df has 11 columns:
# ID IDtool orig_name rectified_name use norm_in norm_out comment
# description parse_info
try:
    _DF_FEAT = pd.read_csv('data/feat_tool.csv', delimiter=';')
except FileNotFoundError:
    _DF_FEAT = pd.read_csv('../data/feat_tool.csv', delimiter=';')


# ##
# ## Top Level Functions
# ##
# ## ##########################################################################

def features(df_ts, show_warnings='ignore', show_progress=False):
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

    Returns
    -------
    df_feat : pandas.dataframe
        nxf pandas feature dataframe, with n being the number of time series
        and f being the number of features
    """
    with warnings.catch_warnings():
        warnings.simplefilter(show_warnings)  # noqa
        df_feat = df_ts.apply(
            lambda ts: single_features(ts, show_progress).squeeze()
        )
    return df_feat.T


def single_features(ts, show_progress=False):
    """Calculate all relevant features for a single time series.

    Gets a (m,)-np.array() time series, m being the number of time steps and
    returns an 1xf feature dataframe, with f being the number of features that
    are calculated for each time series. Invokes `catch22`, `kats`, `tsfel`,
    `tsfresh` and `extra` to do so.

    Note that time series is expected to be normalized by
    `util.norm_maxabs()` and is eventually z-scored automatically for some
    features.

    Parameters
    ----------
    ts : numpy.ndarray
        (m,)-numpy.array time series, m being the number of time steps
    show_progress : bool, default: False
        whether to show progress statusbars/info messages or not

    Returns
    -------
    df_feat : pandas.dataframe
        1xf pandas feature dataframe, with f being the number of features
    """
    feat_dfs = [
        _feat_from_catch22(ts),
        _feat_from_kats(ts),
        _feat_from_tsfel(ts, show_progress),
        _feat_from_tsfresh(ts, show_progress),
        _feat_from_extra(ts)
    ]
    return pd.concat(feat_dfs, axis=1)


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
    result as 1xf feature dataframe.
    The function uses info of the module variable `_DF_FEAT`"""
    # There is a lot of implicit knowledge in this function:
    #   - all vars of catch24 are used
    #   - input time series already normed by norm_maxabs
    #   - all features need this norm or don't care
    feat_dict = pycatch22.catch22_all(ts, catch24=True)
    feat_df = pd.DataFrame(feat_dict['values'], index=feat_dict['names']).T
    _rectify_names(feat_df, 'catch22')
    return feat_df


# ##
# ## Kats
# ##

def _feat_from_kats(ts):
    """Invokes kats feature calculation for the time series `ts` and returns
    result as 1xf feature dataframe.
    The function uses info of the module variable `_DF_FEAT`"""
    # Implicit knowledge in this function:
    #   - all used vars of kats use z-scored ts or don't care

    # z-score data
    ts = util.norm_zscore(ts)
    # generate feature model with selected features
    feat_names = _get_tool_info_tab('kats').orig_name
    model = TsFeatures(selected_features=list(feat_names))
    # transform ts into the kats format
    df_ts = pd.DataFrame(data=dict(time=np.arange(ts.size), data=ts))
    ts_kats = TimeSeriesData(df_ts)
    # call feature calculation and transform output
    feat = model.transform(ts_kats)
    feat_df = pd.DataFrame(data=feat, index=[0])
    _rectify_names(feat_df, 'kats')
    return feat_df


# ##
# ## Tsfel
# ##

def _feat_from_tsfel(ts, show_progress=True):
    """Invokes tsfel feature calculation for the time series `ts` and returns
    result as 1xf feature dataframe.
    The function uses info of the module variable `_DF_FEAT`"""
    # Implicit knowledge in this function
    #   - there are some features that require z-scored data, some require
    #     maxabs normed data, some don't care
    #   - Input `ts` is expected to be normed to max abs
    feat_tab = _get_tool_info_tab('tsfel')
    # call actual extraction routine for subtab with norm_in == norm_maxabs
    feat_df_maxabs = _feat_from_tsfel_subtab(
        ts,
        feat_tab.query("norm_in == 'norm_maxabs' | norm_in == '_dont_care'"),
        show_progress
    )
    # call actual extraction routine for subtab with norm_in == z_score
    feat_df_zscore = _feat_from_tsfel_subtab(
        util.norm_zscore(ts),
        feat_tab.query("norm_in == 'z_score'"),
        show_progress
    )
    # merge both results
    return pd.concat([feat_df_maxabs, feat_df_zscore], axis=1)


def _feat_from_tsfel_subtab(ts, subtab, show_progress=True):
    """Actual invocation routine for tsfel. Calcs features for time series
    `ts` for the features defined in data frame `subtab`. `subtab` is expected
    to be extracted from tsfel-tab by `_feat_from_tsfel` superfunction"""
    cfg = _parse_tsfel_tab_to_dict(subtab)
    df_ts = pd.DataFrame(data=dict(ts=ts))
    feat_df = tsfel.time_series_features_extractor(
        cfg,
        df_ts,  # noqa
        verbose=show_progress
    )
    _strip_0_in_colnames(feat_df)
    _rectify_names(feat_df, 'tsfel')
    return feat_df


def _parse_tsfel_tab_to_dict(tab):
    """Uses the information of the table `tab` about the tsfel vars and
    parses them as a config dict that satisfies the cfg requirements of
    tsfel.
    The `tab` has a column `parse_info` containing a string that encodes
        <domain> <function_name> [<para1>=<val1> <para2>=<val2> ...]"""
    names = list(tab.orig_name)
    parse_info = list(tab.parse_info)
    descriptions = list(tab.description)
    cfg = dict()
    for n, pi, desc in zip(names, parse_info, descriptions):
        # parse parse info `pi` string
        domain, fcn, *paras = pi.split(' ')
        # parse paras list, if existent, else return empty string
        if not paras:
            pdict = ''
        else:
            pdict = {}
            for p in paras:
                k, v = p.split('=')
                pdict[k] = float(v)
        # create domain subdict in cfg superdict, if not existent
        if domain not in cfg:
            cfg[domain] = dict()
        # add feature to domain subdict
        cfg[domain][n] = {
            'complexity': 'constant',
            'description': desc,
            'function': fcn,
            'parameters': pdict,
            'n_features': 1,
            'use': 'yes'
        }
    return cfg


def _strip_0_in_colnames(df):
    """Column names in `df` that are returned by tsfel are prepended by
    `0_`. This is removed."""
    colnames = df.columns.values
    newnames = {old: old.lstrip('0_') for old in colnames}
    df.rename(columns=newnames, inplace=True)


# ##
# ## Tsfresh
# ##

def _feat_from_tsfresh(ts, show_progress=True):
    """Invokes tsfresh feature calculation for the time series `ts` and returns
       result as 1xf feature dataframe.
       The function uses info of the module variable `_DF_FEAT`"""
    # Implicit knowledge in this function
    #   - there is no feature that needs z-scored data, only maxabs or dont
    #     care
    #   - Input `ts` is expected to be normed to max abs
    #   - Some features have a norm_out function encoded in the _DF_FEAT table
    df_ts = pd.DataFrame(data=dict(data=ts, id=[0]*len(ts)))
    cfg = _parse_tsfresh_tab_to_dict(_get_tool_info_tab('tsfresh'))
    feat_df = tsfresh.extract_features(
        df_ts,
        default_fc_parameters=cfg,
        column_id="id",
        disable_progressbar=~show_progress
    )
    _strip_dunder_in_colnames(feat_df)
    _rectify_names(feat_df, src='tsfresh')
    _norm_outputs(feat_df, src='tsfresh', len_ts=len(ts))
    return feat_df


def _parse_tsfresh_tab_to_dict(tab):
    """Uses the information of the table `tab` about the tsfresh vars and
    parses them as a config dict that satisfies the cfg requirements of
    tsfresh.
    The `tab` has a column `parse_info` containing a string that encodes
        [<para1>=<val1> <para2>=<val2> ...]
    or is empty if no para is needed"""
    names = list(tab.orig_name)
    parse_info = list(tab.parse_info)
    cfg = dict()
    for n, pi in zip(names, parse_info):
        # parse paras list, if existent, else return None
        if isinstance(pi, float) and np.isnan(pi):
            pdict = None
        else:
            # parse parse info `pi` string
            paras = pi.split(' ')
            pdict = {}
            for p in paras:
                k, v = p.split('=')
                if v == 'True':
                    pdict[k] = True
                elif v == 'False':
                    pdict[k] = False
                else:
                    pdict[k] = float(v)
                pdict = [pdict]

        # add feature to domain subdict
        cfg[n] = pdict
    return cfg


def _strip_dunder_in_colnames(df):
    """Column names in `df` that are returned by tsfel are prepended by
    `*__` and trailed by '__*'. This is removed."""
    colnames = df.columns.values
    newnames = []
    for n in colnames:
        ind = n.find('__')
        stripped = n[ind+2:]
        ind = stripped.find('__')
        if not ind == -1:
            stripped = stripped[:ind]
        newnames.append(stripped)
    mapping = {old: new for old, new in zip(colnames, newnames)}
    df.rename(columns=mapping, inplace=True)


def _norm_outputs(df_feat, src=None, len_ts=1000):
    """Norms some outputs specified in `tab` (loaded via `src`) from
    absolute count to relative count. Inplace conversion.
    Expects an 1xf dataframe `df_feat` as input.
    Does properly only work for tsfresh without modification."""
    tab = _get_tool_info_tab(src) if src else _DF_FEAT
    mask = tab.norm_out.apply(lambda x: isinstance(x, str))
    mult_mask = ~mask*1 + mask*1/len_ts
    df_feat.loc[:, :] = df_feat.values * mult_mask.values


# ##
# ## Extra Features manually implemented
# ##

def _feat_from_extra(ts):
    """Calculates extra featues for the time series `ts` that are not
    present in any of the above toolboxes and returns result as a 1xf
    feature dataframe. The features are manually implemented below from
    doi:10.3390/s150716225."""
    feat_dict = {
        'arv': _average_rectified_value(ts),
        'crest': _crest(ts),
        'shape': _shape(ts),
        'impulse': _impulse(ts),
        'clearance': _clearance(ts)
    }
    return pd.DataFrame(data=feat_dict, index=[0])


def _crest(ts):
    """determines the crest factor of the ts,
    taken from doi:10.3390/s150716225"""
    rms = np.sqrt(np.mean(ts**2))
    crest = np.max(np.abs(ts))/rms
    return crest


def _shape(ts):
    """calculates the shape factor of the data ,
    taken from doi:10.3390/s150716225"""
    rms = np.sqrt(np.mean(ts**2))
    arv = np.mean(np.abs(ts))
    shape = rms/arv
    return shape


def _impulse(ts):
    """ determines the impusle factor of the ts ,
    taken from doi:10.3390/s150716225"""
    arv = _average_rectified_value(ts)
    impulse = np.max(np.abs(ts))/arv
    return impulse


def _clearance(ts):
    """calculates the clearance factor of the data,
    taken from doi:10.3390/s150716225"""
    clearance = np.max(np.abs(ts))/(np.mean(np.sqrt(np.abs(ts)))**2)
    return clearance


def _average_rectified_value(ts):
    """calculates the arv factor of the data,
    taken from doi:10.3390/s150716225"""
    return float(np.mean(np.abs(ts)))


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


def _rectify_names(df_orig_names, src=None):
    """Changes names of the features in the columns of `df_orig_names` that
    are returned by a toolbox to the rectified names defined in _DF_FEAT.
    If `src` is provided, use a sublist of _DF_FEAT (_get_tool_info_tab()
    is called with this parameter)."""
    if src is not None:
        df_feat = _get_tool_info_tab(src)
    else:
        df_feat = _DF_FEAT

    name_pairs = {orig: rect for orig, rect
                  in zip(df_feat.orig_name, df_feat.rectified_name)}
    old_names = df_orig_names.columns.values
    new_names = {old: name_pairs[old] for old in old_names}
    # inplace, no return
    df_orig_names.rename(columns=new_names, inplace=True)


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

    cfg = tsfel.get_features_by_domain()
    r_df_tsfel = pd.DataFrame(data=dict(r1=r, r2=r))
    feat_tsfel = tsfel.time_series_features_extractor(cfg, r_df_tsfel)  # noqa

    r_df_fresh = pd.DataFrame(data=dict(data=r, id=[1]*len(r)))
    feat_tsfresh = tsfresh.extract_features(r_df_fresh, column_id="id")

    return feat_kats, feat_c24, feat_tsfel, feat_tsfresh
