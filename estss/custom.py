#!/usr/bin/env python3
"""This module provides a concrete example on how to add a custom feature,
manually add it to the feature data frame, and perform the decluster process
again with this feature."""
import copy
import itertools

# from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.interpolate import pchip_interpolate as pchip

from estss import util, dimred, decluster


def curtail_to_zero(ts, rel_val=0):
    minval = min(ts)
    maxval = max(ts)
    curtval = maxval - abs(maxval - minval)*rel_val
    ts = copy.copy(ts)
    ts[ts >= curtval] = curtval
    ts -= curtval
    ts /= -min(ts)
    return ts


def share_zero(ts, treshold=0.00):
    return np.sum(np.abs(ts) <= treshold) / len(ts)


_TRESHOLDS = (0.00, 0.01, 0.02, 0.05, 0.10)
_SUFFIXES = ('', '_01', '_02', '_05', '_10')
_PREFIX = 'share_zero'


def add_share_zero_to_feature_df(
        df_ts='/home/sg/estss/data/manifold_ts_only_neg.pkl',
        df_feat='/home/sg/estss/data/manifold_feat_only_neg.pkl',
        # df_ts='data/init_ts.pkl',
        # df_feat='data/init_feat.pkl',
        tresholds=_TRESHOLDS,
        suffixes=_SUFFIXES,
        prefix=_PREFIX
):
    df_ts = util.read_df_if_string(df_ts)
    df_feat = util.read_df_if_string(df_feat)

    for tres, suf in zip(tresholds, suffixes):
        df_feat[f'{prefix}{suf}'] = df_ts.apply(share_zero, args=(tres,))

    return df_feat


def add_normalized_to_feature_df(df_feat, suffixes=_SUFFIXES, prefix=_PREFIX):
    orig_names = [f'{prefix}{suf}' for suf in suffixes]
    for name in orig_names:
        df_feat[f'{name}_sigmoid'] = \
            dimred._outlier_robust_sigmoid(df_feat[name])  # noqa
        df_feat[f'{name}_whisk'] = \
            dimred._curtail_at_whiskers(df_feat[f'{name}_sigmoid'])  # noqa
        df_feat[f'{name}_minmax'] = \
            util.norm_min_max(df_feat[f'{name}_whisk'])  # noqa
    return df_feat


def plot_share_zero_hists(df_feat, feat_names=None, **hist_kwargs):
    if feat_names is None:
        feat_names = list(f'{_PREFIX}{suf}' for suf in _SUFFIXES)
    sub_df = df_feat[[*feat_names]]
    sub_df.plot.hist(subplots=True, **hist_kwargs)


def prepare_decluster_input(add_feature='share_zero_01'):
    df_ts = pd.read_pickle('data/manifold_ts_only_neg.pkl')
    df_feat = pd.read_pickle('data/custom_feat.pkl')
    norm_space = pd.read_pickle('data/custom_space.pkl')
    return norm_space, df_feat, df_ts, add_feature


def custom_decluster(norm_space=None, df_feat=None, df_ts=None,
                     add_feature='share_zero', set_sizes=(128, 32), seed=1337):
    # Prepare input data
    if norm_space is None or df_feat is None or df_ts is None:
        df_ts = pd.read_pickle('data/manifold_ts_only_neg.pkl')
        df_feat = pd.read_pickle('data/manifold_feat_only_neg.pkl')
        norm_space, cinfo = dimred.dimensional_reduced_feature_space(df_feat)
        df_feat = add_share_zero_to_feature_df(df_ts, df_feat)

    norm_space[add_feature] = df_feat[add_feature]
    sets = decluster.decluster_chain(norm_space, set_sizes, seed=seed)
    sets = decluster.map_sets(sets, df_feat, df_ts)
    return sets


def custom_manifold(df_ts='data/manifold_ts_only_neg.pkl',
                    n_sample=2**18, nout_per_nin=1, seed=42):
    n_out = n_sample*nout_per_nin
    df_ts = util.read_df_if_string(df_ts)
    # df_ts = df_ts.sample(n_sample, random_state=seed, axis='columns',
    #                      ignore_index=True)
    ts_in = df_ts.to_numpy()
    ts_out = np.zeros((ts_in.shape[0], n_out))
    np.random.seed(seed)
    cvals = np.random.rand(n_out)
    ts_ids = [[ts_id]*nout_per_nin for ts_id in range(n_sample)]
    ts_ids = itertools.chain.from_iterable(ts_ids)
    for col_id, (ts_id, cval) in enumerate(zip(ts_ids, cvals)):
        ts_out[:, col_id] = curtail_to_zero(ts_in[:, ts_id], cval)
    return pd.DataFrame(ts_out)


def save_sets_to_mat(sets, savepath='data/curtail_set_2048.mat',
                     n_final=8760, repetitions=1):
    ts = sets['ts'][2048].values
    feat = sets['features'][2048]
    feat_names = feat.columns.values
    feat_vals = feat.values

    n_ts = ts.shape[1]
    n_resample = n_final//repetitions
    ts_res = np.zeros((n_final, n_ts))
    for ii in range(n_ts):
        # Resample repetion
        single_ts_res = _resample(ts[:, ii], n_resample)
        # concat repetitions
        single_ts_concat = np.tile(
            single_ts_res,
            reps=np.array(np.ceil(n_final/n_resample), dtype=int)
        )
        # curtail repetitions
        ts_res[:, ii] = single_ts_concat[:n_final]

    readme = (f'ts = nxm = {n_final}x{n_ts} array\n'
              f'    1st dim n: datapoints\n'
              f'    2nd dim m: number of timeseries\n'
              f'feat = mxf = {n_ts}x{feat.shape[1]} feature array\n'
              f'    1st dim m: number of timeseries\n'
              f'    2nd dim f: number of features\n'
              f'feat_names = fx1 = {feat.shape[1]}x1 names vector')

    matexport = dict(
        ts=ts_res,
        feat=feat_vals,
        feat_names=feat_names,
        info=readme
    )

    sio.savemat(savepath, matexport)


def _resample(ts, n):
    n_ts = len(ts)
    if n_ts < n:
        return _resample_pchip(ts, n)
    elif n_ts == n:
        return ts
    elif n_ts > n:
        return _resample_max(ts, n)
    else:
        raise RuntimeError('If-...-else reached presumably impossible path')


def _resample_max(ts, n):
    n_ts = len(ts)
    interval_edges = np.array(np.round(np.linspace(0, n_ts, n+1)), dtype=int)
    interval_starts = interval_edges[:-1]
    interval_ends = interval_edges[1:]
    ts_out = np.zeros(n)
    for ii, (istart, iend) in enumerate(zip(interval_starts, interval_ends)):
        ts_out[ii] = np.min(ts[istart:iend])
    return ts_out


def _resample_pchip(ts, n):
    yi = ts
    xi = np.arange(len(yi))
    xx = np.linspace(xi[0], xi[-1], n)
    return pchip(xi, yi, xx)


def main():
    df_feat = add_share_zero_to_feature_df()
    df_feat = add_normalized_to_feature_df(df_feat)
    plot_share_zero_hists(df_feat, bins=20)
    plot_share_zero_hists(
        df_feat,
        bins=20,
        feat_names=[f'{_PREFIX}{suf}_sigmoid' for suf in _SUFFIXES]
    )
    plot_share_zero_hists(
        df_feat,
        bins=20,
        feat_names=[f'{_PREFIX}{suf}_whisk' for suf in _SUFFIXES]
    )
    plot_share_zero_hists(
        df_feat,
        bins=20,
        feat_names=[f'{_PREFIX}{suf}_minmax' for suf in _SUFFIXES]
    )
    return df_feat


if __name__ == '__main__':
    sets_ = custom_decluster()
    feat_ = main()
