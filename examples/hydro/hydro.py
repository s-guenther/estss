#!/usr/bin/env python3

import numpy as np
import scipy.io as sio
from scipy.interpolate import pchip_interpolate as pchip


def n2048set_to_mat(sets, savepath='examples/hydro/ts_neg_2048.mat',
                    n_final=8760, repetitions=1):
    """Extracts the 4096 set from `sets`, chooses the only-negative time
    series, repeats the time series `repetitions` times and resamples them to
    `n_final` points and saves the results to `savepath`"""
    ts = sets['ts'][4096].iloc[:, :2048].values
    feat = sets['features'][4096].iloc[:2048, :]
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
    return ts_res, feat_vals, feat_names, readme


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


def _test_month():
    import os
    os.chdir('/home/sg/estss')
    from estss import reduce
    sets = reduce.get_reduced_sets()
    ts, _, _, _ = n2048set_to_mat(
        sets,
        savepath='examples/hydro/n2048_month_debug.mat',
        repetitions=12
    )
    return ts


if __name__ == '__main__':
    _test_month()
