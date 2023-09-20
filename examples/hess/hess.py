#!/usr/bin/env python3

import os
import pickle

from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns

from estss import reduce, init, expand, util, features


def export_to_mat():
    os.chdir('/home/sg/estss')

    # ##
    # ## Init Set and Features
    print('# ## Init Set')
    init_ts = init._ees_ts(  # noqa
        selectionsfile='examples/hess/ees_selections_mod',
        _raw_to_init_fcn=_mod_single_raw_to_init
    )
    # remove init ts that do not comply boundary start = max, end = Min
    csum_init = pd.DataFrame(np.cumsum(init_ts, axis=0))
    valid = csum_init.apply(lambda x: np.all((x <= 0) & (x >= x.iloc[-1])))
    init_ts = init_ts.loc[:, valid]
    init_ts = init_ts.sample(256, axis=1)
    init_feat = features.features(init_ts)

    # ##
    # ## Reduced Set and Features
    print('# ## Reduced Set')
    sets = reduce.get_reduced_sets()
    red_ts = sets['ts'][4096]
    red_feat = sets['features'][4096]

    # ##
    # ## Expanded Set and Features
    print('# ## Expanded Set')
    exp_ts_sets = expand.get_expanded_ts()
    exp_feat_sets = features.get_features()

    # Reindex pos neg feature and ts array with +1e6 (as done by reduce.py)
    exp_ts_posneg = exp_ts_sets[1]
    exp_ts_posneg.columns += int(1e6)
    exp_ts_sets[1] = exp_ts_posneg
    exp_feat_posneg = exp_feat_sets[1]
    exp_feat_posneg.index += int(1e6)
    exp_feat_sets[1] = exp_feat_posneg

    exp_ts = pd.concat(exp_ts_sets, axis=1)
    exp_feat = pd.concat(exp_feat_sets, axis=0)

    exp_ts = exp_ts.sample(2*8192, axis=1)
    exp_feat = exp_feat.loc[exp_ts.columns, :]

    # ##
    # ## Export
    print('# ## Export')
    matexport = dict(
        reduced=red_ts.values,
        expanded=exp_ts.values,
        init=init_ts.values
    )
    matsavepath = 'examples/hess/data.mat'
    sio.savemat(matsavepath, matexport)

    pyexport = dict(
        init_ts=init_ts,
        init_feat=init_feat,
        exp_ts=exp_ts,
        exp_feat=exp_feat,
        red_ts=red_ts,
        red_feat=red_feat
    )
    pysavepath = 'examples/hess/data.pkl'
    with open(pysavepath, 'wb') as file:
        pickle.dump(pyexport, file)

    return pyexport


def import_from_mat_and_merge():
    os.chdir('/home/sg/estss')

    # load orig python data (features and time series)
    with open('examples/hess/data.pkl', 'rb') as od:
        orig_data = pickle.load(od)

    # load matlab data
    structdata = sio.loadmat('examples/hess/hess_data.mat')

    hess_data_raw = dict(
        init_feat=structdata['data'][0, 0]['init'],
        exp_feat=structdata['data'][0, 0]['expanded'],
        red_feat=structdata['data'][0, 0]['reduced'],
    )

    # slice map: temporary for small dataset
    # Create pandas dataframes out of raw matlab arrays
    hess_data = dict()
    for key, val in hess_data_raw.items():
        hess_data[key] = pd.DataFrame(
            val[:, :3],
            columns=['hyb_pot', 'hyb_rel', 'hyb_skew'],
            index=orig_data[key].index
        )

    # merge with original feature dataframes
    for key in hess_data.keys():
        orig_data[key] = pd.concat([orig_data[key], hess_data[key]],
                                   axis=1)

    return orig_data


def _mod_single_raw_to_init(raw_ts, start, stop, endpoint=False, samples=1000):
    """Takes a single raw time series `raw_ts` as numpy array, extracts a
    subsection defined by [`start`, `stop`], resamples to `samples` points,
    normalizes and returns."""
    if endpoint:
        stop += 1

    init_ts = raw_ts[start:stop]
    init_ts = np.nan_to_num(init_ts)
    init_ts = util.resample_ts(init_ts, samples)
    init_ts = util.norm_maxabs(init_ts)

    if np.mean(init_ts) > 0:
        init_ts *= -1

    return init_ts


def plot_mean_hyb_main():
    os.chdir('/home/sg/estss')

    hess = import_from_mat_and_merge()
    red = hess['red_feat']

    fig, ax = plt.subplots(1, 1)

    # calc and plot 50, 75, 90, 95 line
    mod_hyb = red['hyb_pot']/(1 + red['mean'])
    mod_hyb.sort_values(inplace=True, ascending=False)
    quants = [95, 90, 75, 50]
    cols = np.linspace(0.9, 0.7, len(quants))
    for perc, col in zip(quants, cols):
        val = np.percentile(mod_hyb, 100 - perc)
        patch = mpatches.Polygon(
            [(1, 0), (0, 1), (0, val)],
            ec=None, fc=str(col)
        )
        ax.text(0.005, val, perc, va='bottom', ha='left', weight='bold')
        ax.add_patch(patch)

    ax.plot([0, 1], [1, 0], color='0.2')

    # plot scatter
    sns.scatterplot(x=-red['mean'], y=red['hyb_pot'],
                    ax=ax, color='0.3', s=10)
    ax.autoscale(tight=True)

    return fig, ax


_INIT64_INDEX = [
    123, 15, 1, 103, 44, 78, 128, 94, 132, 299, 331, 297, 77,
    97, 118, 110, 75, 301, 96, 108, 131, 0, 361, 280, 136, 362,
    365, 285, 360, 329, 300, 4, 11, 318, 313, 135, 363, 354, 133,
    3, 356, 366, 319, 359, 83, 112, 324, 323, 140, 61, 114, 358,
    138, 126, 134, 292, 100, 295,  79, 289, 279, 72, 2, 137
]


def plot_compare_init_exp_red():
    os.chdir('/home/sg/estss')

    hess = import_from_mat_and_merge()
    init64 = hess['init_feat'].loc[_INIT64_INDEX, :]
    red64 = hess['red_feat'].query(
        '(index <32) | ((index >= 1000000) & (index < 1000032))'
    )
    exp = hess['exp_feat']

    fig, ax = plt.subplots(1, 3)
    sns.scatterplot(x=-init64['mean'], y=init64['hyb_pot'],
                    ax=ax[0], color='0.8', s=10)
    sns.scatterplot(x=-red64['mean'], y=red64['hyb_pot'],
                    ax=ax[1], color='0.8', s=10)
    sns.scatterplot(x=-exp['mean'], y=exp['hyb_pot'],
                    ax=ax[2], color='0.8', s=2)
    ax[0].plot([0, 1], [1, 0], color='0.8')
    ax[0].autoscale(tight=True)
    ax[1].plot([0, 1], [1, 0], color='0.8')
    ax[1].autoscale(tight=True)
    ax[2].plot([0, 1], [1, 0], color='0.8')
    ax[2].autoscale(tight=True)
    fig.tight_layout()
    return fig, ax
