#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio

import estss.decluster


def decluster_chain(df_feat_norm, set_sizes=(3000, 1000, 300, 100), seed=None):
    return estss.decluster.decluster_chain(df_feat_norm, set_sizes, seed)


def map_declustered_features(df_mani_feat, feat_ids):
    return df_mani_feat.loc[feat_ids, :]


def map_declustered_ts(feat_ids, ts_paths=None, n_per_file=64000):
    if ts_paths is None:
        ts_path_base = Path(__file__).parent / 'ts'
        ts_paths = [ts_path_base / f'ts_manifold{ii}.pkl'
                    for ii in range(1, 9)]
    sorted_ids = np.sort(feat_ids)
    # get file and ts id with mod and div
    file_ids = sorted_ids // n_per_file
    # make file id unique, split ts_ids into adequate subarrays
    unique_fids, unique_fid_idx = np.unique(file_ids, return_index=True)
    ts_ids_sublists = np.split(sorted_ids, unique_fid_idx[1:])
    # then crawl each file
    ts_list = []
    for fid, ts_ids_sub in zip(unique_fids, ts_ids_sublists):
        ts_path = ts_paths[fid]
        print(f'Processing file {ts_path}')
        df_ts = pd.read_pickle(ts_path)
        df_ts_sub = df_ts.loc[:, ts_ids_sub].copy()
        del df_ts
        ts_list.append(df_ts_sub)
    # then reorder found ts
    df_ts = pd.concat(ts_list, axis=1)
    df_ts = df_ts.loc[:, feat_ids]
    return df_ts


def plot_nd_hist_grid(init, manifold, feat_sets):
    fig, axs = plt.subplots(
        2, 3,
        gridspec_kw=dict(
            top=1, bottom=0.05, right=1, left=0.1,
            wspace=0.03, hspace=0.15
        )
    )
    titles = [
        'initial set', 'manifold set', 'n3000 set',
        'n1000 set', 'n300 set', 'n100 set'
    ]
    fsets = [init, manifold, *feat_sets]
    for ii, (ax, fset, title) in enumerate(zip(axs.flat, fsets, titles)):
        yticks = True if (ii == 0 or ii == 3) else False
        estss.analyze.plot_nd_hist(fset, ax=ax, title=title, ndigits=2,
                                   yticks=yticks)


def datadict_to_mat(datadict):
    keys = ['df_init_feat', 'df_init_space', 'df_mani_feat',
            'df_mani_space', 'df_init_ts']
    new_dict = dict()
    for key in keys:
        new_dict[key] = datadict[key].to_numpy()
    keys = ['sets_dc_space', 'sets_dc_feat', 'sets_dc_ts']
    nums = [3000, 1000, 300, 100]
    for key in keys:
        for ii, num in enumerate(nums):
            new_dict[key + '_n'+ str(num)] = datadict[key][ii].to_numpy()
    new_dict['feat_names'] = list(datadict['df_init_feat'].columns)
    new_dict['space_names'] = list(datadict['df_init_space'].columns)
    new_dict['init_ids'] = list(datadict['df_init_feat'].index)
    new_dict['dc_ids'] = list(datadict['sets_dc_feat'][0].index)
    sio.savemat('datadict.mat', new_dict)
    return new_dict
