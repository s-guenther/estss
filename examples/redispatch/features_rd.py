#!/usr/bin/env python3

from pathlib import Path
from os import listdir
from os.path import isfile, join
import pandas as pd

import estss


MANIFOLD_PATHS = tuple(
    f'examples/redispatch/ts_manifold{ii}.pkl' for ii in range(1, 9)
)
FEAT_PATHS = tuple(s.replace('ts', 'feat') for s in MANIFOLD_PATHS)
FPATH = 'examples/redispatch/feat_manifold.pkl'


def get_features(path=FPATH):
    return pd.read_pickle(path)


def compute_features(mani_paths=MANIFOLD_PATHS, feat_paths=FEAT_PATHS):
    for ii, (mpath, fpath) in enumerate(zip(mani_paths, feat_paths)):
        print(f'Computing Feature Set #{ii+1}/{len(mani_paths)}')
        df_ts = pd.read_pickle(mpath)
        df_feat = estss.features.features_for_df(df_ts)
        df_feat.to_pickle(fpath)
    return None


def merge_features(feat_paths=FEAT_PATHS):
    df_list = [pd.read_pickle(fpath) for fpath in feat_paths]
    df_feat = pd.concat(df_list, axis=0, ignore_index=True)
    return df_feat


def feature_list_from_path(path=None):
    if path is None:
        path = Path(__file__).parent / 'feat'
    feat_list = [path / f for f in listdir(path)
                 if isfile(path / f)]
    feat_list.sort()
    return feat_list


def dim_red_features(df_feat, threshold=0.5):
    feat_to_drop = [
        'ami_timescale',
        'low_freq_power',
        'stl_spikiness',
        'bocp_conf_max',
        'max',
        'median',
        'min',
        'ecdf01_norm',
        'ecdf05_norm',
        'ecdf20_norm',
        'median_of_signed_diff',
        'median_diff_from_mean',
        'median_of_abs_diff',
        'peak2peak',
        'slope',
        'freq_slope',
        'mean_2nd_diff',
        'fund_freq',
        'mean_of_signed_diff',
        'stl_trough',
        'stl_peak',
        'loc_of_last_min',
    ]
    df_dropped = df_feat.drop(feat_to_drop, axis='columns')
    df_space = estss.dimred.raw_feature_array_to_feature_space(df_dropped)
    corr_mat, cinfo = \
        estss.dimred.hierarchical_corr_mat(df_space, threshold=threshold)
    choose_dim = estss.dimred.get_first_name_per_cluster(cinfo['cluster'])
    df_space2 = df_space[list(choose_dim)]
    return df_space2, corr_mat, cinfo
