#!/usr/bin/env python3
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


def merge_features(feat_paths=FEAT_PATHS, save=False):
    df_list = [pd.read_pickle(fpath) for fpath in feat_paths]
    df_feat = pd.concat(df_list, axis=0, ignore_index=True)
    if save:
        df_feat.to_pickle(save)
    return df_feat
