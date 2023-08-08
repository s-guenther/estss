#!/usr/bin/env python3
"""Minimal testing script to see if `kats`, `tsfresh`, `tsfel` and
`pycatch22` are working."""

import numpy as np
import pandas as pd

from kats.tsfeatures.tsfeatures import TsFeatures
from kats.consts import TimeSeriesData
import pycatch22
import tsfel
import tsfresh

r = np.random.rand(100)

r_df_kats = pd.DataFrame(data=dict(time=np.arange(r.size), data=r))
r_kats = TimeSeriesData(r_df_kats)
feat_kats = TsFeatures().transform(r_kats)

feat_c24 = pycatch22.catch22_all(r, catch24=True)

r_df_fresh = pd.DataFrame(data=dict(data=r, id=[1]*len(r)))
feat_tsfresh = tsfresh.extract_features(r_df_fresh, column_id="id")

cfg = tsfel.get_features_by_domain()
r_df_fel = pd.DataFrame(data=dict(r1=r, r2=r))
feat_tsfel = tsfel.time_series_features_extractor(cfg, r_df_fel)

