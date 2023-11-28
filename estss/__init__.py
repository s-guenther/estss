#!/usr/bin/env/ python3
"""The `estss` (Energy System Time Series Suite) module is a comprehensive
framework designed for the intricate handling, analysis, and processing of time
series data, particularly in the context of energy systems. This module
encompasses a suite of submodules, each specialized in various aspects of time
series data manipulation, from initial data retrieval to advanced feature
engineering, declustering, manifold creation, and dimensional reduction.

Key Functionalities:
--------------------
- Data Preparation:
  Retrieve and generate initial datasets for analysis.
- Manifold Creation:
  Transform datasets through various mathematical and signal processing
  techniques.
- Feature Engineering:
  Calculate a wide array of features from time series data, facilitating data
  analysis.
- Dimensionality Reduction:
  Streamline datasets by retaining critical information and reducing
  complexity.
- Decluster and Optimize:
  Enhance data representation and manageability in feature space.
- Import/Export:
  Handle time series data efficiently, ensuring ease of access and
  manipulation.
- Analysis and Visualization:
  Explore data distributions, correlations, and other analytical aspects.

Submodules in the estss Namespace:
----------------------------------
init: Handles the generation and retrieval of initial time series data.
manifold: Focuses on transforming initial datasets through recombination,
    signal processing, and boundary condition adjustments.
features: Feature calculation using various established toolboxes, essential
    for feature space construction.
decluster: Specializes in declustering and optimizing feature space
    distribution, reducing redundancy and enhancing data uniformity.
analyze: Specializes in analyzing and visualizing data, with a focus on
    feature distribution and dataset heterogeneity.
dimred: Dimensionality Reduction capabilities of feature space needed for
    efficient declustering.
io: Provides functionalities for  import and export of time series data and
    associated information to csv.
util: Generic functions for common data manipulations, applicable across
    different submodules.

Functions in the estss Namespace:
----------------------------------
get_init_ts()
get_manifold_ts(), compute_manifold_ts()
get_features(), compute_features()
get_declustered_sets(), compute_declustered_sets()

Note: the `get_manifold_ts()` and `get_features()` function only works after
invoking the corresponding `compute_{features,declustered_sets}()` function and
manually saving the result. The `get_init()` and `get_declustered_sets()`
functions work right away, as the precomputed data is stored in the git
project.

Pipeline from initial time series to declustered set:
-----------------------------------------------------
    import pickle
    import pandas as pd
    from estss import import *

    # load init
    init = get_init_ts()

    # compute and save manifolded ts
    mani_neg, mani_posneg, mani_info = compute_manifold_ts()
    mani_neg.to_pickle('data/manifold_ts_only_neg.pkl')
    mani_posneg.to_pickle('data/manifold_ts_only_posneg.pkl')

    # compute and save features
    feat_neg, feat_posneg = compute_features()
    feat_neg.to_pickle('data/manifold_feat_only_neg.pkl')
    feat_posneg.to_pickle('data/manifold_feat_only_posneg.pkl')


    # compute and save declustered sets
    sets = compute_declustered_sets()
    with open('data/declustered_sets.pkl', 'wb') as file:
        pickle.dump(sets, file)

    # write data as csv to data/csvexport
    io.to_csv()

Refer to individual submodule and function docstrings for detailed information
and usage instructions.
"""
from estss import init, manifold, decluster, features, io, util, analyze

from .init import get_init_ts
from .manifold import get_manifold_ts, compute_manifold_ts
from .features import get_features, compute_features
from .decluster import get_declustered_sets, compute_declustered_sets
