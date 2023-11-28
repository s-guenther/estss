# ESTSS - Energy System Time Series Suite

A toolbox to generate a de-clustered, application-independent, semi-artificial
load profile benchmark set that exposes a broad feature variety and low
discrepancy within feature space.

---


## Associated Work

The final declustered data is available at\
[zenodo](https://TODO/link/to/zenodo)

The associated research paper is available at\
_\<still in peer review\>_


## Key Functionalities and Overview

### Data Preparation
This component deals with the generation and retrieval of initial time series
datasets. It is designed to serve as the starting point for any time series
analysis within the module, ensuring the availability of standardized and
pre-processed data.

### Manifold Creation
The submodule dedicated to manifold creation focuses on transforming time
series datasets through a series of mathematical and signal processing
techniques. 

### Feature Engineering
This aspect involves the calculation and analysis of a wide range of
features from time series data. The submodule utilizes various established
toolboxes and custom implementations to extract meaningful features from time
series.

### Dimensionality Reduction
To manage the complexity of high-dimensional data, this functionality simplifies
datasets by reducing their dimensionality. The process involves retaining
critical information and features from the original datasets, ensuring the
essence of the data is preserved while making it more manageable for further
processing and analysis.

### Decluster and Optimize
This component enhances the representation of data in feature space. It focuses
on declustering and optimizing datasets, which is vital for reducing redundancy
in data representation and improving the efficiency of data handling and
analysis.

### Analysis and Visualization
The module provides tools for exploring and visualizing data distributions,
correlations, and other analytical aspects of time series data.


## Submodules Overview

- `init`: Generation and retrieval of initial time series data.
- `manifold`: Data transformation through recombination and signal processing.
- `features`: Feature calculation from time series data.
- `decluster`: Optimization of feature space distribution.
- `analyze`: Analysis and visualization of feature distribution.
- `dimred`: Dimensionality reduction in feature space.
- `io`: Import/export functionalities for time series data.
- `util`: General utility functions for data manipulation.


## Requirements

Python 3.8. More details in [Installation](#installation)


## Installation

There is no setup script, `pip` or `conda` package prepared. Part of the reason is
that the toolbox is still in pre-alpha stage (I didn't even dare to provide a
version number 0.1), part is missing knowledge about how to deploy it properly and
another part is that the some packages used for this one are in conflict.
The packages that are in conflict are:

    kats, tsfresh, tsfel, pycatch24

The main issue seems to be a broken or too restrictive dependency tree of `kats`
(as of Aug 23, for a start, see
https://github.com/facebookresearch/Kats/issues/308), making it hard to install
on it's own and harder to get to work with the other ones. I didn't manage to
create a setup without dependency conflicts, however, the following installation
procedure seems to work without errors in use:

    mamba create python=3.8 -n estss
    mamba activate estss
    mamba install numpy pandas convertdate lunarcalendar holidays=0.23 tqdm pystan=2.19.1.1 fbprophet=0.7.1 packaging=21.3 kats=0.2.0 pycatch22 tsfresh ipython -c conda-forge
    pip install tsfel

So, clone the repository, add it to the Python Path, e.g. add to `.bashrc`

    export PYTHONPATH=$PYTHONPATH:$HOME/estss

setup the `venv` with `mamba` as described above and you should be good to go.


## Getting Started

The pipeline from initial time series to the published declustered sets is as follows:

    import pickle
    from estss import *
    
    # Load initial time series data
    init = get_init_ts()
    
    # Compute and save manifolded time series
    mani_neg, mani_posneg, mani_info = compute_manifold_ts()
    mani_neg.to_pickle('data/manifold_ts_only_neg.pkl')
    mani_posneg.to_pickle('data/manifold_ts_only_posneg.pkl')
    
    # Compute and save features
    feat_neg, feat_posneg = compute_features()
    feat_neg.to_pickle('data/manifold_feat_only_neg.pkl')
    feat_posneg.to_pickle('data/manifold_feat_only_posneg.pkl')
    
    # Compute and save declustered sets
    sets = compute_declustered_sets()
    with open('data/declustered_sets.pkl', 'wb') as file:
        pickle.dump(sets, file)
    
    # Export data as CSV
    io.to_csv()


## Documentation

For detailed information and usage instructions, refer to the docstrings within
each submodule and function.


## Contributing

Contributions are welcome! If you have suggestions for improvements or want to
contribute code, please feel free to create an issue or submit a pull request.


## License

This software is licensed under GPLv3, excluding later versions.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

For details see [\$ESTSS/LICENSE](LICENSE).

GPLv3 explicitely allows a commercial usage without any royalty or further
implications. However, any contact to discuss possible cooperations is
appreciated.


## Author

estss - Energy System Time Series Suite\
Copyright (C) 2023\
Sebastian Günther\
sebastian.guenther@ifes.uni-hannover.de

Leibniz Universität Hannover\
Institut für Elektrische Energiesysteme\
Fachgebiet für Elektrische Energiespeichersysteme

Leibniz University Hannover\
Institute of Electric Power Systems\
Electric Energy Storage Systems Section

https://www.ifes.uni-hannover.de/ees.html