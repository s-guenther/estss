#!/usr/bin/env python3
"""
This module provides functionalities to export and import the declustered data
sets related to the Energy System Time Series Suite. It allows for efficient
handling of time series data, their features, and associated information in CSV
format. The module includes functions to export data sets to CSV files, import
them back into Python environments, and validate the consistency of these
operations.

Key Functions:
- to_csv: Exports data sets to CSV files, organizing them by field names and
  sizes.
- from_csv: Imports data sets from CSV files, reconstructing the dictionary of
  data sets.
- _export_readme: Exports a predefined README markdown file to a specified
  path.
- _assert_equal: Validates that data written to and read from CSV files are
  identical.

For more information on the data, see the `_README` variable in the module.
"""

import itertools as it

import pandas as pd

from estss import reduce


_README = """Energy System Time Series Suite - Data Archive
==============================================

This archive contains variously sized sets of declustered time series within
the context of energy systems. These series demonstrate low discrepancy and
high heterogeneity in feature space, resulting in a roughly uniform
distribution within this space.

For detailed information, please refer to the corresponding GitHub project:\\
[ESTSS GitHub Project](https://github.com/s-guenther/estss/)

For associated research, see the paper:\\
[Research Paper Link](https://doi.org/10.TODO/TODO/)

Data is provided in .csv format. The GitHub project includes a Python function
to load this data as a dictionary of pandas data frames.

Should you utilize this data, kindly also cite the associated research paper.
For any queries, please feel free to reach out to us through GitHub or the
contact details provided at the end of this readme file.


## Folder Content

- `ts_*.csv`: Contains declustered load profile time series in tabular format.
  - Size: `(n+1) x (m+1)`, with `n` representing time steps (1000 per series)
    and `m` the number of series.
  - Includes a header row and index column. Headers indicate series id, and the
    index column numbers each time step, starting from `0`.
  - The first half of the series `(m/2)` consistently display a constant sign
    (negative). They are sequentially numbered from 0.
  - The second half `(m/2)` display varying signs. Numbering starts from
    `1,000,000`.
- `features_*.csv`: Tabulates features corresponding to the time series.
  - Size: `(m+1) x (f+1)`, where `m` is the number of time series and `f` is
    the number of features
  - Includes a header row and index column. Indexes represent time series id
    (matching `ts_*.csv` headers), and headers name the features.
- `norm_space_*.csv`: Shows feature vectors in normalized feature space where
  time series are declustered. Provided for completeness; typically not needed
  by users.
  - Size: `(m+1) x (g+1)`, where `m` is the number of timer series and `g` is
    the number of selected features space features. (a subset of `f` from
    `features_*.csv`).
  - Format matches `features_*.csv`.
- `info_*.csv`: Maps declustered datasets to the manifolded dataset. Provided
  for completeness; typically not needed by users.
  - Size: `(m+1) x 2`, with `m` as series count. Columns contain manifolded set
    time series ids.
  - Includes an index column and a header. The index holds the remapped id of
    declustered series. Header `0` is non-significant.

---

Each `ts_*.csv`, `features_*.csv`, `norm_space_*.csv`, and `info_*.csv` file
comes in four versions to accommodate various set sizes:
- `*_4096.csv`
- `*_1024.csv`
- `*_256.csv`
- `*_64.csv`

These represent sets with 4096, 1024, 256, and 64 time series,
respectively,offering different densities in feature space population. The
objective is to balance computational load and resolution for individual
research needs.


## Contact

ESTSS - Energy System Time Series Suite
Copyright (C) 2023\\
Sebastian Günther\\
sebastian.guenther@ifes.uni-hannover.de

Leibniz Universität Hannover\\
Institut für Elektrische Energiesysteme\\
Fachgebiet für Elektrische Energiespeichersysteme

Leibniz University Hannover\\
Institute of Electric Power Systems\\
Electric Energy Storage Systems Section
"""

_SAVEPATH = 'data/csvexport/'
_FIELDNAMES = ['features', 'ts', 'norm_space', 'info']
_SIZES = [4096, 1024, 256, 64]


def _export_readme(savepath=_SAVEPATH):
    """Exports the `_README` variable to the specified `savepath`. Used by
    `to_csv()`."""
    with open(f'{savepath}README.md', 'w') as file:
        file.writelines(_README)


def to_csv(sets=None, savepath=_SAVEPATH):
    """ Exports the given sets of data to CSV files.

    This function iterates over the specified data sets and their corresponding
    sizes, exporting each as a CSV file to the designated save path. If no
    sets are provided, it defaults to fetching reduced sets of data.

    Parameters
    ----------
    sets : dict, optional:
        A dictionary of data sets to be exported. Each key corresponds
        to a field name, and each value is a dictionary where keys are sizes
        and values are data frames. If None, reduced data sets are obtained by
        default.
    savepath : str
        The base path where the CSV files will be saved. Each file is named
        following the pattern '{field}_{size}.csv'.

    The function also calls an internal function to export a corresponding
    readme file to the save path.

    Note - Ensure that the specified save path exists and is writable.
    """
    if sets is None:
        sets = reduce.get_reduced_sets()
    fields = _FIELDNAMES
    sizes = _SIZES
    for field, size in it.product(fields, sizes):
        data = sets[field][size]
        data.to_csv(f'{savepath}{field}_{size}.csv')
    _export_readme(savepath)


def from_csv(savepath=_SAVEPATH):
    """Imports data sets from CSV files located at a specified path.

    This function iterates over predefined field names and sizes, loading each
    corresponding CSV file into a pandas DataFrame. It constructs a dictionary
    of data sets, organized by field names and sizes.

    Parameters
    ----------
    savepath : str, optional
        The base path where the CSV files are located. The function expects
        files named following the pattern '{field}_{size}.csv'.

    Returns
    -------
    sets : dict
        A dictionary containing the imported data sets. Each key corresponds to
        a field name, and each value is a dictionary where keys are sizes
        and values are pandas DataFrames.

    Note - Ensure that the specified save path exists and contains the
    expected CSV files.
    """
    fields = _FIELDNAMES
    sizes = _SIZES
    sets = dict()
    for field, size in it.product(fields, sizes):
        if field not in sets.keys():
            sets[field] = dict()
        data = pd.read_csv(f'{savepath}{field}_{size}.csv', index_col=0)
        if field == 'ts':
            data.columns = data.columns.astype(int)
        if field == 'info':
            data = data.squeeze()
            data.name = None
        sets[field][size] = data
    return sets


def _assert_equal(sets=None, savepath=_SAVEPATH):
    """Writes the `sets` to `savepath` with `to_csv()` and loads them again
    with `from_csv()`. Asserts the the original data and loaded data are
    equal."""
    if sets is None:
        sets = reduce.get_reduced_sets()
    to_csv(sets)
    sets_read = from_csv(savepath)
    fields = _FIELDNAMES
    sizes = _SIZES
    for f, s in it.product(fields, sizes):
        print(f'Checking {f}, {s}')
        if f == 'info':
            assert_equal = pd.testing.assert_series_equal
        else:
            assert_equal = pd.testing.assert_frame_equal
        assert_equal(sets[f][s], sets_read[f][s], rtol=1e-8)
