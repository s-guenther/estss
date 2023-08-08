# ESTSS - Energy System Time Series Suite

A toolbox to generate a de-clustered, application-independent, semi-artificial
load profile benchmark set that exposes a broad feature variety and low
discrepancy within feature space


---

## Description

_tbd_


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

_tbd_


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

ESTSS - Energy System Time Series Suite
Copyright (C) 2023\
Sebastian Günther\
sebastian.guenther@ifes.uni-hannover.de

Leibniz Universität Hannover
Institut für Elektrische Energiesysteme\
Fachgebiet für Elektrische Energiespeichersysteme

Leibniz University Hannover
Institute of Electric Power Systems\
Electric Energy Storage Systems Section

https://www.ifes.uni-hannover.de/ees.html

