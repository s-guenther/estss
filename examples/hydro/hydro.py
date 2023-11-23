#!/usr/bin/env python3

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.interpolate import pchip_interpolate as pchip
import seaborn as sns

from estss import reduce


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


_COLNAMES = [
    'LCOH',
    'Strompreis',
    'CO2_Limit',
    'CO2_FP',
    'Ely_Anlage_Nenn',
    'Ely_Nenn',
    'Peri_Nenn',
    'Ver_Nenn',
    'Speicher_Nenn',
    'WEA_Nenn',
    'PV_Nenn',
    'Ely_VLS',
    'Netzbezugsmenge',
    'Netzbezug_Nenn',
    'Ueberschuss_PV',
    'Ueberschuss_WEA',
    'Ueberschuss_Summe',
    'Ueberschuss_Nenn',
    'PV_Strom_gesamt',
    'WEA_Strom_gesamt',
    'Gesamtkosten',
    'Kosten_Ely_Nenn',
    'Kosten_Verd_Nenn',
    'Kosten_HDS_Nenn',
    'Kosten_OuM_H2_Comp',
    'Kosten_Strombezug_Netz',
    'Kosten_WEA',
    'Kosten_PV',
    'Durchgespeicherte_Wasserstoffmenge',
    'Add_WEA',
    'Add_PV',
    'Add_tot',
    'EE_Ely_Korrelation_mat_index',
    'RES_share',
    'EE_Ely_Korrelation_eigen',
    'EE_Ely_Korrelation_eigen_ex',
    'EE_Ely_Korrelation_eigen_index',
    'EE_Ely_Korrelation_eigen_excess_index'
]


def import_from_mat_and_merge():
    os.chdir('/home/sg/estss')

    monthly = sio.loadmat('examples/hydro/Mix_monthly_sol_arr_PS_9_0_1.mat')
    seasonal = sio.loadmat('examples/hydro/Mix_seasonal_sol_arr_PS_9_0_1.mat')
    yearly = sio.loadmat('examples/hydro/Mix_yearly_sol_arr_PS_9_0_1.mat')

    monthly = monthly['Mix_sol_arr_PS_9_0_1']
    seasonal = seasonal['Mix_sol_arr_PS_9_0_1']
    yearly = yearly['Mix_sol_arr_PS_9_0_1']

    sets = reduce.get_reduced_sets()
    feat = sets['features'][4096].iloc[:2048, :]

    monthly = pd.DataFrame(monthly, index=feat.index, columns=_COLNAMES)
    seasonal = pd.DataFrame(seasonal, index=feat.index, columns=_COLNAMES)
    yearly = pd.DataFrame(yearly, index=feat.index, columns=_COLNAMES)

    monthly = pd.concat([feat, monthly], axis=1)
    seasonal = pd.concat([feat, seasonal], axis=1)
    yearly = pd.concat([feat, yearly], axis=1)

    monthly['seasonality'] = 'monthly'
    seasonal['seasonality'] = 'seasonal'
    yearly['seasonality'] = 'yearly'

    return pd.concat([monthly, seasonal, yearly], axis=0)


def plot_scatter():
    data = import_from_mat_and_merge()
    yearly = data.query('seasonality == "yearly"')

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(3.5, 2.3)
    ax.set_position([0.10, 0.15, 0.899, 0.849])
    sns.scatterplot(x=-yearly['mean'], y=yearly['LCOH'],
                    hue='Netzbezugsmenge', ax=ax, data=yearly,
                    hue_norm=(-20e6, 100e6), palette='Blues')
    ax.set_xlim(0, 1)
    ax.set_ylim(5, 40)
    ax.set_xlabel('Normalized mean $\overline{x}/\hat{x}$',
                  size=8, labelpad=-0)
    ax.set_ylabel('Levelized Cost of Hydrogen $C$',
                  size=8, labelpad=-2)
    ax.tick_params(axis='both', labelsize=8)
    return fig, ax


def plot_sobol():
    paths = ('/home/sg/estss/examples/hydro/sobol_res_10k_mix_LCOH_orig.mat',
             '/home/sg/estss/examples/hydro/sobol_res_10k_mix_LCOH_new.mat')
    data = [sio.loadmat(path) for path in paths]

    fig, axs = plt.subplot_mosaic(
        [['orig', 'new']],
        gridspec_kw=dict(top=0.92, bottom=0.11, left=0.33, right=0.999,
                         wspace=0.15),
        width_ratios=[1, 1],
    )
    fig.set_size_inches(3.5, 3.5)

    ytop = np.arange(9)[::-1] + 0.2
    ybot = np.arange(9)[::-1] - 0.2

    blues = mpl.colormaps.get_cmap('Blues')(
        np.linspace(0, 1, 4)
    )
    b1 = blues[1, :]
    b2 = blues[2, :]
    b3 = blues[3, :]

    err_kw1 = dict(capsize=2, ecolor=b2, clip_on=False)
    err_kw2 = dict(capsize=2, ecolor=b3, clip_on=False)

    for d, ax in zip(data, axs.values()):
        f0 = d['indices_fO'].squeeze()
        f0_err = np.abs(np.vstack([d['conf_int_fO_low'],
                                   d['conf_int_fO_high']]) - f0)

        tot = d['indices_total'].squeeze()
        tot_err = np.abs(np.vstack([d['conf_int_total_low'],
                                    d['conf_int_total_high']]) - tot)

        ax.set_xlabel('Sobol\' Index', size=8)
        ax.set_ylim(-0.4, 8.4)
        ax.set_xlim(0, 0.505)
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_xticklabels(['0', '', '0.2', '', '0.4', ''], size=8)
        ax.grid(axis='x', zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True)

        ax.barh(ytop, f0, xerr=f0_err, height=0.4,
                error_kw=err_kw1, color=b1, clip_on=False, zorder=1)
        ax.barh(ybot, tot, xerr=tot_err, height=0.4,
                error_kw=err_kw2, color=b2, clip_on=False, zorder=1)


    fig.patches.extend(
        [plt.Rectangle(
            (0.01, 0.1), 0.98, 0.10,
            fill=True, ec=b3, fc=b1, alpha=0.2, zorder=0, linewidth=1.5,
            transform=fig.transFigure, figure=fig
        )]
    )
    axs['new'].set_yticks(ytop - 0.2)
    axs['new'].set_yticklabels([])
    axs['orig'].set_yticks(ytop - 0.2)
    axs['orig'].set_yticklabels([
        'Availability of\nRenewable Energy', 'Interest Rate', 'CAPEX RES',
        'CAPEX Electrolyzer',
        'Spec Energy Con-\nsump. of Electrolyzer\nat Nominal Power',
        'CAPEX Storage\nand Compressor', 'Grid Electricity Price',
        'Grid Electricity\nEmission Intensity', 'Demand\nCharacteristic'],
        size=7,
    )
    axs['orig'].tick_params(axis='y', which='major', pad=5)

    axs['orig'].text(0, 8.5, '(a) Original\n      time series', size=8,
                     va='bottom', ha='left', clip_on=False)
    axs['new'].text(0, 8.5, '(b) Declustered\n      time series', size=8,
                    va='bottom', ha='left', clip_on=False)

    return fig, ax
