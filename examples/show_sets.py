#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import patheffects as pe
import cycler

from estss import reduce, features


def plot_nd_hist_matrix(calc=False):
    os.chdir('/home/sg/estss/')

    # ## calc or load data
    if calc:
        # Calculate and save dimensional reduced spaces
        exp_feat = pd.concat(features.get_features(), axis=0, ignore_index=True)
        init_feat = pd.read_pickle('data/init_feat.pkl')
        large_feat = pd.concat([init_feat, exp_feat], axis=0, ignore_index=True)
        large_space, _ = reduce.dimensional_reduced_feature_space(
            large_feat, plot=False
        )

        init_space = large_space.iloc[:init_feat.shape[0], :]
        exp_space, _ = reduce.dimensional_reduced_feature_space(
            exp_feat, plot=False
        )
        init_space.to_pickle('data/init_space.pkl')
        exp_space.to_pickle('data/exp_space.pkl')
    else:
        init_space = pd.read_pickle('data/init_space.pkl')
        exp_space = pd.read_pickle('data/exp_space.pkl')

    set_spaces = reduce.get_reduced_sets()['norm_space']

    # ## create diagram
    fig, ax = plt.subplot_mosaic(
        [['initial', 'expanded'], ['n4096', 'n1024'], ['n256', 'n64'],
         ['cmap', 'cmap'], ['dims', 'dims']],
        gridspec_kw=dict(top=1, bottom=0.11, left=0, right=1, wspace=0.05,
                         hspace=0.20),
        height_ratios=[1, 1, 1, 0.1, 0.01],
    )
    fig.set_size_inches(8.9/2.54, 12/2.54)

    # make cmap
    cmap = mpl.colors.ListedColormap(
        ['darkblue', 'steelblue', 'lightblue', 'whitesmoke',
         'lightsalmon', 'indianred', 'maroon']
    )
    bounds = [0, 0.02, 0.05, 0.07, 0.13, 0.15, 0.18, 1]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    titlelabels = ['initial', 'expanded', 'n4096', 'n1024', 'n256', 'n64']

    # loop through histograms
    sets = [init_space, exp_space, set_spaces[4096], set_spaces[1024],
            set_spaces[256], set_spaces[64]]
    for set_, lbl, letter in zip(sets, titlelabels, 'abcdef'):
        a = ax[lbl]
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['bottom'].set_visible(False)
        a.spines['left'].set_visible(False)
        reduce._plot_nd_hist(  # noqa
            set_,
            bins=10,
            ax=a,
            ndigits=None,
            gridlinewidth=1.5,
            cmap=cmap,
            norm=norm
        )
        a.get_xaxis().set_ticks([])
        a.get_yaxis().set_ticks([])
        ttl_lbl = ( f'({letter}) {lbl} set, '
                    f'H = {reduce._heterogeneity(set_):.3f}' # noqa
        )
        a.set_title(ttl_lbl, {'va': 'top'}, loc='left', y=-0.07, fontsize=8)

    #  make extra info axes
    ax['cmap'].matshow(np.array([[1, 4, 6, 10, 14, 17, 20]])/100,
                       cmap=cmap, norm=norm, aspect='auto')
    perc_labels = ['<2%', '2-5%', '5-8%', '8-12%', '12-15%', '15-18%', '>18%']
    colors = ['white', 'white', 'black', 'black', 'black', 'white', 'white']
    for ii, (perc_lbl, color) in enumerate(zip(perc_labels, colors)):
        ax['cmap'].text(
            ii, 0, perc_lbl,
            ha='center', va='center', clip_on=True, color=color, fontsize=8
        )
    ax['cmap'].axis('off')
    ax['cmap'].set_title(
        f'(g) Color encoding of ND-Histograms',
        loc='left', y=-1.5, fontsize=8
    )
    ax['dims'].axis('off')
    ax['dims'].set_title(
        (f'temporal_centroid, loc_of_last_max, dfa, rs_range, mode5, \n'
         f'share_above_mean, iqr, mean, rcp_num, acf_first_zero, \n'
         f'median_of_abs_diff, freq_mean, mean_2nd_diff, trev\n'
         f'(h) Dimension names of ND-histograms from top to bottom\n'),
        {'va': 'top'},
        loc='left',
        y=-8,
        fontsize=8
    )

    return fig, ax


def plot_n64_ts(yspace=1, xspace=50, **kwargs):
    os.chdir('/home/sg/estss/')

    sets = reduce.get_reduced_sets()
    n64 = sets['ts'][64]

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(18/2.54, 12/2.54)
    ax.set_position([0, 0, 1, 1])
    colormap = plt.get_cmap('Paired')(range(8))
    ax.set_prop_cycle(cycler.cycler(color=colormap))
    ax.set_xlim([0, 4000 + 3*xspace])
    ax.set_ylim([-16*yspace - 0.1, 0.8])
    ax.axis('off')
    for ii in range(64):
        row = ii % 16
        col = ii // 16
        ts = n64.iloc[:, ii]
        x = np.arange(1000) + col * 1000 + col * xspace
        y = ts - yspace * row
        ax.plot([x[0], x[-1]], [-yspace * row, -yspace * row], **kwargs)
        ax.plot(
            x, y,
            path_effects=[pe.Stroke(linewidth=3, foreground='w'), pe.Normal()],
            linewidth=0.5,

            **kwargs
        )

    return fig, ax


if __name__ == '__main__':
    plot_nd_hist_matrix()