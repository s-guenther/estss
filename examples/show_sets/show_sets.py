#!/usr/bin/env python3

import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import patheffects as pe

from estss import decluster, features, init, dimred, analyze


def plot_nd_hist_matrix(calc=False):
    os.chdir('/home/sg/estss/')

    # ## calc or load data
    if calc:
        # Calculate and save dimensional reduced spaces
        exp_feat = pd.concat(features.get_features(),
                             axis=0, ignore_index=True)
        init_feat = pd.read_pickle('data/init_feat.pkl')
        large_feat = pd.concat([init_feat, exp_feat],
                               axis=0, ignore_index=True)
        large_space, _ = dimred.dimensional_reduced_feature_space(large_feat)

        init_space = large_space.iloc[:init_feat.shape[0], :]
        exp_space, _ = dimred.dimensional_reduced_feature_space(exp_feat)
        init_space.to_pickle('data/init_space.pkl')
        exp_space.to_pickle('data/exp_space.pkl')
    else:
        init_space = pd.read_pickle('data/init_space.pkl')
        exp_space = pd.read_pickle('data/manifold_space.pkl')

    set_spaces = decluster.get_declustered_sets()['norm_space']

    # ## create diagram
    fig, ax = plt.subplot_mosaic(
        [['initial', 'expanded', 'n4096'],
         ['n1024', 'n256', 'n64'],
         ['empty', 'cmap', 'cmap']],
        gridspec_kw=dict(top=1, bottom=0.00, left=0, right=1, wspace=0.10,
                         hspace=0.35),
        height_ratios=[1, 1, 0.10],
    )
    fig.set_size_inches(13.12/2.54, 7/2.54)

    # make cmap
    # cmap = mpl.colors.ListedColormap(
    #     ['darkblue', 'steelblue', 'lightblue', 'whitesmoke',
    #      'lightsalmon', 'indianred', 'maroon']
    # )
    nval = 7
    c1 = mpl.colormaps.get_cmap('RdBu')(np.linspace(0, 1, nval))[::-1, :]
    c1[3, :] = [0.93, 0.93, 0.93, 1]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('mycmap', c1, N=nval)
    bounds = [0, 0.02, 0.05, 0.07, 0.13, 0.15, 0.18, 1]
    norm = mpl.colors.BoundaryNorm(bounds, nval)
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
        analyze.plot_nd_hist(
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
        lbl = lbl if lbl != 'expanded' else 'manif.'
        ttl_lbl = (f'({letter}) {lbl} set, '
                   f'$h_{{14,10}}$ = {analyze.heterogeneity(set_):.3f}'  # noqa
                   )
        a.set_title(ttl_lbl, {'va': 'top'}, loc='left', y=-0.13, fontsize=8)

    # disable empty axes
    ax['empty'].set_axis_off()

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
        f'Relative frequency',
        loc='left', y=-0.5, x=-0.32, fontsize=8
    )
    return fig, ax


def plot_n32_ts(yspace=2.0, xspace=100, which='results', **kwargs):
    os.chdir('/home/sg/estss/')

    if which == 'results':
        sets = decluster.get_declustered_sets()
        ts_set = sets['ts'][64].iloc[:, 32:]
    elif which == 'init':
        ts_set = init.get_init_ts().sample(32, axis='columns', random_state=3)
    else:
        raise ValueError(f'Invalid value which = {which}')

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(13.12/2.54, 12/2.54)
    ax.set_position([0, 0, 1, 1])
    # colormap = plt.get_cmap('Paired')(range(2))
    # ax.set_prop_cycle(cycler.cycler(color=colormap))
    c0 = (0.75, 0.75, 0.75)
    c1 = (0, 96/255, 156/255)
    ax.set_xlim([-13, 2000 + 1*xspace])
    ax.set_ylim([-16*yspace + 0.9, 1])
    ax.axis('off')
    for ii in range(32):
        # coordinate transformation
        row = ii % 16
        col = ii // 16
        ts = ts_set.iloc[:, ii]
        x = np.arange(1000) + col * 1000 + col * xspace
        y = ts - yspace * row
        x0, xend, y0, coff = x[0], x[-1], -yspace * row, 0.9

        # plot coordinate system
        ax.plot([x0, xend - 20], [y0, y0], **kwargs,
                linewidth=0.5, color=c0)
        ax.plot([x0, x0], [y0 - coff, y0 + coff - 0.2],
                linewidth=0.5, color=c0)
        # arrows
        t1 = plt.Polygon([(xend, y0),
                          (xend - 40, y0 - 0.15),
                          (xend - 40, y0 + 0.15)],
                         color=c0, ec=None)
        t2 = plt.Polygon([(x0, y0 + coff),
                          (x0 - 13, y0 + coff - 0.4),
                          (x0 + 13, y0 + coff - 0.4)],
                         color=c0, ec=None)
        ax.add_patch(t1)
        ax.add_patch(t2)
        # axes text
        ax.text(xend, y0 - 0.16, '$t$',
                va='top', ha='right', color=c0, size=8, zorder=0)
        ax.text(x0 + 13, y0 + coff + 0.1, '$x(t)$',
                va='top', ha='left', color=c0, size=8, zorder=0)

        # plot time series
        ax.plot(
            x, y,
            path_effects=[pe.Stroke(linewidth=2, foreground='w'),
                          pe.Normal()],
            linewidth=0.5, color=c1,
            **kwargs
        )

    return fig, ax


def plot_corr_mat(calc=False):
    # calc or load data
    if calc:
        df_feat_list = features.get_features()
        df_feat = pd.concat(df_feat_list, axis=0, ignore_index=True)
        fspace = dimred.raw_feature_array_to_feature_space(
            df_feat, special_treatment=True)
        corr_mat, info = decluster.hierarchical_corr_mat(fspace)  # noqa
        with open('/home/sg/estss/data/hier_corr_data.pkl', 'wb') as file:
            pickle.dump((corr_mat, info), file)
    else:
        with open('/home/sg/estss/data/hier_corr_data.pkl', 'rb') as file:
            corr_mat, info = pickle.load(file)

    # Process data
    fig, ax = plt.subplots(1, 2)
    ax[0].set_position((0, 0, 4.2/5.165, 1))
    ax[1].set_position((4.5/5.165, 0, 0.03, 1))
    fig.set_size_inches(5.165, 4.2)
    analyze.plot_hierarchical_corr_mat(  # noqa
        corr_mat, info,
        clust_color='lightsalmon', ax=ax[0], write_clust_names=False,
        selected_feat=dimred._REPRESENTATIVES,  # noqa
        cbar_ax=ax[1],
        cbar_kws=dict(orientation='vertical')
    )
    ax[1].tick_params(axis='both', labelsize=8)
    return fig, ax


if __name__ == '__main__':
    plot_nd_hist_matrix()