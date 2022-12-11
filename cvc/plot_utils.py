from typing import *
import collections
import logging

import numpy as np
import pandas as pd
from anndata import AnnData
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn.metrics as metrics
from adjustText import adjust_text

import cvc.featurization as ft
import cvc.custom_metrics as custom_metrics
import cvc.utils as utils

SAVEFIG_DPI = 300


def plot_sequence_diversity(
    sequences: Sequence[str],
    title: str = "Sequence diversity",
    xlabel: str = "Position in sequence",
    fname: str = "",
):
    """
    Plot sequence diversity
    """
    fixed_len = set([len(s) for s in sequences])
    assert len(fixed_len) == 1
    fixed_len = fixed_len.pop()
    per_pos_counts = custom_metrics.per_position_aa_count(sequences)

    fig, ax = plt.subplots(dpi=SAVEFIG_DPI)
    bottom = np.zeros(fixed_len)
    for i in range(len(ft.AMINO_ACIDS)):
        ax.bar(
            np.arange(fixed_len),
            per_pos_counts.values[:, i],
            bottom=bottom,
            label=ft.AMINO_ACIDS[i],
        )
        bottom += per_pos_counts.values[:, i]
    ax.set_xticks(np.arange(fixed_len))
    ax.set_xticklabels(np.arange(fixed_len) + 1)
    ax.set(title=title, xlabel=xlabel)
    if fname:
        fig.savefig(fname, bbox_inches="tight")
    return fig


def plot_anndata_rep(
    a: AnnData,
    color: str,
    representation: str = "umap",
    representation_axes_label: str = "",
    swap_axes: bool = False,
    cmap: Callable = plt.get_cmap("tab20"),
    direct_label: bool = True,
    adjust: bool = False,
    ax_tick: bool = False,
    title: str = "Title of Figure",
    legend_size: Optional[int] = None,
    figsize: Tuple[float, float] = (6.4, 4.8),
    fname: str = "",
    **kwargs,
):
    """
    Plot the given adata's representation, directly labelling instead of using
    a legend
    """
    rep_key = "X_" + representation
    assert (
        rep_key in a.obsm
    ), f"Representation {representation} not fount in keys {a.obsm.keys()}"

    coords = a.obsm[rep_key]
    if swap_axes:
        coords = coords[:, ::-1]  # Reverse the columns
    assert isinstance(coords, np.ndarray) and len(coords.shape) == 2
    assert coords.shape[0] == a.n_obs
    assert color in a.obs
    color_vals = a.obs[color]
    unique_val = np.unique(color_vals.values)
    #unique_val = color_vals.values.categories.values # for appearance bins
    color_idx = [sorted(list(unique_val)).index(i) for i in color_vals]
    #color_idx = [list(unique_val).index(i) for i in color_vals] # for appearance bins
    # Vector of colors for each point
    color_vec = [cmap(i) for i in color_idx]

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.scatter(
        coords[:, 0], coords[:, 1], s=12000 / coords.shape[0], c=color_vec, alpha=0.9
    )

    if direct_label:
        # Label each cluster
        texts = []
        for v in unique_val:
            v_idx = np.where(color_vals.values == v)
            # Median behaves better with outliers than mean
            v_coords = np.median(coords[v_idx], axis=0)
            t = ax.text(
                *v_coords,
                v,
                horizontalalignment="center",
                verticalalignment="center",
                size=legend_size,
            )
            texts.append(t)
        if adjust:
            adjust_text(
                texts,
                only_move={"texts": "y"},
                force_text=0.01,
                autoalign="y",
                avoid_points=False,
            )
    else:
        patches = []
        for i, val in enumerate(unique_val):
            p = mpatches.Patch(color=cmap(i), label=val)
            patches.append(p)
        ax.legend(handles=patches, prop={"size": legend_size})

    rep_str = representation_axes_label if representation_axes_label else representation
    if not swap_axes:
        ax.set(
            xlabel=f"{rep_str.upper()}1", ylabel=f"{rep_str.upper()}2")
    else:
        ax.set(
            xlabel=f"{rep_str.upper()}2", ylabel=f"{rep_str.upper()}1")
    ax.title.set_text(title)
    ax.set(**kwargs)
    if not ax_tick:
        ax.set(xticks=[], yticks=[])

    if fname:
        fig.savefig(fname, bbox_inches="tight")
    return fig
