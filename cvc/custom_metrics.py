"""
Custom metrics
"""
import os
import functools
import json
import subprocess, shlex
import logging
import collections
import itertools
import tempfile
from typing import *

import numpy as np
import pandas as pd
from sklearn import metrics, mixture
from scipy import stats, spatial
import anndata as ad
from Bio import PDB
import logomaker

import cvc.data_loader as dl
import cvc.featurization as ft
import cvc.muscle as muscle
import cvc.utils as utils


def per_position_aa_count(
    sequences: Sequence[str], *, normalize: bool = False, psuedocount: int = 0,
) -> pd.DataFrame:
    """
    Return a count matrix of (seq_len, n_amino_acids) reflecting
    counts of each amino acid at each base
    """
    seq_mat = np.stack([np.array(list(x)) for x in sequences])
    num_seq, fixed_len = seq_mat.shape
    assert num_seq == len(sequences)
    per_pos_counts = []
    for j in range(fixed_len):
        col = seq_mat[:, j]
        assert len(col) == num_seq
        counter = collections.Counter(col)
        count_vec = np.array([counter[aa] for aa in ft.AMINO_ACIDS])
        count_vec += psuedocount
        per_pos_counts.append(count_vec)
    per_pos_counts = np.stack(per_pos_counts)
    assert per_pos_counts.shape == (seq_mat.shape[1], len(ft.AMINO_ACIDS))
    # assert np.allclose(np.sum(per_pos_counts, 1), np.sum(per_pos_counts[0]))
    if normalize:
        row_sums = set(per_pos_counts.sum(axis=1).tolist())
        assert len(row_sums) == 1
        per_pos_counts = per_pos_counts.astype(np.float64)
        per_pos_counts /= row_sums.pop()
        assert np.all(np.isclose(per_pos_counts.sum(axis=1), 1.0))
    retval = pd.DataFrame(per_pos_counts, columns=list(ft.AMINO_ACIDS))
    return retval


def motif_from_sequences(
    sequences: Iterable[str], normalize_pwm: bool = True, dedup: bool = False, **kwargs
) -> Tuple[pd.DataFrame, logomaker.Logo]:
    """
    Computes a motif for the sequences by running sequences through
    MUSCLE and visualizing using logomaker
    kwargs are forwarded to logomaker

    Returns both the per-position counts and the logo
    """
    if dedup:
        sequences = utils.dedup(sequences)
    msa_seqs = muscle.run_muscle(sequences)
    msa_pwm = per_position_aa_count(msa_seqs, normalize=normalize_pwm)
    logo = logomaker.Logo(msa_pwm, **kwargs)
    return msa_pwm, logo



def tukey_outlier_cutoffs(
    x: np.ndarray, k: int = 3, direction: str = "higher"
) -> Tuple[float, float]:
    """
    Uses tukey method to return the outliers in x
    https://en.wikipedia.org/wiki/Outlier
    Given quarties Q1, Q2 (median), and Q3, outlier cutoffs are
    [Q1 - k(Q3-Q1), Q3 + k(Q3-Q1)]
    Values of k are typically 1.5 for "outlier" and 3 for "far out"

    >>> tukey_outlier_cutoffs(np.array(list(range(10)) + [1000]))
    (-12.5, 22.5)
    """
    if direction not in ("higher", "lower", "both"):
        raise ValueError(f"Unrecognized direction: {direction}")
    q1, q3 = np.percentile(x, [25, 75])  # Q1 and Q3
    iqr = stats.iqr(x)
    assert np.isclose(q3 - q1, iqr)
    bottom_cutoff = q1 - k * iqr
    top_cutoff = q3 + k * iqr
    # idx = np.logical_or(x < bottom_cutoff, x > top_cutoff)
    # return x[idx]
    return bottom_cutoff, top_cutoff


#@functools.cache
def load_blosum(
    fname: str = os.path.join(dl.LOCAL_DATA_DIR, "blosum62.json")
) -> pd.DataFrame:
    """Return the blosum matrix as a dataframe"""
    with open(fname) as source:
        d = json.load(source)
        retval = pd.DataFrame(d)
    retval = pd.DataFrame(0, index=list(ft.AMINO_ACIDS), columns=list(ft.AMINO_ACIDS))
    for x, y in itertools.product(retval.index, retval.columns):
        if x == "U" or y == "U":
            continue
        retval.loc[x, y] = d[x][y]
    retval.drop(index="U", inplace=True)
    retval.drop(columns="U", inplace=True)
    return retval
