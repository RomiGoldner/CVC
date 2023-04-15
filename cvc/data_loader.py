"""
Code for loading data
"""

import os, sys
import shutil
import argparse
import functools
import multiprocessing
import gzip
import inspect
import glob
import json
import itertools
import collections
import logging
from typing import *

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import random

import cvc.featurization as ft
import cvc.utils as utils

LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
assert os.path.isdir(LOCAL_DATA_DIR)

EXTERNAL_EVAL_DIR = os.path.join(os.path.dirname(LOCAL_DATA_DIR), "external_eval")
assert os.path.join(EXTERNAL_EVAL_DIR)

# Names of datasets
DATASET_NAMES = {"CUSTOM_DATASET"} # TODO(P2): add hd5 support


logging.basicConfig(level=logging.INFO)

SEP = "|"

class TcrABSupervisedIdxDataset(Dataset):
    """Dataset that returns TcrAB and label"""

    def __init__(
        self,
        tcr_table: pd.DataFrame,
        label_col: str = "tetramer",
        pos_labels: Collection[str] = ["TetMid", "TetPos"],
        idx_encode: bool = False,
        max_a_len: Optional[int] = None,
        max_b_len: Optional[int] = None,
        disambiguate_labels: bool = True,
    ):
        self.table = tcr_table
        self.label_col = label_col

        if disambiguate_labels:
            logging.info("Deduping and removing examples with conflicting labels")
            lcmv_dedup_ab, self.labels = dedup_lcmv_table(tcr_table)
            self.tras, self.trbs = zip(*lcmv_dedup_ab)
        else:
            raise NotImplementedError(
                "Running withough disambiguating labels causes duplicated and conflicting labels! This was the prior behavior, but is now deprecated"
            )

        tcr_a_lengths = [len(self.get_ith_tcr_a(i)) for i in range(len(self))]
        tcr_b_lengths = [len(self.get_ith_tcr_b(i)) for i in range(len(self))]
        self.max_a_len = max(tcr_a_lengths) if max_a_len is None else max_a_len
        self.max_b_len = max(tcr_b_lengths) if max_b_len is None else max_b_len
        self.idx_encode = idx_encode
        logging.info(f"Maximum TCR A/B lengths: {self.max_a_len} {self.max_b_len}")

        self.pos_labels = pos_labels
        logging.info(f"Positive {label_col} labels: {pos_labels}")

    def __len__(self) -> int:
        return len(self.labels)

    def get_ith_tcr_a(self, idx: int, pad: bool = False) -> str:
        """Gets the ith TRA sequence"""
        seq = self.tras[idx]
        if pad:
            seq = ft.pad_or_trunc_sequence(seq, self.max_a_len, right_align=False)
        return seq

    def get_ith_tcr_b(self, idx: int, pad: bool = False) -> str:
        """Gets the ith TRB sequence"""
        seq = self.trbs[idx]
        if pad:
            seq = ft.pad_or_trunc_sequence(seq, self.max_b_len, right_align=False)
        return seq

    def get_ith_sequence(self, idx: int) -> Tuple[str, str]:
        """Get the ith TRA/TRB pair"""
        return self.tras[idx], self.trbs[idx]

    def get_ith_label(self, idx: int, idx_encode: Optional[bool] = None) -> np.ndarray:
        """Get the ith label"""
        label = self.labels[idx]
        retval = float(np.any([l in label for l in self.pos_labels]))
        retval = np.array([1.0 - retval, retval], dtype=np.float32)
        idx_encode = self.idx_encode if idx_encode is None else idx_encode
        if idx_encode:
            retval = np.where(retval)[0]
        return retval

    def __getitem__(self, idx: int):
        tcr_a_idx = ft.idx_encode(self.get_ith_tcr_a(idx, pad=True))
        tcr_b_idx = ft.idx_encode(self.get_ith_tcr_b(idx, pad=True))

        label = self.get_ith_label(idx)
        return (
            {
                "tcr_a": torch.from_numpy(tcr_a_idx),
                "tcr_b": torch.from_numpy(tcr_b_idx),
            },
            torch.from_numpy(label).type(torch.long).squeeze(),
        )


class TcrABSupervisedOneHotDataset(TcrABSupervisedIdxDataset):
    """Dataset that encodes tcrAB as one hot encoded vectors"""

    def __getitem__(self, idx: int):
        tcr_a_idx = ft.one_hot(self.get_ith_tcr_a(idx, pad=True))
        tcr_b_idx = ft.one_hot(self.get_ith_tcr_b(idx, pad=True))

        label = self.get_ith_label(idx)
        return (
            {
                "tcr_a": torch.from_numpy(tcr_a_idx),
                "tcr_b": torch.from_numpy(tcr_b_idx),
            },
            torch.from_numpy(label).type(torch.long).squeeze(),
        )


class TcrSelfSupervisedDataset(TcrABSupervisedIdxDataset):
    """
    Mostly for compatibility with transformers library
    LineByLineTextDataset returns a dict of "input_ids" -> input_ids
    """

    # Reference: https://github.com/huggingface/transformers/blob/447808c85f0e6d6b0aeeb07214942bf1e578f9d2/src/transformers/data/datasets/language_modeling.py
    def __init__(self, tcr_seqs: Iterable[str], tokenizer, round_len: bool = True):
        self.tcr_seqs = utils.dedup(tcr_seqs)
        logging.info(
            f"Creating self supervised dataset with {len(self.tcr_seqs)} sequences"
        )
        self.max_len = max([len(s) for s in self.tcr_seqs])
        logging.info(f"Maximum sequence length: {self.max_len}")
        if round_len:
            self.max_len = int(utils.min_power_greater_than(self.max_len, 2))
            logging.info(f"Rounded maximum length to {self.max_len}")
        self.tokenizer = tokenizer
        self._has_logged_example = False

    def __len__(self) -> int:
        return len(self.tcr_seqs)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        tcr = self.tcr_seqs[i]
        retval = self.tokenizer.encode(ft.insert_whitespace(tcr))
        if not self._has_logged_example:
            logging.info(f"Example of tokenized input: {tcr} -> {retval}")
            self._has_logged_example = True
        return {"input_ids": torch.tensor(retval, dtype=torch.long)}

    def merge(self, other):
        """Merge this dataset with the other dataset"""
        all_tcrs = utils.dedup(self.tcr_seqs + other.tcr_seqs)
        logging.info(
            f"Merged two self-supervised datasets of sizes {len(self)} {len(other)} for dataset of {len(all_tcrs)}"
        )
        return TcrSelfSupervisedDataset(all_tcrs)


# For TRA & TRB with single cell labels
class TcrABSingleCellSupervisedDataset(TcrABSupervisedIdxDataset):
    # have init function receive a path to a csv file that contains the tcrs and the samples they belong to
    def __init__(self, tcr_seqs_path: pd.DataFrame, round_len: bool = True):
        # read the file
        tcr_seqs_df = pd.read_csv(tcr_seqs_path)
        self.tcr_seqs = tcr_seqs_df["tcr_seqs"].tolist()
        logging.info(
            f"Creating self supervised dataset with {len(self.tcr_seqs)} sequences"
        )
        self.max_len = max([len(s) for s in self.tcr_seqs])
        logging.info(f"Maximum sequence length: {self.max_len}")
        if round_len:
            self.max_len = int(utils.min_power_greater_than(self.max_len, 2))
            logging.info(f"Rounded maximum length to {self.max_len}")
        self.tokenizer = ft.get_aa_bert_tokenizer(self.max_len)
        self._has_logged_example = False

    def __len__(self) -> int:
        return len(self.tcr_seqs)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        tcr = self.tcr_seqs[i]
        # generate the order of the CDR3 sequences randomly
        tcr_sub_seqs = tcr.split("|")
        random.shuffle(tcr_sub_seqs)
        tcr = "|".join(tcr_sub_seqs)

        retval = self.tokenizer.encode(ft.insert_whitespace(tcr))
        if not self._has_logged_example:
            logging.info(f"Example of tokenized input: {tcr} -> {retval}")
            self._has_logged_example = True
        return {"input_ids": torch.tensor(retval, dtype=torch.long)}


class DatasetSplit(Dataset):
    """
    Dataset split. Thin wrapper on top a dataset to provide data split functionality.
    Can also enable dynamic example generation for train fold if supported by
    the wrapped dataset (NOT for valid/test folds) via dynamic_training flag

    kwargs are forwarded to shuffle_indices_train_valid_test
    """

    def __init__(
        self,
        full_dataset: Dataset,
        split: str,
        dynamic_training: bool = False,
        **kwargs,
    ):
        self.dset = full_dataset
        split_to_idx = {"train": 0, "valid": 1, "test": 2}
        assert split in split_to_idx
        self.split = split
        self.dynamic = dynamic_training
        if self.split != "train":
            assert not self.dynamic, "Cannot have dynamic examples for valid/test"
        train_valid_test_lists = shuffle_indices_train_valid_test(
            np.arange(len(self.dset)), **kwargs
        )
        list_index = split_to_idx[self.split]
        self.idx = train_valid_test_lists[list_index]
        logging.info(f"Split {self.split} with {len(self)} examples")

    def all_labels(self, **kwargs) -> np.ndarray:
        """Get all labels"""
        if not hasattr(self.dset, "get_ith_label"):
            raise NotImplementedError("Wrapped dataset must implement get_ith_label")
        labels = [
            self.dset.get_ith_label(self.idx[i], **kwargs) for i in range(len(self))
        ]
        return np.stack(labels)

    def all_sequences(self, **kwargs) -> Union[List[str], List[Tuple[str, str]]]:
        """Get all sequences"""
        if not hasattr(self.dset, "get_ith_sequence"):
            raise NotImplementedError(
                f"Wrapped dataset {type(self.dset)} must implement get_ith_sequence"
            )
        # get_ith_sequence could return a str or a tuple of two str (TRA/TRB)
        sequences = [
            self.dset.get_ith_sequence(self.idx[i], **kwargs) for i in range(len(self))
        ]
        return sequences

    def to_file(self, fname: str, compress: bool = True) -> str:
        """
        Write to the given file
        """
        if not (
            hasattr(self.dset, "get_ith_label")
            and hasattr(self.dset, "get_ith_sequence")
        ):
            raise NotImplementedError(
                "Wrapped dataset must implement both get_ith_label & get_ith_sequence"
            )
        assert fname.endswith(".json")
        all_examples = []
        for idx in range(len(self)):
            seq = self.dset.get_ith_sequence(self.idx[idx])
            label_list = self.dset.get_ith_label(self.idx[idx]).tolist()
            all_examples.append((seq, label_list))

        with open(fname, "w") as sink:
            json.dump(all_examples, sink, indent=4)

        if compress:
            with open(fname, "rb") as source:
                with gzip.open(fname + ".gz", "wb") as sink:
                    shutil.copyfileobj(source, sink)
            os.remove(fname)
            fname += ".gz"
        assert os.path.isfile(fname)
        return os.path.abspath(fname)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, idx: int):
        if (
            self.dynamic
            and self.split == "train"
            and "dynamic" in inspect.getfullargspec(self.dset.__getitem__).args
        ):
            return self.dset.__getitem__(self.idx[idx], dynamic=True)
        return self.dset.__getitem__(self.idx[idx])


def shuffle_indices_train_valid_test(
    idx: np.ndarray, valid: float = 0.15, test: float = 0.15, seed: int = 1234
) -> Tuple[np.ndarray]:
    """
    Given an array of indices, return indices partitioned into train, valid, and test indices
    The following tests ensure that ordering is consistent across different calls
    >>> np.all(shuffle_indices_train_valid_test(np.arange(100))[0] == shuffle_indices_train_valid_test(np.arange(100))[0])
    True
    >>> np.all(shuffle_indices_train_valid_test(np.arange(10000))[1] == shuffle_indices_train_valid_test(np.arange(10000))[1])
    True
    >>> np.all(shuffle_indices_train_valid_test(np.arange(20000))[2] == shuffle_indices_train_valid_test(np.arange(20000))[2])
    True
    >>> np.all(shuffle_indices_train_valid_test(np.arange(1000), 0.1, 0.1)[1] == shuffle_indices_train_valid_test(np.arange(1000), 0.1, 0.1)[1])
    True
    """
    np.random.seed(seed)  # For reproducible subsampling
    indices = np.copy(idx)  # Make a copy because shuffling occurs in place
    np.random.shuffle(indices)  # Shuffles inplace
    num_valid = int(round(len(indices) * valid)) if valid > 0 else 0
    num_test = int(round(len(indices) * test)) if test > 0 else 0
    num_train = len(indices) - num_valid - num_test
    assert num_train > 0 and num_valid >= 0 and num_test >= 0
    assert num_train + num_valid + num_test == len(
        indices
    ), f"Got mismatched counts: {num_train} + {num_valid} + {num_test} != {len(indices)}"

    indices_train = indices[:num_train]
    indices_valid = indices[num_train : num_train + num_valid]
    indices_test = indices[-num_test:]
    assert indices_train.size + indices_valid.size + indices_test.size == len(idx)

    return indices_train, indices_valid, indices_test


def load_custom_dataset(
    #fname: str = os.path.join(LOCAL_DATA_DIR, "custom_dataset", "db_sequences_91_mil.csv.csv"),
    fname: str = os.path.join(LOCAL_DATA_DIR, "custom_dataset", "db_5mil_training_data_2.5Mpub_2.5Mpriv.csv"),
    tra_trb_only: bool = True,
    vocab_check: bool = True,
    addtl_filters: Optional[Dict[str, Iterable[str]]] = None,
    with_antigen_only: bool = False,
) -> pd.DataFrame:
    """
    Load custom dataset. 
    The amino acid sequences should be under "Sequences" column.
    If the TRA/TRB sequences are in different columns, they should be under "TRB_Sequences" or "TRA_Sequences".
    """
    
    if not tra_trb_only:
        raise NotImplementedError
    # read the csv file
    df = pd.read_csv(fname, na_values="-", low_memory=False)
    # Filter out entries that have weird characters in their aa sequences
    if vocab_check:
        ##### uncomment if there are TRA sequences / if they are separate than TRB #####
        # tra_pass = [
        #     pd.isnull(aa) or ft.adheres_to_vocab(aa) for aa in df["TRA_Sequences"]
        # ]
        ##### if TRB sequences are in a separate column - change to TRB_Sequences
        trb_pass = [
            pd.isnull(aa) or ft.adheres_to_vocab(aa) for aa in df["Sequences"]
        ]
        #both_pass = np.logical_and(tra_pass, trb_pass)
        # logging.info(
        #     f"CUSTOM_DATASET: Removing {np.sum(both_pass == False)} entires with non amino acid residues"
        # )
        #df = df.iloc[np.where(both_pass)]
        df = df.iloc[np.where(trb_pass)]
    
    if 'Locus' in df:
        logging.info(f"CUSTOM_DATASET data TRA/TRB instances: {collections.Counter(df['Locus'])}")
    # To train on smaller dataset, take first x rows of df
    #retval = df[:5000000] # First 5m public sequences
    retval = df

    ###### uncomment the sequences are in different columns ######
    ## Report metrics
    # has_tra = ~pd.isnull(df["TRB_Sequences"])
    # has_trb = ~pd.isnull(df["TRA_Sequences"])
    # has_both = np.logical_and(has_tra, has_trb)
    # logging.info(f"CUSTOM_DATASET entries with TRB sequence: {np.sum(has_tra)}")
    # logging.info(f"CUSTOM_DATASET entries with TRB sequence: {np.sum(has_trb)}")
    # logging.info(f"CUSTOM_DATASET entries with TRA and TRB:  {np.sum(has_both)}")

    return retval


def dedup_and_merge_labels(
    sequences: Sequence[str], labels: Sequence[str], sep: str = ","
) -> Tuple[List[str], List[str]]:
    """
    Remove duplicates in sequences and merge labels accordingly
    sep is the label separator, used to split and rejoin labels
    Return is sorted!

    >>> dedup_and_merge_labels(['a', 'b', 'a'], ['x', 'y', 'y'])
    (['a', 'b'], ['x,y', 'y'])
    >>> dedup_and_merge_labels(['a', 'b', 'a', 'a'], ['x', 'y', 'y,x', 'z'])
    (['a', 'b'], ['x,y,z', 'y'])
    >>> dedup_and_merge_labels(['a', 'b', 'd', 'c'], ['x', 'z', 'y', 'n'])
    (['a', 'b', 'c', 'd'], ['x', 'z', 'n', 'y'])
    """
    # unique returns the *sorted* unique elements of an array
    uniq_sequences, inverse_idx, uniq_seq_counts = np.unique(
        sequences, return_inverse=True, return_counts=True
    )
    uniq_labels, agg_count = [], 0
    # Walk through all unique sequences and fetch/merge corresponding labels
    for i, (seq, c) in enumerate(zip(uniq_sequences, uniq_seq_counts)):
        orig_idx = np.where(inverse_idx == i)[0]
        match_labels = utils.dedup([labels[i] for i in orig_idx])
        if len(match_labels) == 1:
            uniq_labels.append(match_labels.pop())
        else:  # Aggregate labels
            aggregated_labels = utils.dedup(
                list(
                    itertools.chain.from_iterable([l.split(sep) for l in match_labels])
                )
            )
            logging.debug(f"Merging {match_labels} -> {sep.join(aggregated_labels)}")
            agg_count += 1
            uniq_labels.append(sep.join(sorted(aggregated_labels)))
    assert len(uniq_sequences) == len(uniq_labels)
    logging.info(
        f"Deduped from {len(sequences)} -> {len(uniq_sequences)} merging {agg_count} labels"
    )
    return list(uniq_sequences), uniq_labels


def chunkify(x: Sequence[Any], chunk_size: int = 128):
    """
    Split list into chunks of given size
    >>> chunkify([1, 2, 3, 4, 5, 6, 7], 3)
    [[1, 2, 3], [4, 5, 6], [7]]
    >>> chunkify([(1, 10), (2, 20), (3, 30), (4, 40)], 2)
    [[(1, 10), (2, 20)], [(3, 30), (4, 40)]]
    """
    retval = [x[i : i + chunk_size] for i in range(0, len(x), chunk_size)]
    return retval

class TcrFineTuneDataset(TcrSelfSupervisedDataset):
    def __init__(
            self,
            tcr_a_seqs: Sequence[str],
            tcr_b_seqs: Sequence[str],
            labels: Optional[np.ndarray] = None,
            label_continuous: bool = False,
            tokenizer: Optional[Callable] = None,
            skorch_mode: bool = True,
            idx_encode: bool = False,
    ):
        assert len(tcr_a_seqs) == len(tcr_b_seqs)
        self.tcr_a = list(tcr_a_seqs)
        self.tcr_b = list(tcr_b_seqs)
        self.max_len = max([len(s) for s in self.tcr_a + self.tcr_b]) + 2

        if tokenizer is None:
            tokenizer = ft.get_aa_bert_tokenizer(self.max_len)
            self.tcr_a_tokenized = [
                tokenizer.encode(
                    ft.insert_whitespace(aa),
                    padding="max_length",
                    max_length=self.max_len,
                )
                for aa in self.tcr_a
            ]
            self.tcr_b_tokenized = [
                tokenizer.encode(
                    ft.insert_whitespace(aa),
                    padding="max_length",
                    max_length=self.max_len,
                )
                for aa in self.tcr_b
            ]
        else:
            logging.info(f"Using pre-supplied tokenizer: {tokenizer}")
            _label, _seq, self.tcr_a_tokenized = tokenizer(list(enumerate(self.tcr_a)))
            _label, _seq, self.tcr_b_tokenized = tokenizer(list(enumerate(self.tcr_b)))

        if labels is not None:
            assert len(labels) == len(tcr_a_seqs)
            self.labels = np.atleast_1d(labels.squeeze())
        else:
            logging.warning(
                "Labels not given, defaulting to False labels (DO NOT USE FOR TRAINING)"
            )
            self.labels = None
        self.continuous = label_continuous
        self.skorch_mode = skorch_mode
        self.idx_encode = idx_encode

    def get_ith_sequence(self, idx: int) -> Tuple[str, str]:
        """Get the ith TRA/TRB pair"""
        return self.tcr_a[idx], self.tcr_b[idx]

    def get_ith_label(self, idx: int, idx_encode: Optional[bool] = None) -> np.ndarray:
        """Get the ith label"""
        if self.labels is None:
            return np.array([0])  # Dummy value
        if not self.continuous:
            label = self.labels[idx]
            if not isinstance(label, np.ndarray):
                label = np.atleast_1d(label)
            if self.skorch_mode and len(label) == 1:
                label = np.array([1.0 - label, label]).squeeze()
            # Take given value if supplied, else default to self.idx_encode
            idx_encode = self.idx_encode if idx_encode is None else idx_encode
            if idx_encode:
                label = np.where(label)[0]
            return label
        else:
            # For the continuous case we simply return the ith value(s)
            return self.labels[idx]

    def __len__(self) -> int:
        return len(self.tcr_a)

    def __getitem__(
            self, idx: int
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        label_dtype = torch.float if self.continuous else torch.long
        tcr_a = self.tcr_a_tokenized[idx]
        tcr_b = self.tcr_b_tokenized[idx]
        label = self.get_ith_label(idx)
        if not self.skorch_mode:
            retval = {
                "tcr_a": utils.ensure_tensor(tcr_a, dtype=torch.long),
                "tcr_b": utils.ensure_tensor(tcr_b, dtype=torch.long),
                "labels": utils.ensure_tensor(label, dtype=label_dtype),
            }
        else:
            model_inputs = {
                "tcr_a": utils.ensure_tensor(tcr_a, dtype=torch.long),
                "tcr_b": utils.ensure_tensor(tcr_b, dtype=torch.long),
            }
            retval = (model_inputs, torch.tensor(label, dtype=label_dtype).squeeze())
        return retval