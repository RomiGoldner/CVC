"""
Various model utils
"""

import os, sys
import tempfile
import subprocess
import json
import logging
from itertools import zip_longest
from typing import *
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd
from scipy.special import softmax

import torch
import torch.nn as nn
import skorch

from transformers import (
    AutoModel,
    BertModel,
    BertForMaskedLM,
    BertForNextSentencePrediction,
    BertForSequenceClassification,
    ConvBertForMaskedLM,
    FillMaskPipeline,
    FeatureExtractionPipeline,
    TextClassificationPipeline,
    Pipeline,
    TrainerCallback,
    TrainerControl,
)

from neptune.experiments import Experiment
from neptune.api_exceptions import ChannelsValuesSendBatchError


import cvc.data_loader as dl
import cvc.featurization as ft
from cvc import utils


class NeptuneHuggingFaceCallback(TrainerCallback):
    """
    Add Neptune support for HuggingFace transformers

    Reference:
    https://huggingface.co/transformers/_modules/transformers/integrations.html#WandbCallback
    """

    def __init__(
        self,
        experiment: Experiment,
        epoch_index: bool = True,
        blacklist_keys: Iterable[str] = [
            "train_runtime",
            "train_samples_per_second",
            "epoch",
            "total_flos",
            "eval_runtime",
            "eval_samples_per_second",
        ],
    ):
        self.experiment = experiment
        self.epoch_index = epoch_index
        self.blacklist_keys = set(blacklist_keys)

    def on_log(
        self, args, state, control: TrainerControl, model=None, logs=None, **kwargs
    ):
        """Log relevant values"""
        # Log only if main process
        if not state.is_world_process_zero:
            return

        for k, v in logs.items():
            if k not in self.blacklist_keys:
                # https://docs-legacy.neptune.ai/api-reference/neptune/experiments/index.html
                i = state.global_step if not self.epoch_index else logs["epoch"]
                try:
                    self.experiment.log_metric(k, i, v)
                except ChannelsValuesSendBatchError:
                    logging.warning(
                        f"Error sending index-value pair {k}:{v} to neptune (expected for end of transformers training)"
                    )


def get_transformer_attentions(
    model_dir: Union[str, nn.Module],
    seqs: Iterable[str],
    seq_pair: Optional[Iterable[str]] = None,
    *,
    layer: int = 0,
    batch_size: int = 256,
    device: int = 0,
) -> List[np.ndarray]:
    """Return a list of attentions"""
    device = utils.get_device(device)
    seqs = [s if ft.is_whitespaced(s) else ft.insert_whitespace(s) for s in seqs]
    if isinstance(model_dir, str):
        tok = ft.get_pretrained_bert_tokenizer(model_dir)
        model = BertModel.from_pretrained(
            model_dir, add_pooling_layer=False, output_attentions=True,
        ).to(device)
    elif isinstance(model_dir, nn.Module):
        tok = ft.get_aa_bert_tokenizer(64)
        model = model_dir.to(device)
    else:
        raise TypeError(f"Unhandled type for model_dir: {type(model_dir)}")

    chunks = dl.chunkify(seqs, batch_size)
    # This defaults to [None] to zip correctly
    chunks_pair = [None]
    if seq_pair is not None:
        assert len(seq_pair) == len(seqs)
        chunks_pair = dl.chunkify(
            [s if ft.is_whitespaced(s) else ft.insert_whitespace(s) for s in seq_pair],
            batch_size,
        )
    # For single input, we get (list of seq, None) items
    # for a duo input, we get (list of seq1, list of seq2)
    chunks_zipped = list(zip_longest(chunks, chunks_pair))
    attentions = []
    with torch.no_grad():
        for seq_chunk in chunks_zipped:
            encoded = tok(
                *seq_chunk, padding="max_length", max_length=64, return_tensors="pt"
            )
            input_mask = encoded["attention_mask"].numpy()
            encoded = {k: v.to(device) for k, v in encoded.items()}
            # encoded contains input attention mask of (batch, seq_len)
            x = model.forward(
                **encoded, output_hidden_states=True, output_attentions=True
            )
            for i in range(len(seq_chunk[0])):
                seq_len = np.sum(input_mask[i])
                # The attentions tuple has length num_layers
                # Each entry in the attentions tuple is of shape (batch, num_attn_heads, 64, 64)
                a = x.attentions[layer][i].cpu().numpy()  # (num_attn_heads, 64, 64)
                # Nonzero entries are (num_attn_heads, 64, seq_len)
                # We subset to (num_attn_heads, seq_len, seq_len)
                # as it appears that's what bertviz does
                attentions.append(a[:, :seq_len, :seq_len])
    return attentions


def get_transformer_embeddings(
    model_dir: str,
    seqs: Iterable[str],
    seq_pair: Optional[Iterable[str]] = None,
    *,
    layers: List[int] = [-1],
    method: Literal["mean", "max", "attn_mean", "cls", "pool"] = "mean",
    batch_size: int = 256,
    device: int = 0,
    pbar = False,
) -> np.ndarray:
    """
    Get the embeddings for the given sequences from the given layers
    Layers should be given as negative integers, where -1 indicates the last
    representation, -2 second to last, etc.
    Returns a matrix of num_seqs x (hidden_dim * len(layers))
    Methods:
    - cls:  value of initial CLS token
    - mean: average of sequence length, excluding initial CLS token
    - max:  maximum over sequence length, excluding initial CLS token
    - attn_mean: mean over sequenced weighted by attention, excluding initial CLS token
    - pool: pooling layer
    If multiple layers are given, applies the given method to each layers
    and concatenate across layers
    """
    device = utils.get_device(device)
    seqs = [s if ft.is_whitespaced(s) else ft.insert_whitespace(s) for s in seqs]
    try:
        tok = ft.get_pretrained_bert_tokenizer(model_dir)
    except OSError:
        logging.warning("Could not load saved tokenizer, loading fresh instance")
        tok = ft.get_aa_bert_tokenizer(64)
    model = BertModel.from_pretrained(model_dir, add_pooling_layer=method == "pool").to(
        device
    )

    chunks = dl.chunkify(seqs, batch_size)
    # This defaults to [None] to zip correctly
    chunks_pair = [None]
    if seq_pair is not None:
        assert len(seq_pair) == len(seqs)
        chunks_pair = dl.chunkify(
            [s if ft.is_whitespaced(s) else ft.insert_whitespace(s) for s in seq_pair],
            batch_size,
        )
    # For single input, we get (list of seq, None) items
    # for a duo input, we get (list of seq1, list of seq2)
    chunks_zipped = list(zip_longest(chunks, chunks_pair))
    embeddings = []
    attentions = []
    with torch.no_grad():
        for seq_chunk in tqdm(chunks_zipped, disable=not pbar):
            encoded = tok(
                *seq_chunk, padding="max_length", max_length=64, return_tensors="pt"
            )
            # manually calculated mask lengths
            # temp = [sum([len(p.split()) for p in pair]) + 3 for pair in zip(*seq_chunk)]
            input_mask = encoded["attention_mask"].numpy()
            encoded = {k: v.to(device) for k, v in encoded.items()}
            # encoded contains input attention mask of (batch, seq_len)
            x = model.forward(
                **encoded, output_hidden_states=True, output_attentions=True
            )

            if method == "pool":
                embeddings.append(x.pooler_output.cpu().numpy().astype(np.float64))
                continue
            # x.hidden_states contains hidden states, num_hidden_layers + 1 (e.g. 13)
            # Each hidden state is (batch, seq_len, hidden_size)
            # x.hidden_states[-1] == x.last_hidden_state
            # x.attentions contains attention, num_hidden_layers
            # Each attention is (batch, attn_heads, seq_len, seq_len)

            for i in range(len(seq_chunk[0])):
                e = []
                for l in layers:
                    # Select the l-th hidden layer for the i-th example
                    h = (
                        x.hidden_states[l][i].cpu().numpy().astype(np.float64)
                    )  # seq_len, hidden
                    # initial 'cls' token
                    if method == "cls":
                        e.append(h[0])
                        continue
                    # Consider rest of sequence
                    if seq_chunk[1] is None:
                        seq_len = len(seq_chunk[0][i].split())  # 'R K D E S' = 5
                    else:
                        seq_len = (
                            len(seq_chunk[0][i].split())
                            + len(seq_chunk[1][i].split())
                            + 1  # For the sep token
                        )
                    seq_hidden = h[1 : 1 + seq_len]  # seq_len * hidden
                    assert len(seq_hidden.shape) == 2
                    if method == "mean":
                        e.append(seq_hidden.mean(axis=0))
                    elif method == "max":
                        e.append(seq_hidden.max(axis=0))
                    elif method == "attn_mean":
                        # (attn_heads, seq_len, seq_len)
                        # columns past seq_len + 2 are all 0
                        # summation over last seq_len dim = 1 (as expected after softmax)
                        attn = x.attentions[l][i, :, :, : seq_len + 2]
                        # print(attn.shape)
                        print(attn.sum(axis=-1))
                        raise NotImplementedError
                    else:
                        raise ValueError(f"Unrecognized method: {method}")
                e = np.hstack(e)
                assert len(e.shape) == 1
                embeddings.append(e)
    if len(embeddings[0].shape) == 1:
        embeddings = np.stack(embeddings)
    else:
        embeddings = np.vstack(embeddings)
    del x
    del model
    torch.cuda.empty_cache()
    return embeddings


def get_esm_embedding(
    seqs: Iterable[str],
    model_key: str = "esm1b_t33_650M_UR50S",
    batch_size: int = 1,
    device: Optional[int] = 0,
) -> np.ndarray:
    """
    Get the embedding from ESM for each of the given sequences

    Resources:
    - https://doi.org/10.1073/pnas.2016239118
    - https://github.com/facebookresearch/esm
    - https://github.com/facebookresearch/esm/blob/master/examples/variant_prediction.ipynb
    """
    esm_device = utils.get_device(device)
    esm_model, esm_alphabet = torch.hub.load("facebookresearch/esm", model_key)
    esm_model = esm_model.to(esm_device)
    batch_converter = esm_alphabet.get_batch_converter()

    seqs_with_faux_labels = list(enumerate(seqs))
    labels, seqs, tokens = batch_converter(seqs_with_faux_labels)
    # Get per-base representations
    reps = []
    with torch.no_grad():
        for batch in dl.chunkify(tokens, chunk_size=batch_size):
            batch = batch.to(device)
            rep = (
                esm_model(batch, repr_layers=[33], return_contacts=True)[
                    "representations"
                ][33]
                .cpu()
                .numpy()
            )
            reps.append(rep)
            del batch  # Try to save some GPU memory
    reps = np.concatenate(reps, axis=0)

    # Get the overall sequence representations
    averaged = []
    for i, (_, seq) in enumerate(seqs_with_faux_labels):
        averaged.append(reps[i, 1 : len(seq) + 1].mean(axis=0))
    averaged = np.vstack(averaged)
    return averaged

