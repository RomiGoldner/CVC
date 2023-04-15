import os, sys
import logging
import json
import itertools
import pandas as pd
import argparse
from typing import *
import numpy as np
import torch
import torch.nn as nn

from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    BertConfig,
    BertForMaskedLM,
    ConvBertConfig,
    ConvBertForMaskedLM,
    EvalPrediction,
)
import git

import cvc.data_loader as dl
import cvc.model_utils as model_utils
import cvc.featurization as ft
import cvc.custom_metrics as custom_metrics
import cvc.utils as utils

BLOSUM = custom_metrics.load_blosum()


def compute_metrics(pred: EvalPrediction, top_n: int = 3) -> Dict[str, float]:
    """
    Compute metrics to report
    top_n controls the top_n accuracy reported
    """
    # labels are -100 for masked tokens and value to predict for masked token
    labels = pred.label_ids.squeeze()  # Shape (n, 47)
    preds = pred.predictions  # Shape (n, 47, 26)

    n_mask_total = 0
    top_one_correct, top_n_correct = 0, 0
    blosum_values = []
    for i in range(labels.shape[0]):
        masked_idx = np.where(labels[i] != -100)[0]
        n_mask = len(masked_idx)  # Number of masked items
        n_mask_total += n_mask
        pred_arr = preds[i, masked_idx]
        truth = labels[i, masked_idx]  # The masked token indices
        # argsort returns indices in ASCENDING order
        pred_sort_idx = np.argsort(pred_arr, axis=1)  # apply along vocab axis
        # Increments by number of correct in top 1
        top_one_correct += np.sum(truth == pred_sort_idx[:, -1])
        top_n_preds = pred_sort_idx[:, -top_n:]
        for truth_idx, top_n_idx in zip(truth, top_n_preds):
            # Increment top n accuracy
            top_n_correct += truth_idx in top_n_idx
            # Check BLOSUM score
            truth_res = ft.AMINO_ACIDS_WITH_ALL_ADDITIONAL[truth_idx]
            pred_res = ft.AMINO_ACIDS_WITH_ALL_ADDITIONAL[top_n_idx[-1]]
            if truth_res in BLOSUM.index and pred_res in BLOSUM.index:
                blosum_values.append(BLOSUM.loc[truth_res, pred_res])
    # These should not exceed each other
    assert top_one_correct <= top_n_correct <= n_mask_total
    retval = {
        f"top_{top_n}_acc": top_n_correct / n_mask_total,
        "acc": top_one_correct / n_mask_total,
        "average_blosum": np.mean(blosum_values),
    }
    return retval


def load_data(identifier: str) -> List[str]:
    if identifier == "CUSTOM_DATASET":
        tab = dl.load_custom_dataset()
        logging.info(f"CUSTOM_DATASET: {len(tab['Sequences'])} TRA/TRB sequences")
        return list(tab["Sequences"])
    else:
        raise ValueError(f"Unrecognized identifier: {identifier}")


def get_cvc_bert_model(bert_variant: str, **kwargs) -> nn.Module:
    """Load the bert model"""
    if bert_variant == "bert":
        logging.info("Loading vanilla BERT model")
        config = BertConfig(
            **kwargs,
            vocab_size=len(ft.AMINO_ACIDS_WITH_ALL_ADDITIONAL),
            pad_token_id=ft.AMINO_ACIDS_WITH_ALL_ADDITIONAL_TO_IDX[ft.PAD],
        )
        model = BertForMaskedLM(config)
    elif bert_variant == "convbert":
        logging.info("Loading ConvBERT model")
        config = ConvBertConfig(
            **kwargs,
            vocab_size=len(ft.AMINO_ACIDS_WITH_ALL_ADDITIONAL),
            pad_token_id=ft.AMINO_ACIDS_WITH_ALL_ADDITIONAL_TO_IDX[ft.PAD],
        )
        model = ConvBertForMaskedLM(config)
    else:
        raise ValueError(f"Unrecognized BERT variant: {bert_variant}")
    return model


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--pathdata",
        required=False,
        help="Path to dataset for training",
    )
    parser.add_argument(
        "-d",
        "--datasets",
        required=False,
        choices=dl.DATASET_NAMES,
        nargs="*",
        help="Datasets to train on",
    )
    parser.add_argument(
        "-b",
        "--bert",
        type=str,
        default="bert",
        choices=["bert", "convbert"],
        help="BERT variant to train",
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Model config json"
    )
    parser.add_argument(
        "-o", "--outdir", type=str, required=True, help="Output directory"
    )
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Max epochs")
    parser.add_argument("--bs", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup", type=float, default=0.0, help="Warmup ratio")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU for debugging")
    parser.add_argument(
        "--holdout",
        type=float,
        default=0.1,
        help="Proportion of data to hold out for eval",
    )
    parser.add_argument(
        "--noneptune", action="store_true", help="Disable neptune logging"
    )
    return parser


def main():
    """Run the script"""
    args = build_parser().parse_args()
    if not os.path.exists(args.outdir):
        logging.info(f"Creating output directory: {args.outdir}")
        os.makedirs(args.outdir)

    # Setup logging
    logger = logging.getLogger()
    fh = logging.FileHandler(os.path.join(args.outdir, "training.log"), "w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # Log CLI invocation
    logging.info(f"Commandline invocation: {' '.join(sys.argv)}")

    # Log git status
    try:
        repo = git.Repo(
            path=os.path.dirname(os.path.abspath(__file__)), search_parent_directories=True
        )
        sha = repo.head.object.hexsha
        logging.info(f"Git commit: {sha}")
    except git.InvalidGitRepositoryError:
        logging.info("Git commit: N/A")

    # Log torch version and params
    with open(os.path.join(args.outdir, "params.json"), "w") as sink:
        json.dump(vars(args), sink, indent=4)
    if torch.cuda.is_available():
        logging.info(f"PyTorch CUDA version: {torch.version.cuda}")
    for arg in vars(args):
        logging.info(f"Parameter {arg}: {getattr(args, arg)}")

    train_dataset = dl.TcrABSingleCellSupervisedDataset(args.pathdata)
    tok = train_dataset.tokenizer
    tok.save_pretrained(args.outdir)

    test_dataset = None
    if args.holdout is not None:
        assert 0.0 < args.holdout < 1.0, "Must hold out a fractional proportion of data"
        # No validation since we aren't really doing MLM elsewhere
        test_dataset = dl.DatasetSplit(
            train_dataset, split="test", valid=0, test=args.holdout
        )
        train_dataset = dl.DatasetSplit(
            train_dataset, split="train", valid=0, test=args.holdout
        )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tok, mlm=True, mlm_probability=0.15
    )

    # https://huggingface.co/transformers/main_classes/trainer.html
    training_args = TrainingArguments(
        output_dir=args.outdir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup,
        evaluation_strategy="epoch" if args.holdout else "no",
        per_device_eval_batch_size=args.bs,
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        dataloader_num_workers=8,
        load_best_model_at_end=True,
        no_cuda=args.cpu,  # Useful for debugging
        skip_memory_metrics=True,
        disable_tqdm=True,
        logging_dir=os.path.join(args.outdir, "logs"),
    )
    
    
    params = utils.load_json_params(args.config)
    model = get_cvc_bert_model(args.bert, **params)

    # Set up neptune logger (this later because we want to also log params for model)
    neptune_logger = None
    if not args.noneptune:
        import neptune
        neptune.init(project_qualified_name="wukevin/cvc")
        experiment = neptune.create_experiment(
            name=f"selfsupervised-{args.bert}",
            params={
                "datasets": args.datasets,
                "epochs": args.epochs,
                "batch_size": args.bs,
                "lr": args.lr,
                "warmup_ratio": args.warmup,
                **params,
            },
            tags=[args.bert, "selfsupervised", "mlm"],
        )
        neptune_logger = model_utils.NeptuneHuggingFaceCallback(experiment)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,  # Defaults to None, see above
        callbacks=[neptune_logger] if neptune_logger is not None else None,
        compute_metrics=compute_metrics,
        # disable neptune logging
    )
    # disable neptune logging
    for i, callback in enumerate(trainer.callback_handler.callbacks):
        import transformers
        if isinstance(callback, transformers.integrations.NeptuneCallback):
            trainer.callback_handler.callbacks.pop(i)
            break
    trainer.train()
    trainer.save_model(args.outdir)


if __name__ == "__main__":
    main()
