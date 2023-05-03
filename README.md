# Count Von Count Models (CVC and scCVC)

CVC is a language model trained on CDR3 T-cell Receptor (TCR) sequences, built using a lightly modified BERT architecture with tweaked pre-training objectives. The model creates meaningful embeddings that can be used for downstream tasks like classification.

scCVC is an updated version of CVC that was trained on single-cell TCR sequences (each single cell was represented by its CDR3 sequences). The model was trained using the same architecture as CVC. It creates meaningful embeddings both for single sequences and single cells.

<p align="center">
<img src="https://github.com/RomiGoldner/CVC/blob/main/CVC-scCVC_pipeline.png" width="300" height="380">
</p>

## Installation and Usage

To install CVC/scCVC:
1. Clone the GitHub repository and create its requisite conda environment as follows.<br />
   Make sure you use a recent conda version, e.g. version=4.10 or above

```bash
conda env create -n my_env_name_py39 python=3.9 --file=environment.yml

conda activate my_env_name_py39
```

2. Upload model into the project base dir.
   The model is shared with a google drive link and can be downloaded using 'gdown'.

```bash
# install gdown
pip install --upgrade --no-cache-dir gdown

# run script to download model
# CVC
python -m scripts.download_cvc --model_type CVC
# scCVC
python -m scripts.download_cvc --model_type scCVC
```

3. To create embeddings: <br />
   a. open notebook: [lab_notebooks/create_embeddings.ipynb](https://github.com/RomiGoldner/CVC/blob/b5434f6ce4419a4dfda299b104064747f0672215/lab_notebooks/create_embeddings.ipynb) <br />
   b. edit cells under "Specify Parameters" with relative paths and model <br />
   c. run notebook <br />
   
## Train Model
The data used to train each model are shared with a google drive link and can be downloaded using the following commands:
```bash
# install gdown
pip install --upgrade --no-cache-dir gdown

# run script to download data
# CVC
python -m scripts.download_cvc_training_data --model_type CVC
# scCVC
python -m scripts.download_cvc_training_data --model_type scCVC
```

To train the model on your own set of sequences, use the '--data_path' flag and give it the data file's path.
```bash
# CVC
# train CVC with default dataset
python3 bin/train_cvc.py --epochs 50 --bs 1024 --noneptune --datasets CUSTOM_DATASET --config ./model_configs/bert_defaults.json --outdir ./output_dir/

# train CVC with custom dataset
python3 bin/train_cvc.py --epochs 50 --bs 1024 --noneptune --datasets CUSTOM_DATASET --config ./model_configs/bert_defaults.json --outdir ./output_dir/ --data_path PATH_TO_CSV

# scCVC
# train scCVC model with default dataset
python -m bin.train_sc_cvc --epochs 50 --bs 128 --noneptune --pathdata ./scDATA_ready_for_training.csv --config ./model_configs/bert_defaults.json --outdir ./output_dir/

# train scCVC model with custom dataset
# use the preprocess_scData_for_training.ipynb notebook to preprocess your data
python -m bin.train_sc_cvc --epochs 50 --bs 128 --noneptune --pathdata PATH_TO_CSV --config ./model_configs/bert_defaults.json --outdir ./output_dir/
```
## Notebooks
The main notebooks used in the paper are under the lab_notebooks folder and single_cell_research folder. <br />

The [lab_notebooks]() folder contains notebooks that are used to create the embeddings (mostly using CVC), analyze TCR data and run several downstream tasks. <br />
Some of the more useful notebooks are: <br />
- [lab_notebooks/create_embeddings.ipynb]() is used to create the embeddings (using either CVC ot scCVC) for a given dataset. <br />
- [lab_notebooks/binary_classifiers.ipynb]() is used to run a binary classification (Public/Private or MAIT) on a given dataset. <br />
- [lab_notebooks/model_train_test_data_creation.ipynb]() is used to create the train/test data for training CVC or other classification tasks. <br />
- [lab_notebooks/Private_Public_labeling.ipynb]() is used to label the data as Public/Private. <br />
- [lab_notebooks/plot_embeddings_MAIT.ipynb]() is used to label the data as MAIT and analyze it. <br />

The rest of the notebooks can be used to re-create the plots displayed in the paper, or new plots the given data. <br />


The [single_cell_research]() folder contains notebooks that are useful for single cell data analysis and creation of embeddings using scCVC. <br />
- [single_cell_research/preprocess_scData_for_training.ipynb]() is used to preprocess the single cell data for training scCVC. <br />
- [single_cell_research/calc_psuedo_perplexity.ipynb]() is used to calculate the psuedo perplexity of the sequences in a given dataset. <br />
- [single_cell_research/embeddings_sc_data.ipynb]() is used to create and plot the embeddings for single cells (concatenated representation). <br />
- [single_cell_research/sequencing_10x_data_scCVC.ipynb]() is used to CDR3 sequencign - extract the CDR3 sequences from the RNA sequence using the Psuedo-Perplexity score. <br />

