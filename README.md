# Count Von Count (CVC)

CVC is a language model trained on CDR3 T-cell receptor sequences, built using a lightly modified BERT architecture with tweaked pre-training objectives. The model creates meaningful embeddings that can be used for downstream tasks like classification.

<p align="center">
<img src="https://github.com/RomiGoldner/CVC/blob/d91d7dfaaaae240393a32ba05cfda2dfc327f4e5/cvc_pipeline.png" width="500" height="350">
</p>

## Installation and Usage

To install CVC:
1. Clone the GitHub repository and create its requisite conda environment as follows.<br />
   Make sure you use a recent conda version, e.g. version=4.10 or above

```bash
conda env create -n my_env_name_py39 python=3.9 --file=environment.yml
```

2. Upload model folder into the project base dir. Place the downloaded folder in the root of the project.
   The model is shared with a google drive link and can be downloaded using 'gdown'.

```bash
# install gdown
pip install --upgrade --no-cache-dir gdown

# run script to download model
python -m scripts.download_cvc
```

3. To create embeddings: <br />
   a. open notebook: [lab_notebooks/create_embeddings.ipynb](https://github.com/RomiGoldner/CVC/blob/b5434f6ce4419a4dfda299b104064747f0672215/lab_notebooks/create_embeddings.ipynb) <br />
   b. edit first cell with relative paths <br />
   c. run notebook <br />
   
## Train Model 
To train the model on your own set of sequences, first upload your csv data file using:
```bash
- rsync -arvP PATH_TO_DATA MACHINE NAME:~/cvc/data/custom_dataset/
```
Then train model by running:
```bash
python3 bin/selfsupervised_transformer.py --epochs 50 --bs 1024 --noneptune --datasets CUSTOM_DATASET --config ./model_configs/bert_defaults.json --outdir ./output_dir/
```
