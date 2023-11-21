# Downloads the CVC model from Google drive and returns the path to the model.
import os
import gdown
import argparse

def download_cvc_train_data(model_type = 'CVC'):
    # download the model data according to the model type (CVC or scCVC)
    if model_type == 'CVC':
        dir_url = "https://drive.google.com/file/d/1hWP_d3NFUZVPUyiZsumPo7qNTEQl_pIY/view?usp=sharing"
    else:
        dir_url = "https://drive.google.com/file/d/1F9QOe8egJAxGaJevj8boa8PQ4QWA7W0C/view?usp=sharing"
    # directory to save the model
    output = 'data'

    # download the whole directory from dir_url
    gdown.download(dir_url,  quiet=False, use_cookies=False, fuzzy=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='CVC')
    args = parser.parse_args()
    download_cvc_train_data(args.model_type)


if __name__ == "__main__":
    main()