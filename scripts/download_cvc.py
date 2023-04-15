# Downloads the CVC model from Google drive and returns the path to the model.
import os
import gdown
import argparse

from lab_notebooks.utils import get_project_root


def download_cvc_model(model_type = 'CVC'):
    # download the model according to the model type (CVC or scCVC)
    if model_type == 'CVC':
        dir_url = "https://drive.google.com/drive/folders/1R4sm0TLl6sHB4Ge4o2m0pPasKhva3T8s?usp=sharing"
    else:
        dir_url = "https://drive.google.com/drive/folders/12XszGPtRfpkInz5GV5RsWKBmDzL5aQZ2?usp=sharing"

    # Make sure we're at the root of the project
    assert os.getcwd() == get_project_root(), f"Current working directory is " \
        f"{os.getcwd()}, but should be {get_project_root()}. rerun from root of project."

    # download the whole directory from dir_url
    gdown.download_folder(dir_url,  quiet=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='CVC')
    args = parser.parse_args()
    download_cvc_model(args.model_type)

if __name__ == "__main__":
    main()