# Downloads the CVC model from Google drive and returns the path to the model.
import os
import gdown

from lab_notebooks.utils import get_project_root


def download_cvc_model():
    # download the model
    dir_url = "https://drive.google.com/drive/folders/1R4sm0TLl6sHB4Ge4o2m0pPasKhva3T8s?usp=sharing"
    # directory to save the model
    output = 'output_5mil_even_priv_pub'

    # Make sure we're at the root of the project
    assert os.getcwd() == get_project_root(), f"Current working directory is " \
        f"{os.getcwd()}, but should be {get_project_root()}. rerun from root of project."

    # download the whole directory from dir_url
    gdown.download_folder(dir_url,  quiet=False)

if __name__ == "__main__":
    download_cvc_model()