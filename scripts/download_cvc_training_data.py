# Downloads the CVC model from Google drive and returns the path to the model.
import os
import gdown

def download_cvc_train_data():
    # download the model
    dir_url = "https://drive.google.com/file/d/1hWP_d3NFUZVPUyiZsumPo7qNTEQl_pIY/view?usp=sharing"
    # directory to save the model
    output = 'data'

    # download the whole directory from dir_url
    gdown.download_folder(dir_url,  quiet=False)

if __name__ == "__main__":
    download_cvc_train_data()