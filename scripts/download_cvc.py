# Downloads the CVC model from Google drive and returns the path to the model.

import os
import gdown

def download_cvc_model():
    # download the model
    # TODO(P0): change the url
    dir_url = "https://drive.google.com/uc?id=1tZ8z1Z1J1yC0X9XWt0oMv5f5z5w7V5mG"
    # directory to save the model
    output = 'output_5mil_even_priv_pub'
    # create the directory if it does not exist
    if not os.path.exists(output):
        os.makedirs(output)

    # download the whole directory from dir_url
    gdown.download(dir_url, output, quiet=False)
