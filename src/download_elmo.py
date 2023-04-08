# Data source : https://aclanthology.org/2020.sltu-1.49.pdf

import os
import urllib.request
import zipfile


def download_elmo_hindi_model():
    os.makedirs("elmo_models", exist_ok=True)
    os.makedirs("elmo_models/hi", exist_ok=True)
    elmo_zip_path = "elmo_models/hi/elmo_archive.zip"

    if not os.path.isfile(elmo_zip_path):
        elmo_zip_url = "https://github.com/iamshnoo/soc_bias/releases/download/v1.0.1/elmo_archive.zip"
        urllib.request.urlretrieve(elmo_zip_url, elmo_zip_path)

    with zipfile.ZipFile(elmo_zip_path, "r") as zip_ref:
        zip_ref.extractall("elmo_models/hi")

if __name__ == "__main__":
    download_elmo_hindi_model()
