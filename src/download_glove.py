# Data source : https://aclanthology.org/2020.sltu-1.49.pdf

import os
import urllib.request
import zipfile
import shutil
import json


def download_glove_hindi_model():
    os.makedirs("glove_models", exist_ok=True)
    os.makedirs("glove_models/hi", exist_ok=True)
    glove_zip_path = "glove_models/hi/hi.zip"

    if not os.path.isfile(glove_zip_path):
        glove_zip_url = "https://www.cfilt.iitb.ac.in/~diptesh/embeddings/monolingual/non-contextual/hi.zip"
        urllib.request.urlretrieve(glove_zip_url, glove_zip_path)

    with zipfile.ZipFile(glove_zip_path, "r") as zip_ref:
        zip_ref.extractall("glove_models")

        # only keep 50/glove, 300/glove
        shutil.rmtree("glove_models/hi/50/cbow")
        shutil.rmtree("glove_models/hi/50/fasttext")
        shutil.rmtree("glove_models/hi/50/sg")
        shutil.rmtree("glove_models/hi/300/cbow")
        shutil.rmtree("glove_models/hi/300/fasttext")
        shutil.rmtree("glove_models/hi/300/sg")

        shutil.rmtree("glove_models/hi/100")
        shutil.rmtree("glove_models/hi/200")


if __name__ == "__main__":
    download_glove_hindi_model()
