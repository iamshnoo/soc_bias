# Data source : https://aclanthology.org/2020.sltu-1.49.pdf

import os
import urllib.request
import zipfile
import shutil


def download_glove_hindi_model():
    os.makedirs("glove_test", exist_ok=True)
    os.makedirs("glove_test/hi", exist_ok=True)
    glove_zip_path = "glove_test/hi/glove_archive.zip"

    if not os.path.isfile(glove_zip_path):
        glove_zip_url = "https://github.com/iamshnoo/soc_bias/releases/download/v1.0.1/glove_archive.zip"
        urllib.request.urlretrieve(glove_zip_url, glove_zip_path)

    with zipfile.ZipFile(glove_zip_path, "r") as zip_ref:
        zip_ref.extractall("glove_test/hi")

    # shutil.rmtree(path="glove_test/hi/__MACOSX", ignore_errors=True)


if __name__ == "__main__":
    download_glove_hindi_model()
