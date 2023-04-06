# Data source : https://aclanthology.org/2020.sltu-1.49.pdf

import os
import urllib.request
import zipfile
import shutil
import json


def download_elmo_hindi_model():
    os.makedirs("elmo_models", exist_ok=True)
    os.makedirs("elmo_models/hi", exist_ok=True)
    elmo_zip_path = "elmo_models/hi/hi.zip"

    if not os.path.isfile(elmo_zip_path):
        elmo_zip_url = "https://www.cfilt.iitb.ac.in/~diptesh/embeddings/monolingual/contextual/hi.zip"
        urllib.request.urlretrieve(elmo_zip_url, elmo_zip_path)

    with zipfile.ZipFile(elmo_zip_path, "r") as zip_ref:
        zip_ref.extractall("elmo_models")

    shutil.move("elmo_models/hi/elmo/hi-d512-elmo.hdf5", "elmo_models/hi/model.hdf5")
    shutil.move(
        "elmo_models/hi/elmo/hi-d512-options.json", "elmo_models/hi/options.json"
    )
    shutil.move("elmo_models/hi/elmo/hi-d512-vocab.txt", "elmo_models/hi/vocab.txt")

    shutil.rmtree("elmo_models/hi/elmo")


if __name__ == "__main__":
    download_elmo_hindi_model()

    ELMO_OPTIONS_FILE = "elmo_models/hi/options.json"
    with open(ELMO_OPTIONS_FILE, "r") as f:
        options = json.load(f)
    options["n_characters"] = 262
    options["char_cnn"]["n_characters"] = 262
    with open(ELMO_OPTIONS_FILE, "w") as f:
        json.dump(options, f, indent=4)
