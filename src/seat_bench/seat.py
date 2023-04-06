# Refers : https://github.com/McGill-NLP/bias-bench

import json
import logging
import os
import random
import re

import numpy as np
from seat_bench import weat
from tqdm import tqdm
import fasttext.util

fasttext.util.download_model("hi", if_exists="ignore")

ELMO_DIR = "elmo_models/hi"
DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GLOVE_50_PATH = os.path.join(DIRECTORY, "glove_models/hi/50/glove", "hi-d50-glove.txt")
GLOVE_300_PATH = os.path.join(
    DIRECTORY, "glove_models/hi/300/glove", "hi-d300-glove.txt"
)


class WEATRunner:
    """Runs WEAT tests for a given FastText or Glove model."""

    def __init__(
        self,
        tests,
        data_dir,
        experiment_id,
        n_samples=1000,
        parametric=False,
        seed=0,
        embedding_model="glove",
    ):

        self._tests = tests
        self._data_dir = data_dir
        self._experiment_id = experiment_id
        self._n_samples = n_samples
        self._parametric = parametric
        self._seed = seed
        self._embedding_model = embedding_model
        if self._embedding_model == "glove":
            # self.hindi_glove_50 = emb_matrix_maker(GLOVE_50_PATH)
            self.hindi_glove_300 = emb_matrix_maker(GLOVE_300_PATH)

    def __call__(self):
        """Runs specified WEAT tests.

        Returns:
            `list` of `dict`s containing the WEAT test results.
        """
        # Extension for files containing WEAT tests.
        TEST_EXT = ".jsonl"

        random.seed(self._seed)
        np.random.seed(self._seed)

        all_tests = sorted(
            [
                entry[: -len(TEST_EXT)]
                for entry in os.listdir(self._data_dir)
                if not entry.startswith(".") and entry.endswith(TEST_EXT)
            ],
            key=_test_sort_key,
        )

        # Use the specified tests, otherwise, run all WEAT tests.
        tests = self._tests or all_tests

        results = []
        for test in tests:
            print(f"Running test {test}")

            # Load the test data.
            encs = _load_json(os.path.join(self._data_dir, f"{test}{TEST_EXT}"))

            if self._embedding_model == "fasttext":
                print("Computing FastText word encodings")
                encs_targ1 = _fasttext_encode(encs["targ1"]["examples"])
                encs_targ2 = _fasttext_encode(encs["targ2"]["examples"])
                encs_attr1 = _fasttext_encode(encs["attr1"]["examples"])
                encs_attr2 = _fasttext_encode(encs["attr2"]["examples"])

            elif self._embedding_model == "glove":
                print("Computing Glove word encodings")
                encs_targ1 = _glove_encode(
                    encs["targ1"]["examples"], model=self.hindi_glove_300
                )
                encs_targ2 = _glove_encode(
                    encs["targ2"]["examples"], model=self.hindi_glove_300
                )
                encs_attr1 = _glove_encode(
                    encs["attr1"]["examples"], model=self.hindi_glove_300
                )
                encs_attr2 = _glove_encode(
                    encs["attr2"]["examples"], model=self.hindi_glove_300
                )
            else:
                raise NotImplementedError("Embedding model not implemented.")

            encs["targ1"]["encs"] = encs_targ1
            encs["targ2"]["encs"] = encs_targ2
            encs["attr1"]["encs"] = encs_attr1
            encs["attr2"]["encs"] = encs_attr2

            print("\tDone!")

            # Run the test on the encodings.
            esize, pval = weat.run_test(
                encs,
                n_samples=self._n_samples,
                parametric=self._parametric,
            )

            results.append(
                {
                    "experiment_id": self._experiment_id,
                    "seed": self._seed,
                    "embedding_model": self._embedding_model,
                    "test": test,
                    "p_value": pval,
                    "effect_size": esize,
                }
            )

        return results


class SEATRunner:
    """Runs SEAT tests for a given FastText or Glove model or ELMO."""

    def __init__(
        self,
        tests,
        data_dir,
        experiment_id,
        n_samples=1000,
        parametric=False,
        seed=0,
        embedding_model="glove",
    ):
        """Initializes a SEAT test runner."""

        self._tests = tests
        self._data_dir = data_dir
        self._experiment_id = experiment_id
        self._n_samples = n_samples
        self._parametric = parametric
        self._seed = seed
        self._embedding_model = embedding_model
        if self._embedding_model == "glove":
            # self.hindi_glove_50 = emb_matrix_maker(GLOVE_50_PATH)
            self.hindi_glove_300 = emb_matrix_maker(GLOVE_300_PATH)
        if self._embedding_model == "elmo":
            from simple_elmo import ElmoModel

            logging.getLogger("simple_elmo").setLevel(logging.ERROR)
            self.model = ElmoModel()
            self.model.load(ELMO_DIR)

    def __call__(self):
        """Runs specified SEAT tests.

        Returns:
            `list` of `dict`s containing the SEAT test results.
        """
        # Extension for files containing WEAT tests.
        TEST_EXT = ".jsonl"

        random.seed(self._seed)
        np.random.seed(self._seed)

        all_tests = sorted(
            [
                entry[: -len(TEST_EXT)]
                for entry in os.listdir(self._data_dir)
                if not entry.startswith(".") and entry.endswith(TEST_EXT)
            ],
            key=_test_sort_key,
        )

        # Use the specified tests, otherwise, run all SEAT tests.
        tests = self._tests or all_tests

        results = []
        for test in tests:
            print(f"Running test {test}")

            # Load the test data.
            encs = _load_json(os.path.join(self._data_dir, f"{test}{TEST_EXT}"))

            if self._embedding_model == "fasttext":
                print("Computing FastText sentence encodings")
                encs_targ1 = _fasttext_sentence_encode(encs["targ1"]["sentences"])
                encs_targ2 = _fasttext_sentence_encode(encs["targ2"]["sentences"])
                encs_attr1 = _fasttext_sentence_encode(encs["attr1"]["sentences"])
                encs_attr2 = _fasttext_sentence_encode(encs["attr2"]["sentences"])

            elif self._embedding_model == "elmo":
                print("Computing ELMO sentence encodings")
                encs_targ1 = _elmo_sentence_encode(
                    encs["targ1"]["sentences"], self.model
                )
                encs_targ2 = _elmo_sentence_encode(
                    encs["targ2"]["sentences"], self.model
                )
                encs_attr1 = _elmo_sentence_encode(
                    encs["attr1"]["sentences"], self.model
                )
                encs_attr2 = _elmo_sentence_encode(
                    encs["attr2"]["sentences"], self.model
                )

            elif self._embedding_model == "glove":
                print("Computing Glove sentence encodings")
                encs_targ1 = _glove_sentence_encode(
                    encs["targ1"]["sentences"], model=self.hindi_glove_300
                )
                encs_targ2 = _glove_sentence_encode(
                    encs["targ2"]["sentences"], model=self.hindi_glove_300
                )
                encs_attr1 = _glove_sentence_encode(
                    encs["attr1"]["sentences"], model=self.hindi_glove_300
                )
                encs_attr2 = _glove_sentence_encode(
                    encs["attr2"]["sentences"], model=self.hindi_glove_300
                )

            else:
                raise NotImplementedError("Embedding model not implemented.")

            encs["targ1"]["encs"] = encs_targ1
            encs["targ2"]["encs"] = encs_targ2
            encs["attr1"]["encs"] = encs_attr1
            encs["attr2"]["encs"] = encs_attr2

            print("\tDone!")

            # Run the test on the encodings.
            esize, pval = weat.run_test(
                encs,
                n_samples=self._n_samples,
                parametric=self._parametric,
            )

            results.append(
                {
                    "experiment_id": self._experiment_id,
                    "seed": self._seed,
                    "embedding_model": self._embedding_model,
                    "test": test,
                    "p_value": pval,
                    "effect_size": esize,
                }
            )

        return results


def _test_sort_key(test):
    """Return tuple to be used as a sort key for the specified test name.
    Break test name into pieces consisting of the integers in the name
    and the strings in between them.
    """
    key = ()
    prev_end = 0
    for match in re.finditer(r"\d+", test):
        key = key + (test[prev_end : match.start()], int(match.group(0)))
        prev_end = match.end()
    key = key + (test[prev_end:],)

    return key


def _load_json(sent_file):
    """Load from json. We expect a certain format later, so do some post processing."""
    print(f"Loading {sent_file}...")
    all_data = json.load(open(sent_file, "r"))
    data = {}
    for k, v in all_data.items():
        examples = v["examples"]
        data[k] = examples
        v["examples"] = examples

    return all_data


def read_lines(path_to_txt, limit=500000):
    with open(path_to_txt, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            yield line.strip()


def emb_matrix_maker(path_to_txt):
    embeddings = {}
    for line in tqdm(read_lines(path_to_txt)):
        values = line.split(" ")
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings[word] = vector
    return embeddings


def _glove_encode(texts, dim=300, model=None):
    embeddings = model
    return {
        text: embeddings.get(text, np.zeros_like(embeddings[next(iter(embeddings))]))
        for text in texts
    }


def _glove_sentence_encode(texts, dim=300, model=None):
    embeddings = model
    result = {}
    for text in texts:
        words = text.split()
        vectors = [
            embeddings.get(word, np.zeros_like(embeddings[next(iter(embeddings))]))
            for word in words
        ]
        result[text] = np.mean(vectors, axis=0)
    return result


def _fasttext_encode(texts, dim=300):
    fasttext_emb = fasttext.load_model("cc.hi.300.bin")
    if dim != 300:
        fasttext_emb = fasttext.util.reduce_model(fasttext_emb, dim)
    return {text: fasttext_emb.get_word_vector(text) for text in texts}


def _fasttext_sentence_encode(texts, dim=300):
    fasttext_emb = fasttext.load_model("cc.hi.300.bin")
    if dim != 300:
        fasttext_emb = fasttext.util.reduce_model(fasttext_emb, dim)
    return {text: fasttext_emb.get_sentence_vector(text) for text in texts}


def _elmo_sentence_encode(texts, model):
    # 512 dimnsion embeddings
    print("-" * 80)
    embeddings = []
    for text in tqdm(texts):
        tokens = text.split(" ")
        vecs = model.get_elmo_vectors([tokens], layers="all")
        tok_embs = vecs[0][0]
        sent_emb = np.mean(tok_embs, axis=0)
        embeddings.append(sent_emb)

    return dict(zip(texts, embeddings))
