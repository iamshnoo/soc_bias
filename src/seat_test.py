# Refers : https://github.com/McGill-NLP/bias-bench

from seat_bench.seat import SEATRunner
from seat_bench.experiment_id import generate_experiment_id

import argparse
import json
import os

DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description="Runs SEAT benchmark.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=DIRECTORY,
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--tests",
    action="store",
    nargs="*",
    help="List of SEAT tests to run. Test files should be in `data_dir` and have "
    "corresponding names with extension .jsonl.",
)
parser.add_argument(
    "--n_samples",
    action="store",
    type=int,
    default=1000,
    help="Number of permutation test samples used when estimating p-values "
    "(exact test is used if there are fewer than this many permutations).",
)
parser.add_argument(
    "--parametric",
    action="store_true",
    help="Use parametric test (normal assumption) to compute p-values.",
)
parser.add_argument(
    "--mode",
    action="store",
    type=str,
    default="lang_spec",
    choices=["lang_spec", "trans"],
    help="Use language specific word lists or translation based word lists.",
)
parser.add_argument(
    "--seed",
    action="store",
    type=int,
    default=0,
    help="Random seed used for reproducibility.",
)
parser.add_argument(
    "--embedding_model",
    action="store",
    type=str,
    default="fasttext",
    choices=["fasttext", "glove", "elmo"],
    help="Embedding model to use.",
)


if __name__ == "__main__":
    args = parser.parse_args()
    args.persistent_dir = DIRECTORY
    args.tests = None
    args.n_samples = 1000
    args.parametric = False
    args.mode = "lang_spec"
    args.seed = 0
    args.embedding_model = "glove"

    experiment_id = generate_experiment_id(
        name=f"seat_all_{args.mode}_{args.embedding_model}"
    )

    print("Running SEAT benchmark:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - tests: {args.tests}")
    print(f" - n_samples: {args.n_samples}")
    print(f" - parametric: {args.parametric}")
    print(f" - seed: {args.seed}")
    print(f" - mode: {args.mode}")
    print(f" - embedding_model: {args.embedding_model}")

    runner = SEATRunner(
        experiment_id=experiment_id,
        tests=args.tests,
        data_dir=f"{args.persistent_dir}/data/seat/hi/{args.mode}",
        n_samples=args.n_samples,
        parametric=args.parametric,
        seed=args.seed,
        embedding_model=args.embedding_model,
    )
    results = runner()
    print(results)

    os.makedirs(f"{args.persistent_dir}/results/seat/hi/{args.mode}", exist_ok=True)
    with open(
        f"{args.persistent_dir}/results/seat/hi/{args.mode}/{experiment_id}.json", "w"
    ) as f:
        json.dump(results, f, indent=4)
