# Refers : https://github.com/McGill-NLP/bias-bench


def generate_experiment_id(
    name,
    bias_type=None,
    seed=None,
):
    experiment_id = f"{name}"
    if isinstance(bias_type, str):
        experiment_id += f"_t-{bias_type}"
    if isinstance(seed, int):
        experiment_id += f"_s-{seed}"

    return experiment_id
