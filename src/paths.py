import os


def define_outputs_dir(dataset_name: str, model_version: int) -> str:
    """
    Define base output path and create if missing.
    """
    out_dir = os.path.join('outputs', dataset_name, str(model_version))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir
