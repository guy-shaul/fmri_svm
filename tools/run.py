import argparse
import json
from data_loader.pre_process import build_data_dict
from model.svm_classifier import SVMClassifier
import warnings
import shura


warnings.filterwarnings("ignore", category=UserWarning)

# Setup logger
log = shura.get_logger(name= "Run (Tool)", level="DEBUG", to_file= False, filename="run.log", file_format="log")


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def override_config_with_args(config, args):
    # Only override if args are not None
    for key in config:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Run SVM on a single pkl file")

    parser.add_argument('--config_path', type=str, default='tools/params_config.json', help='Path to parameters config file')

    # Optional CLI overrides
    parser.add_argument('--NET', type=str)
    parser.add_argument('--SUB_AREA', type=str)
    parser.add_argument('--idx', type=int)
    parser.add_argument('--H', type=str)
    parser.add_argument('--slice', type=str)
    parser.add_argument('--dur', type=int)
    parser.add_argument('--offset', type=int)
    parser.add_argument('--z_norm', type=lambda x: x.lower() == 'true')
    parser.add_argument('--is_rest', type=int)
    parser.add_argument('--kernel', type=str)
    parser.add_argument('--scale', type=lambda x: x.lower() == 'true')
    parser.add_argument('--k_folds', type=int)

    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config_path)
    config = override_config_with_args(config, args)

    log.info("Running Model")

    svm_dict = build_data_dict(
        directory=config["directory"],
        NET=config["NET"],
        SUB_AREA=config["SUB_AREA"],
        idx=config["idx"],
        H=config["H"],
        slice=config["slice"],
        dur=config["dur"],
        offset=config["offset"],
        z_norm=config["z_norm"],
        is_rest=config["is_rest"]
    )

    if "data" in svm_dict:
        X, y = svm_dict["data"]
        clf = SVMClassifier(
            kernel=config["kernel"],
            scale=config["scale"],
            k_folds=config["k_folds"]
        )
        clf.train_and_evaluate(X, y)
    else:
        log.error("Failed to load data from pickle file.")


if __name__ == '__main__':
    main()
