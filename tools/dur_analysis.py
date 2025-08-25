import os
import pickle
import numpy as np
import shura
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
from model.svm_classifier import SVMClassifier
from data_loader.pre_process import build_data_dict
from data_loader.utils import load_config

# Ignore all DeprecationWarning warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Setup Logger ---
log = shura.get_logger(name= "Duration Analysis (Tool)", level="DEBUG", to_file= False, filename="dur_analysis.log", file_format="log")


def run_dur_analysis(config, output_dir, force_run=False):

    # set durs vector
    durs = list(range(5, 66)) if config["is_rest"] == 0 else list(range(5, 19))
    H, NET, SUB_AREA, NET_idx, slice = config["H"], config["NET"],config["SUB_AREA"], config["idx"], config["slice"]

    # Set file path
    save_path = os.path.join(output_dir, f"{H}_{NET}_{SUB_AREA}_{NET_idx}_{slice}_dur_{durs[0]}_{durs[-1]}.pkl")

    # Load from file if exists
    if os.path.exists(save_path) and not force_run:
        log.info(f"Loading saved results from {save_path}")
        with open(save_path, "rb") as f:
            return durs, pickle.load(f)
    else:
        scores = []
        for dur in durs:
            log.info(f"Duration set to: {dur} TR")
            data_dict = build_data_dict(
                directory=config["directory"],
                NET=NET,
                SUB_AREA=SUB_AREA,
                idx=NET_idx,
                H=H,
                slice=slice,
                dur=dur,
                offset=config["offset"],
                z_norm=config["z_norm"],
                is_rest=config["is_rest"]
            )

            if "data" in data_dict:
                X, y = data_dict["data"]
                model = SVMClassifier(
                    kernel=config["kernel"],
                    scale=config["scale"],
                    k_folds=config["k_folds"]
                )
                accs, *_ = model.train_and_evaluate(X, y)
                scores.append(np.mean(accs))
            else:
                scores.append(0.0)  # fallback score

        # Save results
        with open(save_path, "wb") as f:
            pickle.dump(scores, f)
            log.info(f"Saved results as pickle to {save_path}")

        return durs, scores


def plot_scores(config, output_dir, durs, scores):

    # 1. Elbow Method
    distances = []
    for i in range(1, len(scores) - 1):
        a = np.array([durs[0], scores[0]])
        b = np.array([durs[-1], scores[-1]])
        p = np.array([durs[i], scores[i]])
        distance = np.linalg.norm(np.cross(b - a, p - a)) / np.linalg.norm(b - a)
        distances.append(distance)
    elbow = durs[np.argmax(distances) + 1]

    # 2. First Derivative
    slopes = np.diff(scores)
    threshold = 0.01
    first_derivative = next((durs[i+1] for i, s in enumerate(slopes) if s < threshold), durs[-1])

    # 3. Second Derivative
    second_derivative = np.diff(slopes)
    second_point = durs[np.argmin(second_derivative) + 2]

    # 4. Saturation (95%)
    max_score = max(scores)
    saturation_score = 0.95 * max_score
    saturation = next(d for d, score in zip(durs, scores) if score >= saturation_score)

    # Optimization points
    log.info(f"Optimal TR Durations: Elbow Point: {elbow} TR | 1st Derivative Drop: {first_derivative} TR"
             f" | 2nd Derivative Drop: {second_point} TR | 95% Saturation: {saturation} TR")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(durs, scores, marker='o', label="Scores")
    plt.axvline(elbow, color='purple', linestyle='--', label=f"Elbow: {elbow}")
    plt.axvline(first_derivative, color='green', linestyle='--', label=f"First Derivative: {first_derivative}")
    plt.axvline(second_point, color='orange', linestyle='--', label=f"Second Derivative: {second_point}")
    plt.axvline(saturation, color='red', linestyle='--', label=f"Saturation (95%): {saturation}")
    plt.title("Movie duration in training vs Model Accuracy")
    plt.ylim([0,1])
    plt.xlabel("Duration [TR]")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,f"{config["H"]}_{config["NET"]}_{config["SUB_AREA"]}_{config["idx"]}_{config["slice"]}_dur_{durs[0]}_{durs[-1]}"))
    plt.show()


if __name__ == '__main__':
    config = load_config('tools/params_config.json')
    output_dir = os.path.join(config["results_dir"], "Duration_Analysis")
    Path(output_dir).mkdir(exist_ok=True)

    log.info("Running Duration Analysis")
    log.info(f"Input Dir : {config['directory']}")
    log.info(f"Output Dir: {output_dir}")


    durs, scores = run_dur_analysis(config=config, output_dir= output_dir, force_run=config["force_run"])
    plot_scores(config=config, output_dir=output_dir, durs=durs, scores=scores)
