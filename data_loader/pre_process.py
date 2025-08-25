import logging
import os.path as osp
import pickle
import torch
from .utils import z_score, pad_to_length
import shura


# Setup logger
log = shura.get_logger(name= "pre_process", level="DEBUG", to_file= False, filename="pre_process.log", file_format="log")


def load_subject_data(data_vis_subject, movie, dur, slice, offset, z_norm):
    """
    Process a single subject's clip into a tensor.

    Returns:
        torch.Tensor or None
    """

    # TODO: dur parameter to track and log for invalid cases
    try:
        movie_data = data_vis_subject[data_vis_subject['y'] == movie]

        if len(movie_data) == 0:
            log.error(f"No data found for movie {movie}")
            raise ValueError(f"No data found for movie {movie}")

        input_data = movie_data.iloc[:, :-4]  # last 4 columns are metadata

        if slice == 'start':
            input_data = input_data.iloc[offset:offset + dur]
        elif slice == 'end':
            input_data = input_data.iloc[-dur - offset:-offset]
        elif slice == 'middle':
            first_index = len(input_data) // 2 - dur // 2
            input_data = input_data.iloc[first_index:first_index + dur]
        elif slice == 'all':
            input_data = pad_to_length(input_data, target_len=260)
        else:
            log.error(f"Invalid slice value '{slice}' for movie {movie}. Must be one of [start, middle, end, all]")
            raise ValueError(f"Invalid slice value '{slice}'")

        input_array = z_score(input_data.values) if z_norm else input_data.values
        return torch.tensor(input_array)

    except Exception as e:
        log.error(f"Failed to load clip data for movie {movie}: {e}")
        raise


def build_data_dict(directory, NET, SUB_AREA, idx, H, slice, dur, offset, z_norm, is_rest):
    """
    Load and preprocess fMRI data for SVM classification.

    Args:
        directory (str): Path to data directory.
        NET (str): Network name (e.g., 'Cont').
        SUB_AREA (str): Sub-area name (e.g., 'Cing').
        idx (int/str): Region index.
        H (str): Hemisphere ('LH' or 'RH').
        slice (str): Slice method ('start', 'middle', 'end', 'all').
        dur (int): Duration of the time slice.
        z_norm (bool): Whether to apply z-score normalization.
        is_rest (int): Filter for rest condition.

    Returns:
        dict: Dictionary with keys ['data'] -> [X, y] tensors.
    """

    log.debug(f"Build params: NET={NET}, SUB_AREA={SUB_AREA}, idx={idx}, H={H}, slice={slice}, dur={dur}, offset={offset}, z_norm={z_norm}, is_rest={is_rest}")

    svm_dict = {}
    inputs, outputs = [], []

    filename = f'{H}_{NET}_{SUB_AREA}_{idx}.pkl' if SUB_AREA != 'NA' else f'{H}_{NET}_{idx}.pkl'
    file_path = osp.join(directory, filename)

    try:
        with open(file_path, 'rb') as file:
            data_vis = pickle.load(file)
        log.info(f"Successfully Loaded: {filename}")
    except Exception as e:
        log.error(f"Error loading file {file_path}: {e}")
        raise e

    for subject in data_vis['Subject'].unique():
        subject_data = data_vis[(data_vis['Subject'] == subject) & (data_vis['is_rest'] == is_rest)]
        for movie in range(1, 15):
            if movie in [4, 11]:
                continue
            input_tensor = load_subject_data(subject_data, movie, dur, slice, offset, z_norm)
            if input_tensor is not None:
                inputs.append(input_tensor)
                outputs.append(torch.tensor(movie))

    if inputs:
        X = torch.stack(inputs)
        y = torch.stack(outputs)
        svm_dict['data'] = [X, y]
        log.debug(f"PreProcessed Data: X{X.shape}, y{y.shape}, samples={len(inputs)}, movies={torch.unique(y).tolist()}")
    else:
        log.warning("No valid data was loaded.")

    return svm_dict
