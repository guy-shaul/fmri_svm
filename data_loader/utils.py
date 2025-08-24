import pandas as pd
import numpy as np
from collections import defaultdict
import os
import json


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

# ----utilz for preprocessing---- #
def z_score(mat):
    """
    Apply z-score normalization across each row (i.e., timepoint).

    Args:
        mat (np.ndarray): 2D matrix [time, voxels]

    Returns:
        np.ndarray: Normalized matrix
    """
    means = np.mean(mat, axis=1, keepdims=True)
    stds = np.std(mat, axis=1, keepdims=True)
    return (mat - means) / stds


def pad_to_length(df, target_len):
    """
    Pad a DataFrame with zeros to a given number of rows (timepoints).

    Args:
        df (pd.DataFrame): Data to pad
        target_len (int): Desired number of rows

    Returns:
        pd.DataFrame: Padded data
    """
    if len(df) < target_len:
        pad = pd.DataFrame(0, index=range(target_len - len(df)), columns=df.columns)
        df = pd.concat([df, pad], axis=0)
    return df


# ----utilz for analysis tools---- #
def map_brain_areas(root_dir):
    brain_map = {'LH': defaultdict(lambda: defaultdict(list)),
                 'RH': defaultdict(lambda: defaultdict(list))}

    for filename in os.listdir(root_dir):
        if not filename.endswith('.pkl'):
            continue

        parts = filename.replace('.pkl', '').split('_')
        if len(parts) == 4:
            hemisphere, area, sub_area, index = parts
        elif len(parts) == 3:
            hemisphere, area, index = parts
            sub_area = 'NA'
        else:
            continue

        try:
            index = int(index)
            brain_map[hemisphere][area][sub_area].append(index)
        except ValueError:
            continue

    for h in brain_map:
        for a in brain_map[h]:
            for s in brain_map[h][a]:
                brain_map[h][a][s].sort()

    return brain_map

