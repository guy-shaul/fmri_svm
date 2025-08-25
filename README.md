# fMRI-SVM Movie Classification

Predicts which of 14 movies a subject is watching based on fMRI brain scan data using SVM classification.

## Overview

- **Data**: 176 subjects, 14 movies, fMRI scans during movie viewing
- **Model**: SVM classifier (voxels × time → movie number)
- **Goal**: Analyze brain activity patterns across regions and time slices

## Structure

```
├── data/                    # Pickle files (brain area data)
├── data_loader/            # Preprocessing
├── model/                  # SVM classifier
├── tools/                  # Analysis scripts
└── results/                # Outputs
```

## Tools

### 1. Basic Run (`run.py`)
```bash
python tools/run.py
```
Runs SVM with parameters from `params_config.json`

### 2. Slice Analysis (`slice_analysis.py`) 
Compares accuracy across movie slices (start/middle/end). Outputs CSV and plots.

### 3. Duration Analysis (`dur_analysis.py`)
Tests slice durations (5-65 TR) to find optimal length. Shows accuracy vs duration with optimization points.

### 4. Movie Duration Display (`display_movies_dur.py`)
Plots duration distribution of the 14 movies.

## Key Parameters (`params_config.json`)

```json
{
  "directory": "/path/to/data",    # Data directory path
  "results_dir": "/path/to/results", # Results output path
  "NET": "Vis",                    # Brain network 
  "SUB_AREA": "NA",                # Sub-area
  "idx": 17,                       # Region index
  "H": "LH",                       # Hemisphere (LH/RH)
  "slice": "end",                  # Temporal slice (start/middle/end/all)
  "dur": 10,                       # Duration in TR
  "offset": 5,                     # Offset from boundary
  "is_rest": 0,                    # 0=active, 1=rest
  "z_norm": false,                 # Z-score normalization
  "kernel": "sigmoid",             # SVM kernel
  "scale": true,                   # Feature scaling
  "k_folds": 5,                    # Cross-validation folds
  "force_run": false              # Force recomputation of cached results
}
```

## Quick Start

1. Update paths in `params_config.json`
2. Run analysis:
   ```bash
   python tools/slice_analysis.py    # Compare start/middle/end
   python tools/dur_analysis.py      # Optimize duration
   ```

## Dependencies
```bash
pip install -r requirements.txt
```