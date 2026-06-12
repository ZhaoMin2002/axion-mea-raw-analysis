# Axion MEA Raw Analysis

This repository collects Python code and notebooks for parsing Axion BioSystems MEA `.raw` files, windowing electrophysiology signals, and preparing downstream spectral/group analyses.

## Data

Raw `.raw` files are not included because they are large binary electrophysiology recordings.

Local raw data should be stored outside the Git repository, for example:

```text
D:/Scripps/
```

The `.gitignore` excludes `.raw`, `.npy`, `.npz`, `.h5`, `.hdf5`, `.bin`, `.dat`, and generated output folders.

## Repository structure

```text
ad_organoids/              Original analysis package and notebooks
archive/original_zips/     Original downloaded zip archives
docs/                      Reference documents and slides
legacy/                    Older or modified scripts kept for comparison
notebooks/                 Top-level exploratory notebooks
```

## Current goal

The working analysis goal is:

1. Read Axion `.raw` MEA recordings.
2. Convert binary recordings into structured well/electrode/time arrays.
3. Split long recordings into fixed-length windows.
4. Extract frequency-band features.
5. Map wells to experimental groups using a plate map.
6. Compare group-level differences.

## Known issues

* `window_signal(..., n_jobs=1)` needs correction.
* `start_end` slicing in `read_raw()` needs validation.
* Group labels parsed from `.raw` metadata should be checked against an external plate map.
* The full pipeline from raw reading to group comparison is not yet finalized.

## Installation

```bash
pip install -r ad_organoids/requirements.txt
```

Additional dependencies may be needed for AR spectral methods.
