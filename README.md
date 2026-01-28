# Shape Transform Descriptor for Thymus Structures

This repository contains code for reproducing the results in the manuscript *"Shape transform descriptor for thymus structures"*.

> **More User-Friendly (pip installable) Package** A more user-friendly implementation for broader applications is available at [SampEuler](https://github.com/reddevil0623/SampEuler).

## Overview

This repo provides implementations of ECT, SECT, DETECT, and SampEuler for simplicial complexes, along with example usage and the data used in our classification study.

## Repository Contents

### Code

| File | Description |
|------|-------------|
| `My_ECT.py` | ECT-based algorithms for simplicial complexes |
| `shap_example.py` | Example of SHAP analysis using vectorized SampEuler as features |
| `synthetic_network_and_MPEG7.ipynb` | Synthetic network generation, ECT/DETECT/SampEuler computation, MDS visualization, and MPEG7 dataset preprocessing |

### Data

| File | Description |
|------|-------------|
| `Selected_quadrants.zip` | TIFF images of quadrants used in the classification study |
| `Whole_tissue_masks.zip` | Masks classifying thymus into cortex/medulla regions, plus individual binary masks for thymic epithelial cells (TEC) |
| `all_cortex_quadrants.zip` | Separate cortex quadrant images generated from tiling and filtering each binary TEC mask |
| `patch_TEC1hop.csv` | Cell counts for selected cell types within 5μm of TECs for each cortex quadrant |

### Media

| File | Description |
|------|-------------|
| `ECT_video.mp4` | Visualization of how ECT and SECT work for 2D data |
