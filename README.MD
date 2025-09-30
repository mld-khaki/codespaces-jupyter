# Evoked Potential Analysis & Plotting

This repository contains code for analyzing and plotting **cortico-cortical evoked potentials (CCEPs)** from intracranial EEG (**EDF format**).

---

## Features
- Reads EDF files and extracts annotations (no TSV required).
- Decodes stimulation events using `ccep_lib` helper functions.
- Extracts stimulated and evoked channels.
- Supports optional preprocessing (band-pass, notch, detrending, median filtering).
- Segments and normalizes responses, computes average evoked potentials.
- Generates plots with:
  - Stimulated channel (10s window)
  - Evoked channel (10s window)
  - Overlapped segments
  - Average evoked response

---

## Repository Structure
.
├── .devcontainer/ # GitHub Codespaces config
├── data/ # Place EDF files here (organized by subject/session)
├── notebooks/ # Jupyter notebooks
├── ccep_lib.py # Helper functions
├── evoked_potential_analysis_plotting.ipynb # Main analysis notebook
├── requirements.txt # Dependencies
└── README.md


---

## Quick Start (Codespaces)

1. Open the repo in **GitHub Codespaces**.
2. Place EDF data in `./data/sub-<ID>/ses-001/`.
3. Launch the notebook:

jupyter notebook notebooks/evoked_potential_analysis_plotting.ipynb

Run cells to analyze EDF annotations and generate evoked potential plots.

