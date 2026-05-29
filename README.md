# Reservoir Computing with the Human Connectome Project

## Overview
In this project, we explore the intersection of Network Neuroscience and Machine Learning. 

Standard Reservoir Computing (RC) models usually rely on random Gaussian recurrent networks. Here, we replaced the standard random reservoir with an empirical, biologically grounded network derived from the Human Connectome Project (HCP). 

The main goal of the model is to predict the chaotic dynamics of the Lorenz System (Lorenz-63). To push the analysis further, we also simulated structural brain damage by progressively removing nodes and edges from the network, measuring how the topological lesions affect the model's predictive performance and robustness.

## Repository Structure
All the core logic and scripts have been organized in the `src/` directory for reproducibility.

* `img/` - Plots and visual results generated during the experiments.
* `src/` - Core Python modules.
* `pyproject.toml` / `uv.lock` - Project metadata and dependencies (managed via uv).

## Core Modules

* `01_connectome_analysis.py`: Parses the HCP `.graphml` files. It extracts 3D spatial positions and edge features like fractional anisotropy (FA) and fiber counts.
* `02_lorenz_system.py`: Generates the chaotic target signal (Lorenz-63 attractor) using numerical Euler integration.
* `03_reservoir_model.py`: The actual Reservoir model setup. It handles the continuous-time updates, spectral radius normalization, Teacher Forcing washout, and Ridge Regression for the readout training.
* `04_experiments.py`: The pipeline for the structural damage simulation. It drops specific percentages of connections and evaluates the RMSE degradation.
