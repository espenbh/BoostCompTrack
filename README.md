# BoostCompTrack: A multi-purpose tracking framework for salmon welfare monitoring in challenging environments

**Last updated:** July 29, 2025  
**Authors:** Espen Uri Høgstedt, Christian Schellewald, Annette Stahl, Rudolf Mester
**Repository:** BoostCompTrack

## 🐟 Overview

This repository contains the codebase for **BoostCompTrack**, a flexible computer vision-based tracking framework designed for **automated salmon welfare monitoring** in industrial aquaculture net pens.
Built upon [BoostTrack: Boosting the similarity measure and detection confidence for improved multiple object tracking](https://link.springer.com/article/10.1007/s00138-024-01531-5) to address the unique challenges of underwater tracking. 
Our framework demonstrates strong performance in crowded scenes, during salmon turning, and enables monitoring of tail beat wavelength — an important welfare indicator.

## 📂 Project Structure

```
BoostCompTrack/
├── annotation_helpers/            # Tools for visualizing annotations
├── associator/                    # Object association logic
├── detector/                      # Detection modules
├── evaluation/                    # Benchmark visualization and TurnSalmon dataset formatting
├── helpers/                       # Utility functions
├── paper_helpers/                 # Scripts for generating paper figures
├── welfare_helpers/               # Modules for welfare indicators (tail beat wavelength)
├── benchmark_salmon_trackers.ipynb   # Tracker benchmarking notebook
├── extract_tailbeat_period.ipynb     # Tail beat wavelength analysis
├── track_salmon.ipynb                # Salmon tracking in video
```

## 🧠 Key Features

- **Pose-based tracking**: Extracts bounding boxes around salmon and their body parts.
- **Body-part-aware modules**: Specialized components to handle occlusion and turning salmon.
- **Benchmarking**: Outperforms BoostTrack on two novel salmon tracking datasets.
- **Tail beat analysis**: Demonstrates suitability for tail beat-based welfare monitoring.

## 📊 Datasets

We introduce **three novel datasets**:
1. **CrowdedSalmon** – Tests robustness in dense environments.
2. **TurningSalmon** – Evaluates tracking during salmon turning.
3. **TailbeatWavelength** – For calculating tail beat wavelength.


## 📄 Citation

If you use this work in your research, please cite the corresponding paper (link coming soon).

## 📬 Contact

For questions or collaborations, feel free to reach out via GitHub or email.

---

**BoostCompTrack** is developed to support the aquaculture industry in achieving **continuous, automated, and precise** salmon welfare monitoring.
