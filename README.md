# Detection of Exons Using Particle Swarm Optimization (PSO)

A hybrid computational biology framework that optimizes a Convolutional Neural Network (CNN) architecture using Particle Swarm Optimization (PSO) for high-accuracy exon mapping and genomic sequence splice-site detection in eukaryotic genomes.

---

## 📂 Project Structure

```text
├── config.json          # Hyperparameters for the PSO swarm and CNN architecture
├── main.py             # Central pipeline execution script (Data -> PSO -> Train -> Eval)
├── requirements.txt     # Python environment dependency specifications
├── data/               # Modular data management scripts
│   ├── downloader.py   # Automated extraction of NCBI GenBank sequences
│   ├── extractor.py    # Feature extraction and exon/intron coordinate parser
│   └── preprocessor.py # One-hot sequence mapping and window slicing tools
├── model/              # Swarm optimization and deep learning architectures
│   ├── cnn.py          # 1D-CNN tensor graph for sequence classification
│   ├── optimizer.py    # Backpropagation training loop routines
│   └── pso_optimize.py # Swarm logic tracking local/global best positions
├── genbank_files/      # Local repository of reference .gb records
├── training/           # Model calibration routines
│   └── trainer.py      # K-Fold cross-validation orchestration
├── utils/              # Project logging and system configuration loaders
│   ├── config.py
│   └── logger.py
└── output/             # Dynamically generated assets
    ├── data/raw/       # Cached download segments
    ├── logs/           # Detailed execution timestamps
    ├── metrics/        # Cross-validation and test set evaluation profiles
    ├── models/         # Serialized Keras/TensorFlow model weights (.h5)
    └── plots/          # ROC curves and confusion matrices

```

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you are using Python 3.8 or higher. It is highly recommended to deploy within a isolated `venv` or Conda environment.

### 2. Installation

Clone the repository and install the standard scientific computing and machine learning dependencies:

```bash
git clone [https://github.com/mayankbafna04/detection_of_exons_using_pso.git](https://github.com/mayankbafna04/detection_of_exons_using_pso.git)
cd detection_of_exons_using_pso
pip install -r requirements.txt

```

### 3. Configuration

Pipeline hyperparameters are globally accessible via `config.json`. Modify this file to adjust the following execution profiles:

* **Swarm Optimization Parameters:** Swarm population size, cognitive/social coefficients ($c_1, c_2$), and inertial weight ($w$).
* **CNN Graph Layers:** Convolutional filters, kernel kernel dimension constraints, dropout threshold variables, and dense classification sizes.
* **Dataset Dimensions:** Window length configurations and validation split controls.

### 4. Running the Pipeline

Execute the master process workflow handler to initiate GenBank data compilation, sequence vectorizing, and hybrid PSO-CNN architecture training:

```bash
python main.py

```

---

## 📊 Results and Outputs

All training metrics, cross-validation profiles, and performance graphs are exported directly to the `output/` directory:

* **Performance Tracking:** Run checkpoints and accuracy records are written directly to `output/logs/`.
* **Statistical Matrices:** Micro-averaged performance scores and evaluation arrays are saved inside `output/metrics/evaluation_metrics.json`.
* **Serialized Weights:** The optimized deep learning model graph files are written to `output/models/final_model.h5`.

```

```
