Detection of Exons Using Particle Swarm Optimization (PSO)
This project implements a computational biology pipeline to locate and detect exons within DNA/RNA sequences using a combination of Particle Swarm Optimization (PSO) and Convolutional Neural Networks (CNNs).

📂 Project Structure
The repository is organized as follows:

main.py: The central entry point to run the data pipeline, training, and optimization.

config.json: Contains hyperparameters for both the PSO algorithm and the CNN model.

data/: Modular scripts handling the data lifecycle.

downloader.py: Fetches target genetic sequences.

extractor.py: Extracts relevant genomic features.

preprocessor.py: Cleans and formats raw sequences for model compatibility.

model/: Core logic for the neural network and heuristic optimization.

cnn.py: Defines the Convolutional Neural Network architecture used for exon sequence classification.

optimizer.py: Standard training optimization routines.

pso_optimize.py: Implements Particle Swarm Optimization to fine-tune model parameters or sequence alignments.

genbank_files/: Local repository of raw GenBank genetic records (e.g., .gb files).

output/: Dynamically generated results.

data/raw/: Extracted and cached sequencing datasets.

logs/: Detailed timestamps and performance histories for each execution loop.

metrics/: JSON files containing validation, loss accuracy, and final evaluation data.

models/: Serialized Keras/TensorFlow weights (.h5) and architecture maps (.json).

🚀 Getting Started
1. Prerequisites
Ensure you have Python 3.8+ installed along with the required scientific computing and deep learning packages (TensorFlow/Keras, NumPy, and Biopython).

2. Installation
Clone the repository and navigate into the project directory:

Bash
git clone https://github.com/mayankbafna04/detection_of_exons_using_pso.git
cd detection_of_exons_using_pso
3. Configuration
You can adjust execution parameters, swarm size, mutation factors, and CNN architectures directly inside the configuration file:

Open config.json to alter settings before a run.

4. Running the Pipeline
Execute the main script to run the data collection, preprocessing, and PSO-driven network optimization sequence:

Bash
python main.py
📊 Results and Outputs
After a complete run, check the output/ directory to analyze performance:

Review training performance across cross-validation splits in output/metrics/evaluation_metrics.json.

The optimal finalized neural network configuration will be saved to output/models/final_model.h5.
