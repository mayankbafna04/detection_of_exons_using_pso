# main.py
import os
import argparse
import json
import logging
import tensorflow as tf
from datetime import datetime
from typing import List, Dict, Any

# Import modules
from data.downloader import GenBankDownloader
from data.extractor import SequenceExtractor
from data.preprocessor import SequencePreprocessor
from model.cnn import ExonCNN
from model.optimizer import PSOOptimizer
from training.trainer import ModelTrainer
from utils.logger import setup_logger
from utils.config import load_config, save_config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Exon Detection using CNN with PSO Optimization")
    
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--accessions", type=str, nargs="+", help="GenBank accession numbers")
    parser.add_argument("--email", type=str, help="Email for NCBI Entrez")
    parser.add_argument("--api_key", type=str, help="NCBI API key")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--optimize", action="store_true", help="Run PSO optimization")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logger
    log_file = os.path.join(args.output_dir, "logs", f"exon_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logger(log_file)
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = {
            "data": {
                "window_size": 100,
                "step_size": 50,
                "test_size": 0.2
            },
            "model": {
                "n_conv_layers": 2,
                "filters": [32, 64],
                "kernel_sizes": [3, 5],
                "pool_sizes": [2, 2],
                "dense_units": [128],
                "dropout_rate": 0.3,
                "learning_rate": 0.001
            },
            "training": {
                "n_folds": 5,
                "epochs": 100,
                "batch_size": 32
            },
            "pso": {
                "n_particles": 10,
                "max_iterations": 20
            }
        }
        save_config(config, args.config)
    
    # Override config with command line arguments
    if args.accessions:
        config["accessions"] = args.accessions
    if args.email:
        config["email"] = args.email
    if args.api_key:
        config["api_key"] = args.api_key
    if args.random_state:
        config["random_state"] = args.random_state
    
    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"GPU is available: {gpus}")
        try:
            # Set memory growth to avoid allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.warning(f"Error setting memory growth: {str(e)}")
    else:
        logger.warning("No GPU available. Using CPU.")
    
    # Download GenBank files
    if "accessions" in config and "email" in config:
        logger.info("Downloading GenBank files")
        downloader = GenBankDownloader(
            email=config["email"],
            api_key=config.get("api_key"),
            output_dir=os.path.join(args.output_dir, "data", "raw")
        )
        file_paths = downloader.download_genbank_files(config["accessions"])
        logger.info(f"Downloaded {len(file_paths)} GenBank files")
    else:
        logger.warning("Accessions or email not provided. Skipping download.")
        file_paths = []
        for root, _, files in os.walk(os.path.join(args.output_dir, "data", "raw")):
            for file in files:
                if file.endswith(".gb"):
                    file_paths.append(os.path.join(root, file))
        logger.info(f"Found {len(file_paths)} existing GenBank files")
    
    if not file_paths:
        logger.error("No GenBank files found. Exiting.")
        return
    
    # Extract sequences
    logger.info("Extracting sequences from GenBank files")
    extractor = SequenceExtractor(
        window_size=config["data"]["window_size"],
        step_size=config["data"]["step_size"]
    )
    sequences, labels = extractor.extract_from_files(file_paths)
    logger.info(f"Extracted {len(sequences)} sequences with {sum(labels)} exons and {len(labels) - sum(labels)} introns")
    
    # Preprocess data
    logger.info("Preprocessing data")
    preprocessor = SequencePreprocessor(random_state=config["random_state"])
    data = preprocessor.prepare_data(
        sequences=sequences,
        labels=labels,
        test_size=config["data"]["test_size"]
    )
    logger.info(f"Preprocessed data: {data['X_train'].shape[0]} training samples, {data['X_test'].shape[0]} test samples")
    
    # Initialize CNN model builder
    input_shape = (config["data"]["window_size"], 5)  # (sequence_length, one-hot encoding size)
    cnn = ExonCNN(input_shape=input_shape, random_state=config["random_state"])
    
    # Optimize model parameters with PSO
    if args.optimize:
        logger.info("Optimizing model parameters with PSO")
        pso_optimizer = PSOOptimizer(
            input_shape=input_shape,
            build_model_fn=cnn.build_model,
            n_particles=config["pso"]["n_particles"],
            max_iterations=config["pso"]["max_iterations"],
            random_state=config["random_state"]
        )
        best_params, best_auc = pso_optimizer.optimize(data["X_train"], data["y_train"])
        
        # Update config with best parameters
        config["model"] = best_params
        save_config(config, args.config)
        
        logger.info(f"PSO optimization completed. Best AUC: {best_auc:.4f}")
        logger.info(f"Best parameters: {best_params}")
    
    # Train and evaluate model
    if args.train or args.evaluate:
        logger.info("Training and evaluating model")
        trainer = ModelTrainer(
            model_builder=cnn.build_model,
            output_dir=args.output_dir,
            random_state=config["random_state"]
        )
        
        # Cross-validation
        results = trainer.train_with_cross_validation(
            X=data["X_train"],
            y=data["y_train"],
            params=config["model"],
            n_folds=config["training"]["n_folds"],
            epochs=config["training"]["epochs"],
            batch_size=config["training"]["batch_size"]
        )
        
        # Train final model
        final_model = trainer.train_final_model(
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_test=data["X_test"],
            y_test=data["y_test"],
            params=config["model"],
            epochs=config["training"]["epochs"],
            batch_size=config["training"]["batch_size"]
        )
        
        logger.info("Model training and evaluation completed")
    
    logger.info("Exon detection pipeline completed successfully")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")