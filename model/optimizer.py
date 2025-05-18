# model/optimizer.py
import numpy as np
import pyswarms as ps
import logging
import tensorflow as tf
from typing import Dict, Any, List, Tuple, Callable
from sklearn.model_selection import KFold

class PSOOptimizer:
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 build_model_fn: Callable,
                 n_particles: int = 10,
                 max_iterations: int = 20,
                 random_state: int = 42):
        """
        Initialize the PSO optimizer.
        
        Args:
            input_shape: Shape of input data
            build_model_fn: Function to build a model with given parameters
            n_particles: Number of particles in the swarm
            max_iterations: Maximum number of iterations
            random_state: Random seed for reproducibility
        """
        self.input_shape = input_shape
        self.build_model_fn = build_model_fn
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        np.random.seed(random_state)
        
        # Define parameter bounds
        self.bounds = (
            # n_conv_layers, filters (3 values), kernel_sizes (3 values), 
            # pool_sizes (3 values), dense_units, dropout_rate, learning_rate
            np.array([1, 16, 16, 16, 2, 2, 2, 1, 1, 1, 32, 0.1, 0.0001]),  # Lower bounds
            np.array([4, 128, 128, 128, 9, 9, 9, 4, 4, 4, 512, 0.5, 0.1])   # Upper bounds
        )
        
    def _decode_parameters(self, particle: np.ndarray) -> Dict[str, Any]:
        """
        Decode a particle's position into model parameters.
        
        Args:
            particle: Particle position
            
        Returns:
            Dictionary of model parameters
        """
        # Round discrete parameters
        n_conv_layers = int(particle[0])
        filters = [int(particle[1]), int(particle[2]), int(particle[3])][:n_conv_layers]
        kernel_sizes = [int(particle[4]), int(particle[5]), int(particle[6])][:n_conv_layers]
        pool_sizes = [int(particle[7]), int(particle[8]), int(particle[9])][:n_conv_layers]
        dense_units = [int(particle[10])]
        dropout_rate = particle[11]
        learning_rate = particle[12]
        
        return {
            'n_conv_layers': n_conv_layers,
            'filters': filters,
            'kernel_sizes': kernel_sizes,
            'pool_sizes': pool_sizes,
            'dense_units': dense_units,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate
        }
        
    def _evaluate_particle(self, particle: np.ndarray, X: np.ndarray, y: np.ndarray, 
                          n_folds: int = 3, epochs: int = 10, batch_size: int = 32) -> float:
        """
        Evaluate a particle using k-fold cross-validation.
        
        Args:
            particle: Particle position
            X: Feature matrix
            y: Target vector
            n_folds: Number of folds for cross-validation
            epochs: Number of epochs for training
            batch_size: Batch size for training
            
        Returns:
            Negative mean validation AUC (to be minimized)
        """
        params = self._decode_parameters(particle)
        
        # Use k-fold cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        val_aucs = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Clear previous model and build a new one
            tf.keras.backend.clear_session()
            model = self.build_model_fn(params)
            
            # Train the model with early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=5,
                mode='max',
                restore_best_weights=True
            )
            
            try:
                model.fit(
                    X_train_fold, y_train_fold,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val_fold, y_val_fold),
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # Evaluate on validation set
                _, _, val_auc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
                val_aucs.append(val_auc)
                
            except Exception as e:
                self.logger.warning(f"Error training model with parameters {params}: {str(e)}")
                val_aucs.append(0.0)  # Penalize failed configurations
        
        mean_val_auc = np.mean(val_aucs) if val_aucs else 0.0
        return -mean_val_auc  # Negative because PSO minimizes
        
    def _objective_function(self, particles: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Objective function for PSO.
        
        Args:
            particles: Particle positions
            X: Feature matrix
            y: Target vector
            
        Returns:
            Array of fitness values for each particle
        """
        n_particles = particles.shape[0]
        fitness = np.zeros(n_particles)
        
        for i in range(n_particles):
            fitness[i] = self._evaluate_particle(particles[i], X, y)
            self.logger.info(f"Particle {i+1}/{n_particles} evaluated with fitness {-fitness[i]:.4f}")
        
        return fitness
        
    def optimize(self, X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, Any], float]:
        """
        Optimize model parameters using PSO.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (best_parameters, best_auc)
        """
        self.logger.info("Starting PSO optimization")
        
        # Initialize PSO optimizer
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=len(self.bounds[0]),
            options=options,
            bounds=self.bounds
        )
        
        # Run optimization
        best_cost, best_pos = optimizer.optimize(
            lambda particles: self._objective_function(particles, X, y),
            iters=self.max_iterations
        )
        
        # Decode best parameters
        best_params = self._decode_parameters(best_pos)
        best_auc = -best_cost  # Convert back to AUC
        
        self.logger.info(f"PSO optimization completed. Best AUC: {best_auc:.4f}")
        self.logger.info(f"Best parameters: {best_params}")
        
        return best_params, best_auc
