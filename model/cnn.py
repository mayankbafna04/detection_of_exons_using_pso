# model/cnn.py
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from typing import Dict, Any, List, Tuple

class ExonCNN:
    def __init__(self, input_shape: Tuple[int, int], random_state: int = 42):
        """
        Initialize the CNN model.
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            random_state: Random seed for reproducibility
        """
        self.input_shape = input_shape
        self.random_state = random_state
        tf.random.set_seed(random_state)
        
    def build_model(self, params: Dict[str, Any]) -> tf.keras.Model:
        """
        Build a CNN model with the given parameters.
        
        Args:
            params: Dictionary of model parameters
            
        Returns:
            Compiled Keras model
        """
        # Extract parameters
        n_conv_layers = params.get('n_conv_layers', 2)
        filters = params.get('filters', [32, 64])
        kernel_sizes = params.get('kernel_sizes', [3, 5])
        pool_sizes = params.get('pool_sizes', [2, 2])
        dense_units = params.get('dense_units', [128])
        dropout_rate = params.get('dropout_rate', 0.3)
        learning_rate = params.get('learning_rate', 0.001)
        
        # Ensure lists have the right length
        filters = filters[:n_conv_layers] if isinstance(filters, list) else [filters] * n_conv_layers
        kernel_sizes = kernel_sizes[:n_conv_layers] if isinstance(kernel_sizes, list) else [kernel_sizes] * n_conv_layers
        pool_sizes = pool_sizes[:n_conv_layers] if isinstance(pool_sizes, list) else [pool_sizes] * n_conv_layers
        
        # Build model
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=self.input_shape))
        
        # Convolutional layers
        for i in range(n_conv_layers):
            model.add(layers.Conv1D(
                filters=filters[i],
                kernel_size=kernel_sizes[i],
                activation='relu',
                padding='same'
            ))
            model.add(layers.MaxPooling1D(pool_size=pool_sizes[i]))
        
        # Flatten
        model.add(layers.Flatten())
        
        # Dense layers
        for units in dense_units:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
