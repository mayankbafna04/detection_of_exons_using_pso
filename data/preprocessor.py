# data/preprocessor.py
import numpy as np
import logging
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class SequencePreprocessor:
    def __init__(self, random_state: int = 42):
        """
        Initialize the sequence preprocessor.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        self.nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        
    def encode_sequences(self, sequences: List[str]) -> np.ndarray:
        """
        One-hot encode DNA sequences.
        
        Args:
            sequences: List of DNA sequences
            
        Returns:
            One-hot encoded sequences as a numpy array
        """
        # Get the length of the first sequence
        seq_length = len(sequences[0])
        
        # Initialize the one-hot encoded array
        encoded = np.zeros((len(sequences), seq_length, 5), dtype=np.float32)
        
        # Encode each sequence
        for i, seq in enumerate(sequences):
            for j, nucleotide in enumerate(seq):
                # Handle unknown nucleotides
                idx = self.nucleotide_map.get(nucleotide.upper(), 4)
                encoded[i, j, idx] = 1.0
        
        self.logger.info(f"Encoded {len(sequences)} sequences with shape {encoded.shape}")
        return encoded
        
    def balance_classes(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance classes using SMOTE.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (balanced_X, balanced_y)
        """
        # Reshape X for SMOTE (2D array required)
        original_shape = X.shape
        X_reshaped = X.reshape(X.shape[0], -1)
        
        # Apply SMOTE
        self.logger.info(f"Applying SMOTE to balance classes. Original class distribution: {np.bincount(y)}")
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)
        
        # Reshape X back to original shape
        X_resampled = X_resampled.reshape(-1, original_shape[1], original_shape[2])
        
        self.logger.info(f"After SMOTE, class distribution: {np.bincount(y_resampled)}")
        return X_resampled, y_resampled
        
    def prepare_data(self, sequences: List[str], labels: List[int], 
                     test_size: float = 0.2) -> Dict[str, Any]:
        """
        Prepare data for training.
        
        Args:
            sequences: List of DNA sequences
            labels: List of labels
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary containing train and test data
        """
        # Encode sequences
        X = self.encode_sequences(sequences)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Balance training data
        X_train_balanced, y_train_balanced = self.balance_classes(X_train, y_train)
        
        return {
            'X_train': X_train_balanced,
            'y_train': y_train_balanced,
            'X_test': X_test,
            'y_test': y_test
        }
