# training/trainer.py
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import KFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, 
    confusion_matrix, classification_report
)
import logging

class ModelTrainer:
    def __init__(self, 
                 model_builder,
                 output_dir: str = "output",
                 random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            model_builder: Function to build a model with given parameters
            output_dir: Directory to save output files
            random_state: Random seed for reproducibility
        """
        self.model_builder = model_builder
        self.output_dir = output_dir
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
        
    def train_with_cross_validation(self, 
                                   X: np.ndarray, 
                                   y: np.ndarray, 
                                   params: Dict[str, Any],
                                   n_folds: int = 5,
                                   epochs: int = 100,
                                   batch_size: int = 32) -> Dict[str, Any]:
        """
        Train and evaluate model using k-fold cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            params: Model parameters
            n_folds: Number of folds for cross-validation
            epochs: Number of epochs for training
            batch_size: Batch size for training
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info(f"Starting {n_folds}-fold cross-validation")
        
        # Initialize metrics
        fold_metrics = []
        all_y_true = []
        all_y_pred = []
        all_y_prob = []
        
        # Create k-fold cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            self.logger.info(f"Training fold {fold+1}/{n_folds}")
            
            # Split data
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Clear previous model and build a new one
            tf.keras.backend.clear_session()
            model = self.model_builder(params)
            
            # Create callbacks (no early stopping as requested)
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.output_dir, f"models/model_fold_{fold+1}.h5"),
                monitor='val_auc',
                mode='max',
                save_best_only=True
            )
            
            # Train the model
            history = model.fit(
                X_train_fold, y_train_fold,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val_fold, y_val_fold),
                callbacks=[model_checkpoint],
                verbose=1
            )
            
            # Plot training history (accuracy, val_accuracy, loss, val_loss)
            self._plot_training_history(history, fold)
            
            # Evaluate on validation set
            y_val_prob = model.predict(X_val_fold).ravel()
            y_val_pred = (y_val_prob > 0.5).astype(int)
            
            # Evaluate on training set (per-fold training metrics)
            y_train_prob = model.predict(X_train_fold).ravel()
            y_train_pred = (y_train_prob > 0.5).astype(int)
            
            # Calculate metrics
            metrics = {
                'fold': fold + 1,
                'train': {
                    'auc': roc_auc_score(y_train_fold, y_train_prob),
                    'f1': f1_score(y_train_fold, y_train_pred),
                    'precision': precision_score(y_train_fold, y_train_pred),
                    'recall': recall_score(y_train_fold, y_train_pred),
                    'confusion_matrix': confusion_matrix(y_train_fold, y_train_pred).tolist()
                },
                'validation': {
                    'auc': roc_auc_score(y_val_fold, y_val_prob),
                    'f1': f1_score(y_val_fold, y_val_pred),
                    'precision': precision_score(y_val_fold, y_val_pred),
                    'recall': recall_score(y_val_fold, y_val_pred),
                    'confusion_matrix': confusion_matrix(y_val_fold, y_val_pred).tolist()
                }
            }
            
            fold_metrics.append(metrics)
            self.logger.info(
                f"Fold {fold+1} metrics: "
                f"Train AUC={metrics['train']['auc']:.4f}, F1={metrics['train']['f1']:.4f} | "
                f"Val AUC={metrics['validation']['auc']:.4f}, F1={metrics['validation']['f1']:.4f}"
            )
            
            # Save predictions for later analysis (for overall metrics on validation)
            all_y_true.extend(y_val_fold.tolist())
            all_y_pred.extend(y_val_pred.tolist())
            all_y_prob.extend(y_val_prob.tolist())
            
            # Plot ROC curves per fold for train and validation
            self._plot_roc_curve(
                y_train_fold, y_train_prob,
                title=f"ROC Curve - Train Fold {fold+1}"
            )
            self._plot_roc_curve(
                y_val_fold, y_val_prob,
                title=f"ROC Curve - Validation Fold {fold+1}"
            )
            
        # Calculate overall metrics (based on validation predictions)
        overall_metrics = {
            'auc': roc_auc_score(all_y_true, all_y_prob),
            'f1': f1_score(all_y_true, all_y_pred),
            'precision': precision_score(all_y_true, all_y_pred),
            'recall': recall_score(all_y_true, all_y_pred),
            'confusion_matrix': confusion_matrix(all_y_true, all_y_pred).tolist(),
            'classification_report': classification_report(all_y_true, all_y_pred, output_dict=True)
        }
        
        # Plot overall ROC curve (validation across folds)
        self._plot_roc_curve(all_y_true, all_y_prob, title="ROC Curve - Validation (All Folds)")
        
        # Plot confusion matrix (validation across folds)
        self._plot_confusion_matrix(all_y_true, all_y_pred, title="Confusion Matrix - Validation (All Folds)")
        
        # Save metrics
        results = {
            'fold_metrics': fold_metrics,
            'overall_metrics': overall_metrics,
            'model_parameters': params
        }
        
        with open(os.path.join(self.output_dir, "metrics/evaluation_metrics.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Cross-validation completed. Overall AUC: {overall_metrics['auc']:.4f}, F1: {overall_metrics['f1']:.4f}")
        
        return results
        
    def train_final_model(self, 
                         X_train: np.ndarray, 
                         y_train: np.ndarray,
                         X_test: np.ndarray,
                         y_test: np.ndarray,
                         params: Dict[str, Any],
                         epochs: int = 100,
                         batch_size: int = 32) -> tf.keras.Model:
        """
        Train the final model on all training data and evaluate on test data.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            X_test: Test feature matrix
            y_test: Test target vector
            params: Model parameters
            epochs: Number of epochs for training
            batch_size: Batch size for training
            
        Returns:
            Trained model
        """
        self.logger.info("Training final model on all training data")
        
        # Clear previous model and build a new one
        tf.keras.backend.clear_session()
        model = self.model_builder(params)
        
        # Create callbacks (no early stopping as requested)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.output_dir, "models/final_model.h5"),
            monitor='val_auc',
            mode='max',
            save_best_only=True
        )
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,  # Use a small validation set
            callbacks=[model_checkpoint],
            verbose=1
        )
        
        # Plot training history (accuracy, val_accuracy, loss, val_loss)
        self._plot_training_history(history, fold="final")
        
        # Evaluate on test set
        y_prob = model.predict(X_test).ravel()
        y_pred = (y_prob > 0.5).astype(int)
        
        # Calculate metrics
        test_metrics = {
            'auc': roc_auc_score(y_test, y_prob),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Plot ROC curve
        self._plot_roc_curve(y_test, y_prob, title="Final Model ROC Curve (Test Set)")
        
        # Plot confusion matrix
        self._plot_confusion_matrix(y_test, y_pred, title="Final Model Confusion Matrix (Test Set)")
        
        # Save metrics
        with open(os.path.join(self.output_dir, "metrics/final_model_metrics.json"), 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        self.logger.info(f"Final model evaluation. Test AUC: {test_metrics['auc']:.4f}, F1: {test_metrics['f1']:.4f}")
        
        # Save model architecture
        model_json = model.to_json()
        with open(os.path.join(self.output_dir, "models/model_architecture.json"), 'w') as f:
            f.write(model_json)
        
        return model
        
    def _plot_training_history(self, history, fold):
        """Plot training history."""
        plt.figure(figsize=(18, 5))
        
        # Plot accuracy
        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'Model Accuracy (Fold {fold})')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot AUC
        plt.subplot(1, 3, 2)
        plt.plot(history.history['auc'])
        plt.plot(history.history['val_auc'])
        plt.title(f'Model AUC (Fold {fold})')
        plt.ylabel('AUC')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot Loss
        plt.subplot(1, 3, 3)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'Model Loss (Fold {fold})')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"plots/training_history_fold_{fold}.png"))
        plt.close()
        
    def _plot_roc_curve(self, y_true, y_prob, title="ROC Curve"):
        """Plot ROC curve."""
        from sklearn.metrics import roc_curve
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(self.output_dir, f"plots/{title.replace(' ', '_').lower()}.png"))
        plt.close()
        
    def _plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        
        classes = ['Intron', 'Exon']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(self.output_dir, f"plots/{title.replace(' ', '_').lower()}.png"))
        plt.close()
