# data/extractor.py
import os
import logging
import numpy as np
from typing import List, Dict, Tuple, Any
from Bio import SeqIO
from pathlib import Path

class SequenceExtractor:
    def __init__(self, window_size: int = 100, step_size: int = 50):
        """
        Initialize the sequence extractor.
        
        Args:
            window_size: Size of the sliding window
            step_size: Step size for the sliding window
        """
        self.window_size = window_size
        self.step_size = step_size
        self.logger = logging.getLogger(__name__)
        
    def extract_from_file(self, file_path: str) -> Tuple[List[str], List[int]]:
        """
        Extract sequences and labels from a GenBank file.
        
        Args:
            file_path: Path to the GenBank file
            
        Returns:
            Tuple of (sequences, labels)
        """
        try:
            # Parse the GenBank file
            record = SeqIO.read(file_path, "genbank")
            
            # Get the full sequence
            full_sequence = str(record.seq)
            
            # Create a binary mask for exons (or coding regions)
            exon_mask = np.zeros(len(full_sequence), dtype=int)
            
            # Mark exon/CDS regions in the mask, handling fuzzy and compound locations
            positive_feature_types = {"exon", "CDS"}
            for feature in record.features:
                if feature.type in positive_feature_types:
                    location = feature.location
                    try:
                        # CompoundLocation (joined parts)
                        parts = getattr(location, 'parts', None)
                        if parts:
                            for part in parts:
                                start_idx = int(part.start)
                                end_idx = int(part.end)
                                if start_idx < end_idx:
                                    exon_mask[start_idx:end_idx] = 1
                        else:
                            start_idx = int(location.start)
                            end_idx = int(location.end)
                            if start_idx < end_idx:
                                exon_mask[start_idx:end_idx] = 1
                    except Exception as le:
                        self.logger.debug(f"Skipping feature with unparsable location in {file_path}: {le}")
            
            # Apply sliding window
            sequences = []
            labels = []
            
            for i in range(0, len(full_sequence) - self.window_size + 1, self.step_size):
                window_seq = full_sequence[i:i+self.window_size]
                window_mask = exon_mask[i:i+self.window_size]
                
                # Label is 1 if at least 50% of the window is exon/coding
                label = 1 if np.mean(window_mask) >= 0.5 else 0
                
                sequences.append(window_seq)
                labels.append(label)
            
            pos_count = int(np.sum(labels))
            neg_count = len(labels) - pos_count
            self.logger.info(f"Extracted {len(sequences)} sequences from {file_path} (pos={pos_count}, neg={neg_count})")
            return sequences, labels
            
        except Exception as e:
            self.logger.error(f"Error extracting sequences from {file_path}: {str(e)}")
            return [], []
            
    def extract_from_files(self, file_paths: List[str]) -> Tuple[List[str], List[int]]:
        """
        Extract sequences and labels from multiple GenBank files.
        
        Args:
            file_paths: List of paths to GenBank files
            
        Returns:
            Tuple of (sequences, labels)
        """
        all_sequences = []
        all_labels = []
        
        for file_path in file_paths:
            sequences, labels = self.extract_from_file(file_path)
            all_sequences.extend(sequences)
            all_labels.extend(labels)
        
        return all_sequences, all_labels
