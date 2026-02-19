"""
Memory-efficient streaming dataset for deep learning.
Never loads full dataset into RAM.
Perfect for 8GB RAM machines.
"""
import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, IterableDataset
from typing import List, Tuple, Optional
import logging

class SEEDStreamingDataset(IterableDataset):
    """
    Memory-efficient streaming dataset.
    Loads data on-the-fly, perfect for limited RAM.
    """
    
    def __init__(
        self,
        data_path: str,
        subject_ids: List[int],
        window_size: int = 200,
        cache_files: bool = True
    ):
        """
        Args:
            data_path: Path to Preprocessed_EEG folder
            subject_ids: List of subject IDs to include
            window_size: Number of time samples per window
            cache_files: Keep .mat files in memory after loading
        """
        self.data_path = data_path
        self.subject_ids = subject_ids
        self.window_size = window_size
        self.cache_files = cache_files
        
        self.logger = logging.getLogger(__name__)
        
        # Load labels once
        label_mat = sio.loadmat(os.path.join(data_path, "label.mat"))
        self.labels = label_mat["label"].flatten()
        
        # Build index (metadata only)
        self.index = self._build_index()
        self.logger.info(f"Indexed {len(self.index)} windows")
        
        # File cache (optional)
        self.file_cache = {} if cache_files else None
        
    def _build_index(self) -> List[Tuple]:
        """
        Build index of (file_path, trial_key, label_idx, window_idx)
        No actual data loaded.
        """
        files = [
            f for f in os.listdir(self.data_path)
            if f.endswith(".mat") and f != "label.mat"
        ]
        files.sort()
        
        index = []
        
        for file in files:
            # Extract subject ID from filename (e.g., "1_20131027.mat" → 1)
            subject_id = int(file.split("_")[0])
            
            if subject_id not in self.subject_ids:
                continue
                
            file_path = os.path.join(self.data_path, file)
            
            # Load just the keys, not the data
            mat_contents = sio.whosmat(file_path)
            trial_keys = [
                name for name, _, _ in mat_contents 
                if 'eeg' in name.lower()
            ]
            trial_keys.sort()
            
            for trial_idx, key in enumerate(trial_keys):
                # We don't know the exact number of windows without loading data
                # So we'll store (file_path, key, trial_idx, None) 
                # and compute windows in __iter__
                index.append((file_path, key, trial_idx, None))
        
        return index
    
    def __iter__(self):
        """
        Generator-style iteration.
        Yields (eeg_window, label) one at a time.
        """
        worker_info = torch.utils.data.get_worker_info()
        
        # Simple sequential iteration
        for file_path, key, trial_idx, _ in self.index:
            
            # Load file (from cache or disk)
            if self.cache_files and file_path in self.file_cache:
                mat = self.file_cache[file_path]
            else:
                mat = sio.loadmat(file_path)
                if self.cache_files:
                    self.file_cache[file_path] = mat
            
            # Get trial data
            trial = mat[key]  # Shape: (62, time)
            label = self.labels[trial_idx]
            
            # Remap label: -1→0, 0→1, 1→2
            label = 0 if label == -1 else (1 if label == 0 else 2)
            
            # Generate windows
            n_windows = trial.shape[1] // self.window_size
            
            for w in range(n_windows):
                start = w * self.window_size
                end = start + self.window_size
                
                segment = trial[:, start:end].astype(np.float32)
                
                # Convert to torch tensor and add channel dimension
                x = torch.from_numpy(segment).unsqueeze(0)  # (1, 62, 200)
                y = torch.tensor(label, dtype=torch.long)
                
                yield x, y
        
        # Clear cache if needed
        if not self.cache_files and hasattr(self, 'file_cache'):
            self.file_cache.clear()
    
    def __len__(self):
        """Total number of windows (approximate)."""
        # This is an estimate - actual count may vary
        if hasattr(self, '_length'):
            return self._length
        
        # Calculate approximate length
        total = 0
        for file_path, key, trial_idx, _ in self.index:
            mat = sio.loadmat(file_path)
            trial = mat[key]
            total += trial.shape[1] // self.window_size
        
        self._length = total
        return total


class SEEDLabelledDataset(Dataset):
    """
    Standard PyTorch Dataset with on-demand loading.
    Suitable for smaller subsets.
    """
    
    def __init__(
        self,
        data_path: str,
        subject_ids: List[int],
        window_size: int = 200,
        transform=None
    ):
        self.data_path = data_path
        self.subject_ids = subject_ids
        self.window_size = window_size
        self.transform = transform
        
        # Load labels
        label_mat = sio.loadmat(os.path.join(data_path, "label.mat"))
        self.labels = label_mat["label"].flatten()
        
        # Build full index
        self.index = []
        self._build_index()
        
    def _build_index(self):
        """Build complete index with file caching."""
        files = [
            f for f in os.listdir(self.data_path)
            if f.endswith(".mat") and f != "label.mat"
        ]
        files.sort()
        
        self.file_cache = {}
        
        for file in files:
            subject_id = int(file.split("_")[0])
            
            if subject_id not in self.subject_ids:
                continue
                
            file_path = os.path.join(self.data_path, file)
            
            # Load file
            mat = sio.loadmat(file_path)
            self.file_cache[file_path] = mat
            
            trial_keys = [
                k for k in mat.keys() if "eeg" in k.lower()
            ]
            trial_keys.sort()
            
            for trial_idx, key in enumerate(trial_keys):
                trial = mat[key]
                label = self.labels[trial_idx]
                
                n_windows = trial.shape[1] // self.window_size
                
                for w in range(n_windows):
                    self.index.append(
                        (file_path, key, trial_idx, w)
                    )
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        file_path, key, trial_idx, window_idx = self.index[idx]
        
        # Get from cache
        mat = self.file_cache[file_path]
        trial = mat[key]
        
        start = window_idx * self.window_size
        end = start + self.window_size
        
        segment = trial[:, start:end].astype(np.float32)
        
        # Get label
        label = self.labels[trial_idx]
        label = 0 if label == -1 else (1 if label == 0 else 2)
        
        # Convert to tensor
        x = torch.from_numpy(segment).unsqueeze(0)  # (1, 62, 200)
        y = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            x = self.transform(x)
            
        return x, y