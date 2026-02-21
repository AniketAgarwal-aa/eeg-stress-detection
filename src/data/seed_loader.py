"""
SEED dataset loader for classical ML feature extraction.
Loads Preprocessed_EEG files and extracts comprehensive features.
Optimized for speed with vectorized operations.
"""

import os
import numpy as np
import scipy.io as sio
from typing import Tuple, List, Optional
import logging
from scipy import signal, stats


class SEEDLoader:
    """
    Loader for SEED dataset with comprehensive feature extraction.
    
    Features extracted per window:
    - Differential Entropy (5 bands)
    - Band ratios (3 ratios)
    - Hjorth parameters (3)
    - Statistical moments (4)
    - Total: 15 features per channel → 62 * 15 = 930 features
    - Plus region aggregation and global stats
    """
    
    def __init__(
        self,
        data_path: str,
        window_size: int = 200,
        fs: int = 200,
        selected_subjects: Optional[List[int]] = None
    ):
        """
        Args:
            data_path: Path to Preprocessed_EEG folder
            window_size: Number of samples per window (200 = 1 second at 200Hz)
            fs: Sampling frequency in Hz
            selected_subjects: List of subject IDs to load (None = all)
        """
        self.data_path = data_path
        self.window_size = window_size
        self.fs = fs
        self.selected_subjects = selected_subjects
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load labels
        label_path = os.path.join(data_path, "label.mat")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"label.mat not found in {data_path}")
        
        label_mat = sio.loadmat(label_path)
        self.labels = label_mat["label"].flatten()
        self.logger.info(f"Loaded labels: {np.unique(self.labels)}")
        
        # EEG bands for feature extraction
        self.bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        # Brain regions (approximate from channel_62_pos.locs)
        self.regions = {
            'frontal': list(range(0, 15)),
            'central': list(range(15, 30)),
            'parietal': list(range(30, 45)),
            'occipital': list(range(45, 55)),
            'temporal': list(range(55, 62))
        }
        
        self.logger.info(f"SEEDLoader initialized with {data_path}")
        self.logger.info(f"Window size: {window_size} samples, FS: {fs}Hz")
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Main loading function.
        
        Returns:
            X: Feature matrix (n_windows, n_features)
            y: Labels (n_windows,)
            subjects: Subject IDs (n_windows,)
        """
        # Get all subject files
        files = [
            f for f in os.listdir(self.data_path)
            if f.endswith(".mat") and f != "label.mat"
        ]
        files.sort()
        
        self.logger.info(f"Found {len(files)} subject files")
        
        X_list = []
        y_list = []
        subjects_list = []
        
        for idx, file in enumerate(files):
            # Extract subject ID from filename (e.g., "1_20131027.mat" -> 1)
            subject_id = int(file.split("_")[0])
            
            # Skip if not in selected subjects
            if self.selected_subjects and subject_id not in self.selected_subjects:
                continue
            
            self.logger.info(f"Processing {idx+1}/{len(files)}: {file} (Subject {subject_id})")
            
            file_path = os.path.join(self.data_path, file)
            
            try:
                X_subj, y_subj = self._process_subject_file(file_path, subject_id)
                
                X_list.append(X_subj)
                y_list.append(y_subj)
                subjects_list.extend([subject_id] * len(y_subj))
                
            except Exception as e:
                self.logger.error(f"Error processing {file}: {str(e)}")
                continue
        
        # Concatenate all subjects
        X = np.vstack(X_list).astype(np.float32)
        y = np.concatenate(y_list)
        subjects = np.array(subjects_list, dtype=np.int16)
        
        # Show original label distribution
        unique, counts = np.unique(y, return_counts=True)
        self.logger.info(f"Original label distribution: {dict(zip(unique, counts))}")
        
        # Remap labels from [-1,0,1] to [0,1,2] for sklearn compatibility
        y = self.remap_labels(y)
        
        self.logger.info(f"Dataset created: X {X.shape}, y {y.shape}")
        self.logger.info(f"Remapped label distribution: {np.bincount(y)}")
        
        return X, y, subjects
    
    def _process_subject_file(self, file_path: str, subject_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single subject file.
        
        Returns:
            X_subj: Feature matrix for this subject
            y_subj: Labels for this subject
        """
        mat = sio.loadmat(file_path)
        
        # Get all trial keys (eeg1, eeg2, eeg3)
        trial_keys = [k for k in mat.keys() if 'eeg' in k.lower()]
        trial_keys.sort()
        
        X_subj = []
        y_subj = []
        
        for trial_idx, key in enumerate(trial_keys):
            # Get trial data: shape (62, time)
            trial = mat[key]
            label = self.labels[trial_idx]
            
            # Convert to time x channels format
            trial = trial.T  # (time, 62)
            
            # Extract windows
            n_windows = trial.shape[0] // self.window_size
            
            for w in range(n_windows):
                start = w * self.window_size
                end = start + self.window_size
                
                window = trial[start:end, :]
                
                # Extract features
                features = self._extract_features(window)
                
                X_subj.append(features)
                y_subj.append(label)
        
        return np.array(X_subj, dtype=np.float32), np.array(y_subj)
    
    def _extract_features(self, window: np.ndarray) -> np.ndarray:
        """
        Extract all features from a single window.
        
        Args:
            window: EEG window of shape (window_size, n_channels)
        
        Returns:
            Feature vector
        """
        # 1. Compute PSD using Welch's method
        freqs, psd = signal.welch(
            window.T,  # (n_channels, window_size)
            fs=self.fs,
            nperseg=min(64, self.window_size),
            axis=-1
        )  # psd shape: (n_channels, n_freqs)
        
        # 2. Extract band powers
        band_powers = []
        for (low, high) in self.bands.values():
            idx = (freqs >= low) & (freqs <= high)
            band_power = np.mean(psd[:, idx], axis=1)
            band_powers.append(band_power)
        
        band_powers = np.array(band_powers)  # (5, 62)
        
        # 3. Differential Entropy (DE)
        de_features = np.log(band_powers + 1e-8)  # (5, 62)
        
        # 4. Band ratios
        delta, theta, alpha, beta, gamma = band_powers
        
        ratio_beta_alpha = beta / (alpha + 1e-8)
        ratio_theta_beta = theta / (beta + 1e-8)
        engagement = beta / (alpha + theta + 1e-8)
        
        ratios = np.array([
            ratio_beta_alpha,
            ratio_theta_beta,
            engagement
        ])  # (3, 62)
        
        # 5. Hjorth parameters
        activity, mobility, complexity = self._hjorth_parameters(window)
        
        # 6. Statistical moments
        moments = self._statistical_moments(window)  # (4, 62)
        
        # Combine all per-channel features
        per_channel = np.concatenate([
            de_features,           # 5
            ratios,                # 3
            activity.reshape(1, -1),   # 1
            mobility.reshape(1, -1),   # 1
            complexity.reshape(1, -1), # 1
            moments                # 4
        ], axis=0)  # (15, 62)
        
        # 7. Region aggregation
        region_features = self._aggregate_regions(per_channel)  # (75,)
        
        # 8. Global band statistics
        global_stats = self._global_band_stats(band_powers)  # (10,)
        
        # Final feature vector
        features = np.concatenate([
            per_channel.flatten(),  # 15*62 = 930
            region_features,         # 75
            global_stats             # 10
        ])  # Total: 1015 features
        
        return features
    
    def _hjorth_parameters(self, window: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Hjorth parameters: activity, mobility, complexity.
        """
        diff1 = np.diff(window, axis=0)
        diff2 = np.diff(diff1, axis=0)
        
        var0 = np.var(window, axis=0)
        var1 = np.var(diff1, axis=0) if len(diff1) > 0 else np.zeros(window.shape[1])
        var2 = np.var(diff2, axis=0) if len(diff2) > 0 else np.zeros(window.shape[1])
        
        activity = var0
        mobility = np.sqrt(var1 / (var0 + 1e-8))
        complexity = np.sqrt(var2 / (var1 + 1e-8)) / (mobility + 1e-8)
        
        return activity, mobility, complexity
    
    def _statistical_moments(self, window: np.ndarray) -> np.ndarray:
        """
        Compute statistical moments: mean, std, skewness, kurtosis.
        """
        mean = np.mean(window, axis=0)
        std = np.std(window, axis=0)
        skew = stats.skew(window, axis=0)
        kurt = stats.kurtosis(window, axis=0)
        
        return np.array([mean, std, skew, kurt])
    
    def _aggregate_regions(self, per_channel: np.ndarray) -> np.ndarray:
        """
        Aggregate features by brain region.
        """
        region_features = []
        
        for channels in self.regions.values():
            region_mean = np.mean(per_channel[:, channels], axis=1)
            region_features.append(region_mean)
        
        return np.concatenate(region_features)
    
    def _global_band_stats(self, band_powers: np.ndarray) -> np.ndarray:
        """
        Compute global statistics across all channels for each band.
        """
        stats_list = []
        for band in band_powers:
            stats_list.append(np.mean(band))
            stats_list.append(np.std(band))
        
        return np.array(stats_list)
    
    def remap_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Remap labels from [-1, 0, 1] to [0, 1, 2].
        """
        return np.where(y == -1, 0, np.where(y == 0, 1, 2))
