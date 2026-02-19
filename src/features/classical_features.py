"""
Advanced EEG feature extraction for classical ML.
Includes:
- Differential Entropy (5 bands)
- Band ratios
- Hjorth parameters
- Spatial aggregation
- Statistical moments
"""
import numpy as np
from scipy import signal, stats

class EEGFeatureExtractor:
    """
    Production-grade EEG feature extractor.
    Optimized for speed and research reproducibility.
    """
    
    def __init__(self, fs=200, window_size=200):
        self.fs = fs
        self.window_size = window_size
        
        # Standard EEG bands
        self.bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        # Approximate brain regions (from channel_62_pos.locs)
        self.regions = {
            'frontal': list(range(0, 15)),
            'central': list(range(15, 30)),
            'parietal': list(range(30, 45)),
            'occipital': list(range(45, 55)),
            'temporal': list(range(55, 62))
        }
        
    def extract(self, X):
        """
        Main extraction method.
        X shape: (n_windows, window_size, n_channels)
        Returns: (n_windows, n_features)
        """
        features = []
        
        for window in X:
            feats = self._extract_window(window)
            features.append(feats)
            
        return np.array(features, dtype=np.float32)
    
    def _extract_window(self, window):
        """Extract all features from single window."""
        
        # FFT-based features
        freqs, psd = self._compute_psd(window)
        band_powers = self._extract_band_powers(psd, freqs)
        
        # 1. Differential Entropy (5 per channel)
        de_features = np.log(band_powers + 1e-8)
        
        # 2. Band ratios (3 per channel)
        delta, theta, alpha, beta, gamma = band_powers
        ratios = np.array([
            beta / (alpha + 1e-8),
            theta / (beta + 1e-8),
            beta / (alpha + theta + 1e-8)
        ])
        
        # 3. Hjorth parameters (3 per channel)
        activity, mobility, complexity = self._hjorth(window)
        
        # 4. Statistical moments (4 per channel)
        moments = self._statistical_moments(window)
        
        # Combine per-channel features
        per_channel = np.concatenate([
            de_features,           # 5
            ratios,                # 3
            activity[None, :],     # 1
            mobility[None, :],     # 1
            complexity[None, :],   # 1
            moments                 # 4
        ])  # Total: 15 per channel
        
        # 5. Region aggregation (region_means × 15)
        region_features = self._aggregate_regions(per_channel)
        
        # 6. Global band statistics (2 per band × 5 bands = 10)
        global_stats = self._global_band_stats(band_powers)
        
        # Final feature vector
        final_features = np.concatenate([
            per_channel.flatten(),      # 62 × 15 = 930
            region_features,            # 5 × 15 = 75
            global_stats                # 10
        ])
        
        return final_features
    
    def _compute_psd(self, window):
        """Compute PSD using Welch's method."""
        freqs, psd = signal.welch(
            window.T, 
            fs=self.fs, 
            nperseg=min(64, self.window_size),
            axis=-1
        )
        return freqs, psd
    
    def _extract_band_powers(self, psd, freqs):
        """Extract mean power in each frequency band."""
        band_powers = []
        for (low, high) in self.bands.values():
            idx = (freqs >= low) & (freqs <= high)
            band_power = np.mean(psd[:, idx], axis=1)
            band_powers.append(band_power)
        return np.array(band_powers)
    
    def _hjorth(self, window):
        """Compute Hjorth parameters."""
        diff1 = np.diff(window, axis=0)
        diff2 = np.diff(diff1, axis=0)
        
        var0 = np.var(window, axis=0)
        var1 = np.var(diff1, axis=0)
        var2 = np.var(diff2, axis=0)
        
        activity = var0
        mobility = np.sqrt(var1 / (var0 + 1e-8))
        complexity = np.sqrt(var2 / (var1 + 1e-8)) / (mobility + 1e-8)
        
        return activity, mobility, complexity
    
    def _statistical_moments(self, window):
        """Compute statistical moments."""
        mean = np.mean(window, axis=0)
        std = np.std(window, axis=0)
        skew = stats.skew(window, axis=0)
        kurt = stats.kurtosis(window, axis=0)
        return np.array([mean, std, skew, kurt])
    
    def _aggregate_regions(self, per_channel):
        """Aggregate features by brain region."""
        per_channel = per_channel.reshape(15, 62)  # 15 features × 62 channels
        region_features = []
        
        for region_name, channels in self.regions.items():
            region_mean = np.mean(per_channel[:, channels], axis=1)
            region_features.append(region_mean)
            
        return np.concatenate(region_features)
    
    def _global_band_stats(self, band_powers):
        """Global statistics across all channels."""
        stats = []
        for band in band_powers:
            stats.append(np.mean(band))
            stats.append(np.std(band))
        return np.array(stats)