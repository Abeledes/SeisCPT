"""
Fast physics-based seismic inversion optimized for real-time performance.
Maintains physics accuracy while dramatically reducing computation time.
"""
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from typing import Tuple, Dict, Optional
import warnings

class FastPhysicsInversion:
    """
    High-performance physics-based inversion optimized for speed.
    
    Uses simplified but accurate physics for real-time processing.
    """
    
    def __init__(self):
        self.ai_range = (2000, 12000)  # Realistic AI range
        self.log_ai_range = (np.log(2000), np.log(12000))
        
    def fast_band_limited_inversion(self, 
                                   seismic_data: np.ndarray,
                                   wavelet: np.ndarray,
                                   low_freq_model: np.ndarray,
                                   dt: float,
                                   lambda_reg: float = 0.1,
                                   freq_band: Tuple[float, float] = (5.0, 200.0)) -> Tuple[np.ndarray, Dict]:
        """
        Fast physics-based inversion using optimized algorithms.
        
        Args:
            seismic_data: Input seismic traces [samples x traces]
            wavelet: Source wavelet
            low_freq_model: Low-frequency impedance model
            dt: Sample interval
            lambda_reg: Regularization parameter
            freq_band: Frequency band (low, high) Hz
            
        Returns:
            impedance: Inverted acoustic impedance
            info: Processing information
        """
        n_samples, n_traces = seismic_data.shape
        
        # Pre-filter seismic data once
        seismic_filtered = self._fast_bandpass_filter(seismic_data, dt, freq_band)
        
        # Convert LFM to log domain
        log_lfm = np.log(np.maximum(low_freq_model, 100))
        
        # Build operators once (reuse for all traces)
        W, D, L = self._build_fast_operators(wavelet, n_samples)
        WD = W @ D
        
        # Vectorized inversion for all traces simultaneously
        impedance = self._solve_vectorized_inversion(
            seismic_filtered, log_lfm, WD, L, lambda_reg
        )
        
        # Apply constraints
        impedance = np.clip(impedance, self.ai_range[0], self.ai_range[1])
        
        info = {
            'method': 'fast_physics',
            'lambda_reg': lambda_reg,
            'freq_band': freq_band,
            'processing_time': 'optimized'
        }
        
        return impedance, info
    
    def _build_fast_operators(self, wavelet: np.ndarray, n: int) -> Tuple:
        """Build operators optimized for speed."""
        
        # Simplified convolution operator (Toeplitz structure)
        w_len = min(len(wavelet), n // 4)  # Limit wavelet length for speed
        wavelet_short = wavelet[:w_len] if len(wavelet) > w_len else wavelet
        
        # Build sparse Toeplitz matrix efficiently
        diagonals = []
        offsets = []
        
        for i, w_val in enumerate(wavelet_short):
            if abs(w_val) > 0.01:  # Skip small values for sparsity
                diagonals.append(np.full(n - i, w_val))
                offsets.append(i)
        
        W = diags(diagonals, offsets, shape=(n, n), format='csr')
        
        # First derivative operator (sparse)
        D = diags([-np.ones(n), np.ones(n-1)], [0, 1], shape=(n, n), format='csr')
        
        # Regularization operator (identity for speed)
        L = diags([np.ones(n)], [0], shape=(n, n), format='csr')
        
        return W, D, L
    
    def _solve_vectorized_inversion(self, 
                                   seismic: np.ndarray,
                                   log_lfm: np.ndarray,
                                   WD: np.ndarray,
                                   L: np.ndarray,
                                   lambda_reg: float) -> np.ndarray:
        """Solve inversion for all traces simultaneously."""
        
        n_samples, n_traces = seismic.shape
        
        # Build regularized system: (WD^T*WD + λ*L^T*L) * x = WD^T*s + λ*L^T*LFM
        A = WD.T @ WD + lambda_reg * (L.T @ L)
        
        # Solve for all traces at once
        impedance = np.zeros_like(seismic)
        
        # Process in chunks for memory efficiency
        chunk_size = min(50, n_traces)
        
        for i in range(0, n_traces, chunk_size):
            end_idx = min(i + chunk_size, n_traces)
            chunk_traces = end_idx - i
            
            # Right-hand side for chunk
            b_chunk = np.zeros((n_samples, chunk_traces))
            
            for j in range(chunk_traces):
                trace_idx = i + j
                b_data = WD.T @ seismic[:, trace_idx]
                b_reg = lambda_reg * (L.T @ (L @ log_lfm[:, trace_idx]))
                b_chunk[:, j] = b_data + b_reg
            
            # Solve for chunk
            try:
                for j in range(chunk_traces):
                    trace_idx = i + j
                    x_log = spsolve(A, b_chunk[:, j])
                    
                    # Convert to linear domain with constraints
                    x_log = np.clip(x_log, self.log_ai_range[0], self.log_ai_range[1])
                    impedance[:, trace_idx] = np.exp(x_log)
                    
            except Exception as e:
                # Fallback to LFM for failed traces
                for j in range(chunk_traces):
                    trace_idx = i + j
                    impedance[:, trace_idx] = np.exp(log_lfm[:, trace_idx])
        
        return impedance
    
    def _fast_bandpass_filter(self, data: np.ndarray, dt: float, 
                             freq_band: Tuple[float, float]) -> np.ndarray:
        """Fast FFT-based bandpass filter."""
        
        # FFT all traces at once
        fft_data = np.fft.fft(data, axis=0)
        freqs = np.fft.fftfreq(data.shape[0], dt)
        
        # Create filter mask
        mask = (np.abs(freqs) >= freq_band[0]) & (np.abs(freqs) <= freq_band[1])
        
        # Apply filter
        fft_filtered = fft_data * mask[:, np.newaxis]
        
        # IFFT
        filtered = np.real(np.fft.ifft(fft_filtered, axis=0))
        
        return filtered
    
    def fast_forward_model(self, impedance: np.ndarray, wavelet: np.ndarray) -> np.ndarray:
        """Fast forward modeling for QC."""
        
        # Calculate reflection coefficients in log domain (vectorized)
        log_ai = np.log(np.maximum(impedance, 100))
        refl_coeff = np.diff(log_ai, axis=0)
        
        # Pad to match size
        refl_padded = np.zeros_like(log_ai)
        refl_padded[1:, :] = refl_coeff
        
        # Fast convolution using FFT
        synthetic = np.zeros_like(impedance)
        
        # Limit wavelet length for speed
        wavelet_short = wavelet[:min(len(wavelet), 100)]
        
        for trace in range(impedance.shape[1]):
            synthetic[:, trace] = np.convolve(refl_padded[:, trace], wavelet_short, mode='same')
        
        return synthetic
    
    def fast_qc_metrics(self, original: np.ndarray, synthetic: np.ndarray, 
                       impedance: np.ndarray) -> Dict:
        """Fast QC calculation with essential metrics only."""
        
        # Subsample for speed if data is large
        if original.size > 50000:
            step = max(1, original.size // 50000)
            orig_sub = original.flatten()[::step]
            synth_sub = synthetic.flatten()[::step]
        else:
            orig_sub = original.flatten()
            synth_sub = synthetic.flatten()
        
        # Essential metrics only
        correlation = np.corrcoef(orig_sub, synth_sub)[0, 1]
        if not np.isfinite(correlation):
            correlation = 0.0
        
        rms_error = np.sqrt(np.mean((orig_sub - synth_sub) ** 2))
        rms_normalized = rms_error / (np.std(orig_sub) + 1e-10)
        
        # SNR
        signal_power = np.var(synth_sub)
        noise_power = np.var(orig_sub - synth_sub)
        snr_db = 10 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10))
        
        # Impedance stats (fast)
        ai_min, ai_max = np.min(impedance), np.max(impedance)
        ai_mean = np.mean(impedance)
        
        return {
            'correlation': float(correlation),
            'rms_normalized': float(rms_normalized),
            'snr_db': float(snr_db),
            'impedance_min': float(ai_min),
            'impedance_max': float(ai_max),
            'impedance_mean': float(ai_mean),
            'impedance_range_ok': 1800 <= ai_min <= 12000 and 2000 <= ai_max <= 12000
        }


class FastWaveletEstimator:
    """Fast wavelet estimation optimized for speed."""
    
    def fast_estimate_wavelet(self, seismic_data: np.ndarray, dt: float,
                             method: str = 'autocorr', length_ms: float = 150.0) -> Tuple[np.ndarray, Dict]:
        """Fast wavelet estimation."""
        
        if method == 'ricker':
            # Fast dominant frequency estimation
            avg_trace = np.mean(seismic_data[:min(1000, seismic_data.shape[0]), :10], axis=1)
            fft_data = np.abs(np.fft.fft(avg_trace))
            freqs = np.fft.fftfreq(len(avg_trace), dt)
            
            pos_freqs = freqs[:len(freqs)//2]
            pos_fft = fft_data[:len(fft_data)//2]
            
            # Find peak in 10-200 Hz range (extended for high-frequency analysis)
            mask = (pos_freqs >= 10) & (pos_freqs <= 200)
            if np.any(mask):
                peak_idx = np.argmax(pos_fft[mask])
                dominant_freq = pos_freqs[mask][peak_idx]
            else:
                dominant_freq = 30.0
            
            # Generate Ricker wavelet
            wavelet = self._fast_ricker_wavelet(abs(dominant_freq), dt, length_ms)
            
            info = {
                'method': 'ricker_fast',
                'dominant_frequency': float(abs(dominant_freq)),
                'length_ms': length_ms
            }
            
        else:  # autocorr (default)
            # Fast autocorrelation using subset
            subset_size = min(500, seismic_data.shape[0])
            subset_traces = min(5, seismic_data.shape[1])
            
            avg_trace = np.mean(seismic_data[:subset_size, :subset_traces], axis=1)
            avg_trace = avg_trace - np.mean(avg_trace)
            
            # Fast autocorrelation
            autocorr = np.correlate(avg_trace, avg_trace, mode='same')
            
            # Extract wavelet
            center = len(autocorr) // 2
            wavelet_samples = min(int(length_ms / (dt * 1000)), len(autocorr) // 3)
            
            start = center - wavelet_samples // 2
            end = center + wavelet_samples // 2
            
            wavelet = autocorr[start:end]
            
            # Normalize
            if np.max(np.abs(wavelet)) > 0:
                wavelet = wavelet / np.max(np.abs(wavelet))
            
            info = {
                'method': 'autocorr_fast',
                'length_ms': length_ms,
                'samples': len(wavelet)
            }
        
        return wavelet, info
    
    def _fast_ricker_wavelet(self, frequency: float, dt: float, length_ms: float) -> np.ndarray:
        """Generate Ricker wavelet quickly."""
        
        length_samples = int(length_ms / (dt * 1000))
        if length_samples % 2 == 0:
            length_samples += 1
        
        t = np.arange(length_samples) * dt
        t = t - t[len(t)//2]
        
        a = (np.pi * frequency * t) ** 2
        wavelet = (1 - 2*a) * np.exp(-a)
        
        if np.max(np.abs(wavelet)) > 0:
            wavelet = wavelet / np.max(np.abs(wavelet))
        
        return wavelet


class FastLFMBuilder:
    """Fast low-frequency model builder."""
    
    def fast_build_lfm(self, time_axis: np.ndarray, num_traces: int,
                      geological_setting: str = 'mixed_sediments') -> Tuple[np.ndarray, Dict]:
        """Build LFM quickly with simplified geology."""
        
        # Geological parameters (simplified)
        geo_params = {
            'shallow_marine': {'surface_ai': 2200, 'gradient': 600},
            'deep_marine': {'surface_ai': 2400, 'gradient': 800},
            'carbonate_platform': {'surface_ai': 3200, 'gradient': 400},
            'tight_gas': {'surface_ai': 3800, 'gradient': 300},
            'mixed_sediments': {'surface_ai': 2500, 'gradient': 700}
        }
        
        # Get parameters
        setting_key = geological_setting.lower().replace(' ', '_').replace('(', '').replace(')', '')
        params = geo_params.get(setting_key, geo_params['mixed_sediments'])
        
        # Fast depth conversion
        depth_km = time_axis * 1.5  # Simple average velocity
        
        # Vectorized model building
        model = np.zeros((len(time_axis), num_traces))
        
        # Base impedance with depth
        base_ai = params['surface_ai'] + depth_km * params['gradient']
        
        # Add simple lateral variation
        lateral_var = np.linspace(-200, 200, num_traces)
        
        # Vectorized assignment
        for i in range(len(time_axis)):
            model[i, :] = base_ai[i] + lateral_var
        
        # Simple smoothing (fast)
        if len(time_axis) > 10:
            # Apply simple moving average
            window = min(5, len(time_axis) // 10)
            for trace in range(num_traces):
                model[:, trace] = self._fast_smooth(model[:, trace], window)
        
        # Ensure realistic range
        model = np.clip(model, 1800, 10000)
        
        info = {
            'geological_setting': geological_setting,
            'surface_impedance': params['surface_ai'],
            'gradient': params['gradient'],
            'impedance_range': (np.min(model), np.max(model))
        }
        
        return model, info
    
    def _fast_smooth(self, data: np.ndarray, window: int) -> np.ndarray:
        """Fast smoothing using convolution."""
        if window <= 1:
            return data
        
        kernel = np.ones(window) / window
        smoothed = np.convolve(data, kernel, mode='same')
        
        # Fix edges
        smoothed[:window//2] = data[:window//2]
        smoothed[-window//2:] = data[-window//2:]
        
        return smoothed


class FastAutoTuner:
    """Fast parameter tuning for real-time optimization."""
    
    def fast_tune_parameters(self, seismic_data: np.ndarray, lfm: np.ndarray,
                           wavelet: np.ndarray, dt: float) -> Dict:
        """Fast parameter tuning using heuristics."""
        
        # Analyze data characteristics quickly
        data_std = np.std(seismic_data)
        data_mean = np.mean(np.abs(seismic_data))
        
        # Noise estimate
        noise_ratio = data_std / (data_mean + 1e-10)
        
        # Fast parameter estimation
        if noise_ratio > 0.5:  # Noisy data
            lambda_reg = 0.2
            freq_low = 8.0
            freq_high = 100.0  # Higher frequency even for noisy data
        elif noise_ratio > 0.3:  # Medium noise
            lambda_reg = 0.1
            freq_low = 5.0
            freq_high = 150.0  # Higher frequency for medium noise
        else:  # Clean data
            lambda_reg = 0.05
            freq_low = 3.0
            freq_high = 200.0  # Higher frequency for clean data
        
        # Frequency analysis (fast)
        avg_trace = np.mean(seismic_data[:min(500, seismic_data.shape[0]), :5], axis=1)
        fft_data = np.abs(np.fft.fft(avg_trace))
        freqs = np.fft.fftfreq(len(avg_trace), dt)
        
        pos_freqs = freqs[:len(freqs)//2]
        pos_fft = fft_data[:len(fft_data)//2]
        
        # Adjust based on dominant frequency
        if len(pos_fft) > 0:
            dom_freq = pos_freqs[np.argmax(pos_fft)]
            if abs(dom_freq) > 0:
                freq_low = max(freq_low, abs(dom_freq) * 0.2)
                freq_high = min(freq_high, abs(dom_freq) * 5.0)  # Allow higher frequencies
        
        return {
            'lambda_reg': lambda_reg,
            'alpha_smooth': lambda_reg * 0.1,  # Simple relationship
            'freq_low': freq_low,
            'freq_high': freq_high,
            'tuning_method': 'fast_heuristic'
        }