"""
Advanced wavelet estimation for seismic inversion.
Supports Ricker wavelets and statistical autocorrelation methods.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from typing import Tuple, Dict, Optional

class WaveletEstimator:
    """
    Professional wavelet estimation for seismic inversion.
    """
    
    def __init__(self):
        self.wavelet_types = ['ricker', 'autocorr', 'statistical']
        
    def estimate_wavelet(self, seismic_data: np.ndarray, dt: float,
                        method: str = 'autocorr',
                        length_ms: float = 10.0,
                        freq_range: Tuple[float, float] = (5.0, 1500.0)) -> Tuple[np.ndarray, Dict]:
        """
        Estimate source wavelet from seismic data.
        
        Args:
            seismic_data: Input seismic traces [samples x traces]
            dt: Sample interval in seconds
            method: Estimation method ('ricker', 'autocorr', 'statistical')
            length_ms: Wavelet length in milliseconds
            freq_range: Frequency range for analysis
            
        Returns:
            wavelet: Estimated wavelet
            info: Estimation information
        """
        
        if method == 'ricker':
            return self._estimate_ricker_wavelet(seismic_data, dt, length_ms, freq_range)
        elif method == 'autocorr':
            return self._estimate_autocorr_wavelet(seismic_data, dt, length_ms)
        elif method == 'statistical':
            return self._estimate_statistical_wavelet(seismic_data, dt, length_ms, freq_range)
        else:
            raise ValueError(f"Unknown wavelet estimation method: {method}")
    
    def _estimate_ricker_wavelet(self, seismic_data: np.ndarray, dt: float,
                               length_ms: float, freq_range: Tuple[float, float]) -> Tuple[np.ndarray, Dict]:
        """
        Estimate Ricker wavelet by finding optimal dominant frequency.
        """
        
        # Estimate dominant frequency from data
        dominant_freq = self._estimate_dominant_frequency(seismic_data, dt, freq_range)
        
        # Generate Ricker wavelet
        wavelet = self._generate_ricker_wavelet(dominant_freq, dt, length_ms)
        
        # Quality assessment
        quality = self._assess_wavelet_quality(seismic_data, wavelet)
        
        info = {
            'method': 'ricker',
            'dominant_frequency': dominant_freq,
            'length_ms': length_ms,
            'quality_score': quality,
            'parameters': {'frequency': dominant_freq}
        }
        
        return wavelet, info
    
    def _estimate_autocorr_wavelet(self, seismic_data: np.ndarray, dt: float,
                                 length_ms: float) -> Tuple[np.ndarray, Dict]:
        """
        Estimate wavelet using autocorrelation method.
        """
        
        # Calculate average trace
        avg_trace = np.mean(seismic_data, axis=1)
        
        # Remove DC component
        avg_trace = avg_trace - np.mean(avg_trace)
        
        # Calculate autocorrelation
        autocorr = np.correlate(avg_trace, avg_trace, mode='full')
        
        # Extract wavelet around zero lag
        center = len(autocorr) // 2
        wavelet_samples = int(length_ms / (dt * 1000))
        
        start = center - wavelet_samples // 2
        end = center + wavelet_samples // 2
        
        if start < 0 or end >= len(autocorr):
            # Fallback to shorter wavelet
            wavelet_samples = min(wavelet_samples, len(autocorr) // 2)
            start = center - wavelet_samples // 2
            end = center + wavelet_samples // 2
        
        wavelet = autocorr[start:end]
        
        # Normalize
        if np.max(np.abs(wavelet)) > 0:
            wavelet = wavelet / np.max(np.abs(wavelet))
        
        # Ensure zero-phase (symmetric)
        wavelet = self._make_zero_phase(wavelet)
        
        # Quality assessment
        quality = self._assess_wavelet_quality(seismic_data, wavelet)
        
        info = {
            'method': 'autocorr',
            'length_ms': length_ms,
            'quality_score': quality,
            'zero_phase': True,
            'samples': len(wavelet)
        }
        
        return wavelet, info
    
    def _estimate_statistical_wavelet(self, seismic_data: np.ndarray, dt: float,
                                    length_ms: float, freq_range: Tuple[float, float]) -> Tuple[np.ndarray, Dict]:
        """
        Estimate wavelet using statistical optimization.
        """
        
        # Initial guess: Ricker wavelet
        initial_freq = self._estimate_dominant_frequency(seismic_data, dt, freq_range)
        
        # Optimize wavelet parameters
        def objective(freq):
            test_wavelet = self._generate_ricker_wavelet(freq, dt, length_ms)
            return -self._assess_wavelet_quality(seismic_data, test_wavelet)
        
        # Optimize frequency
        result = minimize_scalar(objective, bounds=freq_range, method='bounded')
        optimal_freq = result.x
        
        # Generate optimal wavelet
        wavelet = self._generate_ricker_wavelet(optimal_freq, dt, length_ms)
        
        # Further optimize phase
        wavelet = self._optimize_wavelet_phase(seismic_data, wavelet)
        
        quality = self._assess_wavelet_quality(seismic_data, wavelet)
        
        info = {
            'method': 'statistical',
            'optimal_frequency': optimal_freq,
            'initial_frequency': initial_freq,
            'length_ms': length_ms,
            'quality_score': quality,
            'optimization_success': result.success
        }
        
        return wavelet, info
    
    def _generate_ricker_wavelet(self, frequency: float, dt: float, length_ms: float) -> np.ndarray:
        """
        Generate Ricker wavelet with specified parameters.
        """
        
        # Time vector
        length_samples = int(length_ms / (dt * 1000))
        if length_samples % 2 == 0:
            length_samples += 1  # Ensure odd length for symmetry
        
        t = np.arange(length_samples) * dt
        t = t - t[len(t)//2]  # Center around zero
        
        # Ricker wavelet formula
        a = (np.pi * frequency * t) ** 2
        wavelet = (1 - 2*a) * np.exp(-a)
        
        # Normalize
        if np.max(np.abs(wavelet)) > 0:
            wavelet = wavelet / np.max(np.abs(wavelet))
        
        return wavelet
    
    def _estimate_dominant_frequency(self, seismic_data: np.ndarray, dt: float,
                                   freq_range: Tuple[float, float]) -> float:
        """
        Estimate dominant frequency from seismic data.
        """
        
        # Average trace for analysis
        avg_trace = np.mean(seismic_data, axis=1)
        
        # FFT
        fft_data = np.abs(np.fft.fft(avg_trace))
        freqs = np.fft.fftfreq(len(avg_trace), dt)
        
        # Consider only positive frequencies in range
        pos_freqs = freqs[:len(freqs)//2]
        pos_fft = fft_data[:len(fft_data)//2]
        
        # Filter to frequency range
        mask = (pos_freqs >= freq_range[0]) & (pos_freqs <= freq_range[1])
        if np.any(mask):
            filtered_freqs = pos_freqs[mask]
            filtered_fft = pos_fft[mask]
            
            # Find peak frequency
            peak_idx = np.argmax(filtered_fft)
            dominant_freq = filtered_freqs[peak_idx]
        else:
            # Fallback to middle of range
            dominant_freq = (freq_range[0] + freq_range[1]) / 2
        
        return abs(dominant_freq)
    
    def _assess_wavelet_quality(self, seismic_data: np.ndarray, wavelet: np.ndarray) -> float:
        """
        Assess wavelet quality by forward modeling correlation.
        """
        
        try:
            # Simple forward model test
            n_traces = min(10, seismic_data.shape[1])  # Test on subset
            correlations = []
            
            for i in range(n_traces):
                trace = seismic_data[:, i]
                
                # Simple reflectivity model (derivative)
                refl = np.diff(trace)
                refl = np.concatenate([[0], refl])  # Pad to original length
                
                # Convolve with wavelet
                synthetic = np.convolve(refl, wavelet, mode='same')
                
                # Calculate correlation
                if np.std(trace) > 0 and np.std(synthetic) > 0:
                    corr = np.corrcoef(trace, synthetic)[0, 1]
                    if np.isfinite(corr):
                        correlations.append(abs(corr))
            
            if correlations:
                return np.mean(correlations)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _make_zero_phase(self, wavelet: np.ndarray) -> np.ndarray:
        """
        Convert wavelet to zero-phase (symmetric).
        """
        
        # FFT
        fft_wavelet = np.fft.fft(wavelet)
        
        # Keep magnitude, zero phase
        magnitude = np.abs(fft_wavelet)
        zero_phase_fft = magnitude + 0j
        
        # IFFT
        zero_phase_wavelet = np.real(np.fft.ifft(zero_phase_fft))
        
        # Make symmetric by averaging with time-reversed version
        reversed_wavelet = zero_phase_wavelet[::-1]
        symmetric_wavelet = (zero_phase_wavelet + reversed_wavelet) / 2
        
        # Normalize
        if np.max(np.abs(symmetric_wavelet)) > 0:
            symmetric_wavelet = symmetric_wavelet / np.max(np.abs(symmetric_wavelet))
        
        return symmetric_wavelet
    
    def _optimize_wavelet_phase(self, seismic_data: np.ndarray, wavelet: np.ndarray) -> np.ndarray:
        """
        Optimize wavelet phase for best fit to data.
        """
        
        def phase_objective(phase_shift):
            # Apply phase shift
            fft_wavelet = np.fft.fft(wavelet)
            phase_factor = np.exp(1j * phase_shift)
            shifted_fft = fft_wavelet * phase_factor
            shifted_wavelet = np.real(np.fft.ifft(shifted_fft))
            
            # Assess quality
            return -self._assess_wavelet_quality(seismic_data, shifted_wavelet)
        
        # Optimize phase
        try:
            result = minimize_scalar(phase_objective, bounds=(-np.pi, np.pi), method='bounded')
            optimal_phase = result.x
            
            # Apply optimal phase shift
            fft_wavelet = np.fft.fft(wavelet)
            phase_factor = np.exp(1j * optimal_phase)
            shifted_fft = fft_wavelet * phase_factor
            optimized_wavelet = np.real(np.fft.ifft(shifted_fft))
            
            return optimized_wavelet
            
        except Exception:
            return wavelet
    
    def create_ricker_wavelet(self, frequency: float, dt: float, length_ms: float = 200.0) -> np.ndarray:
        """
        Create Ricker wavelet with specified parameters (public interface).
        """
        return self._generate_ricker_wavelet(frequency, dt, length_ms)
    
    def analyze_wavelet_spectrum(self, wavelet: np.ndarray, dt: float) -> Dict:
        """
        Analyze wavelet frequency spectrum.
        """
        
        # FFT
        fft_wavelet = np.abs(np.fft.fft(wavelet))
        freqs = np.fft.fftfreq(len(wavelet), dt)
        
        # Positive frequencies only
        pos_freqs = freqs[:len(freqs)//2]
        pos_fft = fft_wavelet[:len(fft_wavelet)//2]
        
        # Find dominant frequency
        peak_idx = np.argmax(pos_fft)
        dominant_freq = pos_freqs[peak_idx]
        
        # Calculate bandwidth (frequencies at -3dB)
        peak_amplitude = pos_fft[peak_idx]
        half_power = peak_amplitude / np.sqrt(2)
        
        # Find bandwidth
        above_half_power = pos_fft >= half_power
        if np.any(above_half_power):
            bandwidth_indices = np.where(above_half_power)[0]
            bandwidth = pos_freqs[bandwidth_indices[-1]] - pos_freqs[bandwidth_indices[0]]
        else:
            bandwidth = 0.0
        
        return {
            'dominant_frequency': abs(dominant_freq),
            'bandwidth': bandwidth,
            'peak_amplitude': peak_amplitude,
            'frequency_spectrum': (pos_freqs, pos_fft),
            'length_samples': len(wavelet),
            'length_ms': len(wavelet) * dt * 1000
        }