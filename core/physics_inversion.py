"""
Physics-based seismic inversion using Tikhonov regularization in log(AI) domain.
Implements: min_x ||W*D*x - s||^2 + λ||L(x - x_LFM)||^2
"""
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import lsqr
from scipy.optimize import minimize_scalar
from typing import Tuple, Dict, Optional
import warnings

class PhysicsBasedInversion:
    """
    Professional physics-based seismic inversion with Tikhonov regularization.
    
    Uses log(AI) domain for physical accuracy and geological constraints.
    """
    
    def __init__(self):
        self.ai_range = (2000, 12000)  # Realistic AI range [m/s·g/cm³]
        self.log_ai_range = (np.log(2000), np.log(12000))
        
    def band_limited_inversion_pro(self, 
                                  seismic_data: np.ndarray,
                                  wavelet: np.ndarray,
                                  low_freq_model: np.ndarray,
                                  dt: float,
                                  lambda_reg: float = 0.1,
                                  alpha_smooth: float = 0.01,
                                  freq_band: Tuple[float, float] = (5.0, 200.0)) -> Tuple[np.ndarray, Dict]:
        """
        Professional band-limited inversion using physics-based approach.
        
        Args:
            seismic_data: Input seismic traces [samples x traces]
            wavelet: Source wavelet
            low_freq_model: Low-frequency impedance model [samples x traces]
            dt: Sample interval in seconds
            lambda_reg: Tikhonov regularization parameter
            alpha_smooth: Smoothness regularization parameter
            freq_band: Frequency band (low, high) in Hz
            
        Returns:
            impedance: Inverted acoustic impedance [samples x traces]
            info: Inversion information dictionary
        """
        n_samples, n_traces = seismic_data.shape
        
        # Convert to log domain for physics
        log_lfm = np.log(np.maximum(low_freq_model, 100))  # Avoid log(0)
        
        # Initialize results
        impedance = np.zeros_like(seismic_data)
        inversion_info = {
            'convergence': [],
            'residuals': [],
            'lambda_used': lambda_reg,
            'alpha_used': alpha_smooth,
            'freq_band': freq_band
        }
        
        # Process each trace independently for stability
        for trace_idx in range(n_traces):
            try:
                seismic_trace = seismic_data[:, trace_idx]
                lfm_trace = log_lfm[:, trace_idx]
                
                # Band-limit the seismic data
                seismic_filtered = self._bandpass_filter(seismic_trace, dt, freq_band)
                
                # Solve physics-based inversion
                log_ai_trace, trace_info = self._solve_tikhonov_inversion(
                    seismic_filtered, wavelet, lfm_trace, 
                    lambda_reg, alpha_smooth
                )
                
                # Convert back to linear domain with constraints
                ai_trace = np.exp(log_ai_trace)
                ai_trace = np.clip(ai_trace, self.ai_range[0], self.ai_range[1])
                
                impedance[:, trace_idx] = ai_trace
                
                # Store trace info
                inversion_info['convergence'].append(trace_info['convergence'])
                inversion_info['residuals'].append(trace_info['residual'])
                
            except Exception as e:
                # Fallback to LFM for failed traces
                impedance[:, trace_idx] = low_freq_model[:, trace_idx]
                inversion_info['convergence'].append(False)
                inversion_info['residuals'].append(np.inf)
                warnings.warn(f"Trace {trace_idx} inversion failed: {e}")
        
        # Calculate overall statistics
        inversion_info['success_rate'] = np.mean(inversion_info['convergence'])
        inversion_info['mean_residual'] = np.mean([r for r in inversion_info['residuals'] if np.isfinite(r)])
        
        return impedance, inversion_info
    
    def _solve_tikhonov_inversion(self, 
                                 seismic: np.ndarray,
                                 wavelet: np.ndarray,
                                 log_lfm: np.ndarray,
                                 lambda_reg: float,
                                 alpha_smooth: float) -> Tuple[np.ndarray, Dict]:
        """
        Solve Tikhonov regularized inversion in log(AI) domain.
        
        Minimizes: ||W*D*x - s||^2 + λ||L(x - x_LFM)||^2 + α||∇x||^2
        """
        n = len(seismic)
        
        # Build forward operator W*D (wavelet convolution + derivative)
        W = self._build_convolution_matrix(wavelet, n)
        D = self._build_derivative_matrix(n)
        WD = W @ D
        
        # Build regularization operators
        L_data = eye(n)  # Data fit regularization
        L_smooth = self._build_smoothness_matrix(n)  # Smoothness regularization
        
        # Set up regularized system
        # [W*D; λ*L; α*L_smooth] * x = [s; λ*x_LFM; 0]
        A_data = WD
        A_reg = lambda_reg * L_data
        A_smooth = alpha_smooth * L_smooth
        
        # Stack matrices
        A = np.vstack([A_data.toarray(), A_reg.toarray(), A_smooth.toarray()])
        
        # Stack right-hand side
        b_data = seismic
        b_reg = lambda_reg * log_lfm
        b_smooth = np.zeros(n-1)  # Smoothness target is zero gradient
        
        b = np.concatenate([b_data, b_reg, b_smooth])
        
        # Solve using LSQR for stability
        try:
            x, istop, itn, r1norm = lsqr(A, b, atol=1e-8, btol=1e-8, iter_lim=1000)[:4]
            
            convergence = istop in [1, 2]  # Successful convergence codes
            residual = r1norm
            
        except Exception as e:
            # Fallback solution
            x = log_lfm.copy()
            convergence = False
            residual = np.inf
            warnings.warn(f"LSQR failed: {e}, using LFM")
        
        # Apply physical constraints in log domain
        x = np.clip(x, self.log_ai_range[0], self.log_ai_range[1])
        
        info = {
            'convergence': convergence,
            'residual': residual,
            'iterations': itn if 'itn' in locals() else 0
        }
        
        return x, info
    
    def _build_convolution_matrix(self, wavelet: np.ndarray, n: int) -> np.ndarray:
        """Build convolution matrix for wavelet operator W."""
        from scipy.linalg import toeplitz
        
        # Pad wavelet to match signal length
        w_len = len(wavelet)
        
        # Create Toeplitz matrix for convolution
        # First column: wavelet padded with zeros
        first_col = np.zeros(n)
        first_col[:min(w_len, n)] = wavelet[:min(w_len, n)]
        
        # First row: first element of wavelet, then zeros
        first_row = np.zeros(n)
        first_row[0] = wavelet[0] if w_len > 0 else 0
        
        W = toeplitz(first_col, first_row)
        
        return W
    
    def _build_derivative_matrix(self, n: int) -> np.ndarray:
        """Build first derivative matrix D."""
        # Forward difference: D[i,i] = -1, D[i,i+1] = 1
        diag_main = -np.ones(n)
        diag_upper = np.ones(n-1)
        
        D = diags([diag_main, diag_upper], [0, 1], shape=(n, n))
        
        return D
    
    def _build_smoothness_matrix(self, n: int) -> np.ndarray:
        """Build second derivative matrix for smoothness regularization."""
        # Second derivative: [1, -2, 1] pattern
        if n < 3:
            return eye(max(1, n-1))
        
        diag_main = -2 * np.ones(n-1)
        diag_lower = np.ones(n-2)
        diag_upper = np.ones(n-2)
        
        L = diags([diag_lower, diag_main, diag_upper], [-1, 0, 1], shape=(n-1, n))
        
        return L
    
    def _bandpass_filter(self, data: np.ndarray, dt: float, 
                        freq_band: Tuple[float, float]) -> np.ndarray:
        """Apply bandpass filter using FFT."""
        try:
            from scipy.signal import butter, filtfilt
            
            # Design Butterworth filter
            nyquist = 0.5 / dt
            low_norm = freq_band[0] / nyquist
            high_norm = freq_band[1] / nyquist
            
            # Ensure valid frequency range
            low_norm = max(0.01, min(low_norm, 0.99))
            high_norm = max(low_norm + 0.01, min(high_norm, 0.99))
            
            b, a = butter(4, [low_norm, high_norm], btype='band')
            filtered = filtfilt(b, a, data)
            
            return filtered
            
        except ImportError:
            # Fallback FFT filtering
            return self._fft_bandpass_filter(data, dt, freq_band)
    
    def _fft_bandpass_filter(self, data: np.ndarray, dt: float,
                           freq_band: Tuple[float, float]) -> np.ndarray:
        """FFT-based bandpass filter fallback."""
        # FFT
        fft_data = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data), dt)
        
        # Create filter mask
        mask = (np.abs(freqs) >= freq_band[0]) & (np.abs(freqs) <= freq_band[1])
        
        # Apply filter
        fft_filtered = fft_data * mask
        
        # IFFT
        filtered = np.real(np.fft.ifft(fft_filtered))
        
        return filtered
    
    def forward_model(self, impedance: np.ndarray, wavelet: np.ndarray) -> np.ndarray:
        """
        Forward model: AI -> seismic using physics-based approach.
        
        s = W * D * log(AI)
        """
        # Convert to log domain
        log_ai = np.log(np.maximum(impedance, 100))
        
        # Calculate reflection coefficients (derivative in log domain)
        refl_coeff = np.diff(log_ai, axis=0)
        
        # Pad to match original size
        refl_padded = np.zeros_like(log_ai)
        refl_padded[1:, :] = refl_coeff
        
        # Convolve with wavelet
        synthetic = np.zeros_like(impedance)
        
        for trace in range(impedance.shape[1]):
            synthetic[:, trace] = np.convolve(refl_padded[:, trace], wavelet, mode='same')
        
        return synthetic
    
    def calculate_qc_metrics(self, original_seismic: np.ndarray, 
                           synthetic_seismic: np.ndarray,
                           impedance: np.ndarray) -> Dict:
        """Calculate comprehensive QC metrics."""
        
        # Correlation coefficient
        correlation = np.corrcoef(original_seismic.flatten(), 
                                synthetic_seismic.flatten())[0, 1]
        
        # RMS error
        rms_error = np.sqrt(np.mean((original_seismic - synthetic_seismic) ** 2))
        
        # Normalized RMS
        rms_normalized = rms_error / np.std(original_seismic)
        
        # Signal-to-noise ratio
        signal_power = np.var(synthetic_seismic)
        noise_power = np.var(original_seismic - synthetic_seismic)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # Impedance statistics
        ai_stats = {
            'min': np.min(impedance),
            'max': np.max(impedance),
            'mean': np.mean(impedance),
            'std': np.std(impedance),
            'range_realistic': (np.min(impedance) >= 1500) and (np.max(impedance) <= 15000)
        }
        
        # Frequency domain analysis
        freq_metrics = self._analyze_frequency_content(original_seismic, synthetic_seismic)
        
        return {
            'correlation': correlation,
            'rms_error': rms_error,
            'rms_normalized': rms_normalized,
            'snr_db': snr_db,
            'impedance_stats': ai_stats,
            'frequency_metrics': freq_metrics,
            'overall_quality': self._calculate_overall_quality(correlation, rms_normalized, snr_db)
        }
    
    def _analyze_frequency_content(self, original: np.ndarray, 
                                 synthetic: np.ndarray) -> Dict:
        """Analyze frequency domain characteristics."""
        
        # Average traces for analysis
        orig_avg = np.mean(original, axis=1)
        synth_avg = np.mean(synthetic, axis=1)
        
        # FFT
        orig_fft = np.abs(np.fft.fft(orig_avg))
        synth_fft = np.abs(np.fft.fft(synth_avg))
        
        # Spectral correlation
        spectral_corr = np.corrcoef(orig_fft, synth_fft)[0, 1]
        
        # Dominant frequency
        freqs = np.fft.fftfreq(len(orig_avg))
        orig_dom_freq = freqs[np.argmax(orig_fft[:len(orig_fft)//2])]
        synth_dom_freq = freqs[np.argmax(synth_fft[:len(synth_fft)//2])]
        
        return {
            'spectral_correlation': spectral_corr,
            'original_dominant_freq': orig_dom_freq,
            'synthetic_dominant_freq': synth_dom_freq,
            'frequency_match': abs(orig_dom_freq - synth_dom_freq) < 0.1
        }
    
    def _calculate_overall_quality(self, correlation: float, 
                                 rms_normalized: float, snr_db: float) -> str:
        """Calculate overall inversion quality assessment."""
        
        # Quality thresholds
        if correlation > 0.8 and rms_normalized < 0.3 and snr_db > 15:
            return "Excellent"
        elif correlation > 0.6 and rms_normalized < 0.5 and snr_db > 10:
            return "Good"
        elif correlation > 0.4 and rms_normalized < 0.7 and snr_db > 5:
            return "Fair"
        else:
            return "Poor"