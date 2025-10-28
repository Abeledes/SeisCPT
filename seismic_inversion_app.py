"""
Professional Seismic Inversion Application
Advanced seismic amplitude to acoustic impedance inversion with multiple algorithms.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import io
import base64
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Import our SEG-Y reader
try:
    from segy_reader import read_segy_file, get_segy_info, read_segy_subset
    FULL_SEGY_AVAILABLE = True
except ImportError:
    from basic_segy_reader import read_segy_headers
    FULL_SEGY_AVAILABLE = False
    
    # Placeholder function for subset reading
    def read_segy_subset(filename, max_traces=1000, trace_start=0):
        return {'error': 'Numpy required for subset reading'}

# Try to import scipy for advanced filtering
try:
    from scipy.signal import medfilt, butter, filtfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="SeisCPT - Professional Seismic Inversion",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure Streamlit for large files
import streamlit.web.cli as stcli
import sys

# Set maximum upload size to 10GB programmatically
if hasattr(st, '_config'):
    st._config.set_option('server.maxUploadSize', 10240)
    st._config.set_option('server.maxMessageSize', 1024)

# Professional styling
def apply_professional_styling():
    """Apply professional CSS styling for seismic applications"""
    st.markdown("""
    <style>
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%);
        color: #e8eaed !important;
    }
    
    /* Headers */
    .stMarkdown h1 {
        color: #4fc3f7 !important;
        font-weight: 600;
        text-align: center;
        background: linear-gradient(135deg, #4fc3f7, #29b6f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stMarkdown h2, .stMarkdown h3 {
        color: #81c784 !important;
        font-weight: 600;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f3a 0%, #2d3748 100%);
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: rgba(79, 195, 247, 0.1);
        border: 1px solid rgba(79, 195, 247, 0.3);
        padding: 1rem;
        border-radius: 10px;
    }
    
    /* Info boxes */
    .stInfo {
        background: rgba(129, 199, 132, 0.1);
        border-left: 4px solid #81c784;
    }
    
    /* Success boxes */
    .stSuccess {
        background: rgba(102, 187, 106, 0.1);
        border-left: 4px solid #66bb6a;
    }
    
    /* Warning boxes */
    .stWarning {
        background: rgba(255, 183, 77, 0.1);
        border-left: 4px solid #ffb74d;
    }
    
    /* Error boxes */
    .stError {
        background: rgba(239, 83, 80, 0.1);
        border-left: 4px solid #ef5350;
    }
    </style>
    """, unsafe_allow_html=True)


class SeismicInversion:
    """
    Professional seismic inversion algorithms with geological realism.
    """
    
    def __init__(self):
        self.methods = {
            'recursive': 'Recursive Inversion',
            'model_based': 'Model-Based Inversion', 
            'sparse_spike': 'Sparse Spike Inversion',
            'colored_inversion': 'Colored Inversion',
            'band_limited': 'Band-Limited Inversion'
        }
        
        # Realistic sediment properties (m/s * g/cm³)
        self.sediment_properties = {
            'water': {'ai_min': 1480, 'ai_max': 1520, 'density': 1.0},
            'unconsolidated_sand': {'ai_min': 1800, 'ai_max': 2800, 'density': 1.8},
            'consolidated_sand': {'ai_min': 2800, 'ai_max': 4500, 'density': 2.1},
            'shale': {'ai_min': 2000, 'ai_max': 3500, 'density': 2.2},
            'limestone': {'ai_min': 4000, 'ai_max': 7000, 'density': 2.4},
            'dolomite': {'ai_min': 5000, 'ai_max': 8000, 'density': 2.6},
            'salt': {'ai_min': 4500, 'ai_max': 4700, 'density': 2.1},
            'coal': {'ai_min': 1200, 'ai_max': 2000, 'density': 1.3},
            'tight_gas_sand': {'ai_min': 3500, 'ai_max': 5500, 'density': 2.3}
        }
        
        # Typical velocity-depth relationships (Gardner's equation parameters)
        self.gardner_params = {
            'a': 310,  # Gardner's constant
            'b': 0.25  # Gardner's exponent
        }
        
        # Current geological setting
        self.geological_setting = "Mixed Sediments (Default)"
        
    def _set_geological_setting(self, setting: str):
        """Set geological constraints based on depositional environment"""
        self.geological_setting = setting
        
        # Adjust sediment properties based on setting
        if setting == "Shallow Marine":
            self.ai_range = (1800, 4500)
            self.typical_gradient = 0.6  # km/s per km
        elif setting == "Deep Marine":
            self.ai_range = (2200, 5500)
            self.typical_gradient = 0.8
        elif setting == "Fluvial/Deltaic":
            self.ai_range = (1900, 4200)
            self.typical_gradient = 0.7
        elif setting == "Carbonate Platform":
            self.ai_range = (2800, 7000)
            self.typical_gradient = 0.5
        elif setting == "Tight Gas Reservoir":
            self.ai_range = (3200, 6500)
            self.typical_gradient = 0.4
        elif setting == "Unconventional Shale":
            self.ai_range = (2200, 4800)
            self.typical_gradient = 0.6
        else:  # Mixed Sediments (Default)
            self.ai_range = (1800, 6000)
            self.typical_gradient = 0.7
    
    def recursive_inversion(self, seismic_data: np.ndarray, dt: float, 
                          initial_impedance: float = 2500.0) -> np.ndarray:
        """
        Enhanced recursive seismic inversion with geological constraints.
        
        Args:
            seismic_data: Seismic amplitude data (samples x traces)
            dt: Sample interval in seconds
            initial_impedance: Initial acoustic impedance value
            
        Returns:
            Geologically realistic acoustic impedance section
        """
        impedance = np.zeros_like(seismic_data, dtype=np.float32)
        
        # Create realistic initial model based on depth
        time_axis = np.arange(seismic_data.shape[0]) * dt
        depth_km = time_axis * 1.5  # Approximate depth conversion (1.5 km/s average velocity)
        
        # Initialize with depth-dependent background model
        background_ai = self._create_background_model(depth_km, seismic_data.shape[1])
        impedance[0, :] = background_ai[0, :]
        
        # Enhanced recursive formula with geological constraints
        for i in range(1, seismic_data.shape[0]):
            # Normalize seismic amplitudes to realistic reflection coefficients
            reflection_coeff = self._normalize_amplitudes(seismic_data[i-1, :])
            
            # Apply recursive formula
            impedance[i, :] = impedance[i-1, :] * (1 + reflection_coeff) / (1 - reflection_coeff)
            
            # Apply geological constraints
            impedance[i, :] = self._apply_geological_constraints(
                impedance[i, :], depth_km[i], background_ai[i, :]
            )
        
        # Post-processing: smooth unrealistic variations
        impedance = self._smooth_geological_variations(impedance, dt)
        
        return impedance
    
    def model_based_inversion(self, seismic_data: np.ndarray, dt: float,
                            wavelet: Optional[np.ndarray] = None,
                            initial_model: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Enhanced model-based seismic inversion with geological realism.
        
        Args:
            seismic_data: Seismic amplitude data
            dt: Sample interval
            wavelet: Source wavelet (if None, estimates from data)
            initial_model: Initial impedance model
            
        Returns:
            Geologically realistic acoustic impedance section
        """
        # Estimate wavelet from seismic data if not provided
        if wavelet is None:
            wavelet = self._estimate_wavelet_from_data(seismic_data, dt)
        
        # Create realistic initial model
        if initial_model is None:
            time_axis = np.arange(seismic_data.shape[0]) * dt
            depth_km = time_axis * 1.5
            initial_model = self._create_background_model(depth_km, seismic_data.shape[1])
        
        impedance = initial_model.copy().astype(np.float32)
        
        # Iterative inversion with geological constraints
        for iteration in range(10):  # More iterations for better convergence
            # Forward modeling
            synthetic = self._forward_model_enhanced(impedance, wavelet)
            
            # Calculate weighted residual
            residual = seismic_data - synthetic
            
            # Calculate gradient with regularization
            gradient = self._calculate_gradient(impedance, residual, wavelet)
            
            # Adaptive step size
            step_size = self._calculate_adaptive_step_size(iteration, residual)
            
            # Update model
            impedance_new = impedance + step_size * gradient
            
            # Apply geological constraints
            for i in range(impedance.shape[0]):
                depth = i * dt * 1.5  # Approximate depth
                impedance_new[i, :] = self._apply_geological_constraints(
                    impedance_new[i, :], depth, initial_model[i, :]
                )
            
            # Check convergence
            rms_change = np.sqrt(np.mean((impedance_new - impedance) ** 2))
            impedance = impedance_new
            
            if rms_change < 10.0:  # Convergence threshold
                break
        
        # Final geological smoothing
        impedance = self._smooth_geological_variations(impedance, dt)
        
        return impedance
    
    def sparse_spike_inversion(self, seismic_data: np.ndarray, dt: float,
                             sparsity_factor: float = 0.1) -> np.ndarray:
        """
        Enhanced sparse spike inversion with geological constraints.
        
        Args:
            seismic_data: Seismic amplitude data
            dt: Sample interval
            sparsity_factor: Sparsity constraint (0-1)
            
        Returns:
            High-resolution acoustic impedance section
        """
        # Start with enhanced recursive inversion
        impedance = self.recursive_inversion(seismic_data, dt)
        
        # Calculate impedance contrasts (reflection strength)
        diff_impedance = np.diff(impedance, axis=0)
        
        # Adaptive thresholding based on geological expectations
        # Strong reflectors are geologically significant
        for trace in range(diff_impedance.shape[1]):
            trace_data = diff_impedance[:, trace]
            
            # Calculate adaptive threshold
            trace_std = np.std(trace_data)
            trace_median = np.median(np.abs(trace_data))
            
            # Geological threshold: preserve strong reflectors
            geo_threshold = max(trace_median * 2, trace_std * 0.5)
            
            # Apply sparsity with geological awareness
            threshold = np.percentile(np.abs(trace_data), (1-sparsity_factor) * 100)
            final_threshold = min(threshold, geo_threshold)
            
            # Preserve geologically significant reflectors
            mask = np.abs(trace_data) < final_threshold
            diff_impedance[mask, trace] = 0
        
        # Reconstruct impedance with geological smoothing
        sparse_impedance = np.zeros_like(impedance)
        
        # Initialize with background model
        time_axis = np.arange(impedance.shape[0]) * dt
        depth_km = time_axis * 1.5
        background = self._create_background_model(depth_km, impedance.shape[1])
        sparse_impedance[0, :] = background[0, :]
        
        # Reconstruct with constraints
        for i in range(1, impedance.shape[0]):
            sparse_impedance[i, :] = sparse_impedance[i-1, :] + diff_impedance[i-1, :]
            
            # Apply geological constraints
            sparse_impedance[i, :] = self._apply_geological_constraints(
                sparse_impedance[i, :], depth_km[i], background[i, :]
            )
        
        return sparse_impedance
    
    def colored_inversion(self, seismic_data: np.ndarray, dt: float,
                         low_freq_model: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Enhanced colored inversion with realistic geological trends.
        
        Args:
            seismic_data: Seismic amplitude data
            dt: Sample interval
            low_freq_model: Low-frequency impedance model
            
        Returns:
            Broadband acoustic impedance section
        """
        if low_freq_model is None:
            # Create geologically realistic low-frequency model
            time_axis = np.arange(seismic_data.shape[0]) * dt
            depth_km = time_axis * 1.5
            low_freq_model = self._create_background_model(depth_km, seismic_data.shape[1])
        
        # Extract high-frequency component using enhanced method
        high_freq_impedance = self.recursive_inversion(seismic_data, dt)
        
        # Calculate relative impedance variations
        # Use logarithmic domain for better frequency separation
        log_high_freq = np.log(np.maximum(high_freq_impedance, 100))  # Avoid log(0)
        log_low_freq = np.log(np.maximum(low_freq_model, 100))
        
        # Remove low-frequency trend from high-frequency component
        high_freq_detail = log_high_freq - log_low_freq
        
        # Apply frequency-dependent scaling
        # High frequencies should have smaller amplitude
        freq_scaling = self._calculate_frequency_scaling(high_freq_detail, dt)
        high_freq_detail *= freq_scaling
        
        # Combine in logarithmic domain
        log_colored = log_low_freq + high_freq_detail
        
        # Convert back to linear domain
        colored_impedance = np.exp(log_colored)
        
        # Apply final geological constraints
        for i in range(colored_impedance.shape[0]):
            depth = i * dt * 1.5
            colored_impedance[i, :] = self._apply_geological_constraints(
                colored_impedance[i, :], depth, low_freq_model[i, :]
            )
        
        return colored_impedance
    
    def _calculate_frequency_scaling(self, detail: np.ndarray, dt: float) -> np.ndarray:
        """Calculate frequency-dependent scaling for realistic amplitude spectrum"""
        
        # Create frequency-dependent scaling
        # High frequencies should have lower amplitude in real earth
        scaling = np.ones_like(detail)
        
        # Apply depth-dependent scaling (high freq attenuates with depth)
        for i in range(detail.shape[0]):
            depth_factor = np.exp(-i * dt * 2.0)  # Exponential attenuation
            scaling[i, :] = 0.3 + 0.7 * depth_factor  # Scale between 0.3 and 1.0
        
        return scaling
    
    def band_limited_inversion(self, seismic_data: np.ndarray, dt: float,
                             freq_low: float = 5.0, freq_high: float = 80.0) -> np.ndarray:
        """
        Enhanced band-limited inversion with proper frequency domain processing.
        
        Args:
            seismic_data: Seismic amplitude data
            dt: Sample interval
            freq_low: Low-cut frequency (Hz)
            freq_high: High-cut frequency (Hz)
            
        Returns:
            Acoustic impedance section
        """
        # Apply proper bandpass filtering
        filtered_data = self._bandpass_filter_enhanced(seismic_data, dt, freq_low, freq_high)
        
        # Apply enhanced recursive inversion to filtered data
        impedance = self.recursive_inversion(filtered_data, dt)
        
        # Apply additional frequency-domain constraints
        impedance = self._apply_frequency_constraints(impedance, dt, freq_low, freq_high)
        
        return impedance
    
    def _bandpass_filter_enhanced(self, data: np.ndarray, dt: float, 
                                freq_low: float, freq_high: float) -> np.ndarray:
        """Enhanced bandpass filter using FFT or scipy if available"""
        
        if SCIPY_AVAILABLE:
            # Use proper Butterworth filter
            nyquist = 0.5 / dt
            low_norm = freq_low / nyquist
            high_norm = freq_high / nyquist
            
            # Ensure frequencies are in valid range
            low_norm = max(0.01, min(low_norm, 0.99))
            high_norm = max(low_norm + 0.01, min(high_norm, 0.99))
            
            try:
                b, a = butter(4, [low_norm, high_norm], btype='band')
                
                filtered_data = np.zeros_like(data)
                for trace in range(data.shape[1]):
                    filtered_data[:, trace] = filtfilt(b, a, data[:, trace])
                
                return filtered_data
            except:
                # Fallback to FFT method
                pass
        
        # FFT-based filtering
        filtered_data = np.zeros_like(data)
        
        for trace in range(data.shape[1]):
            trace_data = data[:, trace]
            
            # FFT
            fft_data = np.fft.fft(trace_data)
            freqs = np.fft.fftfreq(len(trace_data), dt)
            
            # Create filter
            filter_mask = np.zeros_like(freqs, dtype=bool)
            filter_mask[(np.abs(freqs) >= freq_low) & (np.abs(freqs) <= freq_high)] = True
            
            # Apply filter
            fft_filtered = fft_data * filter_mask
            
            # IFFT
            filtered_data[:, trace] = np.real(np.fft.ifft(fft_filtered))
        
        return filtered_data
    
    def _apply_frequency_constraints(self, impedance: np.ndarray, dt: float,
                                   freq_low: float, freq_high: float) -> np.ndarray:
        """Apply frequency-domain constraints to impedance"""
        
        constrained = impedance.copy()
        
        # Remove very high frequency noise that's geologically unrealistic
        for trace in range(impedance.shape[1]):
            # Apply light smoothing to remove unrealistic high-frequency variations
            trace_data = impedance[:, trace]
            
            # Simple moving average for high-frequency noise removal
            window_size = max(3, int(0.005 / dt))  # 5ms window
            if window_size % 2 == 0:
                window_size += 1
            
            if SCIPY_AVAILABLE:
                try:
                    constrained[:, trace] = medfilt(trace_data, window_size)
                except:
                    constrained[:, trace] = self._simple_median_filter(trace_data, window_size)
            else:
                constrained[:, trace] = self._simple_median_filter(trace_data, window_size)
        
        return constrained
    
    def _ricker_wavelet(self, dt: float, freq: float, length: float = 0.2) -> np.ndarray:
        """Generate Ricker wavelet"""
        t = np.arange(-length/2, length/2, dt)
        a = (np.pi * freq * t) ** 2
        wavelet = (1 - 2*a) * np.exp(-a)
        return wavelet / np.max(np.abs(wavelet))
    
    def _forward_model(self, impedance: np.ndarray, wavelet: np.ndarray) -> np.ndarray:
        """Forward modeling: impedance to seismic"""
        # Calculate reflection coefficients
        refl_coeff = np.diff(impedance, axis=0) / (impedance[1:, :] + impedance[:-1, :] + 1e-10)
        
        # Convolve with wavelet (simplified 1D convolution for each trace)
        synthetic = np.zeros_like(impedance)
        
        for trace in range(impedance.shape[1]):
            if trace < refl_coeff.shape[1]:
                conv_result = np.convolve(refl_coeff[:, trace], wavelet, mode='same')
                synthetic[:len(conv_result), trace] = conv_result
        
        return synthetic
    
    def _create_background_model(self, depth_km: np.ndarray, num_traces: int) -> np.ndarray:
        """Create geologically realistic background impedance model"""
        
        # Initialize model
        model = np.zeros((len(depth_km), num_traces), dtype=np.float32)
        
        # Get geological setting parameters
        base_min, base_max = getattr(self, 'ai_range', (1800, 6000))
        gradient = getattr(self, 'typical_gradient', 0.7)
        
        for i, depth in enumerate(depth_km):
            # Calculate base impedance using geological setting
            if self.geological_setting == "Shallow Marine":
                base_ai = 2000 + depth * 600 + depth**2 * 100  # Gradual increase
                variation = 200 + depth * 100
            elif self.geological_setting == "Deep Marine":
                base_ai = 2400 + depth * 800 + depth**2 * 50
                variation = 300 + depth * 80
            elif self.geological_setting == "Carbonate Platform":
                base_ai = 3200 + depth * 400 + depth**2 * 200  # Higher base, variable
                variation = 400 + depth * 150
            elif self.geological_setting == "Tight Gas Reservoir":
                base_ai = 3500 + depth * 300 + depth**2 * 100  # High impedance
                variation = 200 + depth * 50  # Less variation
            elif self.geological_setting == "Unconventional Shale":
                base_ai = 2400 + depth * 500 + depth**2 * 80
                variation = 250 + depth * 70
            else:  # Default mixed sediments
                if depth < 0.5:
                    base_ai = 1800 + depth * 1000
                    variation = 300
                elif depth < 1.5:
                    base_ai = 2300 + depth * 800
                    variation = 400
                elif depth < 3.0:
                    base_ai = 3000 + depth * 600
                    variation = 500
                else:
                    base_ai = 4500 + depth * 300
                    variation = 300
            
            # Ensure within geological bounds
            base_ai = np.clip(base_ai, base_min, base_max)
            
            # Add lateral geological variation
            lateral_trend = np.linspace(-variation/2, variation/2, num_traces)
            
            # Add geological structure based on setting
            if self.geological_setting == "Carbonate Platform":
                # More complex structure for carbonates
                structure = variation * 0.4 * (
                    np.sin(np.linspace(0, 4*np.pi, num_traces)) +
                    0.3 * np.sin(np.linspace(0, 12*np.pi, num_traces))
                )
            elif self.geological_setting in ["Tight Gas Reservoir", "Unconventional Shale"]:
                # More uniform for tight formations
                structure = variation * 0.2 * np.sin(np.linspace(0, 2*np.pi, num_traces))
            else:
                # Standard structure
                structure = variation * 0.3 * np.sin(np.linspace(0, 6*np.pi, num_traces))
            
            model[i, :] = base_ai + lateral_trend + structure
        
        return model
    
    def _normalize_amplitudes(self, amplitudes: np.ndarray) -> np.ndarray:
        """Normalize seismic amplitudes to realistic reflection coefficients"""
        
        # Estimate amplitude scaling from data statistics
        amp_std = np.std(amplitudes)
        amp_max = np.max(np.abs(amplitudes))
        
        # Scale to realistic reflection coefficient range (-0.3 to +0.3)
        if amp_max > 0:
            scale_factor = 0.3 / amp_max
            normalized = amplitudes * scale_factor
        else:
            normalized = amplitudes
        
        # Apply additional scaling based on amplitude distribution
        # Strong amplitudes are rare in real data
        normalized = np.tanh(normalized * 2.0) * 0.3
        
        return np.clip(normalized, -0.4, 0.4)
    
    def _apply_geological_constraints(self, impedance: np.ndarray, depth_km: float, 
                                    background: np.ndarray) -> np.ndarray:
        """Apply geological constraints based on setting and depth"""
        
        # Get base range from geological setting
        base_min, base_max = getattr(self, 'ai_range', (1800, 6000))
        
        # Adjust range based on depth and geological setting
        depth_factor = 1.0 + depth_km * getattr(self, 'typical_gradient', 0.7) / 2.0
        
        ai_min = base_min * depth_factor
        ai_max = base_max * depth_factor
        
        # Additional depth-based constraints
        if depth_km < 0.2:  # Very shallow
            ai_min = max(ai_min, 1500)
            ai_max = min(ai_max, 3500)
        elif depth_km > 4.0:  # Very deep
            ai_min = max(ai_min, 3000)
            ai_max = min(ai_max, 8000)
        
        # Clip to realistic ranges
        constrained = np.clip(impedance, ai_min, ai_max)
        
        # Prevent extreme variations from background (geological continuity)
        max_deviation_factor = 0.4 if self.geological_setting == "Carbonate Platform" else 0.5
        max_deviation = background * max_deviation_factor
        
        constrained = np.clip(constrained, 
                            background - max_deviation, 
                            background + max_deviation)
        
        return constrained
    
    def _smooth_geological_variations(self, impedance: np.ndarray, dt: float) -> np.ndarray:
        """Apply geological smoothing to remove unrealistic variations"""
        
        smoothed = impedance.copy()
        
        # Vertical smoothing (geological layering)
        window_size = max(3, int(0.01 / dt))  # 10ms window
        if window_size % 2 == 0:
            window_size += 1
        
        for trace in range(impedance.shape[1]):
            # Apply median filter to remove spikes
            from scipy.signal import medfilt
            try:
                smoothed[:, trace] = medfilt(impedance[:, trace], window_size)
            except:
                # Fallback if scipy not available
                smoothed[:, trace] = self._simple_median_filter(impedance[:, trace], window_size)
        
        # Lateral smoothing (geological continuity)
        lateral_window = max(3, min(11, impedance.shape[1] // 10))
        if lateral_window % 2 == 0:
            lateral_window += 1
        
        for sample in range(impedance.shape[0]):
            try:
                smoothed[sample, :] = medfilt(smoothed[sample, :], lateral_window)
            except:
                smoothed[sample, :] = self._simple_median_filter(smoothed[sample, :], lateral_window)
        
        return smoothed
    
    def _simple_median_filter(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Simple median filter implementation"""
        filtered = data.copy()
        half_window = window_size // 2
        
        for i in range(half_window, len(data) - half_window):
            window = data[i - half_window:i + half_window + 1]
            filtered[i] = np.median(window)
        
        return filtered
    
    def _estimate_wavelet_from_data(self, seismic_data: np.ndarray, dt: float) -> np.ndarray:
        """Estimate source wavelet from seismic data"""
        
        # Use autocorrelation of average trace to estimate wavelet
        avg_trace = np.mean(seismic_data, axis=1)
        
        # Calculate autocorrelation
        autocorr = np.correlate(avg_trace, avg_trace, mode='full')
        center = len(autocorr) // 2
        
        # Extract wavelet around zero lag
        wavelet_length = min(int(0.2 / dt), len(autocorr) // 4)  # 200ms or quarter length
        start = center - wavelet_length // 2
        end = center + wavelet_length // 2
        
        wavelet = autocorr[start:end]
        
        # Normalize
        if np.max(np.abs(wavelet)) > 0:
            wavelet = wavelet / np.max(np.abs(wavelet))
        
        return wavelet
    
    def _forward_model_enhanced(self, impedance: np.ndarray, wavelet: np.ndarray) -> np.ndarray:
        """Enhanced forward modeling with proper convolution"""
        
        # Calculate reflection coefficients
        refl_coeff = np.diff(impedance, axis=0) / (impedance[1:, :] + impedance[:-1, :] + 1e-10)
        
        # Initialize synthetic seismic
        synthetic = np.zeros_like(impedance)
        
        # Convolve each trace with wavelet
        for trace in range(impedance.shape[1]):
            if trace < refl_coeff.shape[1]:
                # Proper convolution
                conv_result = np.convolve(refl_coeff[:, trace], wavelet, mode='same')
                
                # Pad to match original size
                if len(conv_result) < impedance.shape[0]:
                    padded = np.zeros(impedance.shape[0])
                    padded[1:len(conv_result)+1] = conv_result
                    synthetic[:, trace] = padded
                else:
                    synthetic[1:, trace] = conv_result[:impedance.shape[0]-1]
        
        return synthetic
    
    def _calculate_gradient(self, impedance: np.ndarray, residual: np.ndarray, 
                          wavelet: np.ndarray) -> np.ndarray:
        """Calculate gradient for model update"""
        
        gradient = np.zeros_like(impedance)
        
        # Simplified gradient calculation
        # In practice, this would involve the adjoint of the forward operator
        for trace in range(impedance.shape[1]):
            # Cross-correlate residual with wavelet
            if trace < residual.shape[1]:
                grad_trace = np.correlate(residual[:, trace], wavelet, mode='same')
                
                if len(grad_trace) == impedance.shape[0]:
                    gradient[:, trace] = grad_trace
                else:
                    # Handle size mismatch
                    min_len = min(len(grad_trace), impedance.shape[0])
                    gradient[:min_len, trace] = grad_trace[:min_len]
        
        return gradient
    
    def _calculate_adaptive_step_size(self, iteration: int, residual: np.ndarray) -> float:
        """Calculate adaptive step size for optimization"""
        
        # Start with larger steps, decrease with iterations
        base_step = 100.0 / (iteration + 1)
        
        # Scale by residual magnitude
        residual_rms = np.sqrt(np.mean(residual ** 2))
        
        # Adaptive scaling
        if residual_rms > 0.1:
            step_scale = 0.5  # Large residual, smaller steps
        elif residual_rms > 0.05:
            step_scale = 1.0  # Medium residual, normal steps
        else:
            step_scale = 2.0  # Small residual, larger steps
        
        return base_step * step_scale
    
    def _create_low_freq_model(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create geologically realistic low-frequency impedance model"""
        samples, traces = shape
        
        # Create depth axis
        depth_km = np.linspace(0, samples * 0.001, samples)  # Approximate depth
        
        return self._create_background_model(depth_km, traces)
    
    def _bandpass_filter(self, data: np.ndarray, dt: float, 
                        freq_low: float, freq_high: float) -> np.ndarray:
        """Simple bandpass filter (placeholder - would use scipy in full implementation)"""
        # This is a simplified version - in practice would use proper FFT filtering
        # For now, just apply some smoothing to simulate filtering
        
        filtered_data = data.copy()
        
        # Simple smoothing as placeholder for proper filtering
        for trace in range(data.shape[1]):
            # Moving average (very basic low-pass)
            window_size = max(1, int(1.0 / (freq_high * dt)))
            if window_size > 1:
                kernel = np.ones(window_size) / window_size
                filtered_data[:, trace] = np.convolve(data[:, trace], kernel, mode='same')
        
        return filtered_data


def load_seismic_data(uploaded_file) -> Dict:
    """Load seismic data from uploaded SEG-Y file with large file support"""
    if uploaded_file is None:
        return None
    
    file_size = uploaded_file.size
    file_size_gb = file_size / (1024**3)
    
    st.info(f"📁 Processing file: {uploaded_file.name} ({file_size_gb:.2f} GB)")
    
    # Create progress bar for large files
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Save uploaded file with progress tracking
        temp_filename = f"temp_seismic_{hash(uploaded_file.name) % 10000}.sgy"
        
        status_text.text("💾 Saving uploaded file...")
        progress_bar.progress(0.1)
        
        # Write file in chunks for large files
        chunk_size = 64 * 1024 * 1024  # 64MB chunks
        bytes_written = 0
        
        with open(temp_filename, "wb") as f:
            uploaded_file.seek(0)
            while True:
                chunk = uploaded_file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                bytes_written += len(chunk)
                
                # Update progress
                progress = 0.1 + 0.3 * (bytes_written / file_size)
                progress_bar.progress(min(progress, 0.4))
                status_text.text(f"💾 Saving: {bytes_written / (1024**2):.1f} MB / {file_size / (1024**2):.1f} MB")
        
        progress_bar.progress(0.5)
        status_text.text("📖 Reading SEG-Y headers...")
        
        if FULL_SEGY_AVAILABLE:
            # For large files, read headers first to get info
            info = get_segy_info(temp_filename)
            
            progress_bar.progress(0.6)
            status_text.text("🔍 Analyzing file structure...")
            
            # Estimate memory requirements
            num_samples = info.get('num_samples', 0)
            sample_interval = info.get('sample_interval_us', 0)
            
            # Calculate estimated traces and memory usage
            header_size = 3600  # SEG-Y headers
            trace_header_size = 240
            bytes_per_sample = 4  # Assuming float32
            trace_size = trace_header_size + (num_samples * bytes_per_sample)
            
            estimated_traces = (file_size - header_size) // trace_size if trace_size > 0 else 0
            estimated_memory_gb = (estimated_traces * num_samples * 4) / (1024**3)
            
            st.info(f"""
            📊 **File Analysis:**
            - **Estimated traces**: {estimated_traces:,}
            - **Samples per trace**: {num_samples:,}
            - **Sample interval**: {sample_interval} μs
            - **Estimated memory**: {estimated_memory_gb:.2f} GB
            """)
            
            # Memory check and user confirmation for large files
            if estimated_memory_gb > 4.0:
                st.warning(f"⚠️ **Large Dataset Warning**: This file will require ~{estimated_memory_gb:.1f} GB of RAM")
                
                # Offer processing options
                processing_option = st.radio(
                    "Choose processing approach:",
                    [
                        "Load subset (recommended for preview)",
                        "Load full dataset (requires sufficient RAM)",
                        "Headers only (minimal memory)"
                    ]
                )
                
                if processing_option == "Headers only (minimal memory)":
                    headers = read_segy_headers(temp_filename)
                    progress_bar.progress(1.0)
                    status_text.text("✅ Headers loaded successfully!")
                    
                    return {
                        'headers_only': True,
                        'info': headers,
                        'filename': uploaded_file.name,
                        'file_size_gb': file_size_gb,
                        'temp_filename': temp_filename
                    }
                
                elif processing_option == "Load subset (recommended for preview)":
                    # Load only first 1000 traces or 1GB worth, whichever is smaller
                    max_traces = min(1000, int(1.0 * 1024**3 / (num_samples * 4))) if num_samples > 0 else 1000
                    
                    st.info(f"📊 Loading first {max_traces} traces for preview...")
                    
                    # Use modified reader for subset
                    data = read_segy_subset(temp_filename, max_traces=max_traces)
                    
                    progress_bar.progress(1.0)
                    status_text.text(f"✅ Loaded {max_traces} traces successfully!")
                    
                    return {
                        'data': data['data'],
                        'time': data['time'],
                        'dt': data['dt'],
                        'num_traces': data['num_traces'],
                        'num_samples': data['num_samples'],
                        'filename': uploaded_file.name,
                        'is_subset': True,
                        'total_traces': estimated_traces,
                        'file_size_gb': file_size_gb,
                        'temp_filename': temp_filename
                    }
            
            # Load full dataset
            progress_bar.progress(0.7)
            status_text.text("🔄 Loading seismic data...")
            
            data = read_segy_file(temp_filename)
            
            progress_bar.progress(1.0)
            status_text.text("✅ File loaded successfully!")
            
            return {
                'data': data['data'],
                'time': data['time'],
                'dt': data['dt'],
                'num_traces': data['num_traces'],
                'num_samples': data['num_samples'],
                'filename': uploaded_file.name,
                'file_size_gb': file_size_gb,
                'temp_filename': temp_filename
            }
        else:
            # Use basic reader for headers only
            headers = read_segy_headers(temp_filename)
            progress_bar.progress(1.0)
            status_text.text("✅ Headers loaded (install numpy for full functionality)")
            
            st.warning("⚠️ Full SEG-Y reading requires numpy. Showing header information only.")
            return {
                'headers_only': True,
                'info': headers,
                'filename': uploaded_file.name,
                'file_size_gb': file_size_gb,
                'temp_filename': temp_filename
            }
            
    except Exception as e:
        st.error(f"❌ Failed to load SEG-Y file: {str(e)}")
        return None
    
    finally:
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()


def create_seismic_colormap():
    """Create professional seismic colormap"""
    colors = ['#000080', '#0000FF', '#00FFFF', '#FFFFFF', '#FFFF00', '#FF0000', '#800000']
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('seismic_pro', colors, N=n_bins)
    return cmap


def plot_seismic_section(data: np.ndarray, time: np.ndarray, title: str = "Seismic Section",
                        trace_range: Tuple[int, int] = None, time_range: Tuple[float, float] = None,
                        colormap: str = 'seismic', clip_percentile: float = 99.0) -> plt.Figure:
    """Plot seismic section with professional styling"""
    
    if trace_range is None:
        trace_range = (0, min(data.shape[1], 500))  # Limit for performance
    
    if time_range is None:
        time_range = (time[0], time[-1])
    
    # Extract subset
    trace_start, trace_end = trace_range
    time_start_idx = np.argmin(np.abs(time - time_range[0]))
    time_end_idx = np.argmin(np.abs(time - time_range[1]))
    
    plot_data = data[time_start_idx:time_end_idx, trace_start:trace_end]
    plot_time = time[time_start_idx:time_end_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#0a0e27')
    ax.set_facecolor('#1a1f3a')
    
    # Calculate clip values
    clip_val = np.percentile(np.abs(plot_data), clip_percentile)
    
    # Plot seismic data
    if colormap == 'seismic_pro':
        cmap = create_seismic_colormap()
    else:
        cmap = plt.cm.get_cmap(colormap)
    
    im = ax.imshow(plot_data, aspect='auto', cmap=cmap, 
                   vmin=-clip_val, vmax=clip_val,
                   extent=[trace_start, trace_end, plot_time[-1], plot_time[0]])
    
    # Styling
    ax.set_xlabel('Trace Number', color='white', fontsize=12)
    ax.set_ylabel('Time (s)', color='white', fontsize=12)
    ax.set_title(title, color='#4fc3f7', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Amplitude', color='white', fontsize=10)
    cbar.ax.tick_params(colors='white')
    
    plt.tight_layout()
    return fig


def calculate_inversion_qc(original: np.ndarray, inverted: np.ndarray, 
                          seismic: np.ndarray) -> Dict:
    """Calculate quality control metrics for inversion"""
    
    # Correlation coefficient
    correlation = np.corrcoef(original.flatten(), inverted.flatten())[0, 1]
    
    # RMS error
    rms_error = np.sqrt(np.mean((original - inverted) ** 2))
    
    # Signal-to-noise ratio
    signal_power = np.mean(inverted ** 2)
    noise_power = np.mean((original - inverted) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # Impedance statistics
    imp_stats = {
        'min': np.min(inverted),
        'max': np.max(inverted),
        'mean': np.mean(inverted),
        'std': np.std(inverted)
    }
    
    return {
        'correlation': correlation,
        'rms_error': rms_error,
        'snr_db': snr,
        'impedance_stats': imp_stats
    }


def main():
    """Main application"""
    apply_professional_styling()
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(10,14,39,0.8) 0%, rgba(26,31,58,0.8) 100%); border-radius: 20px; margin-bottom: 2rem;">
        <h1>🌊 SeisCPT - Professional Seismic Inversion</h1>
        <h3 style="color: #81c784; font-weight: 400;">Advanced Amplitude to Acoustic Impedance Inversion</h3>
        <div style="display: flex; justify-content: center; gap: 2rem; margin: 1rem 0;">
            <div>
                <div style="font-size: 1.5rem; color: #4fc3f7; font-weight: 700;">5</div>
                <div style="color: #94a3b8;">Algorithms</div>
            </div>
            <div>
                <div style="font-size: 1.5rem; color: #81c784; font-weight: 700;">SEG-Y</div>
                <div style="color: #94a3b8;">Format Support</div>
            </div>
            <div>
                <div style="font-size: 1.5rem; color: #ffb74d; font-weight: 700;">QC</div>
                <div style="color: #94a3b8;">Metrics</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Inversion Parameters")
        
        # Algorithm selection
        inversion_method = st.selectbox(
            "Inversion Algorithm:",
            ['recursive', 'model_based', 'sparse_spike', 'colored_inversion', 'band_limited'],
            format_func=lambda x: {
                'recursive': '🔄 Recursive Inversion',
                'model_based': '🎯 Model-Based Inversion',
                'sparse_spike': '⚡ Sparse Spike Inversion', 
                'colored_inversion': '🌈 Colored Inversion',
                'band_limited': '📡 Band-Limited Inversion'
            }[x]
        )
        
        st.markdown("### 🎛️ Processing Parameters")
        
        # Geological parameters for all methods
        st.markdown("**Geological Setting:**")
        geological_setting = st.selectbox(
            "Expected Geology:",
            [
                "Mixed Sediments (Default)",
                "Shallow Marine",
                "Deep Marine", 
                "Fluvial/Deltaic",
                "Carbonate Platform",
                "Tight Gas Reservoir",
                "Unconventional Shale"
            ]
        )
        
        # Initial impedance for recursive method
        if inversion_method == 'recursive':
            # Set default based on geological setting
            geo_defaults = {
                "Mixed Sediments (Default)": 2500,
                "Shallow Marine": 2200,
                "Deep Marine": 2800,
                "Fluvial/Deltaic": 2300,
                "Carbonate Platform": 3500,
                "Tight Gas Reservoir": 3800,
                "Unconventional Shale": 2600
            }
            
            default_ai = geo_defaults.get(geological_setting, 2500)
            
            initial_impedance = st.slider(
                "Initial Impedance (m/s·g/cm³):",
                1500, 5000, default_ai, 50,
                help=f"Recommended for {geological_setting}: {default_ai}"
            )
        
        # Frequency parameters for band-limited
        if inversion_method == 'band_limited':
            freq_low = st.slider("Low-cut Frequency (Hz):", 1, 20, 5, 1)
            freq_high = st.slider("High-cut Frequency (Hz):", 40, 120, 80, 5)
        
        # Sparsity for sparse spike
        if inversion_method == 'sparse_spike':
            sparsity_factor = st.slider("Sparsity Factor:", 0.01, 0.5, 0.1, 0.01)
        
        st.markdown("### 🎨 Display Options")
        
        colormap = st.selectbox(
            "Colormap:",
            ['seismic', 'seismic_pro', 'RdBu_r', 'coolwarm', 'viridis'],
            index=1
        )
        
        clip_percentile = st.slider(
            "Amplitude Clipping (%):",
            90.0, 99.9, 99.0, 0.1
        )
        
        st.markdown("### 💾 Memory Management")
        
        # Show current memory usage if data is loaded
        if 'seismic_data' in st.session_state:
            data_info = st.session_state['seismic_data']
            file_size = data_info.get('file_size_gb', 0)
            
            if not data_info.get('headers_only', False):
                estimated_memory = file_size * 1.5  # Rough estimate including processing overhead
                st.info(f"📊 **Current Dataset:**\n- File: {file_size:.2f} GB\n- Est. Memory: {estimated_memory:.2f} GB")
                
                if data_info.get('is_subset', False):
                    total_traces = data_info.get('total_traces', 0)
                    loaded_traces = data_info.get('num_traces', 0)
                    st.warning(f"⚠️ **Subset Loaded:** {loaded_traces:,} / {total_traces:,} traces")
        
        # Large file processing tips
        with st.expander("💡 Large File Tips"):
            st.markdown("""
            **For files > 2GB:**
            - Use "Load subset" for preview
            - Process in smaller trace ranges
            - Monitor system memory usage
            - Consider data decimation
            
            **Memory Requirements:**
            - ~1.5x file size for processing
            - Additional memory for inversion
            - 16GB+ RAM recommended for 5GB+ files
            
            **Performance Tips:**
            - Close other applications
            - Use SSD storage for temp files
            - Process during off-peak hours
            """)
        
        st.markdown("### 📁 File Upload Limits")
        st.info("📈 **Maximum file size: 10 GB**\nSupports large 3D seismic volumes")
    
    # File upload
    st.subheader("📁 Load Seismic Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload SEG-Y file (up to 10 GB):",
            type=['sgy', 'segy'],
            help="Upload a SEG-Y format seismic file for inversion. Supports large 3D volumes up to 10 GB."
        )
        
        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024**2)
            file_size_gb = uploaded_file.size / (1024**3)
            
            if file_size_gb < 0.1:
                st.info(f"📁 File size: {file_size_mb:.1f} MB")
            else:
                st.info(f"📁 File size: {file_size_gb:.2f} GB")
                
            if file_size_gb > 5.0:
                st.warning("⚠️ Large file detected. Processing may take several minutes and require significant RAM.")
    
    with col2:
        if st.button("📂 Use Sample Data", type="secondary"):
            # Try to load the existing Profile_168.sgy
            try:
                if FULL_SEGY_AVAILABLE:
                    sample_data = read_segy_file("Profile_168.sgy")
                    st.session_state['seismic_data'] = {
                        'data': sample_data['data'],
                        'time': sample_data['time'],
                        'dt': sample_data['dt'],
                        'num_traces': sample_data['num_traces'],
                        'num_samples': sample_data['num_samples'],
                        'filename': 'Profile_168.sgy',
                        'file_size_gb': 0.26  # Approximate size
                    }
                    st.success("✅ Sample data loaded!")
                else:
                    sample_headers = read_segy_headers("Profile_168.sgy")
                    st.session_state['seismic_data'] = {
                        'headers_only': True,
                        'info': sample_headers,
                        'filename': 'Profile_168.sgy',
                        'file_size_gb': 0.26
                    }
                    st.info("ℹ️ Sample headers loaded (full data requires numpy)")
            except Exception as e:
                st.error(f"❌ Could not load sample data: {str(e)}")
        
        # Add memory cleanup button
        if st.button("🧹 Clear Memory", type="secondary", help="Clear loaded data to free memory"):
            # Clear session state
            for key in list(st.session_state.keys()):
                if key in ['seismic_data', 'inversion_results']:
                    del st.session_state[key]
            
            # Clean up temporary files
            import glob
            import os
            temp_files = glob.glob("temp_seismic_*.sgy")
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except:
                    pass
            
            st.success("✅ Memory cleared and temporary files removed!")
            st.experimental_rerun()
    
    # Load data
    if uploaded_file is not None:
        with st.spinner("📖 Loading seismic data..."):
            seismic_data = load_seismic_data(uploaded_file)
            if seismic_data:
                st.session_state['seismic_data'] = seismic_data
    
    # Process data if available
    if 'seismic_data' in st.session_state:
        data_info = st.session_state['seismic_data']
        
        if data_info.get('headers_only', False):
            # Show header information only
            st.subheader("📋 SEG-Y File Information")
            
            info = data_info['info']
            if 'error' not in info:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("File Size", f"{info['file_size_bytes']:,} bytes")
                
                with col2:
                    bh = info['binary_header']
                    st.metric("Sample Interval", f"{bh.get('sample_interval', 'N/A')} μs")
                
                with col3:
                    st.metric("Samples per Trace", f"{bh.get('num_samples', 'N/A')}")
                
                with col4:
                    st.metric("Estimated Traces", f"{info['estimated_traces']:,}")
                
                # Show textual header
                with st.expander("📄 Textual Header"):
                    st.text(info['textual_header'][:1000])
                
                st.warning("⚠️ Install numpy, matplotlib, and scipy for full inversion capabilities")
            
        else:
            # Full data processing
            st.subheader("📊 Seismic Data Overview")
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Traces", f"{data_info['num_traces']:,}")
            
            with col2:
                st.metric("Samples", f"{data_info['num_samples']:,}")
            
            with col3:
                st.metric("Time Range", f"{data_info['time'][-1]:.2f} s")
            
            with col4:
                st.metric("Sample Rate", f"{data_info['dt']*1000:.1f} ms")
            
            # Display controls
            st.subheader("🎛️ Display Controls")
            
            col1, col2 = st.columns(2)
            
            with col1:
                trace_start = st.number_input("Start Trace:", 0, data_info['num_traces']-1, 0)
                trace_end = st.number_input("End Trace:", trace_start+1, data_info['num_traces'], 
                                          min(trace_start+500, data_info['num_traces']))
            
            with col2:
                time_start = st.number_input("Start Time (s):", 0.0, data_info['time'][-1], 0.0, 0.1)
                time_end = st.number_input("End Time (s):", time_start, data_info['time'][-1], 
                                         min(time_start+1.0, data_info['time'][-1]), 0.1)
            
            # Process inversion
            if st.button("🚀 Run Seismic Inversion", type="primary", use_container_width=True):
                
                with st.spinner("🔄 Running seismic inversion..."):
                    
                    # Initialize inversion engine
                    inverter = SeismicInversion()
                    
                    # Extract data subset for processing
                    seismic_subset = data_info['data'][:, trace_start:trace_end]
                    
                    # Set geological constraints based on setting
                    inverter._set_geological_setting(geological_setting)
                    
                    # Run selected inversion method
                    if inversion_method == 'recursive':
                        impedance = inverter.recursive_inversion(
                            seismic_subset, data_info['dt'], initial_impedance
                        )
                    elif inversion_method == 'model_based':
                        impedance = inverter.model_based_inversion(
                            seismic_subset, data_info['dt']
                        )
                    elif inversion_method == 'sparse_spike':
                        impedance = inverter.sparse_spike_inversion(
                            seismic_subset, data_info['dt'], sparsity_factor
                        )
                    elif inversion_method == 'colored_inversion':
                        impedance = inverter.colored_inversion(
                            seismic_subset, data_info['dt']
                        )
                    elif inversion_method == 'band_limited':
                        impedance = inverter.band_limited_inversion(
                            seismic_subset, data_info['dt'], freq_low, freq_high
                        )
                    
                    # Store results
                    st.session_state['inversion_results'] = {
                        'impedance': impedance,
                        'seismic': seismic_subset,
                        'method': inversion_method,
                        'trace_range': (trace_start, trace_end),
                        'time_range': (time_start, time_end)
                    }
                
                st.success("✅ Inversion completed successfully!")
            
            # Display results if available
            if 'inversion_results' in st.session_state:
                results = st.session_state['inversion_results']
                
                st.subheader("📈 Inversion Results")
                
                # Create plots
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Seismic Data**")
                    fig_seismic = plot_seismic_section(
                        results['seismic'], 
                        data_info['time'],
                        "Seismic Amplitudes",
                        colormap=colormap,
                        clip_percentile=clip_percentile
                    )
                    st.pyplot(fig_seismic)
                
                with col2:
                    st.markdown("**Inverted Acoustic Impedance**")
                    fig_impedance = plot_seismic_section(
                        results['impedance'],
                        data_info['time'],
                        "Acoustic Impedance (m/s·g/cm³)",
                        colormap='viridis',
                        clip_percentile=clip_percentile
                    )
                    st.pyplot(fig_impedance)
                
                # Quality control metrics
                st.subheader("📊 Quality Control")
                
                # Calculate synthetic seismic for comparison
                synthetic_seismic = np.diff(results['impedance'], axis=0) / (
                    results['impedance'][1:, :] + results['impedance'][:-1, :] + 1e-10
                )
                
                # Pad to match original size
                synthetic_padded = np.zeros_like(results['seismic'])
                synthetic_padded[1:, :] = synthetic_seismic
                
                qc_metrics = calculate_inversion_qc(
                    results['seismic'], synthetic_padded, results['seismic']
                )
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Correlation", f"{qc_metrics['correlation']:.3f}")
                
                with col2:
                    st.metric("RMS Error", f"{qc_metrics['rms_error']:.2e}")
                
                with col3:
                    st.metric("SNR (dB)", f"{qc_metrics['snr_db']:.1f}")
                
                with col4:
                    imp_stats = qc_metrics['impedance_stats']
                    st.metric("Impedance Range", 
                             f"{imp_stats['min']:.0f} - {imp_stats['max']:.0f}")
                
                # Export options
                st.subheader("💾 Export Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📊 Download Impedance Data", type="secondary"):
                        # Create CSV data
                        csv_data = []
                        for i, trace in enumerate(range(results['impedance'].shape[1])):
                            for j, sample in enumerate(results['impedance'][:, trace]):
                                csv_data.append(f"{trace},{j*data_info['dt']:.4f},{sample:.2f}")
                        
                        csv_content = "Trace,Time_s,Impedance\n" + "\n".join(csv_data)
                        
                        st.download_button(
                            "Download CSV",
                            csv_content,
                            f"impedance_{results['method']}.csv",
                            "text/csv"
                        )
                
                with col2:
                    if st.button("📈 Download Plots", type="secondary"):
                        st.info("Plot download functionality would be implemented here")
    
    else:
        # Welcome message
        st.info("""
        👋 **Welcome to SeisCPT Professional Seismic Inversion!**
        
        **Features:**
        - 🔄 **Recursive Inversion**: Fast, stable algorithm for real-time processing
        - 🎯 **Model-Based Inversion**: Iterative optimization with geological constraints  
        - ⚡ **Sparse Spike Inversion**: High-resolution results with sparsity constraints
        - 🌈 **Colored Inversion**: Incorporates low-frequency geological models
        - 📡 **Band-Limited Inversion**: Frequency domain processing and filtering
        
        **Getting Started:**
        1. Upload a SEG-Y seismic file or use the sample data
        2. Select your preferred inversion algorithm
        3. Adjust processing parameters in the sidebar
        4. Run the inversion and analyze results
        5. Export impedance data and quality control metrics
        
        **Professional Features:**
        - Full SEG-Y format support (IBM & IEEE floating point)
        - Quality control metrics and validation
        - Multiple visualization options
        - Export capabilities for further analysis
        """)


if __name__ == "__main__":
    main()