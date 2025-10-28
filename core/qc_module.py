"""
Comprehensive Quality Control (QC) module for seismic inversion.
Generates professional QC reports with plots and statistics.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, Tuple, Optional
import io
import base64
from datetime import datetime

class QualityControl:
    """
    Professional quality control system for seismic inversion results.
    """
    
    def __init__(self):
        self.qc_thresholds = {
            'excellent': {'correlation': 0.85, 'rms_norm': 0.25, 'snr_db': 18},
            'good': {'correlation': 0.70, 'rms_norm': 0.40, 'snr_db': 12},
            'fair': {'correlation': 0.50, 'rms_norm': 0.60, 'snr_db': 8},
            'poor': {'correlation': 0.30, 'rms_norm': 1.00, 'snr_db': 3}
        }
    
    def comprehensive_qc_analysis(self, 
                                original_seismic: np.ndarray,
                                inverted_impedance: np.ndarray,
                                synthetic_seismic: np.ndarray,
                                wavelet: np.ndarray,
                                low_freq_model: np.ndarray,
                                dt: float,
                                inversion_params: Dict) -> Dict:
        """
        Perform comprehensive QC analysis of inversion results.
        
        Args:
            original_seismic: Original seismic data
            inverted_impedance: Inverted acoustic impedance
            synthetic_seismic: Forward modeled synthetic seismic
            wavelet: Source wavelet used
            low_freq_model: Low-frequency model
            dt: Sample interval
            inversion_params: Inversion parameters used
            
        Returns:
            qc_report: Comprehensive QC report dictionary
        """
        
        # Basic QC metrics
        basic_metrics = self._calculate_basic_metrics(
            original_seismic, synthetic_seismic, inverted_impedance
        )
        
        # Advanced QC metrics
        advanced_metrics = self._calculate_advanced_metrics(
            original_seismic, synthetic_seismic, inverted_impedance, dt
        )
        
        # Frequency domain analysis
        frequency_analysis = self._analyze_frequency_domain(
            original_seismic, synthetic_seismic, wavelet, dt
        )
        
        # Geological validation
        geological_validation = self._validate_geological_realism(
            inverted_impedance, low_freq_model
        )
        
        # Statistical analysis
        statistical_analysis = self._perform_statistical_analysis(
            original_seismic, synthetic_seismic, inverted_impedance
        )
        
        # Overall quality assessment
        overall_quality = self._assess_overall_quality(basic_metrics, advanced_metrics)
        
        # Generate QC plots
        qc_plots = self._generate_qc_plots(
            original_seismic, synthetic_seismic, inverted_impedance,
            wavelet, dt, basic_metrics
        )
        
        # Compile comprehensive report
        qc_report = {
            'timestamp': datetime.now().isoformat(),
            'basic_metrics': basic_metrics,
            'advanced_metrics': advanced_metrics,
            'frequency_analysis': frequency_analysis,
            'geological_validation': geological_validation,
            'statistical_analysis': statistical_analysis,
            'overall_quality': overall_quality,
            'inversion_parameters': inversion_params,
            'qc_plots': qc_plots,
            'recommendations': self._generate_recommendations(basic_metrics, advanced_metrics),
            'pass_fail_status': self._determine_pass_fail(overall_quality)
        }
        
        return qc_report
    
    def _calculate_basic_metrics(self, 
                               original: np.ndarray,
                               synthetic: np.ndarray,
                               impedance: np.ndarray) -> Dict:
        """Calculate basic QC metrics."""
        
        # Ensure same dimensions
        min_samples = min(original.shape[0], synthetic.shape[0])
        orig_trim = original[:min_samples]
        synth_trim = synthetic[:min_samples]
        
        # Correlation coefficient
        correlation = np.corrcoef(orig_trim.flatten(), synth_trim.flatten())[0, 1]
        if not np.isfinite(correlation):
            correlation = 0.0
        
        # RMS error
        rms_error = np.sqrt(np.mean((orig_trim - synth_trim) ** 2))
        rms_normalized = rms_error / (np.std(orig_trim) + 1e-10)
        
        # Signal-to-noise ratio
        signal_power = np.var(synth_trim)
        noise_power = np.var(orig_trim - synth_trim)
        snr_db = 10 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10))
        
        # Impedance statistics
        ai_stats = {
            'min': float(np.min(impedance)),
            'max': float(np.max(impedance)),
            'mean': float(np.mean(impedance)),
            'std': float(np.std(impedance)),
            'median': float(np.median(impedance)),
            'p10': float(np.percentile(impedance, 10)),
            'p90': float(np.percentile(impedance, 90))
        }
        
        return {
            'correlation': float(correlation),
            'rms_error': float(rms_error),
            'rms_normalized': float(rms_normalized),
            'snr_db': float(snr_db),
            'impedance_stats': ai_stats
        }
    
    def _calculate_advanced_metrics(self, 
                                  original: np.ndarray,
                                  synthetic: np.ndarray,
                                  impedance: np.ndarray,
                                  dt: float) -> Dict:
        """Calculate advanced QC metrics."""
        
        # Trace-by-trace correlation
        trace_correlations = []
        for i in range(min(original.shape[1], synthetic.shape[1])):
            if original.shape[1] > i and synthetic.shape[1] > i:
                corr = np.corrcoef(original[:, i], synthetic[:, i])[0, 1]
                if np.isfinite(corr):
                    trace_correlations.append(corr)
        
        # Predictability analysis
        predictability = self._calculate_predictability(original, synthetic)
        
        # Resolution analysis
        resolution_metrics = self._analyze_resolution(impedance, dt)
        
        # Stability analysis
        stability_metrics = self._analyze_stability(impedance)
        
        return {
            'trace_correlations': {
                'mean': float(np.mean(trace_correlations)) if trace_correlations else 0.0,
                'std': float(np.std(trace_correlations)) if trace_correlations else 0.0,
                'min': float(np.min(trace_correlations)) if trace_correlations else 0.0,
                'max': float(np.max(trace_correlations)) if trace_correlations else 0.0
            },
            'predictability': predictability,
            'resolution': resolution_metrics,
            'stability': stability_metrics
        }
    
    def _analyze_frequency_domain(self, 
                                original: np.ndarray,
                                synthetic: np.ndarray,
                                wavelet: np.ndarray,
                                dt: float) -> Dict:
        """Analyze frequency domain characteristics."""
        
        # Average traces for analysis
        orig_avg = np.mean(original, axis=1)
        synth_avg = np.mean(synthetic, axis=1)
        
        # FFT analysis
        orig_fft = np.abs(np.fft.fft(orig_avg))
        synth_fft = np.abs(np.fft.fft(synth_avg))
        freqs = np.fft.fftfreq(len(orig_avg), dt)
        
        # Positive frequencies only
        pos_freqs = freqs[:len(freqs)//2]
        orig_spectrum = orig_fft[:len(orig_fft)//2]
        synth_spectrum = synth_fft[:len(synth_fft)//2]
        
        # Spectral correlation
        spectral_corr = np.corrcoef(orig_spectrum, synth_spectrum)[0, 1]
        if not np.isfinite(spectral_corr):
            spectral_corr = 0.0
        
        # Dominant frequencies
        orig_dom_freq = pos_freqs[np.argmax(orig_spectrum)]
        synth_dom_freq = pos_freqs[np.argmax(synth_spectrum)]
        
        # Bandwidth analysis
        orig_bandwidth = self._calculate_bandwidth(pos_freqs, orig_spectrum)
        synth_bandwidth = self._calculate_bandwidth(pos_freqs, synth_spectrum)
        
        # Wavelet analysis
        wavelet_analysis = self._analyze_wavelet_spectrum(wavelet, dt)
        
        return {
            'spectral_correlation': float(spectral_corr),
            'original_dominant_freq': float(orig_dom_freq),
            'synthetic_dominant_freq': float(synth_dom_freq),
            'frequency_match_error': float(abs(orig_dom_freq - synth_dom_freq)),
            'original_bandwidth': orig_bandwidth,
            'synthetic_bandwidth': synth_bandwidth,
            'wavelet_analysis': wavelet_analysis
        }
    
    def _validate_geological_realism(self, 
                                   impedance: np.ndarray,
                                   low_freq_model: np.ndarray) -> Dict:
        """Validate geological realism of results."""
        
        # Impedance range validation
        ai_min, ai_max = np.min(impedance), np.max(impedance)
        range_realistic = 1500 <= ai_min <= 15000 and 2000 <= ai_max <= 12000
        
        # Gradient analysis
        gradients = np.abs(np.diff(impedance, axis=0))
        max_gradient = np.max(gradients)
        mean_gradient = np.mean(gradients)
        gradient_realistic = max_gradient < 3000  # Reasonable geological gradient
        
        # Comparison with LFM
        lfm_correlation = np.corrcoef(impedance.flatten(), low_freq_model.flatten())[0, 1]
        if not np.isfinite(lfm_correlation):
            lfm_correlation = 0.0
        
        # Smoothness analysis
        smoothness = np.mean(np.std(gradients, axis=0))
        smooth_enough = smoothness < 800
        
        # Layering analysis
        layering_quality = self._analyze_layering(impedance)
        
        return {
            'impedance_range_realistic': range_realistic,
            'impedance_min': float(ai_min),
            'impedance_max': float(ai_max),
            'gradient_realistic': gradient_realistic,
            'max_gradient': float(max_gradient),
            'mean_gradient': float(mean_gradient),
            'lfm_correlation': float(lfm_correlation),
            'smoothness': float(smoothness),
            'smooth_enough': smooth_enough,
            'layering_quality': layering_quality
        }
    
    def _perform_statistical_analysis(self, 
                                    original: np.ndarray,
                                    synthetic: np.ndarray,
                                    impedance: np.ndarray) -> Dict:
        """Perform statistical analysis of results."""
        
        # Residual analysis
        residuals = original - synthetic
        residual_stats = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'skewness': float(self._calculate_skewness(residuals)),
            'kurtosis': float(self._calculate_kurtosis(residuals))
        }
        
        # Distribution analysis
        impedance_distribution = self._analyze_distribution(impedance)
        
        # Outlier analysis
        outlier_analysis = self._detect_outliers(impedance)
        
        return {
            'residual_statistics': residual_stats,
            'impedance_distribution': impedance_distribution,
            'outlier_analysis': outlier_analysis
        }
    
    def _assess_overall_quality(self, basic_metrics: Dict, advanced_metrics: Dict) -> Dict:
        """Assess overall inversion quality."""
        
        correlation = basic_metrics['correlation']
        rms_norm = basic_metrics['rms_normalized']
        snr_db = basic_metrics['snr_db']
        
        # Determine quality level
        if (correlation >= self.qc_thresholds['excellent']['correlation'] and
            rms_norm <= self.qc_thresholds['excellent']['rms_norm'] and
            snr_db >= self.qc_thresholds['excellent']['snr_db']):
            quality_level = 'Excellent'
            quality_score = 4
        elif (correlation >= self.qc_thresholds['good']['correlation'] and
              rms_norm <= self.qc_thresholds['good']['rms_norm'] and
              snr_db >= self.qc_thresholds['good']['snr_db']):
            quality_level = 'Good'
            quality_score = 3
        elif (correlation >= self.qc_thresholds['fair']['correlation'] and
              rms_norm <= self.qc_thresholds['fair']['rms_norm'] and
              snr_db >= self.qc_thresholds['fair']['snr_db']):
            quality_level = 'Fair'
            quality_score = 2
        else:
            quality_level = 'Poor'
            quality_score = 1
        
        # Calculate composite score
        composite_score = (
            correlation * 0.4 +
            max(0, 1 - rms_norm) * 0.3 +
            min(1, snr_db / 20) * 0.2 +
            advanced_metrics['trace_correlations']['mean'] * 0.1
        )
        
        return {
            'quality_level': quality_level,
            'quality_score': quality_score,
            'composite_score': float(composite_score),
            'meets_professional_standards': quality_score >= 3
        }
    
    def _generate_qc_plots(self, 
                         original: np.ndarray,
                         synthetic: np.ndarray,
                         impedance: np.ndarray,
                         wavelet: np.ndarray,
                         dt: float,
                         basic_metrics: Dict) -> Dict:
        """Generate QC plots as base64 encoded images."""
        
        plots = {}
        
        # Plot 1: Seismic comparison
        plots['seismic_comparison'] = self._plot_seismic_comparison(
            original, synthetic, dt
        )
        
        # Plot 2: Impedance section
        plots['impedance_section'] = self._plot_impedance_section(
            impedance, dt
        )
        
        # Plot 3: Correlation plot
        plots['correlation_plot'] = self._plot_correlation(
            original, synthetic, basic_metrics['correlation']
        )
        
        # Plot 4: Frequency spectra
        plots['frequency_spectra'] = self._plot_frequency_spectra(
            original, synthetic, wavelet, dt
        )
        
        # Plot 5: QC dashboard
        plots['qc_dashboard'] = self._plot_qc_dashboard(basic_metrics)
        
        return plots
    
    def _plot_seismic_comparison(self, original: np.ndarray, synthetic: np.ndarray, dt: float) -> str:
        """Plot seismic data comparison."""
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))
        fig.patch.set_facecolor('#0a0e27')
        
        time_axis = np.arange(original.shape[0]) * dt
        trace_range = min(50, original.shape[1])
        
        # Original seismic
        im1 = ax1.imshow(original[:, :trace_range], aspect='auto', cmap='seismic',
                        extent=[0, trace_range, time_axis[-1], time_axis[0]])
        ax1.set_title('Original Seismic', color='white', fontsize=12)
        ax1.set_xlabel('Trace Number', color='white')
        ax1.set_ylabel('Time (s)', color='white')
        ax1.tick_params(colors='white')
        
        # Synthetic seismic
        im2 = ax2.imshow(synthetic[:, :trace_range], aspect='auto', cmap='seismic',
                        extent=[0, trace_range, time_axis[-1], time_axis[0]])
        ax2.set_title('Synthetic Seismic', color='white', fontsize=12)
        ax2.set_xlabel('Trace Number', color='white')
        ax2.tick_params(colors='white')
        
        # Residual
        residual = original[:, :trace_range] - synthetic[:, :trace_range]
        im3 = ax3.imshow(residual, aspect='auto', cmap='seismic',
                        extent=[0, trace_range, time_axis[-1], time_axis[0]])
        ax3.set_title('Residual', color='white', fontsize=12)
        ax3.set_xlabel('Trace Number', color='white')
        ax3.tick_params(colors='white')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _plot_impedance_section(self, impedance: np.ndarray, dt: float) -> str:
        """Plot impedance section."""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('#0a0e27')
        ax.set_facecolor('#1a1f3a')
        
        time_axis = np.arange(impedance.shape[0]) * dt
        trace_range = min(100, impedance.shape[1])
        
        im = ax.imshow(impedance[:, :trace_range], aspect='auto', cmap='viridis',
                      extent=[0, trace_range, time_axis[-1], time_axis[0]])
        
        ax.set_title('Acoustic Impedance Section', color='white', fontsize=14)
        ax.set_xlabel('Trace Number', color='white')
        ax.set_ylabel('Time (s)', color='white')
        ax.tick_params(colors='white')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Acoustic Impedance (m/s·g/cm³)', color='white')
        cbar.ax.tick_params(colors='white')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _plot_correlation(self, original: np.ndarray, synthetic: np.ndarray, correlation: float) -> str:
        """Plot correlation scatter plot."""
        
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor('#0a0e27')
        ax.set_facecolor('#1a1f3a')
        
        # Sample data for plotting (avoid too many points)
        sample_size = min(5000, original.size)
        indices = np.random.choice(original.size, sample_size, replace=False)
        
        orig_sample = original.flatten()[indices]
        synth_sample = synthetic.flatten()[indices]
        
        ax.scatter(orig_sample, synth_sample, alpha=0.5, c='#4fc3f7', s=1)
        
        # Perfect correlation line
        min_val = min(np.min(orig_sample), np.min(synth_sample))
        max_val = max(np.max(orig_sample), np.max(synth_sample))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Correlation')
        
        ax.set_xlabel('Original Seismic', color='white')
        ax.set_ylabel('Synthetic Seismic', color='white')
        ax.set_title(f'Correlation Plot (R = {correlation:.3f})', color='white', fontsize=14)
        ax.tick_params(colors='white')
        ax.legend()
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _plot_frequency_spectra(self, original: np.ndarray, synthetic: np.ndarray, 
                              wavelet: np.ndarray, dt: float) -> str:
        """Plot frequency spectra comparison."""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.patch.set_facecolor('#0a0e27')
        
        # Average traces
        orig_avg = np.mean(original, axis=1)
        synth_avg = np.mean(synthetic, axis=1)
        
        # FFT
        orig_fft = np.abs(np.fft.fft(orig_avg))
        synth_fft = np.abs(np.fft.fft(synth_avg))
        wavelet_fft = np.abs(np.fft.fft(wavelet))
        
        freqs = np.fft.fftfreq(len(orig_avg), dt)
        pos_freqs = freqs[:len(freqs)//2]
        
        # Amplitude spectra
        ax1.set_facecolor('#1a1f3a')
        ax1.plot(pos_freqs, orig_fft[:len(pos_freqs)], 'b-', label='Original', linewidth=2)
        ax1.plot(pos_freqs, synth_fft[:len(pos_freqs)], 'r-', label='Synthetic', linewidth=2)
        ax1.set_xlabel('Frequency (Hz)', color='white')
        ax1.set_ylabel('Amplitude', color='white')
        ax1.set_title('Amplitude Spectra Comparison', color='white')
        ax1.tick_params(colors='white')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Wavelet spectrum
        ax2.set_facecolor('#1a1f3a')
        wavelet_freqs = np.fft.fftfreq(len(wavelet), dt)
        pos_wavelet_freqs = wavelet_freqs[:len(wavelet_freqs)//2]
        ax2.plot(pos_wavelet_freqs, wavelet_fft[:len(pos_wavelet_freqs)], 'g-', 
                linewidth=2, label='Wavelet')
        ax2.set_xlabel('Frequency (Hz)', color='white')
        ax2.set_ylabel('Amplitude', color='white')
        ax2.set_title('Wavelet Spectrum', color='white')
        ax2.tick_params(colors='white')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _plot_qc_dashboard(self, basic_metrics: Dict) -> str:
        """Plot QC dashboard with key metrics."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor('#0a0e27')
        
        # Correlation gauge
        self._plot_gauge(ax1, basic_metrics['correlation'], 'Correlation', 0, 1)
        
        # RMS gauge
        self._plot_gauge(ax2, basic_metrics['rms_normalized'], 'RMS (Normalized)', 0, 1, reverse=True)
        
        # SNR gauge
        self._plot_gauge(ax3, min(basic_metrics['snr_db'], 30), 'SNR (dB)', 0, 30)
        
        # Impedance histogram
        ax4.set_facecolor('#1a1f3a')
        ai_stats = basic_metrics['impedance_stats']
        ax4.bar(['Min', 'P10', 'Mean', 'Median', 'P90', 'Max'],
               [ai_stats['min'], ai_stats['p10'], ai_stats['mean'], 
                ai_stats['median'], ai_stats['p90'], ai_stats['max']],
               color='#4fc3f7', alpha=0.7)
        ax4.set_title('Impedance Statistics', color='white')
        ax4.set_ylabel('Impedance (m/s·g/cm³)', color='white')
        ax4.tick_params(colors='white')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _plot_gauge(self, ax, value: float, title: str, min_val: float, max_val: float, reverse: bool = False):
        """Plot a gauge-style metric display."""
        
        ax.set_facecolor('#1a1f3a')
        
        # Normalize value
        norm_value = (value - min_val) / (max_val - min_val)
        if reverse:
            norm_value = 1 - norm_value
        
        # Color based on quality
        if norm_value > 0.8:
            color = '#4caf50'  # Green
        elif norm_value > 0.6:
            color = '#ff9800'  # Orange
        else:
            color = '#f44336'  # Red
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        
        ax.plot(x, y, 'white', linewidth=3)
        ax.fill_between(x, 0, y, alpha=0.3, color='gray')
        
        # Value indicator
        value_theta = np.pi * (1 - norm_value)
        ax.plot([0, np.cos(value_theta)], [0, np.sin(value_theta)], 
               color=color, linewidth=4)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{title}\n{value:.3f}', color='white', fontsize=12)
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', facecolor='#0a0e27', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return image_base64
    
    # Helper methods for calculations
    def _calculate_predictability(self, original: np.ndarray, synthetic: np.ndarray) -> float:
        """Calculate predictability metric."""
        return float(1 - np.var(original - synthetic) / np.var(original))
    
    def _analyze_resolution(self, impedance: np.ndarray, dt: float) -> Dict:
        """Analyze vertical resolution."""
        
        # Calculate dominant wavelength
        gradients = np.abs(np.diff(impedance, axis=0))
        mean_gradient = np.mean(gradients)
        
        # Estimate resolution (simplified)
        resolution_samples = int(0.025 / dt)  # ~25ms typical resolution
        
        return {
            'vertical_resolution_ms': resolution_samples * dt * 1000,
            'mean_gradient': float(mean_gradient)
        }
    
    def _analyze_stability(self, impedance: np.ndarray) -> Dict:
        """Analyze inversion stability."""
        
        # Lateral consistency
        lateral_std = np.std(impedance, axis=1)
        stability_metric = 1 / (1 + np.mean(lateral_std) / np.mean(impedance))
        
        return {
            'stability_metric': float(stability_metric),
            'lateral_consistency': float(np.mean(lateral_std))
        }
    
    def _calculate_bandwidth(self, freqs: np.ndarray, spectrum: np.ndarray) -> float:
        """Calculate -3dB bandwidth."""
        
        peak_amplitude = np.max(spectrum)
        half_power = peak_amplitude / np.sqrt(2)
        
        above_half_power = spectrum >= half_power
        if np.any(above_half_power):
            indices = np.where(above_half_power)[0]
            bandwidth = freqs[indices[-1]] - freqs[indices[0]]
        else:
            bandwidth = 0.0
        
        return float(bandwidth)
    
    def _analyze_wavelet_spectrum(self, wavelet: np.ndarray, dt: float) -> Dict:
        """Analyze wavelet spectrum."""
        
        fft_wavelet = np.abs(np.fft.fft(wavelet))
        freqs = np.fft.fftfreq(len(wavelet), dt)
        pos_freqs = freqs[:len(freqs)//2]
        pos_spectrum = fft_wavelet[:len(fft_wavelet)//2]
        
        dominant_freq = pos_freqs[np.argmax(pos_spectrum)]
        bandwidth = self._calculate_bandwidth(pos_freqs, pos_spectrum)
        
        return {
            'dominant_frequency': float(abs(dominant_freq)),
            'bandwidth': bandwidth
        }
    
    def _analyze_layering(self, impedance: np.ndarray) -> Dict:
        """Analyze geological layering quality."""
        
        # Detect layer boundaries
        gradients = np.abs(np.diff(impedance, axis=0))
        mean_gradient = np.mean(gradients, axis=1)
        
        # Find peaks (layer boundaries)
        threshold = np.mean(mean_gradient) + 2 * np.std(mean_gradient)
        boundaries = mean_gradient > threshold
        
        num_layers = np.sum(boundaries)
        layer_quality = min(1.0, num_layers / (impedance.shape[0] * 0.1))  # Expect ~10% boundaries
        
        return {
            'estimated_layers': int(num_layers),
            'layering_quality': float(layer_quality)
        }
    
    def _analyze_distribution(self, data: np.ndarray) -> Dict:
        """Analyze data distribution."""
        
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'skewness': float(self._calculate_skewness(data)),
            'kurtosis': float(self._calculate_kurtosis(data))
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness."""
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return 0.0
        
        skewness = np.mean(((data - mean_val) / std_val) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis."""
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return 0.0
        
        kurtosis = np.mean(((data - mean_val) / std_val) ** 4) - 3
        return kurtosis
    
    def _detect_outliers(self, data: np.ndarray) -> Dict:
        """Detect outliers in impedance data."""
        
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = (data < lower_bound) | (data > upper_bound)
        outlier_percentage = np.sum(outliers) / data.size * 100
        
        return {
            'outlier_percentage': float(outlier_percentage),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'outliers_detected': int(np.sum(outliers))
        }
    
    def _generate_recommendations(self, basic_metrics: Dict, advanced_metrics: Dict) -> list:
        """Generate QC recommendations."""
        
        recommendations = []
        
        if basic_metrics['correlation'] < 0.6:
            recommendations.append("Low correlation - consider different wavelet or geological model")
        
        if basic_metrics['rms_normalized'] > 0.5:
            recommendations.append("High RMS error - increase regularization or check data quality")
        
        if basic_metrics['snr_db'] < 10:
            recommendations.append("Low SNR - apply noise reduction or stronger smoothing")
        
        ai_stats = basic_metrics['impedance_stats']
        if ai_stats['min'] < 1500 or ai_stats['max'] > 12000:
            recommendations.append("Unrealistic impedance range - adjust geological constraints")
        
        if advanced_metrics['trace_correlations']['std'] > 0.3:
            recommendations.append("Inconsistent trace quality - check lateral continuity")
        
        if not recommendations:
            recommendations.append("Results meet professional QC standards")
        
        return recommendations
    
    def _determine_pass_fail(self, overall_quality: Dict) -> Dict:
        """Determine pass/fail status."""
        
        return {
            'overall_pass': overall_quality['meets_professional_standards'],
            'quality_level': overall_quality['quality_level'],
            'professional_grade': overall_quality['quality_score'] >= 3
        }