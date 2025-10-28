"""
Automatic parameter tuning for seismic inversion optimization.
Self-improving system that optimizes f₀, λ, α to maximize correlation.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, Tuple, Callable, Optional
import warnings

class ParameterAutoTuner:
    """
    Automatic parameter tuning system for seismic inversion.
    
    Optimizes inversion parameters to achieve professional QC thresholds.
    """
    
    def __init__(self):
        # QC thresholds for professional results
        self.qc_thresholds = {
            'correlation_min': 0.7,
            'rms_normalized_max': 0.4,
            'snr_min_db': 12.0,
            'impedance_range': (1800, 10000)
        }
        
        # Parameter bounds
        self.parameter_bounds = {
            'dominant_frequency': (5.0, 80.0),
            'lambda_reg': (0.001, 1.0),
            'alpha_smooth': (0.001, 0.5),
            'freq_low': (2.0, 20.0),
            'freq_high': (30.0, 120.0)
        }
        
        # Optimization history
        self.optimization_history = []
    
    def auto_tune_inversion(self, 
                          seismic_data: np.ndarray,
                          low_freq_model: np.ndarray,
                          dt: float,
                          inversion_func: Callable,
                          max_iterations: int = 20,
                          target_correlation: float = 0.8) -> Tuple[Dict, Dict]:
        """
        Automatically tune inversion parameters for optimal results.
        
        Args:
            seismic_data: Input seismic data [samples x traces]
            low_freq_model: Low-frequency impedance model
            dt: Sample interval in seconds
            inversion_func: Inversion function to optimize
            max_iterations: Maximum optimization iterations
            target_correlation: Target correlation coefficient
            
        Returns:
            optimal_params: Optimized parameters
            optimization_info: Optimization information
        """
        
        # Initial parameter guess
        initial_params = self._get_initial_parameters(seismic_data, dt)
        
        # Define objective function
        def objective_function(params):
            return -self._evaluate_inversion_quality(
                params, seismic_data, low_freq_model, dt, inversion_func
            )
        
        # Multi-stage optimization
        best_params, best_score = self._multi_stage_optimization(
            objective_function, initial_params, max_iterations
        )
        
        # Validate final results
        final_validation = self._validate_final_parameters(
            best_params, seismic_data, low_freq_model, dt, inversion_func
        )
        
        # Store optimization history
        optimization_info = {
            'initial_params': initial_params,
            'optimal_params': best_params,
            'best_score': -best_score,
            'target_achieved': -best_score >= target_correlation,
            'iterations_used': len(self.optimization_history),
            'validation': final_validation,
            'optimization_history': self.optimization_history.copy()
        }
        
        return best_params, optimization_info
    
    def _get_initial_parameters(self, seismic_data: np.ndarray, dt: float) -> Dict:
        """
        Get intelligent initial parameter guess based on data analysis.
        """
        
        # Analyze frequency content
        avg_trace = np.mean(seismic_data, axis=1)
        fft_data = np.abs(np.fft.fft(avg_trace))
        freqs = np.fft.fftfreq(len(avg_trace), dt)
        
        # Find dominant frequency
        pos_freqs = freqs[:len(freqs)//2]
        pos_fft = fft_data[:len(fft_data)//2]
        
        if len(pos_fft) > 0:
            dominant_freq = pos_freqs[np.argmax(pos_fft)]
            dominant_freq = abs(dominant_freq)
        else:
            dominant_freq = 30.0  # Default
        
        # Estimate noise level for regularization
        noise_estimate = np.std(seismic_data) / np.mean(np.abs(seismic_data))
        
        # Initial parameters
        initial_params = {
            'dominant_frequency': np.clip(dominant_freq, 10.0, 60.0),
            'lambda_reg': np.clip(noise_estimate * 0.1, 0.01, 0.3),
            'alpha_smooth': np.clip(noise_estimate * 0.05, 0.005, 0.1),
            'freq_low': max(5.0, dominant_freq * 0.3),
            'freq_high': min(100.0, dominant_freq * 3.0)
        }
        
        return initial_params
    
    def _evaluate_inversion_quality(self, 
                                   params: Dict,
                                   seismic_data: np.ndarray,
                                   low_freq_model: np.ndarray,
                                   dt: float,
                                   inversion_func: Callable) -> float:
        """
        Evaluate inversion quality for given parameters.
        """
        
        try:
            # Convert parameter array to dictionary if needed
            if isinstance(params, np.ndarray):
                param_names = ['dominant_frequency', 'lambda_reg', 'alpha_smooth', 'freq_low', 'freq_high']
                params = dict(zip(param_names, params))
            
            # Ensure parameters are within bounds
            params = self._clip_parameters(params)
            
            # Run inversion with current parameters
            impedance, inversion_info = inversion_func(
                seismic_data, low_freq_model, dt, params
            )
            
            # Calculate QC metrics
            qc_metrics = self._calculate_qc_metrics(
                seismic_data, impedance, params, dt
            )
            
            # Calculate composite score
            score = self._calculate_composite_score(qc_metrics)
            
            # Store in history
            self.optimization_history.append({
                'params': params.copy(),
                'qc_metrics': qc_metrics,
                'score': score
            })
            
            return score
            
        except Exception as e:
            warnings.warn(f"Inversion evaluation failed: {e}")
            return 0.0  # Poor score for failed inversion
    
    def _multi_stage_optimization(self, 
                                 objective_func: Callable,
                                 initial_params: Dict,
                                 max_iterations: int) -> Tuple[Dict, float]:
        """
        Multi-stage optimization strategy.
        """
        
        # Stage 1: Global search with differential evolution
        bounds = [
            self.parameter_bounds['dominant_frequency'],
            self.parameter_bounds['lambda_reg'],
            self.parameter_bounds['alpha_smooth'],
            self.parameter_bounds['freq_low'],
            self.parameter_bounds['freq_high']
        ]
        
        # Convert initial params to array
        x0 = np.array([
            initial_params['dominant_frequency'],
            initial_params['lambda_reg'],
            initial_params['alpha_smooth'],
            initial_params['freq_low'],
            initial_params['freq_high']
        ])
        
        try:
            # Global optimization
            result_global = differential_evolution(
                objective_func, bounds, 
                maxiter=max_iterations // 2,
                seed=42,
                atol=1e-6,
                tol=1e-6
            )
            
            global_params = result_global.x
            global_score = result_global.fun
            
        except Exception as e:
            warnings.warn(f"Global optimization failed: {e}")
            global_params = x0
            global_score = objective_func(x0)
        
        # Stage 2: Local refinement
        try:
            result_local = minimize(
                objective_func, global_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iterations // 2}
            )
            
            if result_local.success and result_local.fun < global_score:
                final_params = result_local.x
                final_score = result_local.fun
            else:
                final_params = global_params
                final_score = global_score
                
        except Exception as e:
            warnings.warn(f"Local optimization failed: {e}")
            final_params = global_params
            final_score = global_score
        
        # Convert back to dictionary
        param_names = ['dominant_frequency', 'lambda_reg', 'alpha_smooth', 'freq_low', 'freq_high']
        best_params = dict(zip(param_names, final_params))
        
        return best_params, final_score
    
    def _calculate_qc_metrics(self, 
                            seismic_data: np.ndarray,
                            impedance: np.ndarray,
                            params: Dict,
                            dt: float) -> Dict:
        """
        Calculate comprehensive QC metrics.
        """
        
        # Forward model to get synthetic seismic
        synthetic = self._forward_model_simple(impedance, params, dt)
        
        # Ensure same size
        min_len = min(len(seismic_data), len(synthetic))
        seismic_trim = seismic_data[:min_len]
        synthetic_trim = synthetic[:min_len]
        
        # Correlation coefficient
        if seismic_trim.size > 0 and synthetic_trim.size > 0:
            correlation = np.corrcoef(seismic_trim.flatten(), synthetic_trim.flatten())[0, 1]
            if not np.isfinite(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        # RMS error
        rms_error = np.sqrt(np.mean((seismic_trim - synthetic_trim) ** 2))
        rms_normalized = rms_error / (np.std(seismic_trim) + 1e-10)
        
        # SNR
        signal_power = np.var(synthetic_trim)
        noise_power = np.var(seismic_trim - synthetic_trim)
        snr_db = 10 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10))
        
        # Impedance statistics
        ai_min, ai_max = np.min(impedance), np.max(impedance)
        ai_range_ok = (self.qc_thresholds['impedance_range'][0] <= ai_min and 
                      ai_max <= self.qc_thresholds['impedance_range'][1])
        
        return {
            'correlation': correlation,
            'rms_error': rms_error,
            'rms_normalized': rms_normalized,
            'snr_db': snr_db,
            'impedance_min': ai_min,
            'impedance_max': ai_max,
            'impedance_range_ok': ai_range_ok
        }
    
    def _calculate_composite_score(self, qc_metrics: Dict) -> float:
        """
        Calculate composite quality score.
        """
        
        # Correlation component (most important)
        corr_score = max(0, qc_metrics['correlation']) * 0.5
        
        # RMS component (penalize high RMS)
        rms_score = max(0, 1 - qc_metrics['rms_normalized']) * 0.2
        
        # SNR component
        snr_normalized = min(1, max(0, qc_metrics['snr_db'] / 20.0))
        snr_score = snr_normalized * 0.2
        
        # Impedance range component
        range_score = 0.1 if qc_metrics['impedance_range_ok'] else 0.0
        
        # Composite score
        total_score = corr_score + rms_score + snr_score + range_score
        
        return total_score
    
    def _forward_model_simple(self, impedance: np.ndarray, params: Dict, dt: float) -> np.ndarray:
        """
        Simple forward model for QC evaluation.
        """
        
        try:
            # Create simple Ricker wavelet
            freq = params.get('dominant_frequency', 30.0)
            wavelet_length = int(0.2 / dt)  # 200ms wavelet
            
            t = np.arange(wavelet_length) * dt
            t = t - t[len(t)//2]
            
            a = (np.pi * freq * t) ** 2
            wavelet = (1 - 2*a) * np.exp(-a)
            wavelet = wavelet / np.max(np.abs(wavelet))
            
            # Calculate reflection coefficients
            if impedance.ndim == 1:
                log_ai = np.log(np.maximum(impedance, 100))
                refl_coeff = np.diff(log_ai)
                refl_padded = np.concatenate([[0], refl_coeff])
            else:
                # Handle 2D case - use first trace
                log_ai = np.log(np.maximum(impedance[:, 0], 100))
                refl_coeff = np.diff(log_ai)
                refl_padded = np.concatenate([[0], refl_coeff])
            
            # Convolve with wavelet
            synthetic = np.convolve(refl_padded, wavelet, mode='same')
            
            return synthetic
            
        except Exception as e:
            warnings.warn(f"Forward modeling failed: {e}")
            return np.zeros_like(impedance[:, 0] if impedance.ndim > 1 else impedance)
    
    def _clip_parameters(self, params: Dict) -> Dict:
        """
        Clip parameters to valid bounds.
        """
        
        clipped = {}
        
        for key, value in params.items():
            if key in self.parameter_bounds:
                bounds = self.parameter_bounds[key]
                clipped[key] = np.clip(value, bounds[0], bounds[1])
            else:
                clipped[key] = value
        
        # Ensure freq_low < freq_high
        if 'freq_low' in clipped and 'freq_high' in clipped:
            if clipped['freq_low'] >= clipped['freq_high']:
                clipped['freq_high'] = clipped['freq_low'] + 10.0
        
        return clipped
    
    def _validate_final_parameters(self, 
                                 params: Dict,
                                 seismic_data: np.ndarray,
                                 low_freq_model: np.ndarray,
                                 dt: float,
                                 inversion_func: Callable) -> Dict:
        """
        Validate final optimized parameters.
        """
        
        try:
            # Run final inversion
            impedance, _ = inversion_func(seismic_data, low_freq_model, dt, params)
            
            # Calculate final QC metrics
            final_qc = self._calculate_qc_metrics(seismic_data, impedance, params, dt)
            
            # Check against thresholds
            validation = {
                'correlation_ok': final_qc['correlation'] >= self.qc_thresholds['correlation_min'],
                'rms_ok': final_qc['rms_normalized'] <= self.qc_thresholds['rms_normalized_max'],
                'snr_ok': final_qc['snr_db'] >= self.qc_thresholds['snr_min_db'],
                'impedance_ok': final_qc['impedance_range_ok'],
                'overall_pass': False
            }
            
            # Overall validation
            validation['overall_pass'] = all([
                validation['correlation_ok'],
                validation['rms_ok'],
                validation['snr_ok'],
                validation['impedance_ok']
            ])
            
            validation['final_qc_metrics'] = final_qc
            
            return validation
            
        except Exception as e:
            return {
                'validation_failed': True,
                'error': str(e),
                'overall_pass': False
            }
    
    def suggest_manual_adjustments(self, optimization_info: Dict) -> list:
        """
        Suggest manual parameter adjustments if auto-tuning fails.
        """
        
        suggestions = []
        
        if not optimization_info.get('target_achieved', False):
            final_qc = optimization_info.get('validation', {}).get('final_qc_metrics', {})
            
            if final_qc.get('correlation', 0) < 0.6:
                suggestions.append("Low correlation - try different geological setting or wavelet estimation method")
            
            if final_qc.get('rms_normalized', 1) > 0.5:
                suggestions.append("High RMS error - increase regularization (lambda) or check data quality")
            
            if final_qc.get('snr_db', 0) < 10:
                suggestions.append("Low SNR - apply stronger smoothing (alpha) or bandpass filtering")
            
            if not final_qc.get('impedance_range_ok', False):
                suggestions.append("Unrealistic impedance range - adjust geological constraints or LFM")
        
        if not suggestions:
            suggestions.append("Optimization successful - no manual adjustments needed")
        
        return suggestions