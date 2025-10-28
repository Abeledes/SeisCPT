"""
Low-frequency model (LFM) builder for seismic inversion.
Creates geologically realistic background impedance models.
"""
import numpy as np
from typing import Dict, Tuple, Optional
from scipy.ndimage import gaussian_filter1d

class LowFrequencyModelBuilder:
    """
    Professional low-frequency model builder with geological constraints.
    """
    
    def __init__(self):
        # Geological setting parameters
        self.geological_settings = {
            'shallow_marine': {
                'surface_ai': 2200,
                'gradient': 0.6,  # km/s per km
                'variation': 300,
                'gardner_a': 310,
                'gardner_b': 0.25
            },
            'deep_marine': {
                'surface_ai': 2400,
                'gradient': 0.8,
                'variation': 400,
                'gardner_a': 310,
                'gardner_b': 0.25
            },
            'fluvial_deltaic': {
                'surface_ai': 2100,
                'gradient': 0.7,
                'variation': 350,
                'gardner_a': 300,
                'gardner_b': 0.25
            },
            'carbonate_platform': {
                'surface_ai': 3200,
                'gradient': 0.5,
                'variation': 500,
                'gardner_a': 320,
                'gardner_b': 0.23
            },
            'tight_gas': {
                'surface_ai': 3800,
                'gradient': 0.4,
                'variation': 200,
                'gardner_a': 330,
                'gardner_b': 0.22
            },
            'unconventional_shale': {
                'surface_ai': 2600,
                'gradient': 0.6,
                'variation': 250,
                'gardner_a': 290,
                'gardner_b': 0.26
            },
            'mixed_sediments': {
                'surface_ai': 2500,
                'gradient': 0.7,
                'variation': 400,
                'gardner_a': 310,
                'gardner_b': 0.25
            }
        }
    
    def build_lfm(self, 
                  time_axis: np.ndarray,
                  num_traces: int,
                  geological_setting: str = 'mixed_sediments',
                  use_gardner: bool = True,
                  smoothing_window_ms: float = 250.0,
                  dt: float = 0.001) -> Tuple[np.ndarray, Dict]:
        """
        Build low-frequency impedance model.
        
        Args:
            time_axis: Time axis in seconds
            num_traces: Number of traces
            geological_setting: Geological environment
            use_gardner: Apply Gardner's equation for density
            smoothing_window_ms: Smoothing window in milliseconds
            dt: Sample interval in seconds
            
        Returns:
            lfm: Low-frequency model [samples x traces]
            info: Model information
        """
        
        # Get geological parameters
        geo_params = self.geological_settings.get(
            geological_setting.lower().replace(' ', '_').replace('(', '').replace(')', ''),
            self.geological_settings['mixed_sediments']
        )
        
        # Convert time to approximate depth (assuming average velocity)
        avg_velocity = 2000  # m/s
        depth_km = time_axis * avg_velocity / 2000  # Two-way time to depth
        
        # Build base impedance model
        base_model = self._build_base_impedance_model(
            depth_km, num_traces, geo_params, use_gardner
        )
        
        # Add geological structure
        structured_model = self._add_geological_structure(
            base_model, depth_km, num_traces, geo_params
        )
        
        # Apply smoothing
        smoothed_model = self._apply_smoothing(
            structured_model, smoothing_window_ms, dt
        )
        
        # Ensure realistic impedance range
        final_model = np.clip(smoothed_model, 1500, 12000)
        
        # Model information
        info = {
            'geological_setting': geological_setting,
            'parameters': geo_params,
            'use_gardner': use_gardner,
            'smoothing_ms': smoothing_window_ms,
            'impedance_range': (np.min(final_model), np.max(final_model)),
            'mean_impedance': np.mean(final_model),
            'depth_range_km': (np.min(depth_km), np.max(depth_km))
        }
        
        return final_model, info
    
    def _build_base_impedance_model(self, 
                                   depth_km: np.ndarray,
                                   num_traces: int,
                                   geo_params: Dict,
                                   use_gardner: bool) -> np.ndarray:
        """
        Build base impedance model with depth trend.
        """
        
        model = np.zeros((len(depth_km), num_traces))
        
        for i, depth in enumerate(depth_km):
            if use_gardner:
                # Use Gardner's equation: AI = a * depth^b + surface_ai
                base_ai = (geo_params['surface_ai'] + 
                          geo_params['gardner_a'] * (depth + 0.1) ** geo_params['gardner_b'])
            else:
                # Linear gradient model
                base_ai = geo_params['surface_ai'] + depth * geo_params['gradient'] * 1000
            
            # Apply compaction effects (exponential increase with depth)
            compaction_factor = 1 + 0.1 * (1 - np.exp(-depth * 2))
            base_ai *= compaction_factor
            
            model[i, :] = base_ai
        
        return model
    
    def _add_geological_structure(self, 
                                 base_model: np.ndarray,
                                 depth_km: np.ndarray,
                                 num_traces: int,
                                 geo_params: Dict) -> np.ndarray:
        """
        Add geological structure to base model.
        """
        
        structured_model = base_model.copy()
        variation = geo_params['variation']
        
        # Add lateral variation (geological dip/structure)
        trace_positions = np.linspace(0, 1, num_traces)
        
        for i, depth in enumerate(depth_km):
            # Regional dip
            regional_dip = variation * 0.3 * trace_positions
            
            # Structural features (anticlines, synclines)
            if 'carbonate' in geo_params or depth > 1.0:
                # More complex structure for carbonates and deeper sections
                structure = (variation * 0.4 * np.sin(2 * np.pi * trace_positions * 2) +
                           variation * 0.2 * np.sin(2 * np.pi * trace_positions * 5))
            else:
                # Simpler structure for clastics
                structure = variation * 0.3 * np.sin(2 * np.pi * trace_positions * 1.5)
            
            # Depth-dependent attenuation of structure
            depth_factor = np.exp(-depth * 0.5)  # Structure decreases with depth
            
            # Add to model
            structured_model[i, :] += (regional_dip + structure) * depth_factor
        
        # Add random geological noise
        noise_level = variation * 0.1
        geological_noise = np.random.normal(0, noise_level, structured_model.shape)
        structured_model += geological_noise
        
        return structured_model
    
    def _apply_smoothing(self, 
                        model: np.ndarray,
                        smoothing_window_ms: float,
                        dt: float) -> np.ndarray:
        """
        Apply geological smoothing to model.
        """
        
        # Convert smoothing window to samples
        smoothing_samples = smoothing_window_ms / (dt * 1000)
        sigma = smoothing_samples / 4  # Gaussian sigma
        
        smoothed_model = np.zeros_like(model)
        
        # Apply smoothing to each trace
        for trace in range(model.shape[1]):
            smoothed_model[:, trace] = gaussian_filter1d(
                model[:, trace], sigma=sigma, mode='nearest'
            )
        
        return smoothed_model
    
    def create_layered_model(self, 
                           time_axis: np.ndarray,
                           num_traces: int,
                           layer_interfaces: list,
                           layer_impedances: list,
                           transition_width_ms: float = 50.0,
                           dt: float = 0.001) -> Tuple[np.ndarray, Dict]:
        """
        Create layered impedance model with smooth transitions.
        
        Args:
            time_axis: Time axis in seconds
            num_traces: Number of traces
            layer_interfaces: List of interface times in seconds
            layer_impedances: List of impedance values for each layer
            transition_width_ms: Transition zone width in milliseconds
            dt: Sample interval in seconds
            
        Returns:
            model: Layered impedance model
            info: Model information
        """
        
        model = np.zeros((len(time_axis), num_traces))
        
        # Create base layered model
        for trace in range(num_traces):
            trace_model = np.full(len(time_axis), layer_impedances[0])
            
            for i, interface_time in enumerate(layer_interfaces):
                if i + 1 < len(layer_impedances):
                    # Find interface sample
                    interface_sample = int(interface_time / dt)
                    
                    # Create smooth transition
                    transition_samples = int(transition_width_ms / (dt * 1000))
                    
                    start_sample = max(0, interface_sample - transition_samples // 2)
                    end_sample = min(len(time_axis), interface_sample + transition_samples // 2)
                    
                    # Smooth transition using sigmoid
                    for j in range(start_sample, end_sample):
                        progress = (j - start_sample) / (end_sample - start_sample)
                        sigmoid = 1 / (1 + np.exp(-10 * (progress - 0.5)))
                        
                        trace_model[j] = (layer_impedances[i] * (1 - sigmoid) + 
                                        layer_impedances[i + 1] * sigmoid)
                    
                    # Set values below interface
                    trace_model[end_sample:] = layer_impedances[i + 1]
            
            model[:, trace] = trace_model
        
        # Add lateral variation
        lateral_variation = 100  # Small variation between traces
        for i in range(len(time_axis)):
            variation = np.random.normal(0, lateral_variation, num_traces)
            model[i, :] += variation
        
        # Ensure positive values
        model = np.maximum(model, 1000)
        
        info = {
            'model_type': 'layered',
            'layer_interfaces': layer_interfaces,
            'layer_impedances': layer_impedances,
            'transition_width_ms': transition_width_ms,
            'impedance_range': (np.min(model), np.max(model))
        }
        
        return model, info
    
    def create_gradient_model(self, 
                            time_axis: np.ndarray,
                            num_traces: int,
                            surface_impedance: float = 2500,
                            gradient: float = 0.7,
                            noise_level: float = 100) -> Tuple[np.ndarray, Dict]:
        """
        Create simple gradient impedance model.
        
        Args:
            time_axis: Time axis in seconds
            num_traces: Number of traces
            surface_impedance: Surface impedance value
            gradient: Impedance gradient (m/s·g/cm³ per km)
            noise_level: Random noise level
            
        Returns:
            model: Gradient impedance model
            info: Model information
        """
        
        # Convert time to depth
        depth_km = time_axis * 1.5  # Approximate depth conversion
        
        # Create gradient model
        model = np.zeros((len(time_axis), num_traces))
        
        for i, depth in enumerate(depth_km):
            base_impedance = surface_impedance + depth * gradient * 1000
            
            # Add lateral variation
            lateral_var = np.random.normal(0, noise_level, num_traces)
            model[i, :] = base_impedance + lateral_var
        
        # Ensure realistic range
        model = np.clip(model, 1500, 10000)
        
        info = {
            'model_type': 'gradient',
            'surface_impedance': surface_impedance,
            'gradient': gradient,
            'noise_level': noise_level,
            'impedance_range': (np.min(model), np.max(model))
        }
        
        return model, info
    
    def validate_lfm(self, model: np.ndarray) -> Dict:
        """
        Validate low-frequency model for geological realism.
        """
        
        # Check impedance range
        ai_min, ai_max = np.min(model), np.max(model)
        range_realistic = 1000 <= ai_min <= 15000 and 1500 <= ai_max <= 15000
        
        # Check for unrealistic variations
        gradients = np.abs(np.diff(model, axis=0))
        max_gradient = np.max(gradients)
        gradient_realistic = max_gradient < 2000  # Max 2000 m/s·g/cm³ per sample
        
        # Check for smoothness
        smoothness = np.mean(np.std(gradients, axis=0))
        smooth_enough = smoothness < 500
        
        # Overall validation
        is_valid = range_realistic and gradient_realistic and smooth_enough
        
        return {
            'is_valid': is_valid,
            'impedance_range': (ai_min, ai_max),
            'range_realistic': range_realistic,
            'max_gradient': max_gradient,
            'gradient_realistic': gradient_realistic,
            'smoothness': smoothness,
            'smooth_enough': smooth_enough,
            'recommendations': self._get_validation_recommendations(
                range_realistic, gradient_realistic, smooth_enough
            )
        }
    
    def _get_validation_recommendations(self, 
                                     range_ok: bool,
                                     gradient_ok: bool,
                                     smooth_ok: bool) -> list:
        """Get recommendations for improving LFM."""
        
        recommendations = []
        
        if not range_ok:
            recommendations.append("Adjust impedance range to 1500-12000 m/s·g/cm³")
        
        if not gradient_ok:
            recommendations.append("Reduce impedance gradients - too steep for geology")
        
        if not smooth_ok:
            recommendations.append("Increase smoothing window for more realistic geology")
        
        if not recommendations:
            recommendations.append("Model passes geological validation")
        
        return recommendations