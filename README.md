# 🌊 SeisCPT - Professional Seismic Inversion Application

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Seismic](https://img.shields.io/badge/Seismic-Inversion-orange.svg)](https://github.com/Abeledes/SeisCPT)

A comprehensive, professional-grade seismic inversion application for converting seismic amplitude data to **geologically realistic acoustic impedance** using multiple advanced algorithms with **up to 10GB file support**.

![SeisCPT Demo](https://via.placeholder.com/800x400/0a0e27/4fc3f7?text=SeisCPT+Professional+Seismic+Inversion)

## 🚀 **Live Demo**
Try the application: [SeisCPT Web App](https://your-streamlit-app-url.com) *(Deploy to get live URL)*

## ✨ **Key Features**

### 🎯 **Professional Inversion Algorithms**
- **🔄 Recursive Inversion**: Fast, stable algorithm based on the recursive relationship between impedance and reflection coefficients
- **🎯 Model-Based Inversion**: Iterative optimization approach incorporating geological constraints and wavelet estimation
- **⚡ Sparse Spike Inversion**: High-resolution inversion with sparsity constraints for enhanced vertical resolution
- **🌈 Colored Inversion**: Combines low-frequency geological models with high-frequency seismic data
- **📡 Band-Limited Inversion**: Frequency domain processing with customizable bandpass filtering

### Data Support
- **SEG-Y Format**: Full support for SEG-Y seismic data files
- **Multiple Data Formats**: IBM floating point, IEEE floating point, integer formats
- **2D/3D Data**: Handles both 2D lines and 3D seismic volumes
- **Header Parsing**: Complete textual and binary header interpretation

### Professional Features
- **Quality Control**: Comprehensive QC metrics including correlation, RMS error, and SNR
- **Interactive Visualization**: Professional seismic display with multiple colormaps
- **Parameter Control**: Real-time adjustment of inversion parameters
- **Export Capabilities**: CSV export for impedance data and analysis results
- **Performance Optimization**: Efficient processing for large datasets

## 🚀 Quick Start

### Installation

1. **Clone or download the application files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

**Option 1: Using the startup script (recommended for large files)**
```bash
python run_seismic_app.py
```

**Option 2: Direct Streamlit launch**
```bash
streamlit run seismic_inversion_app.py
```

The application will open in your web browser at `http://localhost:8501`

### Large File Support (Up to 10GB)

The application is configured to handle large seismic files up to **10GB**:

- **Automatic configuration** for large file uploads
- **Memory-efficient processing** with subset loading options
- **Progress tracking** for large file operations
- **System requirements checking** and recommendations
- **Temporary file management** with automatic cleanup

### Using Sample Data

The application supports loading sample seismic data for testing. Due to GitHub file size limits, sample data is not included in the repository, but you can:

1. **Use your own SEG-Y files** (up to 10GB supported)
2. **Download sample data** from public seismic datasets
3. **Generate synthetic data** using the built-in tools

**Recommended sample data characteristics:**
- **Format**: SEG-Y (IBM or IEEE floating point)
- **Size**: Any size up to 10GB
- **Sample interval**: 1-4 ms typical
- **Traces**: 100+ for meaningful results

## 📊 Inversion Methods

### 1. Recursive Inversion
**Best for**: Real-time processing, stable results
- Uses the recursive relationship: `AI(i+1) = AI(i) * (1 + R(i)) / (1 - R(i))`
- Fast computation
- Requires initial impedance value
- Good for preliminary analysis

### 2. Model-Based Inversion
**Best for**: High-quality results with geological constraints
- Iterative optimization approach
- Incorporates source wavelet estimation
- Uses forward modeling and residual minimization
- Produces geologically realistic results

### 3. Sparse Spike Inversion
**Best for**: High vertical resolution
- Applies sparsity constraints to enhance resolution
- Reduces noise while preserving geological features
- Adjustable sparsity factor (0.01 - 0.5)
- Excellent for thin bed detection

### 4. Colored Inversion
**Best for**: Incorporating low-frequency information
- Combines seismic data with geological models
- Adds low-frequency content missing from seismic
- Produces broadband impedance results
- Ideal when well log data is available

### 5. Band-Limited Inversion
**Best for**: Frequency-specific analysis
- Customizable frequency filtering (5-80 Hz typical)
- Removes noise outside seismic bandwidth
- Improves signal-to-noise ratio
- Good for noisy data

## 🎛️ Parameters Guide

### Recursive Inversion
- **Initial Impedance**: Starting impedance value (1500-4000 m/s·g/cm³)
  - Typical values: Sandstone ~2500, Limestone ~3500, Shale ~2000

### Model-Based Inversion
- **Iterations**: Number of optimization iterations (3-10)
- **Convergence**: Automatic convergence detection
- **Wavelet**: Ricker wavelet with dominant frequency estimation

### Sparse Spike Inversion
- **Sparsity Factor**: Controls sparsity level (0.01-0.5)
  - Lower values = more sparse (higher resolution)
  - Higher values = less sparse (more stable)

### Band-Limited Inversion
- **Low-cut Frequency**: Remove low frequencies (1-20 Hz)
- **High-cut Frequency**: Remove high frequencies (40-120 Hz)
- **Typical seismic bandwidth**: 5-80 Hz

## 📈 Quality Control

The application provides comprehensive QC metrics:

### Correlation Coefficient
- Measures similarity between original and synthetic seismic
- Range: -1 to +1 (higher is better)
- Target: > 0.7 for good inversion

### RMS Error
- Root Mean Square error between data and synthetic
- Lower values indicate better fit
- Units match input amplitude units

### Signal-to-Noise Ratio (SNR)
- Expressed in decibels (dB)
- Higher values indicate better quality
- Target: > 20 dB for good results

### Impedance Statistics
- **Range**: Min/max impedance values
- **Mean**: Average impedance
- **Standard Deviation**: Variability measure

## 🎨 Visualization Options

### Colormaps
- **Seismic**: Traditional red-white-blue seismic colormap
- **Seismic Pro**: Enhanced professional seismic colormap
- **Viridis**: Perceptually uniform colormap for impedance
- **RdBu_r**: Red-blue reversed for amplitude data
- **Coolwarm**: Blue-red colormap

### Display Controls
- **Trace Range**: Select subset of traces for display
- **Time Range**: Focus on specific time windows
- **Amplitude Clipping**: Adjust dynamic range (90-99.9%)

## 💾 Export Options

### Impedance Data
- **CSV Format**: Trace, Time, Impedance columns
- **Compatible**: With Excel, MATLAB, Python pandas
- **Includes**: All processing parameters in filename

### Quality Control Reports
- **Metrics Summary**: All QC statistics
- **Processing Parameters**: Complete parameter log
- **Visualization**: Plot exports (future enhancement)

## 📁 Large File Handling

### File Size Limits
- **Maximum upload**: 10GB per file
- **Recommended**: 16GB+ RAM for files > 2GB
- **Supported formats**: SEG-Y (all standard data formats)

### Memory Management
- **Subset loading**: Preview large files with first 1000 traces
- **Chunk processing**: Files loaded in 64MB chunks
- **Progress tracking**: Real-time upload and processing status
- **Memory monitoring**: System RAM usage recommendations
- **Automatic cleanup**: Temporary files removed after processing

### Processing Options for Large Files
1. **Headers Only**: Minimal memory usage, file structure analysis
2. **Subset Loading**: Load first N traces for preview and testing
3. **Full Loading**: Complete dataset (requires sufficient RAM)

### Performance Optimization
- **Efficient algorithms**: Optimized for large datasets
- **Memory pooling**: Reuse allocated memory where possible
- **Parallel processing**: Multi-core utilization for inversion
- **Disk caching**: Temporary file management for large operations

## 🔧 Technical Details

### SEG-Y Reader
- **Textual Header**: EBCDIC to ASCII conversion
- **Binary Header**: Complete field parsing
- **Trace Headers**: All standard fields supported
- **Data Formats**: IBM float, IEEE float, integers
- **Coordinate Systems**: Source/receiver coordinates

### Performance
- **Memory Efficient**: Processes data in chunks
- **Scalable**: Handles large 3D volumes
- **Interactive**: Real-time parameter adjustment
- **Responsive**: Optimized for web interface

### Algorithms
- **Numerical Stability**: Clipping and bounds checking
- **Convergence**: Automatic iteration control
- **Validation**: Input data quality checks
- **Error Handling**: Graceful failure recovery

## 🚨 Troubleshooting

### Common Issues

**"Module not found" errors**:
```bash
pip install -r requirements.txt
```

**Large file processing**:
- Use trace range selection to process subsets
- Increase system memory if needed
- Consider data decimation for preview

**Poor inversion results**:
- Check input data quality
- Adjust clipping percentile
- Try different inversion methods
- Verify initial parameters

**Display issues**:
- Refresh browser page
- Clear browser cache
- Check console for JavaScript errors

## 📚 References

### Seismic Inversion Theory
- Russell, B.H. (1988). Introduction to Seismic Inversion Methods
- Veeken, P.C.H. (2007). Seismic Stratigraphy, Basin Analysis and Reservoir Characterisation
- Chopra, S. & Marfurt, K.J. (2007). Seismic Attributes for Prospect Identification and Reservoir Characterization

### SEG-Y Format
- Society of Exploration Geophysicists (2017). SEG-Y rev 2 Data Exchange Format
- Barry, K.M. et al. (1975). Recommended Standards for Digital Tape Formats

## 📄 License

This application is provided for educational and research purposes. Please ensure compliance with your organization's software policies.

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional inversion algorithms
- 3D visualization capabilities
- Well log integration
- Advanced QC metrics
- Performance optimizations

## 📞 Support

For technical support or feature requests, please refer to the application documentation or contact your system administrator.