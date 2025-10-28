# Contributing to SeisCPT

Thank you for your interest in contributing to SeisCPT! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Issues
- Use the GitHub issue tracker to report bugs
- Include detailed information about your environment
- Provide steps to reproduce the issue
- Include sample data if relevant (small files only)

### Suggesting Enhancements
- Open an issue with the "enhancement" label
- Describe the proposed feature in detail
- Explain the use case and benefits
- Consider implementation complexity

### Code Contributions

#### Getting Started
1. Fork the repository
2. Clone your fork locally
3. Create a new branch for your feature
4. Install dependencies: `pip install -r requirements.txt`

#### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Write tests for new functionality
- Update documentation as needed

#### Pull Request Process
1. Ensure your code passes all tests
2. Update the README if you've added features
3. Add your changes to the changelog
4. Submit a pull request with a clear description

## 🧪 Testing

### Running Tests
```bash
# Install test dependencies
pip install pytest

# Run tests
pytest tests/

# Run with coverage
pytest --cov=. tests/
```

### Test Data
- Use small synthetic datasets for testing
- Do not commit large seismic files
- Include test data generation scripts if needed

## 📝 Documentation

### Code Documentation
- Use clear, descriptive function and variable names
- Add docstrings following Google style
- Include examples in docstrings where helpful

### User Documentation
- Update README.md for new features
- Add usage examples
- Document new parameters and options

## 🌊 Seismic Domain Guidelines

### Algorithm Contributions
- Ensure algorithms are geologically realistic
- Include proper citations for published methods
- Test with realistic seismic data
- Document assumptions and limitations

### Data Handling
- Support standard seismic formats (SEG-Y, etc.)
- Handle large files efficiently
- Include proper error handling
- Validate input data quality

## 🏗️ Project Structure

```
SeisCPT/
├── seismic_inversion_app.py    # Main Streamlit application
├── segy_reader.py              # SEG-Y file handling
├── basic_segy_reader.py        # Lightweight SEG-Y reader
├── run_seismic_app.py          # Application launcher
├── requirements.txt            # Python dependencies
├── .streamlit/                 # Streamlit configuration
├── tests/                      # Test files
└── docs/                       # Documentation
```

## 🎯 Areas for Contribution

### High Priority
- Additional inversion algorithms
- Performance optimizations
- Better error handling
- More geological settings
- Quality control metrics

### Medium Priority
- 3D visualization capabilities
- Well log integration
- Export format options
- Advanced filtering methods
- Batch processing tools

### Future Enhancements
- Machine learning integration
- Cloud deployment options
- Real-time processing
- Advanced QC workflows
- Integration with other tools

## 📊 Code Quality

### Standards
- Maintain test coverage > 80%
- Follow semantic versioning
- Use meaningful commit messages
- Keep functions focused and small
- Handle edge cases gracefully

### Performance
- Profile code for large datasets
- Optimize memory usage
- Use efficient algorithms
- Consider parallel processing
- Test with realistic data sizes

## 🌍 Community

### Communication
- Be respectful and inclusive
- Help newcomers to seismic processing
- Share knowledge and best practices
- Provide constructive feedback

### Recognition
- Contributors will be acknowledged in releases
- Significant contributions may be highlighted
- Academic citations welcome for novel algorithms

## 📚 Resources

### Seismic Processing
- SEG Wiki: https://wiki.seg.org/
- Seismic processing textbooks
- Industry best practices
- Academic publications

### Python Development
- Python documentation
- Streamlit documentation
- NumPy/SciPy guides
- Testing best practices

## 📞 Contact

For questions about contributing:
- Open a GitHub issue
- Start a discussion in the repository
- Contact the maintainers

Thank you for helping make SeisCPT better! 🌊