# 🚀 SeisCPT Deployment Guide

## Local Development

### Quick Start
```bash
# Clone the repository
git clone https://github.com/Abeledes/SeisCPT.git
cd SeisCPT

# Install dependencies
pip install -r requirements.txt

# Run the application
python run_seismic_app.py
```

### Manual Streamlit Launch
```bash
streamlit run seismic_inversion_app.py
```

## Cloud Deployment

### Streamlit Cloud (Recommended)
1. Fork the repository on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy from your forked repository
5. Set main file: `seismic_inversion_app.py`

### Heroku Deployment
```bash
# Install Heroku CLI
# Create Procfile
echo "web: streamlit run seismic_inversion_app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create your-seiscpt-app
git push heroku main
```

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "seismic_inversion_app.py", "--server.address=0.0.0.0"]
```

### AWS/GCP/Azure
- Use container services (ECS, Cloud Run, Container Instances)
- Configure for 10GB file upload limits
- Ensure sufficient memory (8GB+ recommended)

## Configuration

### Environment Variables
```bash
# Optional environment variables
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=10240
export STREAMLIT_SERVER_MAX_MESSAGE_SIZE=1024
```

### Production Settings
- Enable HTTPS for secure file uploads
- Configure proper memory limits
- Set up monitoring and logging
- Implement user authentication if needed

## Performance Optimization

### For Large Files
- Use SSD storage for temporary files
- Configure adequate swap space
- Monitor memory usage
- Consider horizontal scaling for multiple users

### Caching
- Streamlit automatically caches function results
- Large datasets are cached in session state
- Clear cache periodically for memory management

## Security Considerations

### File Upload Security
- Files are processed in isolated temporary directories
- Automatic cleanup of temporary files
- Input validation for SEG-Y format
- No persistent storage of uploaded data

### Network Security
- Use HTTPS in production
- Configure proper CORS settings
- Implement rate limiting if needed
- Monitor for unusual upload patterns

## Monitoring

### Health Checks
```python
# Add to your deployment
def health_check():
    return {"status": "healthy", "version": "1.0.0"}
```

### Logging
- Application logs available in Streamlit
- Monitor memory usage for large files
- Track processing times
- Log inversion algorithm performance

## Troubleshooting

### Common Issues
1. **Memory errors**: Increase available RAM or use subset loading
2. **Upload timeouts**: Check network and file size limits
3. **Processing failures**: Verify SEG-Y file format
4. **Display issues**: Clear browser cache

### Debug Mode
```bash
streamlit run seismic_inversion_app.py --logger.level=debug
```

## Scaling

### Horizontal Scaling
- Deploy multiple instances behind load balancer
- Use shared storage for temporary files
- Implement session affinity for file uploads

### Vertical Scaling
- Increase memory for larger datasets
- Use faster CPUs for processing
- Consider GPU acceleration for future ML features

## Backup and Recovery

### Code Backup
- Repository is backed up on GitHub
- Use branch protection rules
- Tag releases for version control

### Data Backup
- No persistent data storage required
- Users responsible for their own data
- Consider implementing result export/save features