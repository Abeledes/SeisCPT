#!/usr/bin/env python3
"""
Startup script for SeisCPT Seismic Inversion Application
Configures Streamlit for large file handling and launches the app.
"""
import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Setup environment for large file processing"""
    
    # Set environment variables for large file handling
    os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '10240'  # 10GB in MB
    os.environ['STREAMLIT_SERVER_MAX_MESSAGE_SIZE'] = '1024'   # 1GB in MB
    
    # Set memory optimization flags
    os.environ['PYTHONHASHSEED'] = '0'  # Reproducible hashing
    os.environ['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads
    
    # Create .streamlit directory if it doesn't exist
    streamlit_dir = Path('.streamlit')
    streamlit_dir.mkdir(exist_ok=True)
    
    # Ensure config.toml exists
    config_file = streamlit_dir / 'config.toml'
    if not config_file.exists():
        print("Creating Streamlit configuration for large files...")
        
        config_content = """[server]
# Increase file upload size to 10GB
maxUploadSize = 10240

# Increase message size for large data transfers
maxMessageSize = 1024

# Server settings for large files
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
# Professional dark theme
primaryColor = "#4fc3f7"
backgroundColor = "#0a0e27"
secondaryBackgroundColor = "#1a1f3a"
textColor = "#e8eaed"

[runner]
# Memory settings for large seismic data
magicEnabled = true
installTracer = false
fixMatplotlib = true

[client]
showErrorDetails = true
toolbarMode = "auto"
"""
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Configuration created: {config_file}")

def check_dependencies():
    """Check if required dependencies are installed"""
    
    required_packages = [
        'streamlit',
        'numpy', 
        'matplotlib',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        
        print("\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        
        install_now = input("\n🤔 Install missing packages now? (y/n): ").lower().strip()
        
        if install_now == 'y':
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install'
                ] + missing_packages)
                print("✅ Packages installed successfully!")
            except subprocess.CalledProcessError:
                print("❌ Failed to install packages. Please install manually.")
                return False
        else:
            print("⚠️ Some features may not work without required packages.")
    
    return True

def get_system_info():
    """Display system information for large file processing"""
    
    try:
        import psutil
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        # Disk space
        disk = psutil.disk_usage('.')
        disk_free_gb = disk.free / (1024**3)
        
        print(f"\n💻 System Information:")
        print(f"   - Total RAM: {memory_gb:.1f} GB")
        print(f"   - Available RAM: {memory.available / (1024**3):.1f} GB")
        print(f"   - Free disk space: {disk_free_gb:.1f} GB")
        
        # Recommendations
        if memory_gb < 8:
            print("⚠️  Warning: Less than 8GB RAM. Large files may cause issues.")
        elif memory_gb < 16:
            print("ℹ️  Note: 16GB+ RAM recommended for files > 2GB.")
        else:
            print("✅ Sufficient RAM for large seismic files.")
        
        if disk_free_gb < 20:
            print("⚠️  Warning: Low disk space. Ensure sufficient space for temp files.")
        
    except ImportError:
        print("ℹ️  Install 'psutil' for system information: pip install psutil")

def launch_app():
    """Launch the Streamlit application"""
    
    print("\n🚀 Launching SeisCPT Seismic Inversion Application...")
    print("📊 Configured for files up to 10GB")
    print("🌐 Opening in browser...")
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            'seismic_inversion_app.py',
            '--server.port', '8501',
            '--server.headless', 'false'
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
    except Exception as e:
        print(f"❌ Failed to launch application: {e}")

def main():
    """Main startup function"""
    
    print("🌊 SeisCPT - Professional Seismic Inversion")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Show system info
    get_system_info()
    
    # Launch application
    launch_app()

if __name__ == "__main__":
    main()