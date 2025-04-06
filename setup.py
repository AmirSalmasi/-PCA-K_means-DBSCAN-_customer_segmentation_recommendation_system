import os
import sys
import subprocess
import platform

def check_uv_installation():
    """Check if UV is installed."""
    try:
        subprocess.run(['uv', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_uv():
    """Install UV package manager."""
    print("Installing UV...")
    if platform.system() == "Windows":
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'uv'], check=True)
    else:
        subprocess.run(['curl', '-LsSf', 'https://astral.sh/uv/install.sh', '|', 'sh'], check=True)

def create_venv():
    """Create virtual environment using UV."""
    print("Creating virtual environment...")
    subprocess.run(['uv', 'venv'], check=True)

def install_dependencies():
    """Install project dependencies using UV."""
    print("Installing dependencies...")
    subprocess.run(['uv', 'pip', 'install', '-r', 'requirements.txt'], check=True)

def main():
    """Main setup function."""
    if not check_uv_installation():
        install_uv()
    
    create_venv()
    install_dependencies()
    
    print("\nSetup completed successfully!")
    print("\nTo activate the virtual environment:")
    if platform.system() == "Windows":
        print("    .venv\\Scripts\\activate")
    else:
        print("    source .venv/bin/activate")

if __name__ == "__main__":
    main() 