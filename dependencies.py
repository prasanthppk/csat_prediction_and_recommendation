import subprocess
import sys

def install_requirements():
    try:
        # This runs the 'pip install' command from inside your script
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except Exception as e:
        print(f"Failed to install dependencies: {e}")

# Run the installation function
install_requirements()
