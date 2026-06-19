# config.py

import sys

from source.version import __version__


def show():
    """
    Show the build configuration for OpenONDA.
    """
    print("OpenONDA Build Configuration:")
    # You can extend this to include other details such as paths, dependencies, etc.
    print(f"Python version: {sys.version}")
    print(f"OpenONDA version: {__version__}")
    # Add any additional configuration here as needed
