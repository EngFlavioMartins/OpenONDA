# setup.py
from setuptools import setup, find_packages
import os

root = os.getcwd()

def main():
    setup(
        name="OpenONDA",
        version="0.0.1",
        packages=find_packages(where="."),
        package_dir={"": "."},
        install_requires=["numpy", "scipy", "cython"],
        package_data={
            'OpenONDA.solvers.FVM': ['fvmModule.so'],
            'OpenONDA.solvers.VPM': ['fvmModule'],
            'OpenONDA.utilities': ['*'], # includes all files in utilities
        },
    )

if __name__ == "__main__":
    main()