<p align="center">
  <img src="./documentation/marketing_material/Logo_V7_Color.png" width="900px"/>
</p>

# **OpenONDA: Hybrid CFD Solver in Python**

🚀 **OpenONDA** is an advanced and efficient **Computational Fluid Dynamics (CFD) solver** for Python. It enables execution and manipulation of the **Vortex Particle Method (VPM)** and **Finite Volume Method (FVM)** directly within Python.

🔹 **ONDA** (*"wave" in Portuguese*) stands for **"Operator for Numerical Design and Aerodynamics"**. It is a **hybrid solver** with the following capabilities:
   - **Eulerian flow simulations**: OpenFOAM is wrapped and exposed as a Python class: `fvmSolver`.
   - **Lagrangian flow simulations**: An in-house VPM solver with DNS and dynamic LES capabilities, wrapped in a Python class: `vpmSolver`.

<p align="center">
  <img src="./documentation/marketing_material/Diagram.png" width="760px"/>
</p>

> 🚀 **Don’t let "Aerodynamics" fool you!** While the name OpenONDA suggests a focus on aerodynamics, this solver is built for any computational fluid dynamics (CFD) application. Whether you're dealing with airflow, water simulations, or something even more exotic, OpenONDA has you covered! (And yes, we know... changing "Aerodynamics" in the acronym would break the name. But hey, ONDA sounds cool, so let’s roll with it! 😉)


## 🔹 **Solver Capabilities**
✅ 3D unsteady flow simulations  
✅ Large-Eddy Simulation (LES) modeling for both VPM and FVM solvers  
✅ Seamless interfacing with external solvers via Python  
🚧 *(In development)*: Integration between OpenFOAM and VPM solvers  


## 🔹 **Example Simulations**

### **Vortex Filament Flow**  
_Run with:_  
```bash
cd ./examples/lagrangian_module_examples
python E1_vortex_filament_flow.py
```
<p align="center">
<img src="./documentation/examples_of_results/Example_1.gif" alt="Vortex-Filament" width="550px"/>
</p>

### **Vortex Ring Flow**  
_Run with:_  
```bash
cd ./examples/lagrangian_module_examples
python E2_vortex_ring_flow.py
```
<p align="center">
<img src="./documentation/examples_of_results/Example_2.gif" alt="Vortex-Ring" width="550px"/>
</p>


## 🔹 **System Requirements**
✔ **OS**: Ubuntu 22.10 (and, very likely, 22.04 LTS)  
✔ **CFD Framework**: OpenFOAM v2406 (2024)  
✔ **Python Version**: 3.9.13  
✔ **Required Libraries**:  
   - `numpy`, `matplotlib`, `scipy`  
   - `cython`, `libboost-all-dev` (for Boost.Python)  
   - `pyublas`  


## 🔹 **Installation Guide**

### **1️⃣ Install Prerequisites**

Here's your proofread text with improved clarity and flow while keeping all your original content:

#### **1.1 Install Conda**  
- Use the script below to download and install the 2024-10-1 version of **Anaconda** or follow the official 🔗 [installation guide](https://www.anaconda.com/docs/getting-started/anaconda/install#macos-linux-installation) (see also 🔗 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)) for other, more recent version:

  ```bash
  curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
  bash ~/Anaconda3-2024.10-1-Linux-x86_64.sh
  source ~/.bashrc
  ```


- Install necessary system dependencies (**Linux only**):
    ```bash
    sudo apt update
    sudo apt install -y libboost-all-dev
    ```

#### **1.2 Install OpenFOAM**  
- Download and install the precompiled version of 🔗 [OpenFOAM-v2406](https://develop.openfoam.com/Development/openfoam/-/wikis/precompiled/debian) using:

  ```bash
  # Add the repository
  curl https://dl.openfoam.com/add-debian-repo.sh | sudo bash

  # Update the repository information
  sudo apt-get update

  # Install preferred package. Eg,
  sudo apt-get install openfoam2406-default

  # Add OpenFOAM sourcing and alias to ~/.bashrc
  echo 'source /usr/lib/openfoam/openfoam2406/etc/bashrc' >> ~/.bashrc
  echo 'alias of2406="source /usr/lib/openfoam/openfoam2406/etc/bashrc"' >> ~/.bashrc

  # Source to apply the changes immediately
  source ~/.bashrc
  ````

  📌 *Note: OpenONDA has been tested with OpenFOAM v2406. Compatibility with newer versions is not guaranteed. In case to change for more recent version, library links must be modified accordingly.*

  📌 To verify that OpenFOAM is installed and correctly sourced, type `foamVersion` on your terminal. If this prints out **"OpenFOAM v2406"** and you are inside the `OpenONDA` environment, you're all set!

#### **1.3 Clone the Repository**  
  ```bash
  git clone https://github.com/EngFlavioMartins/OpenONDA.git
  cd OpenONDA
  ```

#### **1.4 Set Up the Conda Environment**  
  ```bash
  conda env create --file=./documentation/environment.yml; 
  conda activate OpenONDA
```

### **2️⃣ Install OpenONDA**

📌 Ensure that Anaconda is correctly sourced and that you are inside a Conda environment (e.g., your terminal prompt should display `(OpenONDA)`)! Then, run:

```bash
python setup_environment.py
```

If no error messaged appear, then source your changes using:

```bash
source ~/.bashrc
```

This script will:  
✅ Set up the necessary environment variables in your `~/.bashrc` and source them.  
✅ Activate the `OpenONDA` Conda environment (your terminal prompt should now begin with `(OpenONDA)`).  
✅ Automatically source OpenFOAM for you.  


Now, proceed with the installation by running:

```bash
pip install -e .
```
✔ *This will compile and install all necessary components for OpenONDA.*  


## 🔹 **Running Your First Test Case**

After installation, make sure you are inside the `OpenONDA` Conda environment (`conda activate OpenONDA`). Then, verify that everything is working correctly by running the test scripts:

📌 *Run these commands from within the `tests` directory!*

```bash
pytest -v -s --tb=long ./test1_import_modules.py
pytest -v -s --tb=long ./test2_eulerian_solver_communication.py
```

✅ If no errors appear in the log files, everything is set up correctly.  
✅ Example cases can be found in the `./examples/` directory.

📌 **Tip:** To ensure you have the correct permissions to execute all Bash scripts in this project, run the following command from the `examples` directory:

```bash
chmod u+x ./*.sh */*.sh
```

## 🔹 **Troubleshooting Guide**
📌 **Common Issues & Fixes**

- **Conda command not recognized?** Run:
  ```bash
  conda init
  ```
  Then restart your terminal.
- **Missing dependencies?** Ensure `libboost-all-dev` is installed:
  ```bash
  sudo apt install -y libboost-all-dev
  ```
- **Wrong Python environment?** Check active environments with:
  ```bash
  conda info --envs
  ```
- **Permission issues running scripts?** If you find something like ```[Errno 13] Permission denied: ...``` when running your tests, simply run the command below:
  ```bash
  chmod +x <script-name>.sh
  ```
- 🚧 *Jupyter Notebook compatibility is currently limited to terminal execution (`jupyter notebook`). Support for VS Code and similar IDEs is under development.*


## 🔹 **License**
📄 **OpenONDA** is distributed under the **GNU General Public License (GPL) v3** or later.


## 🔹 **How to Cite**

If you use **OpenONDA** in your work, please cite:

📌 **To cite the VPM solver**:  
- *Martins, F., Lastname, S., & Lastname, C. (2025). FLARE: A hybrid OpenFOAM and vortex particle method for external flow simulations. *Journal of Computational Fluid Dynamics, 12*(3), 123–145. https://doi.org/10.1234/dummy-doi*

📌 **To cite the PyFoamLink Python interface**:  
- *Martins, F., Lastname, S., & Lastname, C. (2025). FLARE: A hybrid OpenFOAM and vortex particle method for external flow simulations. *Journal of Computational Fluid Dynamics, 12*(3), 123–145. https://doi.org/10.1234/dummy-doi*

📌 **To cite the complete OpenONDA framework**:  
- *Martins, F., Lastname, S., & Lastname, C. (2025). FLARE: A hybrid OpenFOAM and vortex particle method for external flow simulations. *Journal of Computational Fluid Dynamics, 12*(3), 123–145. https://doi.org/10.1234/dummy-doi*


## 🔹 **Authors and Contributions**
👨‍💻 **Artur Palha** (2013-2016) - Initial development, pHyFlow  
👨‍💻 **Rention Pasolari** (2022-2024) - Major modifications, pHyFlow  
👨‍💻 **Flavio Martins** (2024-Present) - 3D flow capabilities, Eulerian-Lagrangian communication, VPM solver, debugging, examples, documentation, conversion to OpenONDA  

📩 **Contact:** [Flavio Martins](mailto:f.m.martins@tudelft.nl)


## 🔹 **Support & Contributions**
- *Want to contribute? Found a bug?* Contact via email above or open an **issue** or **pull request** on GitHub!