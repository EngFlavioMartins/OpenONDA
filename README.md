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

> 🚀 **Don’t let "Aerodynamics" fool you!** While the name OpenONDA suggests a focus on aerodynamics, this solver is built for any computational fluid dynamics (CFD) application. *(And yes, we know... changing "Aerodynamics" in the acronym would break the name. But hey, ONDA sounds cool, so let’s roll with it! 😉)*

## 🔹 **Solver Capabilities**
✅ 3D unsteady flow simulations  
✅ Large-Eddy Simulation (LES) modeling for both VPM and FVM solvers  
✅ Seamless interfacing with external solvers via Python  
🚧 *(In development)*: Integration between OpenFOAM and VPM solvers  

## 🔹 **Example Simulations**

### **Vortex Filament Flow**  

<p align="center">
<img src="./documentation/examples_of_results/Example_1.gif" alt="Vortex-Filament" width="550px"/>
</p>

### **Vortex Ring Flow**  

<p align="center">
<img src="./documentation/examples_of_results/Example_2.gif" alt="Vortex-Ring" width="550px"/>
</p>

## 🔹 **System Requirements**
✔ **OS**: Ubuntu 22.10 and 22.04 LTS (and, very likely, other Debian-based OS's)  
✔ **CFD Framework**: OpenFOAM v2406 (2024)  
✔ **Python Version**: 3.12  

## 🔹 **Installation Guide**

### **1️⃣ Install Prerequisites**

#### **1.1 Clone the Repository**  
```bash
git clone https://github.com/EngFlavioMartins/OpenONDA.git
cd OpenONDA
```

#### **1.2 Install OpenFOAM**
Download and install the precompiled version of 🔗 [OpenFOAM-v2406](https://develop.openfoam.com/Development/openfoam/-/wikis/precompiled/debian)

```bash
./install_openfoam.sh
```

📌 If needed, grant execution permission: `chmod +x install_openfoam.sh`

📌 This process may take **5-15 minutes**, depending on your internet speed.

📌 This script will:

✅ Set up necessary environment variables in your `~/.bashrc` and source them.  
✅ Automatically source OpenFOAM.  
✅ Compile all necessary OpenFOAM libraries.


#### **1.3 Set Up the Conda Environment**  
If you **don’t have Conda installed**, use the script below to install Anaconda or follow the official 🔗 [installation guide](https://www.anaconda.com/docs/getting-started/anaconda/install#macos-linux-installation):

```bash
./install_anaconda.sh
```

If Conda is already installed, make sure conda `(base)` is shown in your prompt and create and activate the environment:

```bash
conda env create -f ./documentation/openonda_environment.yml
conda activate OpenONDA
```

Apply the changes immediately:
```bash
source ~/.bashrc
```

📌 Ensure your terminal prompt now shows `(OpenONDA)` before proceeding.  

Verify the installation:
```bash
foamVersion
```
If this prints **"OpenFOAM-v2406"**, and you are inside the `(OpenONDA)` environment, you're all set!  

### **2️⃣ Install OpenONDA**
Now, install OpenONDA:
```bash
pip install --no-cache-dir --use-pep517 .
```
Verify the installation:
```bash
python -c "import OpenONDA; print('\nInstallation successful =)\n')"
```
✔ *This will compile and install all necessary components for OpenONDA.*  

## 🔹 **Running Your First Test Case**
Ensure you are in the `OpenONDA` Conda environment:
```bash
conda activate OpenONDA
```
Run the test scripts:
```bash
pytest -v -s --tb=long ./test1_import_modules.py
pytest -v -s --tb=long ./test2_eulerian_solver_communication.py
```
✅ If no errors appear, everything is set up correctly.  
✅ Example cases are available in `./examples/`.

📌 **Tip:** To ensure correct script execution, run:
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

- **Permission issues running scripts?** If you see `[Errno 13] Permission denied: ...`, run:
  ```bash
  chmod +x <script-name>.sh
  ```

## 🔹 **License**
📄 **OpenONDA** is distributed under the **GNU General Public License (GPL) v3** or later.

## 🔹 **How to Cite**

📌 **To cite OpenONDA or its components, use the following:** *(Replace placeholder names and DOIs!)*

```bibtex
@article{martins2025,
  author    = {Martins, F. and Lastname, S. and Lastname, C.},
  title     = {OpenONDA: A hybrid OpenFOAM and vortex particle method for external flow simulations},
  journal   = {Journal of Computational Fluid Dynamics},
  volume    = {12},
  number    = {3},
  pages     = {123–145},
  year      = {2025},
  doi       = {10.1234/dummy-doi}
}
```

## 🔹 **Authors and Contributions**
👨‍💻 Artur Palha (2013-2016) - Initial development, pHyFlow  
👨‍💻 Rention Pasolari (2022-2024) - Major modifications, pHyFlow  
👨‍💻 Flavio Martins (2024-Present) - 3D flow capabilities, Eulerian-Lagrangian communication, VPM solver, debugging, examples, documentation, OpenONDA conversion  
📩 **Contact:** [Flavio Martins](mailto:f.m.martins@tudelft.nl)

## 🔹 **Support & Contributions**
Want to contribute? Found a bug? Open an **issue** or **pull request** on GitHub!
