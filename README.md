# ADMDynAnlz

[![User Manual](https://img.shields.io/badge/User%20Manual-Open-1f6feb)](Manual.md)
[![Citation](https://img.shields.io/badge/Citation-Open-0a7f5a)](CITATION.md)

ADMDynAnlz is a graphical workflow tool for post-processing molecular dynamics trajectories and generating analysis-ready workflows for transport and dynamical observables.

The application currently provides three GUI modules:

- `Coordinates extraction`
- `Velocities and Dipoles`
- `MSD and NGP / Anisotropic NGP`

Detailed section-by-section instructions are available in [Manual.md](Manual.md).

## Interface Overview

### Module selection

The launcher opens with a simple module-selection window so users can enter the workflow they need directly.

![Module selection](app_images/module_selection.png)

### Coordinates extraction workflow

The `Coordinates extraction` module organizes preprocessing into three stages:

1. `Extraction`
2. `Continuous coordinates`
3. `COM calculation`

It is designed to keep the full trajectory-preparation workflow in one place while still allowing individual steps to be skipped when they were already completed earlier.

![Coordinates extraction module](app_images/coodr_ext_module.png)

This image is one example of the lower control area. The `SLURM Submission Parameters` and `Output Files` sections are available in all modules, and the optional smart optimization tool works module-by-module to recommend suitable inputs for that module's workflow and resource needs.

![Coordinates extraction output and controls](app_images/coodr_ext_module_2.png)

## Definitions

### MSD
MSD stands for **Mean Squared Displacement**. It measures the average squared displacement of particles as a function of time and is commonly used to analyze diffusion and translational dynamics.

### NGP
NGP stands for **Non-Gaussian Parameter**. It measures deviations from ideal Gaussian diffusion and helps identify non-Fickian dynamics.

![alpha2 equation](app_images/equation_alhpa_2.png)

### Anisotropy Parameter
The anisotropy parameter quantifies directional anisotropy in molecular translational motion.

![alpha anisotropy equation](app_images/equation_alhpa_ani.png)

### COM
COM stands for **center of mass**.

## Modules

### 1. Coordinates extraction
This module prepares trajectory-derived data for downstream analysis. It includes three steps:

1. `Extraction`
   Extracts coordinate data from trajectory files using VMD.
2. `Continuous coordinates`
   Reconstructs continuous trajectories by removing periodic-boundary discontinuities.
3. `COM calculation`
   Computes center-of-mass trajectories for molecules or coarse-grained units.

### 2. Velocities and Dipoles
This module generates workflows for:

- extracting center-of-mass velocities from velocity trajectories
- computing individual molecular dipoles
- computing collective dipole signals for selected groups

### 3. MSD and NGP / Anisotropic NGP
This module generates workflows for:

- **MSD**: Mean Squared Displacement
- **NGP**: Non-Gaussian Parameter
- **Anisotropy parameter**

## What the Software Does

ADMDynAnlz is designed to help users:

- preprocess large molecular dynamics trajectories
- organize analysis inputs through a GUI instead of manual script editing
- generate execution-ready analysis scripts
- prepare local or cluster-oriented workflows, including SLURM job files when needed

## Requirements

- Python 3
- `numpy`
- VMD for trajectory-based extraction steps

## Running the GUI

```bash
python3 ADMDynAnlz_launcher.py
```

All entrypoints open a launcher window where you select the module you want to use.

## Running Downloaded Executables

### Linux

If needed, make the executable runnable and launch it from a terminal:

```bash
chmod +x ADMDynAnlz_Linux
./ADMDynAnlz_Linux
```

On Linux, double-click launching can also work when the file is marked as executable and the desktop environment allows launching executable files directly.

### macOS Apple Silicon

If you downloaded the Apple Silicon build, run:

```bash
chmod +x ADMDynAnlz_mac_arm64
xattr -d com.apple.quarantine ADMDynAnlz_mac_arm64 2>/dev/null || true
./ADMDynAnlz_mac_arm64
```

### macOS Intel

If you downloaded the Intel build, run:

```bash
chmod +x ADMDynAnlz_mac_x86_64
xattr -d com.apple.quarantine ADMDynAnlz_mac_x86_64 2>/dev/null || true
./ADMDynAnlz_mac_x86_64
```

### Windows

In most cases you can run the Windows executable directly. If Windows marks the file as downloaded from the internet, open PowerShell in the download folder and run:

```powershell
Unblock-File .\ADMDynAnlz_Windows.exe
.\ADMDynAnlz_Windows.exe
```

## Repository Scope

This public repository contains the GUI application and user-facing project files needed to run or package the software.

## Citation

Per the terms described in `LICENSE.txt`, users of ADMDynAnlz should cite the software when it contributes to published or presented work.

Citation information is provided in [CITATION.md](CITATION.md).

## License

See `LICENSE.txt` for license terms.
