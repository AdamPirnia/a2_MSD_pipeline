# ADMDynAnlz

ADMDynAnlz is a graphical workflow tool for post-processing molecular dynamics trajectories and generating analysis-ready workflows for transport and dynamical observables.

The application currently provides three GUI modules:

- `Coordinates extraction`
- `Velocities and Dipoles`
- `MSD and NGP / Anisotropic NGP`

## Definitions

### MSD
MSD stands for **Mean Squared Displacement**. It measures the average squared displacement of particles as a function of time and is commonly used to analyze diffusion and translational dynamics.

### NGP = α₂(t)
NGP stands for **Non-Gaussian Parameter** and is written here as **α₂(t)**. It measures deviations from ideal Gaussian diffusion and helps identify heterogeneous or non-Fickian dynamics.

### α_{ani}(t)
**α_{ani}(t)** is the **anisotropy parameter**. In this project it is used to quantify directional anisotropy in molecular motion.

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
- **NGP = α₂(t)**: Non-Gaussian Parameter
- **α_{ani}(t)**: anisotropy parameter

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

## Repository Scope

This public repository contains the GUI application and user-facing project files needed to run or package the software.

## Help and Project Page

Public project page:

`https://github.com/AdamPirnia/ADMDynAnlz`

The GUI launcher also includes a `?` button that opens this page in the user’s default browser.

## License

See `LICENSE.txt` for license terms.
