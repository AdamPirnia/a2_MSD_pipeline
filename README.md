# ADMDynAnlz

[<img src="app_images/button_prep_run.svg" alt="Prep. Run" width="220">](Run_Prep.md) Download, setup, and launch steps.<br>
[<img src="app_images/button_user_manual.svg" alt="User Manual" width="220">](Manual.md) Module-by-module usage guide.<br>
[<img src="app_images/button_citation.svg" alt="Citation" width="220">](CITATION.md) Reference information for citing the software.

ADMDynAnlz is a graphical workflow tool for post-processing molecular dynamics trajectories and generating analysis-ready workflows for transport and dynamical observables.

The current distributed application is the Qt GUI.

The application currently provides six GUI modules:

- `Coordinates extraction`
- `Velocities and Dipoles`
- `MSD and NGP / Anisotropic NGP`
- `Correlation Functions`
- `Diffusion constant`
- `Static Structure Factor`

Detailed section-by-section instructions are available in [Manual.md](Manual.md).
A dedicated preparation and download guide is available in [Run_Prep.md](Run_Prep.md).

## What the Software Does

ADMDynAnlz is designed to help users:

- preprocess large molecular dynamics trajectories
- organize analysis inputs through a GUI instead of manual script editing
- generate execution-ready analysis scripts
- prepare local or cluster-oriented workflows, including SLURM job files when needed


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

### Anisotropic NGP
Anisotropic NGP quantifies directional anisotropy in molecular translational motion.

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

Typical outputs and units:

- extracted atomic coordinates: same length unit as the input trajectory coordinates
- continuous / unwrapped atomic coordinates: same length unit as the input trajectory coordinates
- center-of-mass coordinates: same length unit as the input trajectory coordinates

### 2. Velocities and Dipoles
This module generates workflows for:

- extracting center-of-mass velocities from velocity trajectories
- computing individual molecular dipoles
- computing collective dipole signals for selected groups

Current behavior highlights:

- velocity extraction is controlled by `Target Selection` together with `Grouping Unit`
- velocity extraction does not use a module-level particle-count field
- individual dipole calculations use their own `Number of particles` field
- stride control is available for velocity extraction and individual dipole calculations
- individual dipole magnitude output is optional
- individual dipole workflows assume atoms are ordered in consistent repeating blocks, such as `O H H | O H H | ...`

Typical outputs and units:

- COM velocities: same velocity unit as the input velocity trajectory
- individual dipole vectors: Debye
- individual dipole magnitudes: Debye
- collective dipole magnitude: Debye
- collective dipole vector components: VMD internal dipole units (Debye or e·Å)

### 3. MSD and NGP / Anisotropic NGP
This module generates workflows for:

- **MSD**: Mean Squared Displacement
- **NGP**: Non-Gaussian Parameter
- **Anisotropic NGP**

`NGP α₂(t)`

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="app_images/equation_alhpa_2_dark.png">
    <img src="app_images/equation_alhpa_2.png" alt="alpha2 equation" width="62%">
  </picture>
</p>

`Anisotropic NGP`

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="app_images/equation_alhpa_ani_dark.png">
    <img src="app_images/equation_alhpa_ani.png" alt="alpha anisotropy equation" width="72%">
  </picture>
</p>

Typical outputs and units:

- MSD: squared length in the square of the coordinate unit
- NGP `α₂(t)`: dimensionless
- directional anisotropy outputs and anisotropic NGP: dimensionless

### 4. Correlation Functions
This module generates workflows for time-correlation analysis from numeric data arrays.

It supports:

- scalar-scalar correlation
- vector-vector correlation
- single-variable mode
- multiple-variable mode averaged across variables
- `Subtracting Mean` mode with mean subtraction
- `Not Subtracting Mean` mode without mean subtraction

Current behavior highlights:

- the reported correlation function is written as `C(t)/C(0)`
- the separate variance output stores `C(0) * Coefficient^2`
- `Step (between frames)` determines the unit of the horizontal axis, such as time

Typical outputs and units:

- time column: same unit implied by `Step (between frames)`
- correlation column: product of the units of the two input arrays

### 5. Diffusion constant
This module generates workflows for diffusion-constant analysis from saved VACF data, saved MSD data, or both.

It currently supports:

- a VACF-based route
- an MSD-based route
- optional infinite-size correction for both routes
- multiple temperatures in one generated script

`VACF-based diffusion constant`

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="app_images/DvACF_dark.png">
    <img src="app_images/DvACF.png" alt="VACF-based diffusion constant equation" width="58%">
  </picture>
</p>

`Velocity relaxation time`

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="app_images/tau_v_dark.png">
    <img src="app_images/tau_v.png" alt="Velocity relaxation time equation" width="66%">
  </picture>
</p>

`MSD-based diffusion constant`

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="app_images/D_MSD_dark.png">
    <img src="app_images/D_MSD.png" alt="MSD-based diffusion constant equation" width="46%">
  </picture>
</p>

Typical outputs and units:

- velocity relaxation time `tau`: same time unit as the saved VACF spacing input
- diffusion constant: `A^2/ps` and `cm^2/s`
- VACF variance comparison: `A^2/ps^2`

Input-handling highlights:

- relative VACF, MSD, and output paths are resolved from `Base Directory`
- if a time axis exists, the first column is treated as time and the second as data
- if a time axis does not exist, the module reconstructs time from the saved-frame timestep
- normalized VACF input disables velocity-unit conversion and uses the equipartition variance for the diffusion estimate
- if multiple temperatures match one common-term list exactly, temperatures are paired with that common term instead of nested separately

### 6. Static Structure Factor
This module generates workflows for isotropic static-structure-factor analysis from saved coordinate arrays.

Current inputs:

- `Base Directory`
- `Common Terms`
- `Coordinate Path`
- `Output Path`
- `k Max`, `L_x`, `L_y`, `L_z`, and `tolerance`

The generated workflow computes isotropic `S(k)` from the coordinate dataset and writes the result to the user-specified output path.

`Isotropic static structure factor`

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="app_images/static_structure_factor_dark.png">
    <img src="app_images/static_structure_factor.png" alt="Static structure factor equation" width="66%">
  </picture>
</p>


## Citation

Per the terms described in `LICENSE.txt`, users of ADMDynAnlz should cite the software when it contributes to published or presented work.

Citation information is provided in [CITATION.md](CITATION.md).

## License

See `LICENSE.txt` for license terms.
