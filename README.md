# ADMDynAnlz

[![User Manual](https://img.shields.io/badge/User%20Manual-Open-1f6feb)](Manual.md)
[![Citation](https://img.shields.io/badge/Citation-Open-0a7f5a)](CITATION.md)

ADMDynAnlz is a graphical workflow tool for post-processing molecular dynamics trajectories and generating analysis-ready workflows for transport and dynamical observables.

The current distributed application is the Qt GUI.

The application currently provides five GUI modules:

- `Coordinates extraction`
- `Velocities and Dipoles`
- `MSD and NGP / Anisotropic NGP`
- `Correlation Functions`
- `Diffusion constant`

Detailed section-by-section instructions are available in [Manual.md](Manual.md).

## What the Software Does

ADMDynAnlz is designed to help users:

- preprocess large molecular dynamics trajectories
- organize analysis inputs through a GUI instead of manual script editing
- generate execution-ready analysis scripts
- prepare local or cluster-oriented workflows, including SLURM job files when needed

## Requirements

- Python 3.10 or newer
- Python 3.12 recommended
- `numpy`
- VMD for trajectory-based extraction steps

For any `VMD Path` field, enter the actual VMD executable path or an actual runnable launcher command. Do not rely on shell aliases.

Platform notes:

- macOS: use the real executable or launcher path from the VMD application bundle, not an alias such as `vmd` defined only in your interactive shell
- Windows: use the real `vmd.exe` path, not a desktop shortcut or a shell-only wrapper command

Generic examples:

- macOS: `/Applications/VMD ... .app/Contents/MacOS/...`
- Windows: `C:\Program Files\VMD\...\vmd.exe`

## How and what to download

You need to download two different things from this project:

- the repository source code, if you want the scripts, documentation, and editable project files
- a prebuilt executable from the GitHub Releases page, if you only want to run the GUI application

Choose the executable that matches your system:

- Linux: `ADMDynAnlz_Linux`
- macOS Apple Silicon: `ADMDynAnlz_mac_arm64`
- macOS Intel: `ADMDynAnlz_mac_x86_64`
- Windows: `ADMDynAnlz_Windows.exe`

Repository page:

- `https://github.com/adampirnia/a2_MSD_source`

Releases page:

- `https://github.com/adampirnia/a2_MSD_source/releases`

### Web browser

To download the repository source code in a browser:

1. Open `https://github.com/adampirnia/a2_MSD_source`
2. Click `Code`
3. Choose either:
   - `Download ZIP` to download the repository as an archive
   - the repository URL if you only want to copy the clone link for later use

To download a prebuilt executable in a browser:

1. Open `https://github.com/adampirnia/a2_MSD_source/releases`
2. Open the latest release, or another release if you need a specific version
3. In the `Assets` section, download the file for your operating system

### Command line

To download the repository source code with Git:

```bash
git clone https://github.com/adampirnia/a2_MSD_source.git
cd a2_MSD_source
```

If you prefer GitHub CLI, you can also clone with:

```bash
gh repo clone adampirnia/a2_MSD_source
cd a2_MSD_source
```

To download a release executable with GitHub CLI:

```bash
gh release download --repo adampirnia/a2_MSD_source --pattern "ADMDynAnlz_Linux"
```

Replace `ADMDynAnlz_Linux` with the file you actually want:

- `ADMDynAnlz_mac_arm64`
- `ADMDynAnlz_mac_x86_64`
- `ADMDynAnlz_Windows.exe`

If you do not use GitHub CLI, you can download directly with `curl`. Replace `TAG_NAME` with the release tag you want, such as `v1.0.0`, and replace the filename with the correct asset name:

```bash
curl -LO https://github.com/adampirnia/a2_MSD_source/releases/download/TAG_NAME/ADMDynAnlz_Linux
```

Equivalent `wget` example:

```bash
wget https://github.com/adampirnia/a2_MSD_source/releases/download/TAG_NAME/ADMDynAnlz_Linux
```

## Python Environment Setup

ADMDynAnlz now uses Python features that require Python 3.10 or newer. The recommended setup is a Conda environment named `admdyn` with Python 3.12 so you do not need to change your system-default Python.

### macOS and Linux

If you do not already have Conda installed, install Miniconda or Anaconda first. Then create and activate the environment:

```bash
conda create -n admdyn python=3.12
conda activate admdyn
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy pandas psutil Pillow PySide6 pyinstaller scipy matplotlib
```

Optional verification:

```bash
python --version
python -c "import numpy, pandas, psutil, PIL, PySide6; print('ok')"
```

When the environment is active, your shell prompt usually starts with `(admdyn)`.

To leave the environment later:

```bash
conda deactivate
```

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

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="app_images/equation_alhpa_2_dark.png">
    <img src="app_images/equation_alhpa_2.png" alt="alpha2 equation" width="62%">
  </picture>
</p>

### Anisotropy Parameter
The anisotropy parameter quantifies directional anisotropy in molecular translational motion.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="app_images/equation_alhpa_ani_dark.png">
    <img src="app_images/equation_alhpa_ani.png" alt="alpha anisotropy equation" width="72%">
  </picture>
</p>

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

- velocity extraction uses the module-level `Number of Particles` setting
- stride control is available for velocity extraction and individual dipole calculations
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
- **Anisotropy parameter**

Typical outputs and units:

- MSD: squared length in the square of the coordinate unit
- NGP `α₂(t)`: dimensionless
- directional anisotropy outputs and anisotropy parameter: dimensionless

### 4. Correlation Functions
This module generates workflows for time-correlation analysis from numeric data arrays.

It supports:

- scalar-scalar correlation
- vector-vector correlation
- single-variable mode
- multiple-variable mode averaged across variables
- `acf` mode with mean subtraction
- `fluctuation` mode without mean subtraction

The reported correlation values are not normalized by the `t = 0` correlation.

Typical outputs and units:

- time column: same time unit as `Time per Lag (t1)`
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


## Citation

Per the terms described in `LICENSE.txt`, users of ADMDynAnlz should cite the software when it contributes to published or presented work.

Citation information is provided in [CITATION.md](CITATION.md).

## License

See `LICENSE.txt` for license terms.
