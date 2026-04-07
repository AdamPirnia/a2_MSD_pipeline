# ADMDynAnlz

[![User Manual](https://img.shields.io/badge/User%20Manual-Open-1f6feb)](Manual.md)
[![Citation](https://img.shields.io/badge/Citation-Open-0a7f5a)](CITATION.md)

ADMDynAnlz is a graphical workflow tool for post-processing molecular dynamics trajectories and generating analysis-ready workflows for transport and dynamical observables.

The current distributed application is the Qt GUI.

The application currently provides four GUI modules:

- `Coordinates extraction`
- `Velocities and Dipoles`
- `MSD and NGP / Anisotropic NGP`
- `Correlation Functions`

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
- collective dipole vector components: same dipole unit returned by VMD for the collective vector output

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
This module generates workflows for time-correlation analysis on numeric data arrays.

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

## Running the GUI

```bash
python ADMDynAnlz_launcher.py
```

All entrypoints open a launcher window where you select the module you want to use.

If a generated workflow cannot find VMD even though `vmd` works in your terminal, the most common reason is that your shell is using an alias or wrapper that the generated scripts cannot see. In that case, enter the actual VMD executable path in the GUI instead.

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

Large Linux binaries may be distributed as GitHub release assets rather than regular repository files.

For maintainers updating the Linux build:

- the Linux executable is published as a GitHub release asset rather than being committed into repository history
- this avoids rapid repository bloat from repeated large binary updates

To publish the Linux executable as a release asset, authentication can be provided by:

- `PUBLIC_REPO_TOKEN`
- `GITHUB_TOKEN`
- `GH_TOKEN`
- `gh auth token` after logging in with GitHub CLI

To create a GitHub token:

1. Sign in to GitHub.
2. Open `Settings`.
3. Open `Developer settings`.
4. Open `Personal access tokens`.
5. Create a token with repository access for the target public repo.

Example:

```bash
export PUBLIC_REPO_TOKEN=your_token_here
./publish_linux_to_public.sh
```

## Citation

Per the terms described in `LICENSE.txt`, users of ADMDynAnlz should cite the software when it contributes to published or presented work.

Citation information is provided in [CITATION.md](CITATION.md).

## License

See `LICENSE.txt` for license terms.
