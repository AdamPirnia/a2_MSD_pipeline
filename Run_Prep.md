# Preparation and Running Software

This chapter explains how to prepare the runtime environment for a downloaded executable and how to launch the software on each supported platform.

The distributed executable is the GUI itself. The Python environment described here is still important because the generated analysis scripts use Python packages such as `numpy`, `pandas`, `psutil`, `Pillow`, `PySide6`, `scipy`, and `matplotlib`.

## Step 1: Download what you need

Users normally need:

- one platform executable from the GitHub release assets
- the public repository files or public ZIP, which include documentation and helper files such as `configure.sh`

Choose the executable that matches your platform:

- Linux: `ADMDynAnlz_Linux`
- macOS Apple Silicon: `ADMDynAnlz_mac_arm64`
- macOS Intel: `ADMDynAnlz_mac_x86_64`
- Windows: `ADMDynAnlz_Windows.exe`

## Recommended method: use `configure.sh`

The easiest preparation path is to use `configure.sh`.

From a terminal in the folder containing `configure.sh` and the downloaded executable, run:

```bash
chmod +x configure.sh
./configure.sh
```

What these commands do:

- `chmod +x configure.sh`
  makes the helper script executable on Unix-like systems such as Linux and macOS
- `./configure.sh`
  starts the interactive configurator

What the configurator does:

- asks which platform executable you downloaded
- creates a local Python environment
- installs the required Python packages
- verifies that the Python environment works
- prepares the executable for the selected platform
- prints the exact launch command

## Manual method

If you prefer to configure everything yourself, follow the manual steps below.

## Python Environment Setup

ADMDynAnlz uses Python features that require Python 3.10 or newer. The recommended setup is a Python 3.12 environment named `admdyn`.

### macOS and Linux

If you do not already have Conda installed, first install Miniconda or Anaconda.

Then create and activate the environment:

```bash
conda create -n admdyn python=3.12
conda activate admdyn
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy pandas psutil Pillow PySide6 pyinstaller scipy matplotlib
```

What each command does:

- `conda create -n admdyn python=3.12`
  creates a new isolated Conda environment called `admdyn` with Python 3.12
- `conda activate admdyn`
  activates that environment so later commands use it instead of your system Python
- `python -m pip install --upgrade pip setuptools wheel`
  updates the Python packaging tools inside the active environment
- `python -m pip install ...`
  installs the packages needed by ADMDynAnlz and by the generated analysis scripts

Optional verification:

```bash
python --version
python -c "import numpy, pandas, psutil, PIL, PySide6; print('ok')"
```

What these checks do:

- `python --version`
  confirms which Python version is active
- `python -c "import ..."`
  confirms that the required Python packages import successfully

When the environment is active, your shell prompt usually starts with `(admdyn)`.

To leave the environment later:

```bash
conda deactivate
```

What that does:

- returns your shell to the default environment

### Windows

The public README currently focuses on Conda-based setup for macOS and Linux, but the same general idea applies on Windows: use a dedicated Python environment before running generated analysis scripts.

If you already have Conda on Windows, you can use the same environment name and package set:

```powershell
conda create -n admdyn python=3.12
conda activate admdyn
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy pandas psutil Pillow PySide6 pyinstaller scipy matplotlib
```

The meaning of the commands is the same as above.

## Running Downloaded Executables

After preparing the Python environment, run the downloaded GUI executable that matches your system.

### Linux

If needed, make the executable runnable and launch it from a terminal:

```bash
chmod +x ADMDynAnlz_Linux
./ADMDynAnlz_Linux
```

What these commands do:

- `chmod +x ADMDynAnlz_Linux`
  adds the executable permission bit so Linux can launch the file directly
- `./ADMDynAnlz_Linux`
  launches the GUI from the current folder

On Linux, double-click launching can also work when the file is marked executable and the desktop environment allows launching executable files directly.

### macOS Apple Silicon

If you downloaded the Apple Silicon build, run:

```bash
chmod +x ADMDynAnlz_mac_arm64
xattr -d com.apple.quarantine ADMDynAnlz_mac_arm64 2>/dev/null || true
./ADMDynAnlz_mac_arm64
```

What these commands do:

- `chmod +x ADMDynAnlz_mac_arm64`
  makes the file runnable
- `xattr -d com.apple.quarantine ADMDynAnlz_mac_arm64 2>/dev/null || true`
  removes the macOS quarantine flag if it exists, which helps avoid Gatekeeper blocking a downloaded executable
- `./ADMDynAnlz_mac_arm64`
  launches the Apple Silicon build

### macOS Intel

If you downloaded the Intel build, run:

```bash
chmod +x ADMDynAnlz_mac_x86_64
xattr -d com.apple.quarantine ADMDynAnlz_mac_x86_64 2>/dev/null || true
./ADMDynAnlz_mac_x86_64
```

What these commands do:

- `chmod +x ADMDynAnlz_mac_x86_64`
  makes the file runnable
- `xattr -d com.apple.quarantine ADMDynAnlz_mac_x86_64 2>/dev/null || true`
  removes the quarantine flag if present
- `./ADMDynAnlz_mac_x86_64`
  launches the Intel build

### Windows

In many cases you can run the Windows executable directly.

If Windows marks the file as downloaded from the internet, open PowerShell in the download folder and run:

```powershell
Unblock-File .\ADMDynAnlz_Windows.exe
.\ADMDynAnlz_Windows.exe
```

What these commands do:

- `Unblock-File .\ADMDynAnlz_Windows.exe`
  removes the Windows download block marker if one is present
- `.\ADMDynAnlz_Windows.exe`
  launches the executable from the current directory

## Practical note about generated analysis scripts

The GUI executable launches the software interface. Later, when the GUI generates analysis scripts, those scripts rely on the Python environment you prepared earlier.

That is why environment setup is still recommended even when you are using a prebuilt GUI executable instead of running the source code directly.
