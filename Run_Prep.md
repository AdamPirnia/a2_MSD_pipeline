# Preparation and Running Software

## Requirements

- Python 3.10 or newer
- Python 3.12 recommended
- `numpy`
- VMD for trajectory-based extraction steps

For any `VMD Path` field, enter the actual VMD executable path that can be run directly by the software. Do not rely on shell aliases.

Platform notes:

- macOS: use the real executable from the VMD application bundle, not `startup.command.csh` and not an alias such as `vmd` defined only in your interactive shell
- macOS: inspect the bundle with `ls -l "/Applications/VMD ... .app/Contents/vmd"` and use the actual binary you find there, such as `vmd_MACOSXARM64`
- macOS: if a trajectory-extraction step creates zero-byte `.rawf32` / `.shape` files, do not use `startup.command.csh`; use the actual VMD executable inside the app bundle
- Linux: use the real VMD executable path, or a real executable/symlink discoverable by `which -a vmd`
- Linux: if `vmd` works in your terminal, run `which -a vmd` and enter that executable path in the software
- Windows: use the real `vmd.exe` path, not a desktop shortcut or a shell-only wrapper command
- Windows: locate the installed `vmd.exe`, typically under `C:\Program Files\VMD\...`, and enter that full path

Generic examples:

- macOS: `/Applications/VMD ... .app/Contents/vmd/vmd_MACOSXARM64`
- Linux: `/usr/local/bin/vmd`
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

- `https://github.com/AdamPirnia/ADMDynAnlz`

Releases page:

- `https://github.com/AdamPirnia/ADMDynAnlz/releases`

### Web browser

To download the repository source code in a browser:

1. Open `https://github.com/AdamPirnia/ADMDynAnlz`
2. Click `Code`
3. Choose either:
   - `Download ZIP` to download the repository as an archive
   - the repository URL if you only want to copy the clone link for later use

To download a prebuilt executable in a browser:

1. Open `https://github.com/AdamPirnia/ADMDynAnlz/releases`
2. Open the latest release, or another release if you need a specific version
3. In the `Assets` section, download the file for your operating system

### Command line

To download the repository source code with Git:

```bash
git clone https://github.com/AdamPirnia/ADMDynAnlz.git
cd ADMDynAnlz
```

If you prefer GitHub CLI, you can also clone with:

```bash
gh repo clone AdamPirnia/ADMDynAnlz
cd ADMDynAnlz
```

To download a release executable with GitHub CLI:

```bash
gh release download --repo AdamPirnia/ADMDynAnlz --pattern "ADMDynAnlz_Linux"
```

Replace `ADMDynAnlz_Linux` with the file you actually want:

- `ADMDynAnlz_mac_arm64`
- `ADMDynAnlz_mac_x86_64`
- `ADMDynAnlz_Windows.exe`

If you do not use GitHub CLI, you can download directly with `curl`. Replace `TAG_NAME` with the release tag you want, such as `v1.0.0`, and replace the filename with the correct asset name:

```bash
curl -LO https://github.com/AdamPirnia/ADMDynAnlz/releases/download/TAG_NAME/ADMDynAnlz_Linux
```

Equivalent `wget` example:

```bash
wget https://github.com/AdamPirnia/ADMDynAnlz/releases/download/TAG_NAME/ADMDynAnlz_Linux
```

## Setup and Running

The recommended way to prepare a downloaded executable is to use [configure.sh](configure.sh).

From the folder that contains both `configure.sh` and your downloaded executable, run:

```bash
chmod +x configure.sh
./configure.sh
```

The configurator will:

- ask which executable you downloaded
- create a local Python environment
- install the Python packages needed by the generated analysis scripts
- prepare the downloaded executable for your platform
- print the command needed to launch the software

This chapter also explains how to prepare the runtime environment for a downloaded executable and how to launch the software on each supported platform.

The distributed executable is the GUI itself. The Python environment described here is still important because the generated analysis scripts use Python packages such as `numpy`, `pandas`, `psutil`, `Pillow`, `PySide6`, `scipy`, and `matplotlib`.

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

macOS note for generated analysis scripts:

- On macOS, first try `python3 your_generated_script.py`.
- If you get `ModuleNotFoundError` such as missing `numpy`, `scipy`, `pandas`, `PySide6`, or similar, run the script with `python your_generated_script.py` instead.
- If `python` works and `python3` does not, your active environment is attached to `python` but not to `python3`.

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
