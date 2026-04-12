#!/usr/bin/env bash

set -euo pipefail

VENV_DIR="${VENV_DIR:-.admdyn-venv}"
PIP_PACKAGES=(
  numpy
  pandas
  psutil
  Pillow
  PySide6
  pyinstaller
  scipy
  matplotlib
)

print_header() {
  echo "========================================"
  echo "ADMDynAnlz Runtime Configurator"
  echo "========================================"
  echo
  echo "Select the executable you downloaded:"
  echo "1- LINUX"
  echo "2- MAC OS ARM"
  echo "3- MAC INTEL"
  echo "4- WINDOWS"
  echo
}

resolve_python() {
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  if command -v py >/dev/null 2>&1; then
    echo "py -3"
    return
  fi
  echo "No suitable Python interpreter was found." >&2
  exit 1
}

create_venv() {
  local python_cmd
  python_cmd="$(resolve_python)"

  echo "Creating Python environment in ${VENV_DIR} ..."
  if [[ "$python_cmd" == "py -3" ]]; then
    py -3 -m venv "$VENV_DIR"
  else
    "$python_cmd" -m venv "$VENV_DIR"
  fi
}

activate_venv() {
  if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # Linux/macOS
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    return
  fi
  if [[ -f "${VENV_DIR}/Scripts/activate" ]]; then
    # Git Bash / similar Windows shells
    # shellcheck disable=SC1091
    source "${VENV_DIR}/Scripts/activate"
    return
  fi
  echo "Could not find an activation script in ${VENV_DIR}." >&2
  exit 1
}

install_packages() {
  echo "Upgrading pip/setuptools/wheel ..."
  python -m pip install --upgrade pip setuptools wheel
  echo "Installing required Python packages ..."
  python -m pip install "${PIP_PACKAGES[@]}"
}

verify_packages() {
  echo "Verifying Python package installation ..."
  python --version
  python -c "import numpy, pandas, psutil, PIL, PySide6; print('ok')"
}

prepare_linux() {
  local exec_name="ADMDynAnlz_Linux"
  [[ -f "$exec_name" ]] || { echo "Executable not found: $exec_name" >&2; exit 1; }
  chmod +x "$exec_name"
  echo
  echo "Linux executable prepared."
  echo "To run the software:"
  echo "./$exec_name"
}

prepare_mac_arm() {
  local exec_name="ADMDynAnlz_mac_arm64"
  [[ -f "$exec_name" ]] || { echo "Executable not found: $exec_name" >&2; exit 1; }
  chmod +x "$exec_name"
  xattr -d com.apple.quarantine "$exec_name" 2>/dev/null || true
  echo
  echo "macOS Apple Silicon executable prepared."
  echo "To run the software:"
  echo "./$exec_name"
}

prepare_mac_intel() {
  local exec_name="ADMDynAnlz_mac_x86_64"
  [[ -f "$exec_name" ]] || { echo "Executable not found: $exec_name" >&2; exit 1; }
  chmod +x "$exec_name"
  xattr -d com.apple.quarantine "$exec_name" 2>/dev/null || true
  echo
  echo "macOS Intel executable prepared."
  echo "To run the software:"
  echo "./$exec_name"
}

prepare_windows() {
  local exec_name="ADMDynAnlz_Windows.exe"
  [[ -f "$exec_name" ]] || { echo "Executable not found: $exec_name" >&2; exit 1; }
  if command -v powershell.exe >/dev/null 2>&1; then
    powershell.exe -NoProfile -Command "Unblock-File '.\\${exec_name}'" >/dev/null 2>&1 || true
  elif command -v pwsh >/dev/null 2>&1; then
    pwsh -NoProfile -Command "Unblock-File '.\\${exec_name}'" >/dev/null 2>&1 || true
  fi
  echo
  echo "Windows executable prepared."
  echo "To run the software:"
  echo "./$exec_name"
  echo
  echo "If double-clicking is blocked by Windows, open PowerShell in this folder and run:"
  echo "Unblock-File .\\${exec_name}"
  echo ".\\${exec_name}"
}

main() {
  print_header
  read -r -p "Enter 1, 2, 3, or 4: " choice

  case "$choice" in
    1) platform_label="LINUX" ;;
    2) platform_label="MAC OS ARM" ;;
    3) platform_label="MAC INTEL" ;;
    4) platform_label="WINDOWS" ;;
    *)
      echo "Invalid selection." >&2
      exit 1
      ;;
  esac

  echo
  echo "Selected platform: ${platform_label}"
  echo

  create_venv
  activate_venv
  install_packages
  verify_packages

  case "$choice" in
    1) prepare_linux ;;
    2) prepare_mac_arm ;;
    3) prepare_mac_intel ;;
    4) prepare_windows ;;
  esac

  echo
  echo "Environment location: ${VENV_DIR}"
  echo "To activate it later:"
  if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    echo "source ${VENV_DIR}/bin/activate"
  else
    echo "source ${VENV_DIR}/Scripts/activate"
  fi
}

main "$@"
