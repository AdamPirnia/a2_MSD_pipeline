from __future__ import annotations

import os
from glob import glob
from pathlib import Path

import numpy as np

NAMD_INTERNAL_TO_ANGSTROM_PER_PS = 20.45482706
ANGSTROM2_PER_PS2_PER_M2_PER_S2 = 1.0e-4
AVOGADRO = 6.02214076e23
BOLTZMANN = 1.380649e-23


def _resolve_vacf_files(vacf_path):
    path = Path(vacf_path)

    if path.is_file():
        return [path]

    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file())

    return [Path(p) for p in sorted(glob(str(vacf_path)))]


def _extract_vacf_series(file_path):
    data = np.loadtxt(file_path)
    data = np.asarray(data)

    if data.ndim == 0:
        return np.array([float(data)], dtype=np.float64)

    if data.ndim == 1:
        return data.astype(np.float64)

    return data[:, -1].astype(np.float64)


def _resolve_analysis_file_paths(base_directory, vacf_path, num_vacf=None):
    if base_directory:
        candidate = Path(vacf_path)
        if not candidate.is_absolute():
            vacf_path = str(Path(base_directory) / candidate)

    file_paths = _resolve_vacf_files(vacf_path)
    if not file_paths:
        raise FileNotFoundError(f"No VACF files found for: {vacf_path}")

    if num_vacf is not None:
        requested = int(num_vacf)
        if requested <= 0:
            raise ValueError("num_vacf must be a positive integer.")
        if len(file_paths) < requested:
            raise ValueError(
                f"Requested {requested} VACF files but found only {len(file_paths)} for: {vacf_path}"
            )
        file_paths = file_paths[:requested]

    return file_paths


def _velocity_scale_from_units(velocity_units):
    normalized = velocity_units.strip().lower().replace("-", "_").replace("/", "_")

    if normalized in {"namd", "namd_internal", "namd_internal_unit", "namd_internal_units"}:
        return NAMD_INTERNAL_TO_ANGSTROM_PER_PS

    if normalized in {"a_ps", "angstrom_ps", "angstrom_per_ps", "ang_per_ps"}:
        return 1.0

    raise ValueError("velocity_units must be 'namd_internal' or 'angstrom_per_ps'.")


def _equipartition_variance_ang2_per_ps2(temperature_k, molar_mass_g_mol):
    particle_mass_kg = (float(molar_mass_g_mol) / 1000.0) / AVOGADRO
    variance_si = 3.0 * BOLTZMANN * float(temperature_k) / particle_mass_kg
    return variance_si * ANGSTROM2_PER_PS2_PER_M2_PER_S2


def compute_diffusion_from_vacf_files(
    vacf_path,
    saved_frame_dt_ps,
    velocity_units,
    vacf_is_normalized,
    temperature_k,
    output_file=None,
    molar_mass_g_mol=18.01528,
    base_directory="",
    num_vacf=None,
):
    if saved_frame_dt_ps <= 0:
        raise ValueError("saved_frame_dt_ps must be positive.")
    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive.")
    if molar_mass_g_mol <= 0:
        raise ValueError("molar_mass_g_mol must be positive.")

    file_paths = _resolve_analysis_file_paths(
        base_directory=base_directory,
        vacf_path=vacf_path,
        num_vacf=num_vacf,
    )

    velocity_scale = _velocity_scale_from_units(velocity_units)

    per_file_results = []

    for file_path in file_paths:
        vacf = _extract_vacf_series(file_path)
        if vacf.size == 0:
            continue

        truncated = np.asarray(vacf, dtype=np.float64)
        if truncated.size == 0:
            continue

        if vacf_is_normalized:
            raw_for_integration = truncated.copy()
            normalized_for_integration = truncated.copy()
            vacf0_physical = np.nan
        else:
            raw_for_integration = truncated * velocity_scale ** 2
            vacf0_physical = float(raw_for_integration[0])
            if np.isclose(vacf0_physical, 0.0):
                raise ValueError(f"VACF[0] is zero in {file_path}; cannot normalize this file.")
            normalized_for_integration = raw_for_integration / vacf0_physical

        file_time_ps = np.arange(len(normalized_for_integration), dtype=np.float64) * float(saved_frame_dt_ps)
        tau_ps = float(np.trapz(normalized_for_integration, file_time_ps))
        equipartition_variance = _equipartition_variance_ang2_per_ps2(
            temperature_k=temperature_k,
            molar_mass_g_mol=molar_mass_g_mol,
        )

        if vacf_is_normalized:
            variance_for_diffusion = float(equipartition_variance)
            variance_source = "equipartition"
            variance_ratio = np.nan
            variance_percent_difference = np.nan
        else:
            variance_for_diffusion = vacf0_physical
            variance_source = "vacf0"
            variance_ratio = vacf0_physical / equipartition_variance
            variance_percent_difference = 100.0 * (
                vacf0_physical - equipartition_variance
            ) / equipartition_variance

        per_file_results.append(
            {
                "file": str(file_path),
                "tau_ps": float(tau_ps),
                "diffusion_constant_ang2_per_ps": float(variance_for_diffusion * tau_ps / 3.0),
                "input_vacf0_mean": float(raw_for_integration[0]),
                "vacf0_variance_ang2_per_ps2": float(vacf0_physical),
                "equipartition_variance_ang2_per_ps2": float(equipartition_variance),
                "variance_ratio_vacf0_over_equipartition": float(variance_ratio),
                "variance_percent_difference": float(variance_percent_difference),
                "variance_source_for_diffusion": variance_source,
            }
        )

    if not per_file_results:
        raise ValueError("Resolved VACF files were empty after loading/truncation.")

    equipartition_variance = float(
        _equipartition_variance_ang2_per_ps2(
            temperature_k=temperature_k,
            molar_mass_g_mol=molar_mass_g_mol,
        )
    )
    tau_ps = float(np.mean([item["tau_ps"] for item in per_file_results]))
    diffusion_constant_ang2_per_ps = float(
        np.mean([item["diffusion_constant_ang2_per_ps"] for item in per_file_results])
    )
    avg_vacf0_variance = float(np.nanmean([item["vacf0_variance_ang2_per_ps2"] for item in per_file_results]))
    variance_ratio = float(
        np.nanmean([item["variance_ratio_vacf0_over_equipartition"] for item in per_file_results])
    )
    variance_percent_difference = float(
        np.nanmean([item["variance_percent_difference"] for item in per_file_results])
    )
    variance_source = (
        "equipartition" if vacf_is_normalized else "vacf0"
    )
    if vacf_is_normalized:
        variance_note = (
            "Each VACF file was treated as already normalized. Tau and diffusion were computed per file "
            "using the equipartition variance, then averaged across files."
        )
    else:
        variance_note = (
            "Each VACF file was normalized by its own VACF[0] before integration. Tau and diffusion were "
            "computed per file, then averaged across files."
        )

    result = {
        "files_used": [item["file"] for item in per_file_results],
        "n_files": len(per_file_results),
        "base_directory": str(base_directory),
        "num_vacf_requested": None if num_vacf is None else int(num_vacf),
        "vacf_path": str(vacf_path),
        "saved_frame_dt_ps": float(saved_frame_dt_ps),
        "velocity_units": velocity_units,
        "velocity_scale_to_angstrom_per_ps": float(velocity_scale),
        "vacf_is_normalized": bool(vacf_is_normalized),
        "temperature_k": float(temperature_k),
        "molar_mass_g_mol": float(molar_mass_g_mol),
        "tau_ps": float(tau_ps),
        "diffusion_constant_ang2_per_ps": float(diffusion_constant_ang2_per_ps),
        "diffusion_constant_cm2_per_s": float(diffusion_constant_ang2_per_ps * 1.0e-4),
        "variance_source_for_diffusion": variance_source,
        "input_vacf0_mean": float(np.nanmean([item["input_vacf0_mean"] for item in per_file_results])),
        "vacf0_variance_ang2_per_ps2": float(avg_vacf0_variance),
        "equipartition_variance_ang2_per_ps2": float(equipartition_variance),
        "variance_ratio_vacf0_over_equipartition": float(variance_ratio),
        "variance_percent_difference": float(variance_percent_difference),
        "variance_note": variance_note,
        "per_file_results": per_file_results,
    }

    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# VACF-based diffusion analysis",
            "#",
            "# Inputs",
            f"# base_directory = {base_directory}",
            f"# source = {vacf_path}",
            f"# files_used = {len(per_file_results)}",
            f"# num_vacf_requested = {num_vacf}",
            f"# saved_frame_dt_ps = {float(saved_frame_dt_ps):.12e}",
            f"# velocity_units = {velocity_units}",
            f"# velocity_scale_to_angstrom_per_ps = {float(velocity_scale):.12e}",
            f"# vacf_is_normalized = {bool(vacf_is_normalized)}",
            f"# temperature_k = {float(temperature_k):.12e}",
            f"# molar_mass_g_mol = {float(molar_mass_g_mol):.12e}",
            "#",
            "# Variance comparison",
            f"# vacf0_variance_ang2_per_ps2 = {float(avg_vacf0_variance):.12e}",
            f"# equipartition_variance_ang2_per_ps2 = {float(equipartition_variance):.12e}",
            f"# variance_ratio_vacf0_over_equipartition = {float(variance_ratio):.12e}",
        ]
        if not np.isnan(variance_percent_difference):
            lines.append(f"# variance_percent_difference = {float(variance_percent_difference):.12e}")
        if len(per_file_results) == 1:
            item = per_file_results[0]
            lines.extend([
                "#",
                "Results:",
                f"file = {os.path.basename(item['file'])}",
                f"diffusion_constant_ang2_per_ps = {float(item['diffusion_constant_ang2_per_ps']):.12e}",
                f"diffusion_constant_cm2_per_s = {float(item['diffusion_constant_ang2_per_ps'] * 1.0e-4):.12e}",
                f"velocity_relaxation_time_ps = {float(item['tau_ps']):.12e}",
                f"vacf0_variance_ang2_per_ps2 = {float(item['vacf0_variance_ang2_per_ps2']):.12e}",
                f"equipartition_variance_ang2_per_ps2 = {float(item['equipartition_variance_ang2_per_ps2']):.12e}",
            ])
        else:
            lines.extend([
                "#",
                "Results per VACF file:",
            ])
            for index, item in enumerate(per_file_results):
                lines.append(
                    "file_{idx} = {file} tau_ps:{tau:.12e} D_ang2_per_ps:{diff:.12e} "
                    "vacf0:{vacf0:.12e} equipartition:{equip:.12e}".format(
                        idx=index,
                        file=os.path.basename(item["file"]),
                        tau=item["tau_ps"],
                        diff=item["diffusion_constant_ang2_per_ps"],
                        vacf0=item["vacf0_variance_ang2_per_ps2"],
                        equip=item["equipartition_variance_ang2_per_ps2"],
                    )
                )
            lines.extend([
                "",
                "Average results:",
                f"diffusion_constant_ang2_per_ps = {float(diffusion_constant_ang2_per_ps):.12e}",
                f"diffusion_constant_cm2_per_s = {float(diffusion_constant_ang2_per_ps * 1.0e-4):.12e}",
                f"velocity_relaxation_time_ps = {float(tau_ps):.12e}",
            ])
        output_path.write_text("\n".join(lines) + "\n")

    return result
