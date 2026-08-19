from __future__ import annotations

import os
from math import pi
from glob import glob
from pathlib import Path

import numpy as np
try:
    from scipy.constants import Boltzmann as LIBRARY_BOLTZMANN
except Exception:  # pragma: no cover
    LIBRARY_BOLTZMANN = 1.380649e-23

NAMD_INTERNAL_TO_ANGSTROM_PER_PS = 20.45482706
ANGSTROM2_PER_PS2_PER_M2_PER_S2 = 1.0e-4
AVOGADRO = 6.02214076e23
BOLTZMANN = 1.380649e-23
# If |C(t_end)/C(0)| exceeds this, the VACF has not visibly decayed by the end
# of the supplied data, so trapezoidal integration to t_end underestimates tau.
VACF_TAIL_DECAY_WARNING_THRESHOLD = 0.01


def _resolve_vacf_files(vacf_path):
    path = Path(vacf_path)

    if path.is_file():
        return [path]

    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file())

    return [Path(p) for p in sorted(glob(str(vacf_path)))]


def _extract_vacf_series(file_path, time_axis_exists=False, stride=1):
    data = np.loadtxt(file_path)
    data = np.asarray(data)

    stride = int(stride)
    if stride <= 0:
        raise ValueError("VACF stride must be a positive integer.")

    if data.ndim == 0:
        series = np.array([float(data)], dtype=np.float64)
        return None, series

    if data.ndim == 1:
        if time_axis_exists:
            raise ValueError(f"VACF file {file_path} must contain at least two columns when 'Time Axis Exist' is enabled.")
        return None, data.astype(np.float64)[::stride]

    if time_axis_exists:
        return data[::stride, 0].astype(np.float64), data[::stride, 1].astype(np.float64)

    return None, data[::stride, -1].astype(np.float64)


def _extract_scalar_series(file_path):
    data = np.loadtxt(file_path)
    data = np.asarray(data)

    if data.ndim == 0:
        return np.array([float(data)], dtype=np.float64)

    if data.ndim == 1:
        return data.astype(np.float64)

    return data[:, -1].astype(np.float64)


def _extract_time_value_series(file_path, label, time_axis_exists=False, stride=1):
    data = np.loadtxt(file_path)
    data = np.asarray(data)

    stride = int(stride)
    if stride <= 0:
        raise ValueError(f"{label} stride must be a positive integer.")

    if data.ndim == 0:
        return None, np.array([float(data)], dtype=np.float64)

    if data.ndim == 1:
        if time_axis_exists:
            raise ValueError(f"{label} file {file_path} must contain at least two columns when 'Time Axis Exist' is enabled.")
        return None, data.astype(np.float64)[::stride]

    if time_axis_exists:
        return data[::stride, 0].astype(np.float64), data[::stride, 1].astype(np.float64)

    return None, data[::stride, -1].astype(np.float64)


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


def _resolve_single_analysis_file_path(base_directory, file_path, label):
    candidate = Path(file_path)
    if base_directory and not candidate.is_absolute():
        candidate = Path(base_directory) / candidate

    file_paths = _resolve_vacf_files(str(candidate))
    if not file_paths:
        raise FileNotFoundError(f"No {label} files found for: {candidate}")
    if len(file_paths) != 1:
        raise ValueError(f"Expected exactly one {label} file for {candidate}, found {len(file_paths)}.")
    return file_paths[0]


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


def apply_infinite_size_correction(
    raw_diffusion_constant_ang2_per_ps,
    temperature_k,
    viscosity_cp,
    cubic_box_length_angstrom,
):
    if viscosity_cp <= 0:
        raise ValueError("viscosity_cp must be positive.")
    if cubic_box_length_angstrom <= 0:
        raise ValueError("cubic_box_length_angstrom must be positive.")
    correction_term = (
        2.837297 * LIBRARY_BOLTZMANN * float(temperature_k) * 1.0e21
    ) / (6.0 * pi * float(viscosity_cp) * float(cubic_box_length_angstrom))
    corrected = float(raw_diffusion_constant_ang2_per_ps) + float(correction_term)
    return {
        "raw_diffusion_constant_ang2_per_ps": float(raw_diffusion_constant_ang2_per_ps),
        "correction_term_ang2_per_ps": float(correction_term),
        "corrected_diffusion_constant_ang2_per_ps": corrected,
        "corrected_diffusion_constant_cm2_per_s": float(corrected * 1.0e-4),
        "viscosity_cp": float(viscosity_cp),
        "cubic_box_length_angstrom": float(cubic_box_length_angstrom),
    }


def compute_diffusion_from_msd_file(
    msd_path,
    time_step_ps,
    base_directory="",
    time_axis_exists=False,
    time_axis_unit="ps",
    msd_stride=1,
    msd_fit_skip_frames=0,
):
    if not time_axis_exists and time_step_ps <= 0:
        raise ValueError("time_step_ps must be positive.")

    fit_skip_frames = int(msd_fit_skip_frames)
    if fit_skip_frames < 0:
        raise ValueError("msd_fit_skip_frames must be a non-negative integer.")

    file_path = _resolve_single_analysis_file_path(
        base_directory=base_directory,
        file_path=msd_path,
        label="MSD",
    )
    time_axis, msd_series = _extract_time_value_series(
        file_path,
        label="MSD",
        time_axis_exists=time_axis_exists,
        stride=msd_stride,
    )
    if msd_series.size == 0:
        raise ValueError(f"MSD file is empty: {file_path}")

    if time_axis is None:
        times_ps = np.arange(msd_series.size, dtype=np.float64) * float(time_step_ps)
    else:
        times_ps = np.asarray(time_axis, dtype=np.float64)

    if fit_skip_frames >= times_ps.size:
        raise ValueError(
            f"msd_fit_skip_frames ({fit_skip_frames}) skips all {times_ps.size} available "
            "MSD point(s); reduce it so at least one point remains for the fit."
        )

    fit_times_ps = times_ps[fit_skip_frames:]
    fit_msd_series = msd_series[fit_skip_frames:]
    denominator = float(np.dot(fit_times_ps, fit_times_ps))
    if np.isclose(denominator, 0.0):
        raise ValueError("MSD series must contain at least two time points for the linear fit.")

    slope_ang2_per_ps = float(np.dot(fit_times_ps, fit_msd_series) / denominator)
    diffusion_constant_ang2_per_ps = float(slope_ang2_per_ps / 6.0)
    return {
        "msd_file": str(file_path),
        "msd_time_step_ps": float(time_step_ps),
        "msd_time_axis_exists": bool(time_axis_exists),
        "msd_time_axis_unit": str(time_axis_unit),
        "msd_stride": int(msd_stride),
        "msd_fit_skip_frames": fit_skip_frames,
        "msd_fit_points_used": int(fit_msd_series.size),
        "msd_fit_slope_ang2_per_ps": slope_ang2_per_ps,
        "msd_diffusion_constant_ang2_per_ps": diffusion_constant_ang2_per_ps,
        "msd_diffusion_constant_cm2_per_s": float(diffusion_constant_ang2_per_ps * 1.0e-4),
    }


def compute_diffusion_from_vacf_files(
    vacf_path,
    saved_frame_dt_ps,
    velocity_units,
    vacf_is_normalized,
    vacf_variance_ang2_per_ps2,
    temperature_k,
    output_file=None,
    molar_mass_g_mol=18.01528,
    base_directory="",
    num_vacf=None,
    *,
    vacf_enabled=True,
    msd_enabled=False,
    msd_path=None,
    vacf_stride=1,
    msd_stride=1,
    msd_fit_skip_frames=0,
    msd_time_step_ps=None,
    msd_time_axis_exists=False,
    msd_time_axis_unit="ps",
    time_axis_exists=False,
    time_axis_unit="ps",
    apply_size_correction=False,
    correction_viscosity_cp=None,
    correction_cubic_box_length_angstrom=None,
):
    if not vacf_enabled and not msd_enabled:
        raise ValueError("At least one of VACF or MSD diffusion analysis must be enabled.")
    if vacf_enabled and not time_axis_exists and saved_frame_dt_ps <= 0:
        raise ValueError("saved_frame_dt_ps must be positive.")
    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive.")
    if molar_mass_g_mol <= 0:
        raise ValueError("molar_mass_g_mol must be positive.")
    if vacf_enabled and vacf_is_normalized and vacf_variance_ang2_per_ps2 <= 0:
        raise ValueError("vacf_variance_ang2_per_ps2 must be positive when vacf_is_normalized is True.")

    file_paths = []
    velocity_scale = 1.0
    per_file_results = []
    equipartition_variance = np.nan
    tau_ps = np.nan
    diffusion_constant_ang2_per_ps = np.nan
    avg_vacf0_variance = np.nan
    variance_percent_difference = np.nan
    variance_source = "vacf0"
    variance_note = ""

    if vacf_enabled:
        file_paths = _resolve_analysis_file_paths(
            base_directory=base_directory,
            vacf_path=vacf_path,
            num_vacf=num_vacf,
        )

        velocity_scale = 1.0 if vacf_is_normalized else _velocity_scale_from_units(velocity_units)

        for file_path in file_paths:
            file_time_ps, vacf = _extract_vacf_series(file_path, time_axis_exists=time_axis_exists, stride=vacf_stride)
            if vacf.size == 0:
                continue

            truncated = np.asarray(vacf, dtype=np.float64)
            if truncated.size == 0:
                continue

            if vacf_is_normalized:
                raw_for_integration = truncated.copy()
                normalized_for_integration = truncated.copy()
                vacf_variance = float(vacf_variance_ang2_per_ps2)
            else:
                raw_for_integration = truncated * velocity_scale ** 2
                vacf_variance = float(raw_for_integration[0])
                if np.isclose(vacf_variance, 0.0):
                    raise ValueError(f"VACF[0] is zero in {file_path}; cannot normalize this file.")
                normalized_for_integration = raw_for_integration / vacf_variance

            if file_time_ps is None:
                file_time_ps = np.arange(len(normalized_for_integration), dtype=np.float64) * float(saved_frame_dt_ps)
            else:
                file_time_ps = np.asarray(file_time_ps, dtype=np.float64)
            tau_ps = float(np.trapezoid(normalized_for_integration, file_time_ps))
            tail_ratio = float(abs(normalized_for_integration[-1]))
            if tail_ratio > VACF_TAIL_DECAY_WARNING_THRESHOLD:
                print(
                    f"⚠ VACF in {file_path} has not decayed by the end of the data: "
                    f"|C(t_end)/C(0)| = {tail_ratio:.4f} (threshold {VACF_TAIL_DECAY_WARNING_THRESHOLD}). "
                    "tau and D from this file may be underestimated; consider supplying a longer VACF."
                )
            equipartition_variance = _equipartition_variance_ang2_per_ps2(
                temperature_k=temperature_k,
                molar_mass_g_mol=molar_mass_g_mol,
            )

            if vacf_is_normalized:
                variance_for_diffusion = float(vacf_variance)
                variance_source = "user"
                variance_percent_difference = 100.0 * (
                    vacf_variance - equipartition_variance
                ) / equipartition_variance
            else:
                variance_for_diffusion = vacf_variance
                variance_source = "vacf0"
                variance_percent_difference = 100.0 * (
                    vacf_variance - equipartition_variance
                ) / equipartition_variance

            per_file_results.append(
                {
                    "file": str(file_path),
                    "tau_ps": float(tau_ps),
                    "diffusion_constant_ang2_per_ps": float(variance_for_diffusion * tau_ps / 3.0),
                    "input_vacf0_mean": float(raw_for_integration[0]),
                    "vacf_variance_ang2_per_ps2": float(vacf_variance),
                    "equipartition_variance_ang2_per_ps2": float(equipartition_variance),
                    "variance_percent_difference": float(variance_percent_difference),
                    "variance_source_for_diffusion": variance_source,
                    "vacf_tail_decay_ratio": tail_ratio,
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
        avg_vacf0_variance = float(np.nanmean([item["vacf_variance_ang2_per_ps2"] for item in per_file_results]))
        variance_percent_difference = float(
            np.nanmean([item["variance_percent_difference"] for item in per_file_results])
        )
        variance_source = "user" if vacf_is_normalized else "vacf0"
        if vacf_is_normalized:
            variance_note = (
                "Each VACF file was treated as already normalized. Tau and diffusion were computed per file "
                "using the user-provided variance, then averaged across files."
            )
        else:
            variance_note = (
                "Each VACF file was normalized by its own VACF[0] before integration. Tau and diffusion were "
                "computed per file, then averaged across files."
            )

    result = {
        "vacf_enabled": bool(vacf_enabled),
        "msd_enabled": bool(msd_enabled),
        "files_used": [item["file"] for item in per_file_results],
        "n_files": len(per_file_results),
        "base_directory": str(base_directory),
        "num_vacf_requested": None if num_vacf is None else int(num_vacf),
        "vacf_path": str(vacf_path),
        "saved_frame_dt_ps": float(saved_frame_dt_ps),
        "vacf_stride": int(vacf_stride),
        "time_axis_exists": bool(time_axis_exists),
        "time_axis_unit": str(time_axis_unit),
        "velocity_units": velocity_units,
        "velocity_scale_to_angstrom_per_ps": float(velocity_scale),
        "vacf_is_normalized": bool(vacf_is_normalized),
        "vacf_input_variance_ang2_per_ps2": float(vacf_variance_ang2_per_ps2),
        "temperature_k": float(temperature_k),
        "molar_mass_g_mol": float(molar_mass_g_mol),
        "tau_ps": float(tau_ps),
        "diffusion_constant_ang2_per_ps": float(diffusion_constant_ang2_per_ps),
        "diffusion_constant_cm2_per_s": float(diffusion_constant_ang2_per_ps * 1.0e-4),
        "variance_source_for_diffusion": variance_source,
        "input_vacf0_mean": float(np.nanmean([item["input_vacf0_mean"] for item in per_file_results])) if per_file_results else np.nan,
        "vacf_variance_ang2_per_ps2": float(avg_vacf0_variance),
        "equipartition_variance_ang2_per_ps2": float(equipartition_variance),
        "variance_percent_difference": float(variance_percent_difference),
        "variance_note": variance_note,
        "per_file_results": per_file_results,
    }

    msd_result = None
    if msd_enabled and (msd_path is not None or msd_time_step_ps is not None):
        if not msd_path or msd_time_step_ps is None:
            raise ValueError("Both msd_path and msd_time_step_ps are required for MSD diffusion analysis.")
        msd_result = compute_diffusion_from_msd_file(
            msd_path=msd_path,
            time_step_ps=msd_time_step_ps,
            base_directory=base_directory,
            time_axis_exists=msd_time_axis_exists,
            time_axis_unit=msd_time_axis_unit,
            msd_stride=msd_stride,
            msd_fit_skip_frames=msd_fit_skip_frames,
        )
        result.update(msd_result)

    vacf_size_correction = None
    msd_size_correction = None
    if apply_size_correction:
        if correction_viscosity_cp is None or correction_cubic_box_length_angstrom is None:
            raise ValueError(
                "Infinite size correction requires both viscosity and averaged cubic side length."
            )
        if vacf_enabled:
            vacf_size_correction = apply_infinite_size_correction(
                raw_diffusion_constant_ang2_per_ps=diffusion_constant_ang2_per_ps,
                temperature_k=temperature_k,
                viscosity_cp=correction_viscosity_cp,
                cubic_box_length_angstrom=correction_cubic_box_length_angstrom,
            )
            result.update({
                "vacf_corrected_diffusion_constant_ang2_per_ps": vacf_size_correction["corrected_diffusion_constant_ang2_per_ps"],
                "vacf_corrected_diffusion_constant_cm2_per_s": vacf_size_correction["corrected_diffusion_constant_cm2_per_s"],
            })
        if msd_result is not None:
            msd_size_correction = apply_infinite_size_correction(
                raw_diffusion_constant_ang2_per_ps=msd_result["msd_diffusion_constant_ang2_per_ps"],
                temperature_k=temperature_k,
                viscosity_cp=correction_viscosity_cp,
                cubic_box_length_angstrom=correction_cubic_box_length_angstrom,
            )
            result.update({
                "msd_corrected_diffusion_constant_ang2_per_ps": msd_size_correction["corrected_diffusion_constant_ang2_per_ps"],
                "msd_corrected_diffusion_constant_cm2_per_s": msd_size_correction["corrected_diffusion_constant_cm2_per_s"],
            })
    result["apply_size_correction"] = bool(apply_size_correction)
    result["correction_viscosity_cp"] = None if correction_viscosity_cp is None else float(correction_viscosity_cp)
    result["correction_cubic_box_length_angstrom"] = (
        None if correction_cubic_box_length_angstrom is None else float(correction_cubic_box_length_angstrom)
    )

    if output_file is not None:
        output_path = Path(output_file)
        if base_directory and not output_path.is_absolute():
            output_path = Path(base_directory) / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        def _bool_text(value):
            return "yes" if value else "no"

        lines = [
            "Diffusion analysis",
            "",
            "Inputs:",
        ]
        if vacf_enabled:
            lines.extend([
                f"source = {vacf_path}",
                f"time axis exists = {_bool_text(bool(time_axis_exists))}",
                f"time axis unit = {time_axis_unit}",
                f"VACF timestep, ps = {float(saved_frame_dt_ps):.12e}",
                f"velocity units = {velocity_units}",
                f"velocity scale to angstrom/ps = {float(velocity_scale):.12e}",
                f"VACF already normalized = {_bool_text(bool(vacf_is_normalized))}",
            ])
        if msd_result is not None:
            lines.extend([
                f"MSD source = {msd_path}",
                f"MSD timestep, ps = {float(msd_time_step_ps):.12e}",
                f"MSD time axis exists = {_bool_text(bool(msd_time_axis_exists))}",
                f"MSD time axis unit = {msd_time_axis_unit}",
            ])
        lines.extend([
            f"D VACF enabled = {_bool_text(bool(vacf_enabled))}",
            f"D MSD enabled = {_bool_text(bool(msd_enabled))}",
            f"T, K = {float(temperature_k):.12e}",
            f"molar mass, g/mol = {float(molar_mass_g_mol):.12e}",
        ])
        if apply_size_correction:
            lines.extend([
                f"viscosity, cP = {float(correction_viscosity_cp):.12e}",
                f"averaged cubic side length, ang = {float(correction_cubic_box_length_angstrom):.12e}",
            ])
        lines.extend([
            "",
            "",
        ])
        if vacf_enabled:
            lines.append("Results:")
        if vacf_enabled and len(per_file_results) == 1:
            item = per_file_results[0]
            lines.extend([
                f"D VACF, ang^2/ps = {float(item['diffusion_constant_ang2_per_ps']):.12e}",
                f"D VACF, cm^2/s = {float(item['diffusion_constant_ang2_per_ps'] * 1.0e-4):.12e}",
                f"tau, ps = {float(item['tau_ps']):.12e}",
            ])
            if not vacf_is_normalized:
                lines.append(f"VACF var, ang^2/ps^2 = {float(item['vacf_variance_ang2_per_ps2']):.12e}")
            lines.append(f"equipartition var, ang^2/ps^2 = {float(item['equipartition_variance_ang2_per_ps2']):.12e}")
            if not np.isnan(item["variance_percent_difference"]):
                lines.append(f"variance difference, % = {float(item['variance_percent_difference']):.12e}")
            if msd_result is not None:
                lines.extend([
                    f"D MSD, ang^2/ps = {float(msd_result['msd_diffusion_constant_ang2_per_ps']):.12e}",
                    f"D MSD, cm^2/s = {float(msd_result['msd_diffusion_constant_cm2_per_s']):.12e}",
                ])
        elif vacf_enabled:
            lines.extend([
                "Results per VACF file:",
            ])
            for index, item in enumerate(per_file_results):
                line = (
                    "{idx}. {file} | tau, ps: {tau:.12e} | D VACF, ang^2/ps: {diff:.12e} | "
                    "equipartition var, ang^2/ps^2: {equip:.12e}"
                ).format(
                    idx=index,
                    file=os.path.basename(item["file"]),
                    tau=item["tau_ps"],
                    diff=item["diffusion_constant_ang2_per_ps"],
                    equip=item["equipartition_variance_ang2_per_ps2"],
                )
                if not vacf_is_normalized:
                    line += " | VACF var, ang^2/ps^2: {vacf0:.12e}".format(
                        vacf0=item["vacf_variance_ang2_per_ps2"],
                    )
                lines.append(line)
            lines.extend([
                "",
                "Average results:",
                f"D VACF, ang^2/ps = {float(diffusion_constant_ang2_per_ps):.12e}",
                f"D VACF, cm^2/s = {float(diffusion_constant_ang2_per_ps * 1.0e-4):.12e}",
                f"tau, ps = {float(tau_ps):.12e}",
            ])
            if not vacf_is_normalized:
                lines.append(f"VACF var, ang^2/ps^2 = {float(avg_vacf0_variance):.12e}")
            lines.append(f"equipartition var, ang^2/ps^2 = {float(equipartition_variance):.12e}")
            if not np.isnan(variance_percent_difference):
                lines.append(f"variance difference, % = {float(variance_percent_difference):.12e}")
            if msd_result is not None:
                lines.extend([
                    f"D MSD, ang^2/ps = {float(msd_result['msd_diffusion_constant_ang2_per_ps']):.12e}",
                    f"D MSD, cm^2/s = {float(msd_result['msd_diffusion_constant_cm2_per_s']):.12e}",
                ])
        elif msd_result is not None:
            lines.extend([
                "Results:",
                f"D MSD, ang^2/ps = {float(msd_result['msd_diffusion_constant_ang2_per_ps']):.12e}",
                f"D MSD, cm^2/s = {float(msd_result['msd_diffusion_constant_cm2_per_s']):.12e}",
            ])
        if apply_size_correction:
            lines.extend([
                "",
                "D with Infinite size correction",
            ])
            if vacf_size_correction is not None:
                lines.append(
                    f"D VACF = {float(vacf_size_correction['corrected_diffusion_constant_ang2_per_ps']):.12e}"
                )
            if msd_size_correction is not None:
                lines.append(
                    f"D MSD = {float(msd_size_correction['corrected_diffusion_constant_ang2_per_ps']):.12e}"
                )
        output_path.write_text("\n".join(lines) + "\n")

    return result
