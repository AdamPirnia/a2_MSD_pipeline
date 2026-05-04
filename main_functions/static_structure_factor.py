from __future__ import annotations

import ast
import os
import re
import sys
import time
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np

try:
    from .numeric_io import load_numeric_array, save_numeric_array
    from .path_utils import expand_path_pattern, validate_path_pattern
except ImportError:
    from numeric_io import load_numeric_array, save_numeric_array
    from path_utils import expand_path_pattern, validate_path_pattern


def _resolve_three_tier_k_parameters(
    k_max_primary: float,
    k_max_secondary: float | None,
    k_max_tertiary: float | None,
    k_resolution_primary: Any,
    k_resolution_secondary: Any | None,
    k_resolution_tertiary: Any | None,
) -> tuple[float, float, float, float, float, float]:
    k1 = float(k_max_primary)
    k2_raw = None if k_max_secondary is None else float(k_max_secondary)
    k3_raw = None if k_max_tertiary is None else float(k_max_tertiary)

    def _missing(value: float | None) -> bool:
        return value is None or value <= 0.0

    k2 = k1 if _missing(k2_raw) else float(k2_raw)
    k3 = k2 if _missing(k3_raw) else float(k3_raw)

    s1 = float(k_resolution_primary)
    s2_raw = None if k_resolution_secondary is None else float(k_resolution_secondary)
    s3_raw = None if k_resolution_tertiary is None else float(k_resolution_tertiary)
    s2 = s1 if s2_raw is None or s2_raw <= 0.0 else s2_raw
    s3 = s2 if s3_raw is None or s3_raw <= 0.0 else s3_raw

    return k1, k2, k3, s1, s2, s3


def isotropic_structure_factor_db_density(
    r_flat: np.ndarray,
    k_vals: np.ndarray,
    box: np.ndarray | None = None,
    cutoff: float | None = None,
    cell_size: float | None = None,
    frame_chunk: int = 10,
    i_chunk: int = 256,
    j_chunk: int = 256,
    k_chunk: int = 64,
    dtype: Any = np.float64,
    include_leading_one: bool = True,
    normalize_by_frames: bool = True,
    status_logger: _StatusLogger | None = None,
) -> np.ndarray:
    k_vals = np.asarray(k_vals, dtype=dtype)
    r_arr = np.asarray(r_flat, dtype=dtype)

    if r_arr.ndim == 2:
        if r_arr.shape[1] % 3 != 0:
            raise ValueError("Flattened coordinate arrays must contain 3 columns per atom.")
        natms = r_arr.shape[1] // 3
        r = r_arr.reshape(r_arr.shape[0], natms, 3)
    elif r_arr.ndim == 3:
        if r_arr.shape[2] != 3:
            raise ValueError("3D coordinate arrays must have shape (frames, atoms, 3).")
        natms = int(r_arr.shape[1])
        r = r_arr
    else:
        raise ValueError("Coordinate arrays must be 2D or 3D.")

    if natms <= 0:
        raise ValueError("Number of atoms must be positive.")

    if box is not None:
        box = np.asarray(box, dtype=dtype)
        if box.shape != (3,):
            raise ValueError("box must contain exactly three lengths: Lx, Ly, Lz.")
        if np.any(box <= 0):
            raise ValueError("All box lengths must be positive.")

    total_frames = int(r.shape[0])
    if total_frames <= 0:
        raise ValueError("Coordinate arrays must contain at least one frame.")
    cutoff, cell_size = _validate_cutoff_and_cell_size(cutoff, cell_size, label="Density structure factor")

    inv_pi = dtype(1.0 / np.pi)
    s_accum = np.zeros(k_vals.size, dtype=dtype)
    cutoff_sq = None if cutoff is None else float(cutoff) ** 2
    processed_frames = 0

    for f0 in range(0, total_frames, frame_chunk):
        f1 = min(total_frames, f0 + frame_chunk)
        r_chunk = np.ascontiguousarray(r[f0:f1])

        for r_frame in r_chunk:
            processed_frames += 1
            pair_sum_k = np.zeros(k_vals.size, dtype=dtype)

            if cutoff_sq is not None and box is not None:
                for dr_use in _iter_cutoff_displacement_blocks(
                    np.asarray(r_frame, dtype=np.float64),
                    np.asarray(box, dtype=np.float64),
                    float(cutoff),
                    cell_size,
                    int(i_chunk),
                    int(j_chunk),
                ):
                    dist = np.sqrt(np.sum(dr_use * dr_use, axis=1))
                    for k0 in range(0, k_vals.size, k_chunk):
                        k1 = min(k_vals.size, k0 + k_chunk)
                        x = np.outer(k_vals[k0:k1], dist)
                        pair_sum_k[k0:k1] += np.sinc(x * inv_pi).sum(axis=1)
                if include_leading_one:
                    s_accum += 1.0 + (2.0 / natms) * pair_sum_k
                else:
                    s_accum += (2.0 / natms) * pair_sum_k
                if status_logger is not None:
                    status_logger.density(processed_frames, natms, int(k_vals.size))
                continue

            for i0 in range(0, natms, i_chunk):
                i1 = min(natms, i0 + i_chunk)
                ri = r_frame[i0:i1]

                dr = ri[:, None, :] - ri[None, :, :]
                if box is not None:
                    dr -= box * np.rint(dr / box)
                dist = np.sqrt(np.sum(dr * dr, axis=2)).ravel()

                ni = i1 - i0
                a_idx, b_idx = np.triu_indices(ni, k=1)
                keep = a_idx * ni + b_idx
                if cutoff_sq is not None:
                    dist_sq = np.sum(dr * dr, axis=2).ravel()
                    keep = keep[dist_sq[keep] <= cutoff_sq]
                dist_use = dist[keep]

                for k0 in range(0, k_vals.size, k_chunk):
                    k1 = min(k_vals.size, k0 + k_chunk)
                    x = np.outer(k_vals[k0:k1], dist_use)
                    pair_sum_k[k0:k1] += np.sinc(x * inv_pi).sum(axis=1)

                for j0 in range(i1, natms, j_chunk):
                    j1 = min(natms, j0 + j_chunk)
                    rj = r_frame[j0:j1]
                    dr = ri[:, None, :] - rj[None, :, :]
                    if box is not None:
                        dr -= box * np.rint(dr / box)
                    dist_sq = np.sum(dr * dr, axis=2).ravel()
                    if cutoff_sq is not None:
                        dist_sq = dist_sq[dist_sq <= cutoff_sq]
                    dist = np.sqrt(dist_sq)

                    for k0 in range(0, k_vals.size, k_chunk):
                        k1 = min(k_vals.size, k0 + k_chunk)
                        x = np.outer(k_vals[k0:k1], dist)
                        pair_sum_k[k0:k1] += np.sinc(x * inv_pi).sum(axis=1)

            if include_leading_one:
                s_accum += 1.0 + (2.0 / natms) * pair_sum_k
            else:
                s_accum += (2.0 / natms) * pair_sum_k
            if status_logger is not None:
                status_logger.density(processed_frames, natms, int(k_vals.size))

    if status_logger is not None:
        status_logger.density(total_frames, natms, int(k_vals.size), force=True)
    if normalize_by_frames:
        return s_accum / float(total_frames)
    return s_accum


def isotropic_k_magnitudes_three_tier(
    Lx: float,
    Ly: float,
    Lz: float,
    kmax1: float,
    kmax2: float,
    kmax3: float,
    dtype: Any = np.float64,
) -> np.ndarray:
    kmax1, kmax2, kmax3 = float(kmax1), float(kmax2), float(kmax3)
    if kmax1 < 0 or kmax2 <= 0 or kmax3 <= 0:
        raise ValueError("kmax values must satisfy kmax1 >= 0, kmax2 > 0, and kmax3 > 0.")
    if kmax1 > kmax2 or kmax2 > kmax3:
        raise ValueError("kmax values must satisfy kmax1 <= kmax2 <= kmax3.")

    two_pi = np.array(2.0 * np.pi, dtype=dtype)
    nxmax = int(np.ceil(kmax3 * Lx / two_pi))
    nymax = int(np.ceil(kmax3 * Ly / two_pi))
    nzmax = int(np.ceil(kmax3 * Lz / two_pi))

    kmags: list[float] = []
    for nx in range(1, nxmax + 1):
        for ny in range(1, nymax + 1):
            for nz in range(1, nzmax + 1):
                kvec = two_pi * np.array([nx / Lx, ny / Ly, nz / Lz], dtype=dtype)
                kmag = float(np.linalg.norm(kvec))
                if 0.0 < kmag <= float(kmax3):
                    kmags.append(kmag)

    if not kmags:
        return np.empty(0, dtype=dtype)

    k_mags = np.asarray(kmags, dtype=dtype)
    k_mags = k_mags[np.argsort(k_mags)]
    return k_mags.astype(dtype, copy=False)


def directional_k_vectors_three_tier(
    Lx: float,
    Ly: float,
    Lz: float,
    kmax1: float,
    kmax2: float,
    kmax3: float,
    active_axes: tuple[str, ...] | None = None,
    dtype: Any = np.float64,
) -> np.ndarray:
    kmax1, kmax2, kmax3 = float(kmax1), float(kmax2), float(kmax3)
    if kmax1 < 0 or kmax2 <= 0 or kmax3 <= 0:
        raise ValueError("kmax values must satisfy kmax1 >= 0, kmax2 > 0, and kmax3 > 0.")
    if kmax1 > kmax2 or kmax2 > kmax3:
        raise ValueError("kmax values must satisfy kmax1 <= kmax2 <= kmax3.")

    axis_order = ("x", "y", "z")
    active = set(active_axes or axis_order)
    length_map = {"x": float(Lx), "y": float(Ly), "z": float(Lz)}
    limits = {
        axis: int(np.ceil(kmax3 * length_map[axis] / (2.0 * np.pi)))
        for axis in axis_order
    }

    vectors: list[tuple[float, float, float]] = []
    for nx in range(0, limits["x"] + 1 if "x" in active else 1):
        for ny in range(0, limits["y"] + 1 if "y" in active else 1):
            for nz in range(0, limits["z"] + 1 if "z" in active else 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                kvec = 2.0 * np.pi * np.array(
                    [nx / float(Lx), ny / float(Ly), nz / float(Lz)],
                    dtype=dtype,
                )
                mag = float(np.linalg.norm(kvec))
                if 0.0 < mag <= float(kmax3):
                    vectors.append((float(kvec[0]), float(kvec[1]), float(kvec[2])))

    if not vectors:
        return np.empty((0, 3), dtype=dtype)

    k_vectors_array = np.asarray(vectors, dtype=dtype)
    magnitudes = np.linalg.norm(k_vectors_array, axis=1)
    sort_order = np.lexsort(
        (
            k_vectors_array[:, 2],
            k_vectors_array[:, 1],
            k_vectors_array[:, 0],
            magnitudes,
        )
    )
    k_vectors_array = k_vectors_array[sort_order]
    return k_vectors_array.astype(dtype, copy=False)


def isotropic_k_magnitudes(Lx: float, Ly: float, Lz: float, kmax: float, dtype: Any = np.float64) -> np.ndarray:
    return isotropic_k_magnitudes_three_tier(Lx, Ly, Lz, 0.0, kmax, kmax, dtype=dtype)


def directional_k_vectors(
    Lx: float,
    Ly: float,
    Lz: float,
    kmax: float,
    active_axes: tuple[str, ...] | None = None,
    dtype: Any = np.float64,
) -> np.ndarray:
    return directional_k_vectors_three_tier(
        Lx,
        Ly,
        Lz,
        0.0,
        kmax,
        kmax,
        active_axes=active_axes,
        dtype=dtype,
    )


def directional_structure_factor(
    r_flat: np.ndarray,
    k_vectors_array: np.ndarray,
    box: np.ndarray | None = None,
    cutoff: float | None = None,
    cell_size: float | None = None,
    frame_chunk: int = 10,
    atom_chunk: int = 256,
    k_chunk: int = 64,
    dtype: Any = np.float64,
    normalize_by_frames: bool = True,
    status_logger: _StatusLogger | None = None,
) -> np.ndarray:
    k_vectors_array = np.asarray(k_vectors_array, dtype=dtype)
    r_arr = np.asarray(r_flat, dtype=dtype)

    if r_arr.ndim == 2:
        if r_arr.shape[1] % 3 != 0:
            raise ValueError("Flattened coordinate arrays must contain 3 columns per atom.")
        natms = r_arr.shape[1] // 3
        r = r_arr.reshape(r_arr.shape[0], natms, 3)
    elif r_arr.ndim == 3:
        if r_arr.shape[2] != 3:
            raise ValueError("3D coordinate arrays must have shape (frames, atoms, 3).")
        natms = int(r_arr.shape[1])
        r = r_arr
    else:
        raise ValueError("Coordinate arrays must be 2D or 3D.")

    if natms <= 0:
        raise ValueError("Number of atoms must be positive.")
    if k_vectors_array.ndim != 2 or k_vectors_array.shape[1] != 3:
        raise ValueError("k_vectors_array must have shape (n_k, 3).")
    if k_vectors_array.shape[0] <= 0:
        raise ValueError("At least one k vector is required.")

    if box is not None:
        box = np.asarray(box, dtype=dtype)
        if box.shape != (3,):
            raise ValueError("box must contain exactly three lengths: Lx, Ly, Lz.")
        if np.any(box <= 0):
            raise ValueError("All box lengths must be positive.")

    total_frames = int(r.shape[0])
    if total_frames <= 0:
        raise ValueError("Coordinate arrays must contain at least one frame.")
    cutoff, cell_size = _validate_cutoff_and_cell_size(cutoff, cell_size, label="Density structure factor")

    s_accum = np.zeros(k_vectors_array.shape[0], dtype=np.float64)
    cutoff_sq = None if cutoff is None else float(cutoff) ** 2
    processed_frames = 0

    for f0 in range(0, total_frames, frame_chunk):
        f1 = min(total_frames, f0 + frame_chunk)
        r_chunk = np.ascontiguousarray(r[f0:f1])

        for r_frame in r_chunk:
            processed_frames += 1
            if box is not None:
                r_frame_use = np.mod(r_frame, box)
            else:
                r_frame_use = r_frame

            if cutoff_sq is None:
                for k0 in range(0, k_vectors_array.shape[0], k_chunk):
                    k1 = min(k_vectors_array.shape[0], k0 + k_chunk)
                    rho = np.zeros(k1 - k0, dtype=np.complex128)
                    for a0 in range(0, natms, atom_chunk):
                        a1 = min(natms, a0 + atom_chunk)
                        phase = np.matmul(r_frame_use[a0:a1], k_vectors_array[k0:k1].T)
                        rho += np.exp(1j * phase).sum(axis=0)
                    s_accum[k0:k1] += (rho * np.conjugate(rho)).real / float(natms)
                continue

            if box is not None:
                pair_sum_k = np.zeros(k_vectors_array.shape[0], dtype=np.float64)
                for dr_use in _iter_cutoff_displacement_blocks(
                    np.asarray(r_frame_use, dtype=np.float64),
                    np.asarray(box, dtype=np.float64),
                    float(cutoff),
                    cell_size,
                    int(atom_chunk),
                    int(atom_chunk),
                ):
                    for k0 in range(0, k_vectors_array.shape[0], k_chunk):
                        k1 = min(k_vectors_array.shape[0], k0 + k_chunk)
                        phase = np.matmul(dr_use, k_vectors_array[k0:k1].T)
                        pair_sum_k[k0:k1] += np.cos(phase).sum(axis=0)

                s_accum += 1.0 + (2.0 / float(natms)) * pair_sum_k
                if status_logger is not None:
                    status_logger.density(processed_frames, natms, int(k_vectors_array.shape[0]))
                continue

            pair_sum_k = np.zeros(k_vectors_array.shape[0], dtype=np.float64)
            for i0 in range(0, natms, atom_chunk):
                i1 = min(natms, i0 + atom_chunk)
                ri = r_frame_use[i0:i1]

                dr = ri[:, None, :] - ri[None, :, :]
                if box is not None:
                    dr -= box * np.rint(dr / box)
                dist_sq = np.sum(dr * dr, axis=2).ravel()

                ni = i1 - i0
                a_idx, b_idx = np.triu_indices(ni, k=1)
                keep = a_idx * ni + b_idx
                if cutoff_sq is not None:
                    keep = keep[dist_sq[keep] <= cutoff_sq]
                dr_use = dr.reshape(-1, 3)[keep]

                for k0 in range(0, k_vectors_array.shape[0], k_chunk):
                    k1 = min(k_vectors_array.shape[0], k0 + k_chunk)
                    if dr_use.size == 0:
                        continue
                    phase = np.matmul(dr_use, k_vectors_array[k0:k1].T)
                    pair_sum_k[k0:k1] += np.cos(phase).sum(axis=0)

                for j0 in range(i1, natms, atom_chunk):
                    j1 = min(natms, j0 + atom_chunk)
                    rj = r_frame_use[j0:j1]
                    dr = ri[:, None, :] - rj[None, :, :]
                    if box is not None:
                        dr -= box * np.rint(dr / box)
                    dist_sq = np.sum(dr * dr, axis=2).reshape(-1)
                    keep = dist_sq <= cutoff_sq
                    dr_use = dr.reshape(-1, 3)[keep]

                    for k0 in range(0, k_vectors_array.shape[0], k_chunk):
                        k1 = min(k_vectors_array.shape[0], k0 + k_chunk)
                        if dr_use.size == 0:
                            continue
                        phase = np.matmul(dr_use, k_vectors_array[k0:k1].T)
                        pair_sum_k[k0:k1] += np.cos(phase).sum(axis=0)

            s_accum += 1.0 + (2.0 / float(natms)) * pair_sum_k
            if status_logger is not None:
                status_logger.density(processed_frames, natms, int(k_vectors_array.shape[0]))

    if status_logger is not None:
        status_logger.density(total_frames, natms, int(k_vectors_array.shape[0]), force=True)
    if normalize_by_frames:
        return s_accum / float(total_frames)
    return s_accum


def parse_k_component_selection(selection: str) -> list[tuple[str, ...]]:
    raw = str(selection or "").strip().replace(";", ",")
    if not raw:
        raise ValueError("At least one k-component selection is required.")

    parsed: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    axis_order = {"x": 0, "y": 1, "z": 2}
    for item in [part.strip().lower() for part in raw.split(",") if part.strip()]:
        compact = item.replace(" ", "")
        if not compact:
            continue
        parts = compact.split("+")
        if any(part not in axis_order for part in parts):
            raise ValueError(
                "Unsupported k-component selection. Use combinations of x, y, and z such as x, y, z, x+y, or x+y+z."
            )
        axes = tuple(sorted(set(parts), key=lambda axis: axis_order[axis]))
        if not axes:
            continue
        if axes not in seen:
            seen.add(axes)
            parsed.append(axes)

    if not parsed:
        raise ValueError("At least one valid k-component selection is required.")
    return parsed


def _eval_trajectory_selection_expr(selection_str: str) -> list[int]:
    allowed_nodes = (
        ast.Expression,
        ast.List,
        ast.Tuple,
        ast.Set,
        ast.ListComp,
        ast.SetComp,
        ast.GeneratorExp,
        ast.comprehension,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.UnaryOp,
        ast.USub,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.FloorDiv,
        ast.Mod,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.IfExp,
    )

    tree = ast.parse(selection_str, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsupported expression element: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id != "range":
                raise ValueError("Only range() function calls are allowed")

    value = eval(compile(tree, "<trajectory_selection>", "eval"), {"__builtins__": {}}, {"range": range})
    if isinstance(value, range):
        return list(value)
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]
    return [int(item) for item in value]


def parse_trajectory_selection(selection_str: str, num_trajectories: int) -> list[int]:
    if not str(selection_str or "").strip():
        return list(range(num_trajectories))

    trajectory_indices: list[int] = []
    selection_str = str(selection_str).strip()

    try:
        if selection_str.startswith(("[", "(", "{")) or "for" in selection_str:
            trajectory_indices = _eval_trajectory_selection_expr(selection_str)
            for idx in trajectory_indices:
                if idx < 0 or idx >= num_trajectories:
                    raise ValueError(f"Trajectory index {idx} is outside valid range (0-{num_trajectories-1})")
            trajectory_indices = sorted(set(int(i) for i in trajectory_indices))
            if not trajectory_indices:
                raise ValueError("No valid trajectory indices found")
            return trajectory_indices

        if selection_str.startswith("range(") and selection_str.endswith(")"):
            range_content = selection_str[6:-1]
            parts = [p.strip() for p in range_content.split(",")]
            if len(parts) == 1:
                trajectory_indices = list(range(int(parts[0])))
            elif len(parts) == 2:
                trajectory_indices = list(range(int(parts[0]), int(parts[1])))
            elif len(parts) == 3:
                trajectory_indices = list(range(int(parts[0]), int(parts[1]), int(parts[2])))
            else:
                raise ValueError(f"Invalid range format: {selection_str}")
            for idx in trajectory_indices:
                if idx < 0 or idx >= num_trajectories:
                    raise ValueError(
                        f"Range {selection_str} contains index {idx} outside valid trajectory range (0-{num_trajectories-1})"
                    )
        else:
            ranges = [r.strip() for r in selection_str.split(",")]
            for range_str in ranges:
                if "-" in range_str:
                    start_text, end_text = range_str.split("-", 1)
                    start = int(start_text.strip())
                    end = int(end_text.strip())
                    if start > end:
                        raise ValueError(f"Invalid range: {range_str} (start > end)")
                    if start < 0 or end >= num_trajectories:
                        raise ValueError(f"Range {range_str} is outside valid trajectory range (0-{num_trajectories-1})")
                    trajectory_indices.extend(range(start, end + 1))
                else:
                    index = int(range_str.strip())
                    if index < 0 or index >= num_trajectories:
                        raise ValueError(f"Trajectory index {index} is outside valid range (0-{num_trajectories-1})")
                    trajectory_indices.append(index)

        trajectory_indices = sorted(set(trajectory_indices))
        if not trajectory_indices:
            raise ValueError("No valid trajectory indices found")
        return trajectory_indices
    except ValueError as exc:
        if "invalid literal for int()" in str(exc):
            raise ValueError(
                f"Invalid trajectory selection format: '{selection_str}'. Use formats like '4-10', "
                "'4-6,8-10', 'range(10,20)', or a Python expression"
            ) from exc
        raise


def _selected_trajectory_indices(total_count: int, num_trajectories: int | None, trajectory_selection: str | None) -> list[int]:
    if num_trajectories is not None and int(num_trajectories) <= 0:
        raise ValueError("num_trajectories must be positive when provided.")
    available_count = min(total_count, int(num_trajectories)) if num_trajectories is not None else total_count
    return parse_trajectory_selection(trajectory_selection or "", available_count)


def component_label(axes: tuple[str, ...]) -> str:
    return "k" + "".join(axes)


def _aggregate_by_tolerance(
    k_magnitudes: np.ndarray,
    values: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_magnitudes = np.asarray(k_magnitudes, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if k_magnitudes.size != values.size:
        raise ValueError("k magnitudes and values must have the same size.")
    if k_magnitudes.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    order = np.argsort(k_magnitudes)
    k_sorted = k_magnitudes[order]
    v_sorted = values[order]

    grouped_k: list[float] = []
    grouped_v: list[float] = []
    start = 0
    last = k_sorted[0]
    for index in range(1, k_sorted.size):
        if k_sorted[index] - last > tolerance:
            grouped_k.append(float(np.mean(k_sorted[start:index])))
            grouped_v.append(float(np.mean(v_sorted[start:index])))
            start = index
        last = k_sorted[index]
    grouped_k.append(float(np.mean(k_sorted[start:])))
    grouped_v.append(float(np.mean(v_sorted[start:])))
    return np.asarray(grouped_k, dtype=np.float64), np.asarray(grouped_v, dtype=np.float64)


def unique_with_tolerance(x: np.ndarray, eps: float, keep: str = "first") -> np.ndarray:
    values = np.sort(np.asarray(x, dtype=np.float64))
    if values.size == 0:
        return values

    keep_mask = np.empty(values.size, dtype=bool)
    keep_mask[0] = True
    last = values[0]

    for index in range(1, values.size):
        if values[index] - last > eps:
            keep_mask[index] = True
            last = values[index]
        else:
            keep_mask[index] = False

    output = values[keep_mask]
    if keep == "last":
        return unique_with_tolerance(values[::-1], eps, keep="first")[::-1]
    return output


def _resolve_data_path(base_directory: str, pattern: str, file_index: int | None = None) -> str:
    expanded = expand_path_pattern(pattern, "", file_index)
    if base_directory and not os.path.isabs(expanded):
        return os.path.join(base_directory, expanded)
    return expanded


def _resolve_output_path(base_directory: str, output_path: str) -> str:
    if base_directory and output_path and not os.path.isabs(output_path):
        return os.path.join(base_directory, output_path)
    return output_path


def _status_log_path_for_output(output_path: str) -> str:
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, "status.log")
    return "status.log"


def _dipoles_count_directory_for_output(output_path: str) -> str:
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
        counts_directory = os.path.join(directory, "dipole_counts")
    else:
        counts_directory = "dipole_counts"
    os.makedirs(counts_directory, exist_ok=True)
    return counts_directory


def _single_trajectory_report_path(output_path: str, index: int) -> str:
    root, _ext = os.path.splitext(output_path)
    return f"{root}_{index}.dat"


def _dipoles_count_file_for_trajectory(output_path: str, trajectory_index: int) -> str:
    return os.path.join(_dipoles_count_directory_for_output(output_path), f"dipole_count_{int(trajectory_index)}.dat")


def _write_trajectory_dipoles_count_file(
    output_path: str,
    trajectory_index: int,
    counts_by_cutoff: np.ndarray,
    cutoff_primary: float | None,
    cutoff_secondary: float | None,
    cutoff_tertiary: float | None,
) -> str:
    counts = np.asarray(counts_by_cutoff, dtype=np.int64)
    if counts.ndim != 2 or counts.shape[1] != 3:
        raise ValueError("Dipoles count output requires a frame-by-cutoff array with exactly 3 cutoff columns.")
    counts_path = _dipoles_count_file_for_trajectory(output_path, trajectory_index)
    with open(counts_path, "w", encoding="utf-8") as handle:
        handle.write(
            "# Number of dipoles within each user-set cutoff distance from the charge sites for this trajectory.\n"
        )
        handle.write(f"# Trajectory index: {int(trajectory_index)}\n")
        handle.write(f"# CO-1 cutoff distance: {_format_header_value(cutoff_primary)}\n")
        handle.write(f"# CO-2 cutoff distance: {_format_header_value(cutoff_secondary)}\n")
        handle.write(f"# CO-3 cutoff distance: {_format_header_value(cutoff_tertiary)}\n")
        handle.write(f"{'Frame':>12} {'CO-1':>12} {'CO-2':>12} {'CO-3':>12}\n")
        for offset, row in enumerate(counts, start=1):
            handle.write(f"{offset:12d} {int(row[0]):12d} {int(row[1]):12d} {int(row[2]):12d}\n")
    return counts_path


def _format_elapsed_hms(elapsed_s: float) -> str:
    total_seconds = max(0, int(round(float(elapsed_s))))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class _StatusLogger:
    def __init__(
        self,
        path: str,
        start_epoch: float,
        interval_s: float = 600.0,
        trajectory_index: int | None = None,
    ) -> None:
        self.path = path
        self.start_epoch = float(start_epoch)
        self.interval_s = float(interval_s)
        self.last_emit_epoch = float(start_epoch)
        self.trajectory_index = None if trajectory_index is None else int(trajectory_index)
        self.enabled = True
        self._warned = False
        directory = os.path.dirname(path)
        try:
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.path, "a", encoding="utf-8"):
                pass
        except OSError as exc:
            self.enabled = False
            self._warn_once(exc)

    def _warn_once(self, exc: OSError) -> None:
        if self._warned:
            return
        self._warned = True
        print(
            f"Warning: disabling status logging after I/O failure for {self.path}: {exc}",
            file=sys.stderr,
            flush=True,
        )

    def _write(self, line: str) -> None:
        if not self.enabled:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError as exc:
            self.enabled = False
            self._warn_once(exc)

    def _prefix(self) -> str:
        if self.trajectory_index is None:
            return ""
        return f"Traj.: {self.trajectory_index}, "

    def trajectory_total_frames(self, total_frames: int) -> None:
        if not self.enabled:
            return
        self._write(f"{self._prefix()}Total Num. Frame: {int(total_frames)}\n")

    def density(self, num_frames: int, coords_count: int, k_count: int, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.time()
        if not force and now - self.last_emit_epoch < self.interval_s:
            return
        elapsed = _format_elapsed_hms(now - self.start_epoch)
        self._write(
            f"{self._prefix()}S(k): density: "
            f"Time: {elapsed}, "
            f"Num. Frame: {int(num_frames)}, "
            f"Num. Coord: {int(coords_count)}, "
            f"Num. k vec.: {int(k_count)}\n"
        )
        self.last_emit_epoch = now

    def charge_dipole(
        self,
        num_frames: int,
        charges_count: int,
        k_count: int,
        dipoles_count: int,
        *,
        force: bool = False,
    ) -> None:
        if not self.enabled:
            return
        now = time.time()
        if not force and now - self.last_emit_epoch < self.interval_s:
            return
        elapsed = _format_elapsed_hms(now - self.start_epoch)
        self._write(
            f"{self._prefix()}Sqp(k): "
            f"Time: {elapsed}, "
            f"Num. Frame: {int(num_frames)}, "
            f"Num. charge: {int(charges_count)}, "
            f"Num. k vec.: {int(k_count)}, "
            f"Num. dipoles: {int(dipoles_count)}\n"
        )
        self.last_emit_epoch = now


def _array_to_single_line(values: np.ndarray) -> str:
    array = np.asarray(values)
    return np.array2string(
        array,
        precision=15,
        max_line_width=10**9,
        separator=", ",
        suppress_small=False,
    ).replace("\n", " ")


def _write_single_trajectory_output(
    output_path: str,
    trajectory_index: int,
    output_array: np.ndarray,
    io_spec: dict[str, Any] | None,
    *,
    default_mode: str,
    default_precision: str,
    default_decimals: int | None = None,
    delimiter: str = " ",
    header: str | None = None,
) -> str:
    report_path = _single_trajectory_report_path(output_path, trajectory_index)
    save_numeric_array(
        report_path,
        output_array,
        io_spec,
        default_mode=default_mode,
        default_precision=default_precision,
        default_decimals=default_decimals,
        delimiter=delimiter,
        header=header,
    )
    return report_path


def _format_header_value(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.15g}"
    return str(value)


def _charge_dipole_isotropic_output_header(
    *,
    charge_values_path: str,
    charge_coords_pattern: str,
    charge_coords_stride: int,
    dipole_positions_pattern: str,
    dipole_positions_stride: int,
    dipole_vectors_pattern: str,
    dipole_vectors_stride: int,
    k_max_primary: float,
    k_max_secondary: float,
    k_max_tertiary: float,
    k_resolution_primary: float,
    k_resolution_secondary: float,
    k_resolution_tertiary: float,
    Lx: float,
    Ly: float,
    Lz: float,
    cutoff_primary: float | None,
    cutoff_secondary: float | None,
    cutoff_tertiary: float | None,
    cell_size_primary: float | None,
    cell_size_secondary: float | None,
    cell_size_tertiary: float | None,
    charge_count: int,
    dipole_count: int | str,
    frame_count: int,
    trajectory_count: int,
    delete_residue_index: int | None,
    frame_window: tuple[int, int | None, int] | None,
    trajectory_selection: str | None,
) -> str:
    if frame_window is None:
        frame_window_text = "full trajectory"
    else:
        frame_window_text = f"range({frame_window[0]}, {frame_window[1]}, {frame_window[2]})"
    lines = [
        "This file contains the magnitude of wave vectors, raw accumulated charge dipole structure factor, and number of dipoles in each cutoff value averaged over processed frames.",
        "k      Sqp(k)      Num. m (CO-1)      Num. m (CO-2)      Num. m (CO-3)",
        "######################",
        "Calculation mode: isotropic charge-dipole structure factor",
        f"Charge values path: {charge_values_path}",
        f"Charge coordinates path: {charge_coords_pattern}",
        f"Charge coordinates stride: {int(charge_coords_stride)}",
        f"Dipole positions path: {dipole_positions_pattern}",
        f"Dipole positions stride: {int(dipole_positions_stride)}",
        f"Dipole vectors path: {dipole_vectors_pattern}",
        f"Dipole vectors stride: {int(dipole_vectors_stride)}",
        f"Box lengths: Lx={_format_header_value(float(Lx))}, Ly={_format_header_value(float(Ly))}, Lz={_format_header_value(float(Lz))}",
        f"k max values: K1={_format_header_value(float(k_max_primary))}, K2={_format_header_value(float(k_max_secondary))}, K3={_format_header_value(float(k_max_tertiary))}",
        f"k resolutions: res1={_format_header_value(float(k_resolution_primary))}, res2={_format_header_value(float(k_resolution_secondary))}, res3={_format_header_value(float(k_resolution_tertiary))}",
        f"Cutoffs: CO-1={_format_header_value(cutoff_primary)}, CO-2={_format_header_value(cutoff_secondary)}, CO-3={_format_header_value(cutoff_tertiary)}",
        f"Cell sizes: CS-1={_format_header_value(cell_size_primary)}, CS-2={_format_header_value(cell_size_secondary)}, CS-3={_format_header_value(cell_size_tertiary)}",
        f"Charge sites: {int(charge_count)}",
        f"Dipoles per frame: {dipole_count}",
        f"Total frames accumulated: {int(frame_count)}",
        f"Trajectories summed: {int(trajectory_count)}",
        f"Delete residue index: {_format_header_value(delete_residue_index)}",
        f"Trajectory desired length: {frame_window_text}",
        f"Trajectory selection: {_format_header_value(trajectory_selection)}",
    ]
    return "\n".join(lines)


def _run_report_tasks(
    *,
    task_args: list[tuple[Any, ...]],
    worker: Any,
    max_workers: int,
    dataset_label: str,
) -> dict[str, Any]:
    total_frames = 0
    reports: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    def record(item: dict[str, Any]) -> None:
        nonlocal total_frames
        if item.get("error"):
            failed.append(
                {
                    "trajectory_index": int(item.get("trajectory_index", 0) or 0),
                    "file_path": str(item.get("file_path", "")),
                    "error": str(item.get("error", "")),
                }
            )
            return
        frame_count = int(item["frame_count"])
        total_frames += frame_count
        reports.append(
            {
                "trajectory_index": int(item["trajectory_index"]),
                "file_path": str(item.get("file_path", "")),
                "frame_count": frame_count,
                "coord_count": int(item.get("coord_count", 0) or 0),
                "elapsed_s": float(item.get("elapsed_s", 0.0) or 0.0),
                "output_file": str(item["output_file"]),
            }
        )

    if int(max_workers) > 1 and len(task_args) > 1:
        with ProcessPoolExecutor(max_workers=min(int(max_workers), len(task_args))) as executor:
            futures = [executor.submit(worker, args) for args in task_args]
            for future in as_completed(futures):
                record(future.result())
    else:
        for args in task_args:
            record(worker(args))

    if total_frames <= 0:
        raise ValueError(f"No usable trajectories were found for {dataset_label}.")

    reports.sort(key=lambda item: int(item["trajectory_index"]))
    return {
        "reports": reports,
        "failed": failed,
        "total_frames": total_frames,
        "success_count": len(reports),
        "failure_count": len(failed),
        "success_indices": [int(item["trajectory_index"]) for item in reports],
        "trajectory_results": reports,
    }


def _average_saved_trajectory_outputs(
    *,
    reports: list[dict[str, Any]],
    value_columns: tuple[int, ...],
    io_spec: dict[str, Any] | None,
    default_mode: str,
    default_precision: str,
    expected_prefix: np.ndarray | None = None,
    expected_prefix_columns: tuple[int, ...] | None = None,
) -> np.ndarray:
    weighted_values: np.ndarray | None = None
    prefix_values: np.ndarray | None = None
    total_frames = 0

    for report in reports:
        report_path = str(report["output_file"])
        frame_count = int(report["frame_count"])
        trajectory_output = load_numeric_array(
            report_path,
            io_spec,
            default_mode=default_mode,
            default_precision=default_precision,
        )
        trajectory_output = np.asarray(trajectory_output, dtype=np.float64)
        if trajectory_output.ndim == 1:
            trajectory_output = trajectory_output.reshape(1, -1)
        if trajectory_output.shape[1] <= max(value_columns):
            raise ValueError(f"Per-trajectory output has too few columns: {report_path}")

        if expected_prefix is not None and expected_prefix_columns is not None:
            current_prefix = trajectory_output[:, list(expected_prefix_columns)]
            if current_prefix.shape != expected_prefix.shape or not np.allclose(
                current_prefix,
                expected_prefix,
                rtol=1.0e-10,
                atol=1.0e-12,
            ):
                raise ValueError(f"Per-trajectory k values do not match the current run: {report_path}")
            prefix_values = expected_prefix
        elif prefix_values is None:
            prefix_end = min(value_columns)
            prefix_values = trajectory_output[:, :prefix_end].copy()
        elif not np.allclose(
            trajectory_output[:, : prefix_values.shape[1]],
            prefix_values,
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError(f"Per-trajectory coordinate columns do not match the current run: {report_path}")

        values = trajectory_output[:, list(value_columns)]
        if weighted_values is None:
            weighted_values = np.zeros_like(values, dtype=np.float64)
        if values.shape != weighted_values.shape:
            raise ValueError(f"Per-trajectory value shape does not match the current run: {report_path}")
        weighted_values += values * float(frame_count)
        total_frames += frame_count

    if total_frames <= 0 or weighted_values is None:
        raise ValueError("No saved per-trajectory outputs were available for averaging.")
    return weighted_values / float(total_frames)


def _sum_saved_trajectory_outputs(
    *,
    reports: list[dict[str, Any]],
    value_columns: tuple[int, ...],
    io_spec: dict[str, Any] | None,
    default_mode: str,
    default_precision: str,
    expected_prefix: np.ndarray | None = None,
    expected_prefix_columns: tuple[int, ...] | None = None,
) -> np.ndarray:
    summed_values: np.ndarray | None = None
    prefix_values: np.ndarray | None = None

    for report in reports:
        report_path = str(report["output_file"])
        trajectory_output = load_numeric_array(
            report_path,
            io_spec,
            default_mode=default_mode,
            default_precision=default_precision,
        )
        trajectory_output = np.asarray(trajectory_output, dtype=np.float64)
        if trajectory_output.ndim == 1:
            trajectory_output = trajectory_output.reshape(1, -1)
        if trajectory_output.shape[1] <= max(value_columns):
            raise ValueError(f"Per-trajectory output has too few columns: {report_path}")

        if expected_prefix is not None and expected_prefix_columns is not None:
            current_prefix = trajectory_output[:, list(expected_prefix_columns)]
            if current_prefix.shape != expected_prefix.shape or not np.allclose(
                current_prefix,
                expected_prefix,
                rtol=1.0e-10,
                atol=1.0e-12,
            ):
                raise ValueError(f"Per-trajectory k values do not match the current run: {report_path}")
            prefix_values = expected_prefix
        elif prefix_values is None:
            prefix_end = min(value_columns)
            prefix_values = trajectory_output[:, :prefix_end].copy()
        elif not np.allclose(
            trajectory_output[:, : prefix_values.shape[1]],
            prefix_values,
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError(f"Per-trajectory coordinate columns do not match the current run: {report_path}")

        values = trajectory_output[:, list(value_columns)]
        if summed_values is None:
            summed_values = np.zeros_like(values, dtype=np.float64)
        if values.shape != summed_values.shape:
            raise ValueError(f"Per-trajectory value shape does not match the current run: {report_path}")
        summed_values += values

    if summed_values is None:
        raise ValueError("No saved per-trajectory outputs were available for summing.")
    return summed_values


def _natural_sort_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def _discover_coordinate_files(base_directory: str, pattern: str) -> list[str]:
    expanded_pattern = expand_path_pattern(pattern, "", None)
    resolved_pattern = expanded_pattern
    if base_directory and not os.path.isabs(resolved_pattern):
        resolved_pattern = os.path.join(base_directory, resolved_pattern)

    has_index_placeholder = bool(re.search(r"\{[^}]+\}", resolved_pattern))
    glob_pattern = re.sub(r"\{[^}]+\}", "*", resolved_pattern)

    if has_index_placeholder or any(token in glob_pattern for token in ["*", "?", "["]):
        matches = sorted(glob(glob_pattern), key=_natural_sort_key)
        files = [match for match in matches if os.path.isfile(match)]
    elif os.path.isfile(resolved_pattern):
        files = [resolved_pattern]
    else:
        files = []

    if not files:
        raise FileNotFoundError(f"No coordinate files found for pattern: {resolved_pattern}")
    return files


def _prepare_coordinate_array(array: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    if values.ndim == 2:
        if values.shape[1] % 3 != 0:
            raise ValueError("Flattened coordinates must have 3 columns per atom.")
        natms = values.shape[1] // 3
        return values, natms

    if values.ndim == 3 and values.shape[2] == 3:
        natms = values.shape[1]
        return values, natms

    raise ValueError("Coordinate arrays must be shaped as (frames, 3*atoms) or (frames, atoms, 3).")


def _apply_frame_window(coords: np.ndarray, frame_window: tuple[int, int | None, int] | None) -> np.ndarray:
    if frame_window is None:
        return coords
    start, stop, step = frame_window
    sliced = coords[slice(int(start), None if stop is None else int(stop), int(step))]
    if int(sliced.shape[0]) <= 0:
        raise ValueError("Requested frame range produced an empty coordinate selection.")
    return sliced


def _apply_frame_stride(coords: np.ndarray, stride: int, *, label: str) -> np.ndarray:
    stride = int(stride)
    if stride <= 0:
        raise ValueError(f"{label} stride must be a positive integer.")
    if stride == 1:
        return coords
    sliced = coords[::stride]
    if int(sliced.shape[0]) <= 0:
        raise ValueError(f"{label} stride produced an empty frame selection.")
    return sliced


def _validate_cutoff_and_cell_size(
    cutoff: float | None,
    cell_size: float | None,
    *,
    label: str,
) -> tuple[float | None, float | None]:
    cutoff_value = None if cutoff is None else float(cutoff)
    cell_size_value = None if cell_size is None else float(cell_size)

    if cutoff_value is not None and cutoff_value <= 0:
        raise ValueError(f"{label} cutoff must be positive when provided.")
    if cell_size_value is not None and cell_size_value <= 0:
        raise ValueError(f"{label} Cell Size must be positive when provided.")
    if cell_size_value is not None and cutoff_value is None:
        raise ValueError(f"{label} Cell Size requires Cutoff to be set.")
    if cutoff_value is not None and cell_size_value is None:
        cell_size_value = cutoff_value

    return cutoff_value, cell_size_value


def _resolve_three_tier_cutoffs(
    cutoff_primary: float | None = None,
    cutoff_secondary: float | None = None,
    cutoff_tertiary: float | None = None,
    *,
    legacy_cutoff: float | None = None,
) -> tuple[float | None, float | None, float | None]:
    if cutoff_primary is None and cutoff_secondary is None and cutoff_tertiary is None and legacy_cutoff is not None:
        cutoff_primary = legacy_cutoff
        cutoff_secondary = legacy_cutoff
        cutoff_tertiary = legacy_cutoff

    cutoffs = tuple(
        None if value is None else float(value)
        for value in (cutoff_primary, cutoff_secondary, cutoff_tertiary)
    )
    for value in cutoffs:
        if value is not None and value <= 0:
            raise ValueError("cutoff values must be positive when provided.")
    return cutoffs


def _resolve_three_tier_cell_sizes(
    cutoff_primary: float | None,
    cutoff_secondary: float | None,
    cutoff_tertiary: float | None,
    cell_size_primary: float | None = None,
    cell_size_secondary: float | None = None,
    cell_size_tertiary: float | None = None,
    *,
    legacy_cell_size: float | None = None,
) -> tuple[float | None, float | None, float | None]:
    if cell_size_primary is None and cell_size_secondary is None and cell_size_tertiary is None and legacy_cell_size is not None:
        cell_size_primary = legacy_cell_size
        cell_size_secondary = legacy_cell_size
        cell_size_tertiary = legacy_cell_size

    resolved: list[float | None] = []
    for cutoff_value, cell_size_value in (
        (cutoff_primary, cell_size_primary),
        (cutoff_secondary, cell_size_secondary),
        (cutoff_tertiary, cell_size_tertiary),
    ):
        _, resolved_cell_size = _validate_cutoff_and_cell_size(
            cutoff_value,
            cell_size_value,
            label="Structure factor",
        )
        resolved.append(resolved_cell_size)
    return tuple(resolved)  # type: ignore[return-value]


def _three_tier_masks_from_magnitudes(
    magnitudes: np.ndarray,
    k_max_primary: float,
    k_max_secondary: float,
    k_max_tertiary: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(magnitudes, dtype=np.float64)
    return (
        (values > 0.0) & (values <= float(k_max_primary)),
        (values > float(k_max_primary)) & (values <= float(k_max_secondary)),
        (values > float(k_max_secondary)) & (values <= float(k_max_tertiary)),
    )


def _build_periodic_cell_list(
    positions: np.ndarray,
    box: np.ndarray,
    cell_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[tuple[int, int, int], np.ndarray]]:
    wrapped_positions = np.mod(np.asarray(positions, dtype=np.float64), box)
    cell_counts = np.maximum(1, np.floor(box / float(cell_size)).astype(int))
    cell_widths = box / cell_counts
    cell_indices = np.floor(wrapped_positions / cell_widths).astype(int)
    cell_indices = np.minimum(cell_indices, cell_counts - 1)

    cells: dict[tuple[int, int, int], list[int]] = {}
    for index, cell in enumerate(cell_indices):
        key = (int(cell[0]), int(cell[1]), int(cell[2]))
        cells.setdefault(key, []).append(index)

    return (
        wrapped_positions,
        cell_counts,
        cell_widths,
        {key: np.asarray(indices, dtype=np.int64) for key, indices in cells.items()},
    )


def _cell_key_linear_index(cell: tuple[int, int, int], cell_counts: np.ndarray) -> int:
    return (
        int(cell[0]) * int(cell_counts[1]) * int(cell_counts[2])
        + int(cell[1]) * int(cell_counts[2])
        + int(cell[2])
    )


def _iter_cutoff_displacement_blocks(
    positions: np.ndarray,
    box: np.ndarray,
    cutoff: float,
    cell_size: float | None,
    i_chunk: int,
    j_chunk: int,
) -> Any:
    wrapped_positions, cell_counts, cell_widths, cells = _build_periodic_cell_list(
        positions,
        box,
        float(cell_size if cell_size is not None else cutoff),
    )
    search_span = np.ceil(float(cutoff) / cell_widths).astype(int)
    cutoff_sq = float(cutoff) ** 2
    seen_pairs: set[tuple[int, int]] = set()

    for cell in sorted(cells):
        indices_i_all = cells[cell]
        cell_id_i = _cell_key_linear_index(cell, cell_counts)
        neighbor_cells: set[tuple[int, int, int]] = set()
        for dx in range(-int(search_span[0]), int(search_span[0]) + 1):
            for dy in range(-int(search_span[1]), int(search_span[1]) + 1):
                for dz in range(-int(search_span[2]), int(search_span[2]) + 1):
                    neighbor_cells.add(
                        (
                            int((cell[0] + dx) % cell_counts[0]),
                            int((cell[1] + dy) % cell_counts[1]),
                            int((cell[2] + dz) % cell_counts[2]),
                        )
                    )

        for neighbor in sorted(neighbor_cells):
            if neighbor not in cells:
                continue
            cell_id_j = _cell_key_linear_index(neighbor, cell_counts)
            pair_key = (min(cell_id_i, cell_id_j), max(cell_id_i, cell_id_j))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            indices_j_all = cells[neighbor]

            if cell == neighbor:
                for a0 in range(0, int(indices_i_all.size), int(i_chunk)):
                    a1 = min(int(indices_i_all.size), a0 + int(i_chunk))
                    block_i = indices_i_all[a0:a1]
                    ri = wrapped_positions[block_i]

                    dr = ri[:, None, :] - ri[None, :, :]
                    dr -= box * np.rint(dr / box)
                    dr_flat = dr.reshape(-1, 3)
                    dist_sq = np.sum(dr_flat * dr_flat, axis=1)

                    ni = int(block_i.size)
                    a_idx, b_idx = np.triu_indices(ni, k=1)
                    keep = a_idx * ni + b_idx
                    keep = keep[dist_sq[keep] <= cutoff_sq]
                    if keep.size > 0:
                        yield np.ascontiguousarray(dr_flat[keep], dtype=np.float64)

                    for b0 in range(a1, int(indices_i_all.size), int(j_chunk)):
                        b1 = min(int(indices_i_all.size), b0 + int(j_chunk))
                        block_j = indices_i_all[b0:b1]
                        rj = wrapped_positions[block_j]
                        dr = ri[:, None, :] - rj[None, :, :]
                        dr -= box * np.rint(dr / box)
                        dr_flat = dr.reshape(-1, 3)
                        dist_sq = np.sum(dr_flat * dr_flat, axis=1)
                        keep = dist_sq <= cutoff_sq
                        if np.any(keep):
                            yield np.ascontiguousarray(dr_flat[keep], dtype=np.float64)
                continue

            for a0 in range(0, int(indices_i_all.size), int(i_chunk)):
                a1 = min(int(indices_i_all.size), a0 + int(i_chunk))
                block_i = indices_i_all[a0:a1]
                ri = wrapped_positions[block_i]
                for b0 in range(0, int(indices_j_all.size), int(j_chunk)):
                    b1 = min(int(indices_j_all.size), b0 + int(j_chunk))
                    block_j = indices_j_all[b0:b1]
                    rj = wrapped_positions[block_j]
                    dr = ri[:, None, :] - rj[None, :, :]
                    dr -= box * np.rint(dr / box)
                    dr_flat = dr.reshape(-1, 3)
                    dist_sq = np.sum(dr_flat * dr_flat, axis=1)
                    keep = dist_sq <= cutoff_sq
                    if np.any(keep):
                        yield np.ascontiguousarray(dr_flat[keep], dtype=np.float64)


def _resolve_chunk_length(value: int | None, total: int) -> int:
    if value is None:
        return int(total)
    chunk = int(value)
    if chunk <= 0:
        raise ValueError("Chunk sizes must be positive integers when provided.")
    return min(chunk, int(total))


def _load_charge_values(
    charge_file: str,
    input_io_spec: dict[str, Any] | None,
    delete_index: int | None,
) -> np.ndarray:
    values = load_numeric_array(
        charge_file,
        input_io_spec,
        default_mode="text",
        default_precision="double",
    )
    charge_array = np.asarray(values, dtype=np.float64)
    if charge_array.ndim == 2 and charge_array.shape[1] > 1:
        charge_values = charge_array[:, -1]
    else:
        charge_values = charge_array.reshape(-1)
    if delete_index is not None:
        charge_values = np.delete(charge_values, int(delete_index), axis=0)
    if charge_values.size <= 0:
        raise ValueError("Charge values file produced an empty charge array.")
    return np.asarray(charge_values, dtype=np.float64)


def _active_charge_site_count(charge_values: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(charge_values, dtype=np.float64) != 0.0))


def _prepare_xyz_frames(
    array: np.ndarray,
    *,
    name: str,
    delete_index: int | None = None,
) -> np.ndarray:
    coords, _ = _prepare_coordinate_array(array)
    coords_3d = np.asarray(coords, dtype=np.float64)
    if coords_3d.ndim == 2:
        coords_3d = coords_3d.reshape(coords_3d.shape[0], coords_3d.shape[1] // 3, 3)
    if coords_3d.ndim != 3 or coords_3d.shape[2] != 3:
        raise ValueError(f"{name} must be shaped as (frames, atoms, 3) or (frames, 3*atoms).")
    if delete_index is not None:
        coords_3d = np.delete(coords_3d, int(delete_index), axis=1)
    if int(coords_3d.shape[1]) <= 0:
        raise ValueError(f"{name} has no points after applying the optional deletion.")
    return np.ascontiguousarray(coords_3d, dtype=np.float64)


def _normalize_vectors(vectors: np.ndarray, *, name: str) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=2, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError(f"Found zero-magnitude vector in {name}.")
    return np.ascontiguousarray(vectors / norms, dtype=np.float64)


def _build_khat(k_vectors_array: np.ndarray) -> np.ndarray:
    k_vectors_use = np.asarray(k_vectors_array, dtype=np.float64)
    magnitudes = np.linalg.norm(k_vectors_use, axis=1, keepdims=True)
    khat = np.zeros_like(k_vectors_use, dtype=np.float64)
    valid = magnitudes[:, 0] > 0.0
    khat[valid] = k_vectors_use[valid] / magnitudes[valid]
    return khat


def charge_dipole_k_vectors_three_tier(
    Lx: float,
    Ly: float,
    Lz: float,
    kmax1: float,
    kmax2: float,
    kmax3: float,
    res1: float,
    res2: float,
    res3: float,
    dtype: Any = np.float64,
    include_zero: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if kmax1 < 0 or kmax2 <= 0 or kmax3 <= 0:
        raise ValueError("kmax values must satisfy kmax1 >= 0, kmax2 > 0, and kmax3 > 0.")
    if kmax1 > kmax2 or kmax2 > kmax3:
        raise ValueError("kmax values must satisfy kmax1 <= kmax2 <= kmax3.")
    if float(res1) <= 0 or float(res2) <= 0 or float(res3) <= 0:
        raise ValueError("k resolutions must be positive.")

    two_pi = np.array(2.0 * np.pi, dtype=dtype)
    nxmax = int(np.ceil(float(kmax3) * float(Lx) / float(two_pi)))
    nymax = int(np.ceil(float(kmax3) * float(Ly) / float(two_pi)))
    nzmax = int(np.ceil(float(kmax3) * float(Lz) / float(two_pi)))

    kmags: list[float] = []
    kvecs: list[tuple[float, float, float]] = []
    for nx in range(-nxmax, nxmax + 1):
        for ny in range(-nymax, nymax + 1):
            for nz in range(-nzmax, nzmax + 1):
                if not include_zero and nx == 0 and ny == 0 and nz == 0:
                    continue

                kvec = two_pi * np.array(
                    [nx / float(Lx), ny / float(Ly), nz / float(Lz)],
                    dtype=dtype,
                )
                kmag = float(np.linalg.norm(kvec))
                if 0.0 < kmag <= float(kmax3):
                    kmags.append(kmag)
                    kvecs.append((float(kvec[0]), float(kvec[1]), float(kvec[2])))

    if not kmags:
        return np.empty(0, dtype=dtype), np.empty((0, 3), dtype=dtype)

    k_mags = np.asarray(kmags, dtype=dtype)
    k_vectors_array = np.asarray(kvecs, dtype=dtype)
    sort_order = np.lexsort(
        (
            k_vectors_array[:, 2],
            k_vectors_array[:, 1],
            k_vectors_array[:, 0],
            k_mags,
        )
    )
    k_mags = k_mags[sort_order]
    k_vectors_array = k_vectors_array[sort_order]
    unique_k = np.unique(np.asarray(k_mags, dtype=dtype))
    tier_1 = _filter_by_resolution(unique_k[(unique_k > 0.0) & (unique_k <= float(kmax1))], float(res1), dtype=dtype)
    tier_2 = _filter_by_resolution(unique_k[(unique_k > float(kmax1)) & (unique_k <= float(kmax2))], float(res2), dtype=dtype)
    tier_3 = _filter_by_resolution(unique_k[(unique_k > float(kmax2)) & (unique_k <= float(kmax3))], float(res3), dtype=dtype)
    selected_magnitudes = np.concatenate([values for values in (tier_1, tier_2, tier_3) if values.size > 0]).astype(dtype, copy=False)
    if selected_magnitudes.size == 0:
        return np.empty(0, dtype=dtype), np.empty((0, 3), dtype=dtype)
    keep = np.isin(np.round(k_mags, 12), np.round(selected_magnitudes, 12))
    return k_mags[keep].astype(dtype, copy=False), k_vectors_array[keep].astype(dtype, copy=False)


def charge_dipole_k_magnitudes_by_resolution_three_tier(
    Lx: float,
    Ly: float,
    Lz: float,
    kmax1: float,
    kmax2: float,
    kmax3: float,
    res1: float,
    res2: float,
    res3: float,
    dtype: Any = np.float64,
) -> np.ndarray:
    if kmax1 < 0 or kmax2 <= 0 or kmax3 <= 0:
        raise ValueError("kmax values must satisfy kmax1 >= 0, kmax2 > 0, and kmax3 > 0.")
    if kmax1 > kmax2 or kmax2 > kmax3:
        raise ValueError("kmax values must satisfy kmax1 <= kmax2 <= kmax3.")
    if float(res1) <= 0 or float(res2) <= 0 or float(res3) <= 0:
        raise ValueError("k resolutions must be positive.")

    two_pi = float(2.0 * np.pi)
    nxmax = int(np.ceil(float(kmax3) * float(Lx) / two_pi))
    nymax = int(np.ceil(float(kmax3) * float(Ly) / two_pi))
    nzmax = int(np.ceil(float(kmax3) * float(Lz) / two_pi))

    z_terms = (np.arange(0, nzmax + 1, dtype=np.float64) / float(Lz)) ** 2
    kmax3_float = float(kmax3)
    chunks: list[np.ndarray] = []
    for nx in range(0, nxmax + 1):
        x_term = (float(nx) / float(Lx)) ** 2
        for ny in range(0, nymax + 1):
            if nx == 0 and ny == 0:
                nz_start = 1
            else:
                nz_start = 0
            y_term = (float(ny) / float(Ly)) ** 2
            mags = two_pi * np.sqrt(x_term + y_term + z_terms[nz_start:])
            mags = mags[(mags > 0.0) & (mags <= kmax3_float)]
            if mags.size > 0:
                chunks.append(mags.astype(dtype, copy=False))

    if not chunks:
        return np.empty(0, dtype=dtype)

    unique_k = np.unique(np.concatenate(chunks).astype(dtype, copy=False))
    if unique_k.size == 0:
        return unique_k

    tier_1 = _filter_by_resolution(unique_k[(unique_k > 0.0) & (unique_k <= float(kmax1))], float(res1), dtype=dtype)
    tier_2 = _filter_by_resolution(unique_k[(unique_k > float(kmax1)) & (unique_k <= float(kmax2))], float(res2), dtype=dtype)
    tier_3 = _filter_by_resolution(unique_k[(unique_k > float(kmax2)) & (unique_k <= float(kmax3))], float(res3), dtype=dtype)
    return np.concatenate([values for values in (tier_1, tier_2, tier_3) if values.size > 0]).astype(dtype, copy=False)


def _filter_by_resolution(values: np.ndarray, resolution: float, *, dtype: Any = np.float64) -> np.ndarray:
    values_array = np.asarray(values, dtype=dtype)
    if values_array.size == 0:
        return values_array
    selected = [float(values_array[0])]
    for value in values_array[1:]:
        if float(value) - selected[-1] >= float(resolution):
            selected.append(float(value))
    return np.asarray(selected, dtype=dtype)


def density_k_magnitudes_by_resolution_three_tier(
    Lx: float,
    Ly: float,
    Lz: float,
    kmax1: float,
    kmax2: float,
    kmax3: float,
    res1: float,
    res2: float,
    res3: float,
    dtype: Any = np.float64,
) -> np.ndarray:
    unique_k = np.unique(
        isotropic_k_magnitudes_three_tier(
            Lx,
            Ly,
            Lz,
            kmax1,
            kmax2,
            kmax3,
            dtype=dtype,
        )
    )
    if unique_k.size == 0:
        return unique_k.astype(dtype, copy=False)

    tier_1 = _filter_by_resolution(unique_k[(unique_k > 0.0) & (unique_k <= float(kmax1))], float(res1), dtype=dtype)
    tier_2 = _filter_by_resolution(unique_k[(unique_k > float(kmax1)) & (unique_k <= float(kmax2))], float(res2), dtype=dtype)
    tier_3 = _filter_by_resolution(unique_k[(unique_k > float(kmax2)) & (unique_k <= float(kmax3))], float(res3), dtype=dtype)
    return np.concatenate([values for values in (tier_1, tier_2, tier_3) if values.size > 0]).astype(dtype, copy=False)


def directional_k_vectors_by_resolution_three_tier(
    Lx: float,
    Ly: float,
    Lz: float,
    kmax1: float,
    kmax2: float,
    kmax3: float,
    res1: float,
    res2: float,
    res3: float,
    active_axes: tuple[str, ...] | None = None,
    dtype: Any = np.float64,
) -> np.ndarray:
    all_vectors = directional_k_vectors_three_tier(
        Lx,
        Ly,
        Lz,
        kmax1,
        kmax2,
        kmax3,
        active_axes=active_axes,
        dtype=dtype,
    )
    if all_vectors.size == 0:
        return all_vectors

    magnitudes = np.linalg.norm(all_vectors, axis=1)
    unique_k = np.unique(np.asarray(magnitudes, dtype=dtype))
    tier_1 = _filter_by_resolution(unique_k[(unique_k > 0.0) & (unique_k <= float(kmax1))], float(res1), dtype=dtype)
    tier_2 = _filter_by_resolution(unique_k[(unique_k > float(kmax1)) & (unique_k <= float(kmax2))], float(res2), dtype=dtype)
    tier_3 = _filter_by_resolution(unique_k[(unique_k > float(kmax2)) & (unique_k <= float(kmax3))], float(res3), dtype=dtype)
    selected_magnitudes = np.concatenate([values for values in (tier_1, tier_2, tier_3) if values.size > 0]).astype(dtype, copy=False)
    if selected_magnitudes.size == 0:
        return np.empty((0, 3), dtype=dtype)

    keep = np.isin(np.round(magnitudes, 12), np.round(selected_magnitudes, 12))
    return np.asarray(all_vectors[keep], dtype=dtype)


def _shell_average_complex(k_magnitudes: np.ndarray, values: np.ndarray, dk: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_magnitudes = np.asarray(k_magnitudes, dtype=np.float64)
    complex_values = np.asarray(values, dtype=np.complex128)
    if k_magnitudes.size != complex_values.size:
        raise ValueError("k magnitudes and charge-dipole values must have the same size.")
    if float(dk) <= 0:
        raise ValueError("dk_shell must be positive.")
    if k_magnitudes.size == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.complex128),
            np.empty(0, dtype=np.int64),
        )

    labels = np.round(k_magnitudes / float(dk)).astype(np.int64)
    unique_labels = np.unique(labels)
    k_shell = np.empty(unique_labels.size, dtype=np.float64)
    value_shell = np.empty(unique_labels.size, dtype=np.complex128)
    counts = np.empty(unique_labels.size, dtype=np.int64)

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        k_shell[idx] = float(np.mean(k_magnitudes[mask]))
        value_shell[idx] = np.mean(complex_values[mask])
        counts[idx] = int(np.sum(mask))

    return k_shell, value_shell, counts


def _spherical_bessel_j1(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    small = np.abs(x) < 1.0e-6
    if np.any(small):
        xs = x[small]
        x2 = xs * xs
        out[small] = xs / 3.0 - (xs * x2) / 30.0 + (xs * x2 * x2) / 840.0
    if np.any(~small):
        xl = x[~small]
        out[~small] = np.sin(xl) / (xl * xl) - np.cos(xl) / xl
    return out


def charge_dipole_structure_factor_isotropic(
    k_magnitudes: np.ndarray,
    charge_positions: np.ndarray,
    charge_values: np.ndarray,
    dipole_positions: np.ndarray,
    dipole_directions: np.ndarray,
    cutoff: float | None = None,
    cell_size: float | None = None,
    box: np.ndarray | None = None,
    frame_chunk: int | None = None,
    charge_chunk: int | None = None,
    dipole_chunk: int | None = None,
    k_chunk: int = 256,
    dtype: Any = np.float64,
    normalize_by_frames: bool = True,
    status_logger: _StatusLogger | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    k_vals_use = np.asarray(k_magnitudes, dtype=dtype)
    rq = np.asarray(charge_positions, dtype=dtype)
    z = np.asarray(charge_values, dtype=dtype)
    rp = np.asarray(dipole_positions, dtype=dtype)
    ehat = np.asarray(dipole_directions, dtype=dtype)

    if rq.ndim != 3 or rq.shape[2] != 3:
        raise ValueError("Charge positions must have shape (frames, charges, 3).")
    if rp.ndim != 3 or rp.shape[2] != 3:
        raise ValueError("Dipole positions must have shape (frames, dipoles, 3).")
    if ehat.shape != rp.shape:
        raise ValueError("Dipole directions must match the dipole-position shape.")
    if z.ndim != 1:
        raise ValueError("Charge values must be one-dimensional.")
    if rq.shape[1] != z.shape[0]:
        raise ValueError("Charge-value count must match the number of charge positions.")
    if rq.shape[0] != rp.shape[0]:
        raise ValueError("Charge and dipole trajectories must have the same number of frames.")
    if k_vals_use.ndim != 1:
        raise ValueError("k_magnitudes must be one-dimensional.")
    if k_vals_use.shape[0] <= 0:
        raise ValueError("At least one isotropic charge-dipole k magnitude is required.")
    if box is not None:
        box = np.asarray(box, dtype=dtype)
        if box.shape != (3,):
            raise ValueError("box must contain exactly three lengths: Lx, Ly, Lz.")
        if np.any(box <= 0):
            raise ValueError("All box lengths must be positive.")

    frame_count = int(rq.shape[0])
    nonzero_charge_mask = z != 0.0
    if not np.all(nonzero_charge_mask):
        rq = np.ascontiguousarray(rq[:, nonzero_charge_mask, :], dtype=dtype)
        z = np.ascontiguousarray(z[nonzero_charge_mask], dtype=dtype)
    if int(z.shape[0]) == 0:
        return np.zeros(k_vals_use.shape[0], dtype=dtype), np.zeros(frame_count, dtype=np.int64)

    cutoff, cell_size = _validate_cutoff_and_cell_size(cutoff, cell_size, label="Charge-dipole structure factor")

    cutoff_sq = None if cutoff is None else float(cutoff) ** 2
    out_total = np.zeros(k_vals_use.shape[0], dtype=np.float64)
    frame_chunk_use = _resolve_chunk_length(frame_chunk, frame_count)
    charge_chunk_use = _resolve_chunk_length(charge_chunk, int(rq.shape[1]))
    dipole_chunk_use = _resolve_chunk_length(dipole_chunk, int(rp.shape[1]))
    k_chunk_use = _resolve_chunk_length(k_chunk, int(k_vals_use.shape[0]))
    dipole_count = int(rp.shape[1])
    box_float = None if box is None else np.asarray(box, dtype=np.float64)
    last_used_dipole_count = 0
    frame_dipole_counts = np.zeros(frame_count, dtype=np.int64)

    for f0 in range(0, frame_count, frame_chunk_use):
        f1 = min(frame_count, f0 + frame_chunk_use)
        for frame_index in range(f0, f1):
            rq_frame = np.asarray(rq[frame_index], dtype=np.float64)
            rp_frame = np.asarray(rp[frame_index], dtype=np.float64)
            ehat_frame = np.asarray(ehat[frame_index], dtype=np.float64)
            used_dipoles_mask = None if cutoff_sq is None else np.zeros(dipole_count, dtype=bool)
            frame_out = np.zeros(k_vals_use.shape[0], dtype=np.float64)

            wrapped_rp_frame = None
            cell_counts = None
            cell_widths = None
            dipole_cells = None
            search_span = None
            if cutoff_sq is not None and box is not None and cell_size is not None:
                wrapped_rp_frame, cell_counts, cell_widths, dipole_cells = _build_periodic_cell_list(
                    rp_frame,
                    box_float,
                    float(cell_size),
                )
                search_span = np.ceil(float(cutoff) / cell_widths).astype(int)

            for q0 in range(0, int(rq_frame.shape[0]), charge_chunk_use):
                q1 = min(int(rq_frame.shape[0]), q0 + charge_chunk_use)
                for charge_index in range(q0, q1):
                    zq = float(z[charge_index])
                    charge_position = rq_frame[charge_index]
                    if dipole_cells is not None and wrapped_rp_frame is not None and cell_counts is not None and cell_widths is not None and search_span is not None:
                        charge_wrapped = np.mod(np.asarray(charge_position, dtype=np.float64), box_float)
                        charge_cell = np.floor(charge_wrapped / cell_widths).astype(int)
                        charge_cell = np.minimum(charge_cell, cell_counts - 1)
                        neighbor_cells: set[tuple[int, int, int]] = set()
                        for dx in range(-int(search_span[0]), int(search_span[0]) + 1):
                            for dy in range(-int(search_span[1]), int(search_span[1]) + 1):
                                for dz in range(-int(search_span[2]), int(search_span[2]) + 1):
                                    neighbor_cells.add(
                                        (
                                            int((charge_cell[0] + dx) % cell_counts[0]),
                                            int((charge_cell[1] + dy) % cell_counts[1]),
                                            int((charge_cell[2] + dz) % cell_counts[2]),
                                        )
                                    )
                        candidate_lists = [dipole_cells[cell] for cell in neighbor_cells if cell in dipole_cells]
                        if not candidate_lists:
                            continue
                        candidate_indices = np.concatenate(candidate_lists)
                        ehat_candidates_all = ehat_frame[candidate_indices]
                        dr_all = wrapped_rp_frame[candidate_indices] - charge_wrapped[None, :]
                        dr_all -= box_float * np.rint(dr_all / box_float)
                        keep = np.sum(dr_all * dr_all, axis=1) <= cutoff_sq
                        if not np.any(keep):
                            continue
                        if used_dipoles_mask is not None:
                            used_dipoles_mask[candidate_indices[keep]] = True
                        dr_candidates_all = dr_all[keep]
                        ehat_candidates_all = ehat_candidates_all[keep]
                    else:
                        dr_all = rp_frame - charge_position[None, :]
                        if box_float is not None:
                            dr_all -= box_float * np.rint(dr_all / box_float)
                        if cutoff_sq is not None:
                            keep = np.sum(dr_all * dr_all, axis=1) <= cutoff_sq
                            if not np.any(keep):
                                continue
                            if used_dipoles_mask is not None:
                                used_dipoles_mask[keep] = True
                            dr_candidates_all = dr_all[keep]
                            ehat_candidates_all = ehat_frame[keep]
                        else:
                            dr_candidates_all = dr_all
                            ehat_candidates_all = ehat_frame

                    for d0 in range(0, int(dr_candidates_all.shape[0]), dipole_chunk_use):
                        d1 = min(int(dr_candidates_all.shape[0]), d0 + dipole_chunk_use)
                        dr_block = np.asarray(dr_candidates_all[d0:d1], dtype=np.float64)
                        ehat_block = np.asarray(ehat_candidates_all[d0:d1], dtype=np.float64)
                        r_sq = np.sum(dr_block * dr_block, axis=1)
                        valid = r_sq > 1.0e-24
                        if not np.any(valid):
                            continue
                        dr_valid = dr_block[valid]
                        ehat_valid = ehat_block[valid]
                        r = np.sqrt(r_sq[valid])
                        r_hat = dr_valid / r[:, None]
                        weights = zq * np.sum(ehat_valid * r_hat, axis=1)

                        for k0 in range(0, int(k_vals_use.shape[0]), k_chunk_use):
                            k1 = min(int(k_vals_use.shape[0]), k0 + k_chunk_use)
                            kr = np.outer(k_vals_use[k0:k1], r)
                            frame_out[k0:k1] += np.sum(weights[None, :] * _spherical_bessel_j1(kr), axis=1)
            last_used_dipole_count = dipole_count if used_dipoles_mask is None else int(np.count_nonzero(used_dipoles_mask))
            frame_dipole_counts[frame_index] = last_used_dipole_count
            out_total += frame_out
            if status_logger is not None:
                status_logger.charge_dipole(
                    frame_index + 1,
                    int(z.shape[0]),
                    int(k_vals_use.shape[0]),
                    last_used_dipole_count,
                )

    out_total *= -1.0
    if normalize_by_frames:
        out_total /= float(frame_count)
    if status_logger is not None:
        status_logger.charge_dipole(frame_count, int(z.shape[0]), int(k_vals_use.shape[0]), last_used_dipole_count, force=True)

    return out_total.astype(dtype, copy=False), frame_dipole_counts


def charge_dipole_structure_factor_directional(
    k_vectors_array: np.ndarray,
    charge_positions: np.ndarray,
    charge_values: np.ndarray,
    dipole_positions: np.ndarray,
    dipole_directions: np.ndarray,
    cutoff: float | None = None,
    cell_size: float | None = None,
    box: np.ndarray | None = None,
    frame_chunk: int | None = None,
    charge_chunk: int | None = None,
    dipole_chunk: int | None = None,
    k_chunk: int = 256,
    dtype: Any = np.float64,
    normalize_by_frames: bool = True,
    status_logger: _StatusLogger | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    k_vectors_use = np.asarray(k_vectors_array, dtype=dtype)
    rq = np.asarray(charge_positions, dtype=dtype)
    z = np.asarray(charge_values, dtype=dtype)
    rp = np.asarray(dipole_positions, dtype=dtype)
    ehat = np.asarray(dipole_directions, dtype=dtype)

    if rq.ndim != 3 or rq.shape[2] != 3:
        raise ValueError("Charge positions must have shape (frames, charges, 3).")
    if rp.ndim != 3 or rp.shape[2] != 3:
        raise ValueError("Dipole positions must have shape (frames, dipoles, 3).")
    if ehat.shape != rp.shape:
        raise ValueError("Dipole directions must match the dipole-position shape.")
    if z.ndim != 1:
        raise ValueError("Charge values must be one-dimensional.")
    if rq.shape[1] != z.shape[0]:
        raise ValueError("Charge-value count must match the number of charge positions.")
    if rq.shape[0] != rp.shape[0]:
        raise ValueError("Charge and dipole trajectories must have the same number of frames.")
    if k_vectors_use.ndim != 2 or k_vectors_use.shape[1] != 3:
        raise ValueError("k_vectors_array must have shape (n_k, 3).")
    if k_vectors_use.shape[0] <= 0:
        raise ValueError("At least one charge-dipole k vector is required.")
    if box is not None:
        box = np.asarray(box, dtype=dtype)
        if box.shape != (3,):
            raise ValueError("box must contain exactly three lengths: Lx, Ly, Lz.")
        if np.any(box <= 0):
            raise ValueError("All box lengths must be positive.")

    frame_count = int(rq.shape[0])
    nonzero_charge_mask = z != 0.0
    if not np.all(nonzero_charge_mask):
        rq = np.ascontiguousarray(rq[:, nonzero_charge_mask, :], dtype=dtype)
        z = np.ascontiguousarray(z[nonzero_charge_mask], dtype=dtype)
    if int(z.shape[0]) == 0:
        return np.zeros(k_vectors_use.shape[0], dtype=np.complex128), np.zeros(frame_count, dtype=np.int64)

    cutoff, cell_size = _validate_cutoff_and_cell_size(cutoff, cell_size, label="Charge-dipole structure factor")

    khat = _build_khat(k_vectors_use)
    cutoff_sq = None if cutoff is None else float(cutoff) ** 2

    out_re = np.empty(k_vectors_use.shape[0], dtype=np.float64)
    out_im = np.empty(k_vectors_use.shape[0], dtype=np.float64)
    frame_chunk_use = _resolve_chunk_length(frame_chunk, frame_count)
    charge_chunk_use = _resolve_chunk_length(charge_chunk, int(rq.shape[1]))
    dipole_chunk_use = _resolve_chunk_length(dipole_chunk, int(rp.shape[1]))
    k_chunk_use = _resolve_chunk_length(k_chunk, int(k_vectors_use.shape[0]))
    dipole_count = int(rp.shape[1])
    inv_dipole_count = 1.0 / float(dipole_count)
    box_float = None if box is None else np.asarray(box, dtype=np.float64)
    last_used_dipole_count = 0
    frame_dipole_counts = np.zeros(frame_count, dtype=np.int64)

    out_re.fill(0.0)
    out_im.fill(0.0)

    for f0 in range(0, frame_count, frame_chunk_use):
        f1 = min(frame_count, f0 + frame_chunk_use)
        for frame_index in range(f0, f1):
            rq_frame = np.asarray(rq[frame_index], dtype=np.float64)
            rp_frame = np.asarray(rp[frame_index], dtype=np.float64)
            ehat_frame = np.asarray(ehat[frame_index], dtype=np.float64)
            used_dipoles_mask = np.zeros(dipole_count, dtype=bool)
            wrapped_rp_frame = None
            cell_counts = None
            cell_widths = None
            dipole_cells = None
            search_span = None
            if cutoff_sq is not None and box is not None and cell_size is not None:
                wrapped_rp_frame, cell_counts, cell_widths, dipole_cells = _build_periodic_cell_list(
                    rp_frame,
                    box_float,
                    float(cell_size),
                )
                search_span = np.ceil(float(cutoff) / cell_widths).astype(int)

            for k0 in range(0, k_vectors_use.shape[0], k_chunk_use):
                k1 = min(k_vectors_use.shape[0], k0 + k_chunk_use)
                k_block = np.asarray(k_vectors_use[k0:k1], dtype=np.float64)
                kh_block = np.asarray(khat[k0:k1], dtype=np.float64)

                if cutoff_sq is None:
                    used_dipoles_mask[:] = True
                    q_cos = np.zeros(k1 - k0, dtype=np.float64)
                    q_sin = np.zeros(k1 - k0, dtype=np.float64)
                    for q0 in range(0, int(rq_frame.shape[0]), charge_chunk_use):
                        q1 = min(int(rq_frame.shape[0]), q0 + charge_chunk_use)
                        q_phase = np.matmul(rq_frame[q0:q1], k_block.T)
                        z_block = z[q0:q1]
                        q_cos += np.sum(z_block[:, None] * np.cos(q_phase), axis=0)
                        q_sin += np.sum(z_block[:, None] * np.sin(q_phase), axis=0)

                    p_cos = np.zeros(k1 - k0, dtype=np.float64)
                    p_sin = np.zeros(k1 - k0, dtype=np.float64)
                    for d0 in range(0, int(rp_frame.shape[0]), dipole_chunk_use):
                        d1 = min(int(rp_frame.shape[0]), d0 + dipole_chunk_use)
                        p_phase = np.matmul(rp_frame[d0:d1], k_block.T)
                        weights = np.matmul(ehat_frame[d0:d1], kh_block.T)
                        p_cos += np.sum(weights * np.cos(p_phase), axis=0)
                        p_sin += np.sum(weights * np.sin(p_phase), axis=0)

                    out_re[k0:k1] += q_sin * p_cos - q_cos * p_sin
                    out_im[k0:k1] += q_cos * p_cos + q_sin * p_sin
                    continue

                continue

            if cutoff_sq is None:
                last_used_dipole_count = dipole_count
                frame_dipole_counts[frame_index] = last_used_dipole_count
                continue

            for q0 in range(0, int(rq_frame.shape[0]), charge_chunk_use):
                q1 = min(int(rq_frame.shape[0]), q0 + charge_chunk_use)
                for charge_index in range(q0, q1):
                    zq = float(z[charge_index])
                    charge_position = rq_frame[charge_index]
                    if dipole_cells is not None and wrapped_rp_frame is not None and cell_counts is not None and cell_widths is not None and search_span is not None:
                        charge_wrapped = np.mod(np.asarray(charge_position, dtype=np.float64), box_float)
                        charge_cell = np.floor(charge_wrapped / cell_widths).astype(int)
                        charge_cell = np.minimum(charge_cell, cell_counts - 1)
                        neighbor_cells: set[tuple[int, int, int]] = set()
                        for dx in range(-int(search_span[0]), int(search_span[0]) + 1):
                            for dy in range(-int(search_span[1]), int(search_span[1]) + 1):
                                for dz in range(-int(search_span[2]), int(search_span[2]) + 1):
                                    neighbor_cells.add(
                                        (
                                            int((charge_cell[0] + dx) % cell_counts[0]),
                                            int((charge_cell[1] + dy) % cell_counts[1]),
                                            int((charge_cell[2] + dz) % cell_counts[2]),
                                        )
                                    )
                        candidate_lists = [dipole_cells[cell] for cell in neighbor_cells if cell in dipole_cells]
                        if not candidate_lists:
                            continue
                        candidate_indices = np.concatenate(candidate_lists)
                        rp_candidates_wrapped = wrapped_rp_frame[candidate_indices]
                        ehat_candidates_all = ehat_frame[candidate_indices]
                        dr_candidates = rp_candidates_wrapped - charge_wrapped[None, :]
                        dr_candidates -= box_float * np.rint(dr_candidates / box_float)
                        cutoff_keep = np.sum(dr_candidates * dr_candidates, axis=1) <= cutoff_sq
                        if not np.any(cutoff_keep):
                            continue
                        used_dipoles_mask[candidate_indices[cutoff_keep]] = True
                        rp_candidates_all = rp_frame[candidate_indices][cutoff_keep]
                        ehat_candidates_all = ehat_candidates_all[cutoff_keep]
                    else:
                        dr_candidates = rp_frame - charge_position[None, :]
                        if box_float is not None:
                            dr_candidates -= box_float * np.rint(dr_candidates / box_float)
                        cutoff_keep = np.sum(dr_candidates * dr_candidates, axis=1) <= cutoff_sq
                        if not np.any(cutoff_keep):
                            continue
                        used_dipoles_mask[cutoff_keep] = True
                        rp_candidates_all = rp_frame[cutoff_keep]
                        ehat_candidates_all = ehat_frame[cutoff_keep]

                    for k0 in range(0, k_vectors_use.shape[0], k_chunk_use):
                        k1 = min(k_vectors_use.shape[0], k0 + k_chunk_use)
                        k_block = np.asarray(k_vectors_use[k0:k1], dtype=np.float64)
                        kh_block = np.asarray(khat[k0:k1], dtype=np.float64)
                        charge_phase = np.matmul(charge_position, k_block.T)
                        for d0 in range(0, int(rp_candidates_all.shape[0]), dipole_chunk_use):
                            d1 = min(int(rp_candidates_all.shape[0]), d0 + dipole_chunk_use)
                            rp_candidates = rp_candidates_all[d0:d1]
                            ehat_candidates = ehat_candidates_all[d0:d1]
                            dipole_phase = np.matmul(rp_candidates, k_block.T)
                            weights = np.matmul(ehat_candidates, kh_block.T)
                            phase_diff = dipole_phase - charge_phase[None, :]

                            out_re[k0:k1] += -zq * np.sum(weights * np.sin(phase_diff), axis=0)
                            out_im[k0:k1] += zq * np.sum(weights * np.cos(phase_diff), axis=0)
            last_used_dipole_count = int(np.count_nonzero(used_dipoles_mask))
            frame_dipole_counts[frame_index] = last_used_dipole_count
            if status_logger is not None:
                status_logger.charge_dipole(
                    frame_index + 1,
                    int(z.shape[0]),
                    int(k_vectors_use.shape[0]),
                    last_used_dipole_count,
                )

    scale = inv_dipole_count / float(frame_count) if normalize_by_frames else inv_dipole_count
    out_re *= scale
    out_im *= scale
    if status_logger is not None:
        status_logger.charge_dipole(frame_count, int(z.shape[0]), int(k_vectors_use.shape[0]), last_used_dipole_count, force=True)

    return out_re + 1j * out_im, frame_dipole_counts


def compute_charge_dipole_structure_factor_from_files(
    *,
    baseDir: str,
    charge_values_path: str,
    charge_coords_pattern: str,
    charge_coords_stride: int = 1,
    dipole_positions_pattern: str,
    dipole_positions_stride: int = 1,
    dipole_vectors_pattern: str,
    dipole_vectors_stride: int = 1,
    isotropic_output_file: str,
    directional_output_file: str,
    k_max_primary: float,
    k_max_secondary: float,
    k_max_tertiary: float,
    k_resolution_primary: float,
    k_resolution_secondary: float,
    k_resolution_tertiary: float,
    Lx: float,
    Ly: float,
    Lz: float,
    shell_width: float,
    cutoff_primary: float | None = None,
    cutoff_secondary: float | None = None,
    cutoff_tertiary: float | None = None,
    cell_size_primary: float | None = None,
    cell_size_secondary: float | None = None,
    cell_size_tertiary: float | None = None,
    cutoff: float | None = None,
    cell_size: float | None = None,
    num_trajectories: int | None = None,
    trajectory_selection: str | None = None,
    delete_residue_index: int | None = None,
    frame_chunk: int | None = None,
    charge_chunk: int | None = None,
    dipole_chunk: int | None = None,
    k_chunk: int | None = 64,
    frame_window: tuple[int, int | None, int] | None = None,
    charge_values_io_spec: dict[str, Any] | None = None,
    charge_coords_io_spec: dict[str, Any] | None = None,
    dipole_positions_io_spec: dict[str, Any] | None = None,
    dipole_vectors_io_spec: dict[str, Any] | None = None,
    isotropic_output_io_spec: dict[str, Any] | None = None,
    directional_output_io_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overall_start = time.time()
    k_max_primary, k_max_secondary, k_max_tertiary, k_resolution_primary, k_resolution_secondary, k_resolution_tertiary = _resolve_three_tier_k_parameters(
        k_max_primary,
        k_max_secondary,
        k_max_tertiary,
        k_resolution_primary,
        k_resolution_secondary,
        k_resolution_tertiary,
    )
    box = np.array([float(Lx), float(Ly), float(Lz)], dtype=np.float64)
    if np.any(box <= 0):
        raise ValueError("Lx, Ly, and Lz must all be positive.")
    if float(k_max_primary) < 0 or float(k_max_secondary) <= 0 or float(k_max_tertiary) <= 0:
        raise ValueError("k max values must satisfy k_max_primary >= 0, k_max_secondary > 0, and k_max_tertiary > 0.")
    if float(k_max_primary) > float(k_max_secondary) or float(k_max_secondary) > float(k_max_tertiary):
        raise ValueError("k max values must satisfy k_max_primary <= k_max_secondary <= k_max_tertiary.")
    if float(k_resolution_primary) <= 0 or float(k_resolution_secondary) <= 0 or float(k_resolution_tertiary) <= 0:
        raise ValueError("k resolution values must be positive.")
    if int(charge_coords_stride) <= 0 or int(dipole_positions_stride) <= 0 or int(dipole_vectors_stride) <= 0:
        raise ValueError("Charge-dipole coordinate strides must be positive integers.")
    cutoff_primary, cutoff_secondary, cutoff_tertiary = _resolve_three_tier_cutoffs(
        cutoff_primary,
        cutoff_secondary,
        cutoff_tertiary,
        legacy_cutoff=cutoff,
    )
    cell_size_primary, cell_size_secondary, cell_size_tertiary = _resolve_three_tier_cell_sizes(
        cutoff_primary,
        cutoff_secondary,
        cutoff_tertiary,
        cell_size_primary,
        cell_size_secondary,
        cell_size_tertiary,
        legacy_cell_size=cell_size,
    )
    if float(shell_width) <= 0:
        raise ValueError("shell_width must be positive.")
    for label, value in [
        ("frame_chunk", frame_chunk),
        ("charge_chunk", charge_chunk),
        ("dipole_chunk", dipole_chunk),
        ("k_chunk", k_chunk),
    ]:
        if value is not None and int(value) <= 0:
            raise ValueError(f"{label} must be positive when provided.")

    for label, pattern in [
        ("Charge Coordinates Path", charge_coords_pattern),
        ("Dipole Positions Path", dipole_positions_pattern),
        ("Dipole Vectors Path", dipole_vectors_pattern),
    ]:
        is_valid, error_message = validate_path_pattern(pattern)
        if not is_valid:
            raise ValueError(f"Invalid {label}: {error_message}")

    charge_values_file = _resolve_data_path(baseDir, charge_values_path)
    if not os.path.isfile(charge_values_file):
        raise FileNotFoundError(f"Charge values file not found: {charge_values_file}")

    charge_coordinate_files = _discover_coordinate_files(baseDir, charge_coords_pattern)
    dipole_position_files = _discover_coordinate_files(baseDir, dipole_positions_pattern)
    dipole_vector_files = _discover_coordinate_files(baseDir, dipole_vectors_pattern)
    file_counts = {
        "charge coordinate": len(charge_coordinate_files),
        "dipole position": len(dipole_position_files),
        "dipole vector": len(dipole_vector_files),
    }
    if len(set(file_counts.values())) != 1:
        raise ValueError(
            "Charge-dipole structure-factor inputs must resolve to the same number of files. "
            f"Discovered counts: {file_counts}"
        )
    trajectory_indices = _selected_trajectory_indices(len(charge_coordinate_files), num_trajectories, trajectory_selection)
    selected_file_indices = [index + 1 for index in trajectory_indices]
    charge_coordinate_files = [charge_coordinate_files[index] for index in trajectory_indices]
    dipole_position_files = [dipole_position_files[index] for index in trajectory_indices]
    dipole_vector_files = [dipole_vector_files[index] for index in trajectory_indices]

    k_magnitudes, k_vectors_array = charge_dipole_k_vectors_three_tier(
        float(Lx),
        float(Ly),
        float(Lz),
        float(k_max_primary),
        float(k_max_secondary),
        float(k_max_tertiary),
        res1=float(k_resolution_primary),
        res2=float(k_resolution_secondary),
        res3=float(k_resolution_tertiary),
    )
    if k_vectors_array.shape[0] == 0:
        raise ValueError("No charge-dipole k vectors were generated. Increase k max values or verify the box lengths.")
    isotropic_output_path = _resolve_output_path(baseDir, isotropic_output_file)
    directional_output_path = _resolve_output_path(baseDir, directional_output_file)

    charge_values = _load_charge_values(charge_values_file, charge_values_io_spec, delete_residue_index)
    active_charge_count = _active_charge_site_count(charge_values)
    total_frames = 0
    per_file_stats: list[dict[str, Any]] = []
    isotropic_single_trajectory_reports: list[str] = []
    directional_single_trajectory_reports: list[str] = []
    directional_reports: list[dict[str, Any]] = []
    dipoles_count_files: list[str] = []

    print("=" * 60, flush=True)
    print("CHARGE-DIPOLE STRUCTURE FACTOR CALCULATION", flush=True)
    print("=" * 60, flush=True)
    print(f"Charge values path: {charge_values_path}", flush=True)
    print(f"Charge coordinates path: {charge_coords_pattern}", flush=True)
    print(f"Charge coordinates stride: {int(charge_coords_stride)}", flush=True)
    print(f"Dipole positions path: {dipole_positions_pattern}", flush=True)
    print(f"Dipole positions stride: {int(dipole_positions_stride)}", flush=True)
    print(f"Dipole vectors path: {dipole_vectors_pattern}", flush=True)
    print(f"Dipole vectors stride: {int(dipole_vectors_stride)}", flush=True)
    print(f"Isotropic output path: {isotropic_output_file}", flush=True)
    print(f"Directional output path: {directional_output_file}", flush=True)
    print(f"k max primary: {float(k_max_primary):.8f}", flush=True)
    print(f"k max secondary: {float(k_max_secondary):.8f}", flush=True)
    print(f"k max tertiary: {float(k_max_tertiary):.8f}", flush=True)
    print(f"k resolution primary: {float(k_resolution_primary):.8f}", flush=True)
    print(f"k resolution secondary: {float(k_resolution_secondary):.8f}", flush=True)
    print(f"k resolution tertiary: {float(k_resolution_tertiary):.8f}", flush=True)
    print(f"directional k vectors: {int(k_vectors_array.shape[0])}", flush=True)
    print(f"shell width: {float(shell_width):.8f}", flush=True)
    for label, value in [
        ("cutoff 1", cutoff_primary),
        ("cutoff 2", cutoff_secondary),
        ("cutoff 3", cutoff_tertiary),
    ]:
        print(f"{label}: {float(value):.8f}" if value is not None else f"{label}: none", flush=True)
    for label, value in [
        ("cell size 1", cell_size_primary),
        ("cell size 2", cell_size_secondary),
        ("cell size 3", cell_size_tertiary),
    ]:
        print(f"{label}: {float(value):.8f}" if value is not None else f"{label}: none", flush=True)
    print(
        "chunk sizes: "
        f"frame={frame_chunk if frame_chunk is not None else 'off'}, "
        f"charges={charge_chunk if charge_chunk is not None else 'off'}, "
        f"dipoles={dipole_chunk if dipole_chunk is not None else 'off'}, "
        f"k={k_chunk if k_chunk is not None else 'off'}",
        flush=True,
    )
    print(f"delete residue index: {delete_residue_index if delete_residue_index is not None else 'none'}", flush=True)
    print(f"active nonzero charge sites: {active_charge_count}", flush=True)
    if frame_window is None:
        print("trajectory desired length: full trajectory", flush=True)
    else:
        print(
            f"trajectory desired length: range({frame_window[0]}, {frame_window[1]}, {frame_window[2]})",
            flush=True,
        )
    print(f"file sets used: {len(selected_file_indices)}", flush=True)
    if trajectory_selection:
        print(f"trajectory selection: {trajectory_indices}", flush=True)

    for file_index, rq_file, rp_file, mu_file in zip(
        selected_file_indices,
        charge_coordinate_files,
        dipole_position_files,
        dipole_vector_files,
        strict=False,
    ):
        trajectory_start = time.time()
        print(f"\nProcessing charge-dipole file set {file_index} ({len(per_file_stats) + 1}/{len(charge_coordinate_files)})", flush=True)
        print(f"  charge coordinates: {rq_file}", flush=True)
        print(f"  dipole positions:   {rp_file}", flush=True)
        print(f"  dipole vectors:     {mu_file}", flush=True)

        charge_position_array = load_numeric_array(
            rq_file,
            charge_coords_io_spec,
            default_mode="text",
            default_precision="double",
        )
        dipole_position_array = load_numeric_array(
            rp_file,
            dipole_positions_io_spec,
            default_mode="text",
            default_precision="double",
        )
        dipole_vector_array = load_numeric_array(
            mu_file,
            dipole_vectors_io_spec,
            default_mode="text",
            default_precision="double",
        )

        charge_positions = _prepare_xyz_frames(
            charge_position_array,
            name=str(rq_file),
            delete_index=delete_residue_index,
        )
        dipole_positions = _prepare_xyz_frames(
            dipole_position_array,
            name=str(rp_file),
        )
        dipole_vectors = _prepare_xyz_frames(
            dipole_vector_array,
            name=str(mu_file),
        )

        charge_positions = _apply_frame_stride(charge_positions, charge_coords_stride, label="Charge Coordinates Path")
        dipole_positions = _apply_frame_stride(dipole_positions, dipole_positions_stride, label="Dipole Positions Path")
        dipole_vectors = _apply_frame_stride(dipole_vectors, dipole_vectors_stride, label="Dipole Vectors Path")
        charge_positions = _apply_frame_window(charge_positions, frame_window)
        dipole_positions = _apply_frame_window(dipole_positions, frame_window)
        dipole_vectors = _apply_frame_window(dipole_vectors, frame_window)

        frame_counts = (
            int(charge_positions.shape[0]),
            int(dipole_positions.shape[0]),
            int(dipole_vectors.shape[0]),
        )
        frame_count = min(frame_counts)
        if frame_count <= 0:
            raise ValueError(f"No frames available after preprocessing for file set {file_index}.")
        if len(set(frame_counts)) != 1:
            print(
                f"  frame mismatch charge/dipole-pos/dipole-vec={frame_counts}, using min={frame_count}",
                flush=True,
            )

        charge_positions = np.ascontiguousarray(charge_positions[:frame_count], dtype=np.float64)
        dipole_positions = np.ascontiguousarray(dipole_positions[:frame_count], dtype=np.float64)
        dipole_vectors = np.ascontiguousarray(dipole_vectors[:frame_count], dtype=np.float64)
        dipole_directions = _normalize_vectors(dipole_vectors, name=str(mu_file))

        if int(charge_positions.shape[1]) != int(charge_values.shape[0]):
            raise ValueError(
                "Charge-value count mismatch for file set "
                f"{file_index}: len(Z)={int(charge_values.shape[0])} but charge positions contain "
                f"{int(charge_positions.shape[1])} sites."
            )

        directional_values = np.zeros(k_vectors_array.shape[0], dtype=np.complex128)
        frame_dipole_counts = np.zeros((frame_count, 3), dtype=np.int64)
        tier_masks = _three_tier_masks_from_magnitudes(
            k_magnitudes,
            k_max_primary,
            k_max_secondary,
            k_max_tertiary,
        )
        for tier_index, (mask, tier_cutoff, tier_cell_size) in enumerate(zip(
            tier_masks,
            (cutoff_primary, cutoff_secondary, cutoff_tertiary),
            (cell_size_primary, cell_size_secondary, cell_size_tertiary),
            strict=False,
        )):
            if not np.any(mask):
                continue
            tier_values, tier_counts = charge_dipole_structure_factor_directional(
                k_vectors_array[mask],
                charge_positions,
                charge_values,
                dipole_positions,
                dipole_directions,
                cutoff=tier_cutoff,
                cell_size=tier_cell_size,
                box=box,
                frame_chunk=frame_chunk,
                charge_chunk=charge_chunk,
                dipole_chunk=dipole_chunk,
                k_chunk=k_chunk,
                normalize_by_frames=False,
                status_logger=None,
            )
            directional_values[mask] = tier_values
            frame_dipole_counts[:, tier_index] = np.asarray(tier_counts, dtype=np.int64)
        dipoles_count_file = _write_trajectory_dipoles_count_file(
            isotropic_output_path,
            file_index,
            frame_dipole_counts,
            cutoff_primary,
            cutoff_secondary,
            cutoff_tertiary,
        )
        dipoles_count_files.append(dipoles_count_file)
        total_frames += int(frame_count)
        elapsed_s = float(time.time() - trajectory_start)
        directional_raw_single = np.asarray(directional_values, dtype=np.complex128)
        per_file_stats.append(
            {
                "charge_coordinate_file": rq_file,
                "dipole_position_file": rp_file,
                "dipole_vector_file": mu_file,
                "frame_count": int(frame_count),
                "dipole_count": int(dipole_positions.shape[1]),
                "dipoles_count_file": dipoles_count_file,
                "elapsed_s": elapsed_s,
            }
        )
        k_shell_single, isotropic_single, shell_counts_single = _shell_average_complex(
            k_magnitudes,
            directional_raw_single,
            float(shell_width),
        )
        isotropic_single_trajectory_reports.append(
            _write_single_trajectory_output(
                isotropic_output_path,
                file_index,
                np.column_stack((k_shell_single, isotropic_single.real, isotropic_single.imag, shell_counts_single)),
                isotropic_output_io_spec,
                default_mode="text",
                default_precision="double",
            )
        )
        directional_report_path = _write_single_trajectory_output(
            directional_output_path,
            file_index,
            np.column_stack((k_vectors_array, k_magnitudes, directional_raw_single.real, directional_raw_single.imag)),
            directional_output_io_spec,
            default_mode="text",
            default_precision="double",
        )
        directional_single_trajectory_reports.append(directional_report_path)
        directional_reports.append(
            {
                "trajectory_index": int(file_index),
                "file_path": str(rq_file),
                "frame_count": int(frame_count),
                "coord_count": int(dipole_positions.shape[1]),
                "elapsed_s": elapsed_s,
                "output_file": directional_report_path,
            }
        )
        del charge_position_array, dipole_position_array, dipole_vector_array
        del charge_positions, dipole_positions, dipole_vectors, dipole_directions
        del directional_values, directional_raw_single, frame_dipole_counts
        print(f"  accumulated frames: {total_frames}", flush=True)

    if total_frames <= 0:
        raise ValueError("No frames were accumulated for the charge-dipole structure-factor calculation.")

    directional_components = _sum_saved_trajectory_outputs(
        reports=directional_reports,
        value_columns=(4, 5),
        io_spec=directional_output_io_spec,
        default_mode="text",
        default_precision="double",
        expected_prefix=np.column_stack((k_vectors_array, k_magnitudes)),
        expected_prefix_columns=(0, 1, 2, 3),
    )
    directional_raw_total = directional_components[:, 0] + 1j * directional_components[:, 1]
    k_shell, isotropic_shell, shell_counts = _shell_average_complex(k_magnitudes, directional_raw_total, float(shell_width))

    isotropic_output_array = np.column_stack((k_shell, isotropic_shell.real, isotropic_shell.imag, shell_counts))
    directional_output_array = np.column_stack((k_vectors_array, k_magnitudes, directional_raw_total.real, directional_raw_total.imag))

    save_numeric_array(
        isotropic_output_path,
        isotropic_output_array,
        isotropic_output_io_spec,
        default_mode="text",
        default_precision="double",
    )
    save_numeric_array(
        directional_output_path,
        directional_output_array,
        directional_output_io_spec,
        default_mode="text",
        default_precision="double",
    )
    overall_elapsed = time.time() - overall_start
    print(f"\nSaved isotropic charge-dipole output: {isotropic_output_path}", flush=True)
    print(f"Saved directional charge-dipole output: {directional_output_path}", flush=True)
    print(f"Saved dipoles count outputs: {_dipoles_count_directory_for_output(isotropic_output_path)}", flush=True)
    print(f"total file sets used: {len(per_file_stats)}", flush=True)
    print(f"total frames used: {total_frames}", flush=True)
    print(f"total elapsed time: {overall_elapsed:.2f} s", flush=True)

    return {
        "mode": "directional",
        "directional_k_count": int(k_vectors_array.shape[0]),
        "isotropic_shell_count": int(k_shell.shape[0]),
        "k_vectors": k_vectors_array,
        "k_magnitudes": k_magnitudes,
        "directional_values": directional_raw_total,
        "k_shell": k_shell,
        "isotropic_values": isotropic_shell,
        "shell_counts": shell_counts,
        "total_frames": int(total_frames),
        "isotropic_output_file": isotropic_output_path,
        "directional_output_file": directional_output_path,
        "dipoles_count_files": dipoles_count_files,
        "per_file_stats": per_file_stats,
        "isotropic_single_trajectory_reports": isotropic_single_trajectory_reports,
        "directional_single_trajectory_reports": directional_single_trajectory_reports,
    }


def _compute_single_charge_dipole_isotropic(args: tuple[Any, ...]) -> dict[str, Any]:
    (
        file_index,
        charge_values_path,
        rq_file,
        rp_file,
        mu_file,
        charge_values,
        k_magnitudes,
        box,
        k_max_primary,
        k_max_secondary,
        k_max_tertiary,
        k_resolution_primary,
        k_resolution_secondary,
        k_resolution_tertiary,
        cutoff_primary,
        cutoff_secondary,
        cutoff_tertiary,
        cell_size_primary,
        cell_size_secondary,
        cell_size_tertiary,
        charge_coords_stride,
        dipole_positions_stride,
        dipole_vectors_stride,
        delete_residue_index,
        frame_chunk,
        charge_chunk,
        dipole_chunk,
        k_chunk,
        frame_window,
        charge_coords_io_spec,
        dipole_positions_io_spec,
        dipole_vectors_io_spec,
        isotropic_output_path,
        isotropic_output_io_spec,
    ) = args

    try:
        trajectory_start = time.time()

        print(f"\nProcessing isotropic charge-dipole file set {file_index}", flush=True)
        print(f"  charge coordinates: {rq_file}", flush=True)
        print(f"  dipole positions:   {rp_file}", flush=True)
        print(f"  dipole vectors:     {mu_file}", flush=True)

        charge_position_array = load_numeric_array(
            rq_file,
            charge_coords_io_spec,
            default_mode="text",
            default_precision="double",
        )
        dipole_position_array = load_numeric_array(
            rp_file,
            dipole_positions_io_spec,
            default_mode="text",
            default_precision="double",
        )
        dipole_vector_array = load_numeric_array(
            mu_file,
            dipole_vectors_io_spec,
            default_mode="text",
            default_precision="double",
        )

        charge_positions = _prepare_xyz_frames(
            charge_position_array,
            name=str(rq_file),
            delete_index=delete_residue_index,
        )
        dipole_positions = _prepare_xyz_frames(
            dipole_position_array,
            name=str(rp_file),
        )
        dipole_vectors = _prepare_xyz_frames(
            dipole_vector_array,
            name=str(mu_file),
        )

        charge_positions = _apply_frame_stride(charge_positions, charge_coords_stride, label="Charge Coordinates Path")
        dipole_positions = _apply_frame_stride(dipole_positions, dipole_positions_stride, label="Dipole Positions Path")
        dipole_vectors = _apply_frame_stride(dipole_vectors, dipole_vectors_stride, label="Dipole Vectors Path")
        charge_positions = _apply_frame_window(charge_positions, frame_window)
        dipole_positions = _apply_frame_window(dipole_positions, frame_window)
        dipole_vectors = _apply_frame_window(dipole_vectors, frame_window)

        frame_counts = (
            int(charge_positions.shape[0]),
            int(dipole_positions.shape[0]),
            int(dipole_vectors.shape[0]),
        )
        frame_count = min(frame_counts)
        if frame_count <= 0:
            raise ValueError(f"No frames available after preprocessing for file set {file_index}.")
        if len(set(frame_counts)) != 1:
            print(
                f"  frame mismatch charge/dipole-pos/dipole-vec={frame_counts}, using min={frame_count}",
                flush=True,
            )

        charge_positions = np.ascontiguousarray(charge_positions[:frame_count], dtype=np.float64)
        dipole_positions = np.ascontiguousarray(dipole_positions[:frame_count], dtype=np.float64)
        dipole_vectors = np.ascontiguousarray(dipole_vectors[:frame_count], dtype=np.float64)
        dipole_directions = _normalize_vectors(dipole_vectors, name=str(mu_file))
        charge_values_array = np.asarray(charge_values, dtype=np.float64)
        active_charge_count = _active_charge_site_count(charge_values_array)
        k_magnitudes_array = np.asarray(k_magnitudes, dtype=np.float64)
        box_array = np.asarray(box, dtype=np.float64)

        if int(charge_positions.shape[1]) != int(charge_values_array.shape[0]):
            raise ValueError(
                "Charge-value count mismatch for file set "
                f"{file_index}: len(Z)={int(charge_values_array.shape[0])} but charge positions contain "
                f"{int(charge_positions.shape[1])} sites."
            )

        isotropic_values_total = np.zeros(k_magnitudes_array.shape[0], dtype=np.float64)
        frame_dipole_counts = np.zeros((frame_count, 3), dtype=np.int64)
        tier_count_sums = np.zeros(3, dtype=np.float64)
        tier_count_frames = np.zeros(3, dtype=np.int64)
        tier_masks = _three_tier_masks_from_magnitudes(
            k_magnitudes_array,
            k_max_primary,
            k_max_secondary,
            k_max_tertiary,
        )
        for tier_index, (mask, tier_cutoff, tier_cell_size) in enumerate(zip(
            tier_masks,
            (cutoff_primary, cutoff_secondary, cutoff_tertiary),
            (cell_size_primary, cell_size_secondary, cell_size_tertiary),
            strict=False,
        )):
            if not np.any(mask):
                continue
            tier_total, tier_counts = charge_dipole_structure_factor_isotropic(
                k_magnitudes_array[mask],
                charge_positions,
                charge_values_array,
                dipole_positions,
                dipole_directions,
                cutoff=tier_cutoff,
                cell_size=tier_cell_size,
                box=box_array,
                frame_chunk=frame_chunk,
                charge_chunk=charge_chunk,
                dipole_chunk=dipole_chunk,
                k_chunk=k_chunk,
                normalize_by_frames=False,
                status_logger=None,
            )
            isotropic_values_total[mask] = tier_total
            tier_count_sums[tier_index] += float(np.sum(tier_counts))
            tier_count_frames[tier_index] += int(np.asarray(tier_counts).shape[0])
            frame_dipole_counts[:, tier_index] = np.asarray(tier_counts, dtype=np.int64)

        tier_count_averages = np.full(3, np.nan, dtype=np.float64)
        np.divide(
            tier_count_sums,
            tier_count_frames,
            out=tier_count_averages,
            where=tier_count_frames > 0,
        )
        tier_count_columns = np.tile(tier_count_averages, (k_magnitudes_array.shape[0], 1))
        report_header = _charge_dipole_isotropic_output_header(
            charge_values_path=str(charge_values_path),
            charge_coords_pattern=str(rq_file),
            charge_coords_stride=int(charge_coords_stride),
            dipole_positions_pattern=str(rp_file),
            dipole_positions_stride=int(dipole_positions_stride),
            dipole_vectors_pattern=str(mu_file),
            dipole_vectors_stride=int(dipole_vectors_stride),
            k_max_primary=float(k_max_primary),
            k_max_secondary=float(k_max_secondary),
            k_max_tertiary=float(k_max_tertiary),
            k_resolution_primary=float(k_resolution_primary),
            k_resolution_secondary=float(k_resolution_secondary),
            k_resolution_tertiary=float(k_resolution_tertiary),
            Lx=float(box_array[0]),
            Ly=float(box_array[1]),
            Lz=float(box_array[2]),
            cutoff_primary=cutoff_primary,
            cutoff_secondary=cutoff_secondary,
            cutoff_tertiary=cutoff_tertiary,
            cell_size_primary=cell_size_primary,
            cell_size_secondary=cell_size_secondary,
            cell_size_tertiary=cell_size_tertiary,
            charge_count=active_charge_count,
            dipole_count=int(dipole_positions.shape[1]),
            frame_count=int(frame_count),
            trajectory_count=1,
            delete_residue_index=delete_residue_index,
            frame_window=frame_window,
            trajectory_selection=str(file_index),
        )

        report_path = _write_single_trajectory_output(
            isotropic_output_path,
            int(file_index),
            np.column_stack(
                (
                    k_magnitudes_array,
                    np.asarray(isotropic_values_total, dtype=np.float64),
                    tier_count_columns,
                )
            ),
            isotropic_output_io_spec,
            default_mode="text",
            default_precision="double",
            header=report_header,
        )
        dipoles_count_file = _write_trajectory_dipoles_count_file(
            isotropic_output_path,
            int(file_index),
            frame_dipole_counts,
            cutoff_primary,
            cutoff_secondary,
            cutoff_tertiary,
        )
        elapsed_s = float(time.time() - trajectory_start)
        print(f"  completed isotropic charge-dipole file set {file_index} in {elapsed_s:.2f} s", flush=True)
        return {
            "trajectory_index": int(file_index),
            "charge_coordinate_file": rq_file,
            "dipole_position_file": rp_file,
            "dipole_vector_file": mu_file,
            "frame_count": int(frame_count),
            "dipole_count": int(dipole_positions.shape[1]),
            "dipoles_count_file": dipoles_count_file,
            "dipoles_in_cutoff_tier_sums": np.asarray(tier_count_sums, dtype=np.float64),
            "dipoles_in_cutoff_tier_frames": np.asarray(tier_count_frames, dtype=np.int64),
            "elapsed_s": elapsed_s,
            "output_file": report_path,
            "error": "",
        }
    except Exception as exc:
        return {
            "trajectory_index": int(file_index),
            "charge_coordinate_file": rq_file,
            "dipole_position_file": rp_file,
            "dipole_vector_file": mu_file,
            "error": str(exc),
        }


def compute_charge_dipole_structure_factor_isotropic_from_files(
    *,
    baseDir: str,
    charge_values_path: str,
    charge_coords_pattern: str,
    charge_coords_stride: int = 1,
    dipole_positions_pattern: str,
    dipole_positions_stride: int = 1,
    dipole_vectors_pattern: str,
    dipole_vectors_stride: int = 1,
    isotropic_output_file: str,
    k_max_primary: float,
    k_max_secondary: float,
    k_max_tertiary: float,
    k_resolution_primary: float,
    k_resolution_secondary: float,
    k_resolution_tertiary: float,
    Lx: float,
    Ly: float,
    Lz: float,
    cutoff_primary: float | None = None,
    cutoff_secondary: float | None = None,
    cutoff_tertiary: float | None = None,
    cell_size_primary: float | None = None,
    cell_size_secondary: float | None = None,
    cell_size_tertiary: float | None = None,
    cutoff: float | None = None,
    cell_size: float | None = None,
    num_trajectories: int | None = None,
    trajectory_selection: str | None = None,
    delete_residue_index: int | None = None,
    frame_chunk: int | None = None,
    charge_chunk: int | None = None,
    dipole_chunk: int | None = None,
    k_chunk: int | None = 64,
    max_workers: int = 1,
    frame_window: tuple[int, int | None, int] | None = None,
    charge_values_io_spec: dict[str, Any] | None = None,
    charge_coords_io_spec: dict[str, Any] | None = None,
    dipole_positions_io_spec: dict[str, Any] | None = None,
    dipole_vectors_io_spec: dict[str, Any] | None = None,
    isotropic_output_io_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overall_start = time.time()
    k_max_primary, k_max_secondary, k_max_tertiary, k_resolution_primary, k_resolution_secondary, k_resolution_tertiary = _resolve_three_tier_k_parameters(
        k_max_primary,
        k_max_secondary,
        k_max_tertiary,
        k_resolution_primary,
        k_resolution_secondary,
        k_resolution_tertiary,
    )
    box = np.array([float(Lx), float(Ly), float(Lz)], dtype=np.float64)
    if np.any(box <= 0):
        raise ValueError("Lx, Ly, and Lz must all be positive.")
    if float(k_max_primary) < 0 or float(k_max_secondary) <= 0 or float(k_max_tertiary) <= 0:
        raise ValueError("k max values must satisfy k_max_primary >= 0, k_max_secondary > 0, and k_max_tertiary > 0.")
    if float(k_max_primary) > float(k_max_secondary) or float(k_max_secondary) > float(k_max_tertiary):
        raise ValueError("k max values must satisfy k_max_primary <= k_max_secondary <= k_max_tertiary.")
    if float(k_resolution_primary) <= 0 or float(k_resolution_secondary) <= 0 or float(k_resolution_tertiary) <= 0:
        raise ValueError("k resolution values must be positive.")
    if int(charge_coords_stride) <= 0 or int(dipole_positions_stride) <= 0 or int(dipole_vectors_stride) <= 0:
        raise ValueError("Charge-dipole coordinate strides must be positive integers.")
    cutoff_primary, cutoff_secondary, cutoff_tertiary = _resolve_three_tier_cutoffs(
        cutoff_primary,
        cutoff_secondary,
        cutoff_tertiary,
        legacy_cutoff=cutoff,
    )
    cell_size_primary, cell_size_secondary, cell_size_tertiary = _resolve_three_tier_cell_sizes(
        cutoff_primary,
        cutoff_secondary,
        cutoff_tertiary,
        cell_size_primary,
        cell_size_secondary,
        cell_size_tertiary,
        legacy_cell_size=cell_size,
    )
    for label, value in [
        ("frame_chunk", frame_chunk),
        ("charge_chunk", charge_chunk),
        ("dipole_chunk", dipole_chunk),
        ("k_chunk", k_chunk),
    ]:
        if value is not None and int(value) <= 0:
            raise ValueError(f"{label} must be positive when provided.")
    if int(max_workers) <= 0:
        raise ValueError("max_workers must be positive.")

    for label, pattern in [
        ("Charge Coordinates Path", charge_coords_pattern),
        ("Dipole Positions Path", dipole_positions_pattern),
        ("Dipole Vectors Path", dipole_vectors_pattern),
    ]:
        is_valid, error_message = validate_path_pattern(pattern)
        if not is_valid:
            raise ValueError(f"Invalid {label}: {error_message}")

    charge_values_file = _resolve_data_path(baseDir, charge_values_path)
    if not os.path.isfile(charge_values_file):
        raise FileNotFoundError(f"Charge values file not found: {charge_values_file}")

    charge_coordinate_files = _discover_coordinate_files(baseDir, charge_coords_pattern)
    dipole_position_files = _discover_coordinate_files(baseDir, dipole_positions_pattern)
    dipole_vector_files = _discover_coordinate_files(baseDir, dipole_vectors_pattern)
    file_counts = {
        "charge coordinate": len(charge_coordinate_files),
        "dipole position": len(dipole_position_files),
        "dipole vector": len(dipole_vector_files),
    }
    if len(set(file_counts.values())) != 1:
        raise ValueError(
            "Charge-dipole structure-factor inputs must resolve to the same number of files. "
            f"Discovered counts: {file_counts}"
        )
    trajectory_indices = _selected_trajectory_indices(len(charge_coordinate_files), num_trajectories, trajectory_selection)
    selected_file_indices = [index + 1 for index in trajectory_indices]
    charge_coordinate_files = [charge_coordinate_files[index] for index in trajectory_indices]
    dipole_position_files = [dipole_position_files[index] for index in trajectory_indices]
    dipole_vector_files = [dipole_vector_files[index] for index in trajectory_indices]

    k_magnitudes = charge_dipole_k_magnitudes_by_resolution_three_tier(
        float(Lx),
        float(Ly),
        float(Lz),
        float(k_max_primary),
        float(k_max_secondary),
        float(k_max_tertiary),
        float(k_resolution_primary),
        float(k_resolution_secondary),
        float(k_resolution_tertiary),
    )
    if k_magnitudes.shape[0] == 0:
        raise ValueError("No isotropic charge-dipole k magnitudes were generated. Increase k max values or verify the box lengths.")
    isotropic_output_path = _resolve_output_path(baseDir, isotropic_output_file)

    charge_values = _load_charge_values(charge_values_file, charge_values_io_spec, delete_residue_index)
    active_charge_count = _active_charge_site_count(charge_values)
    dipoles_in_cutoff_tier_sums = np.zeros(3, dtype=np.float64)
    dipoles_in_cutoff_tier_frames = np.zeros(3, dtype=np.int64)
    total_frames = 0
    per_file_stats: list[dict[str, Any]] = []
    isotropic_reports: list[dict[str, Any]] = []
    isotropic_single_trajectory_reports: list[str] = []
    dipoles_count_files: list[str] = []

    print("=" * 60, flush=True)
    print("CHARGE-DIPOLE ISOTROPIC STRUCTURE FACTOR CALCULATION", flush=True)
    print("=" * 60, flush=True)
    print(f"Charge values path: {charge_values_path}", flush=True)
    print(f"Charge coordinates path: {charge_coords_pattern}", flush=True)
    print(f"Charge coordinates stride: {int(charge_coords_stride)}", flush=True)
    print(f"Dipole positions path: {dipole_positions_pattern}", flush=True)
    print(f"Dipole positions stride: {int(dipole_positions_stride)}", flush=True)
    print(f"Dipole vectors path: {dipole_vectors_pattern}", flush=True)
    print(f"Dipole vectors stride: {int(dipole_vectors_stride)}", flush=True)
    print(f"Isotropic output path: {isotropic_output_file}", flush=True)
    print(f"k max primary: {float(k_max_primary):.8f}", flush=True)
    print(f"k max secondary: {float(k_max_secondary):.8f}", flush=True)
    print(f"k max tertiary: {float(k_max_tertiary):.8f}", flush=True)
    print(f"k resolution primary: {float(k_resolution_primary):.8f}", flush=True)
    print(f"k resolution secondary: {float(k_resolution_secondary):.8f}", flush=True)
    print(f"k resolution tertiary: {float(k_resolution_tertiary):.8f}", flush=True)
    print(f"isotropic k values: {int(k_magnitudes.shape[0])}", flush=True)
    for label, value in [
        ("cutoff 1", cutoff_primary),
        ("cutoff 2", cutoff_secondary),
        ("cutoff 3", cutoff_tertiary),
    ]:
        print(f"{label}: {float(value):.8f}" if value is not None else f"{label}: none", flush=True)
    for label, value in [
        ("cell size 1", cell_size_primary),
        ("cell size 2", cell_size_secondary),
        ("cell size 3", cell_size_tertiary),
    ]:
        print(f"{label}: {float(value):.8f}" if value is not None else f"{label}: none", flush=True)
    print(
        "chunk sizes: "
        f"frame={frame_chunk if frame_chunk is not None else 'off'}, "
        f"charges={charge_chunk if charge_chunk is not None else 'off'}, "
        f"dipoles={dipole_chunk if dipole_chunk is not None else 'off'}, "
        f"k={k_chunk if k_chunk is not None else 'off'}",
        flush=True,
    )
    print(f"max workers: {int(max_workers)}", flush=True)
    print(f"delete residue index: {delete_residue_index if delete_residue_index is not None else 'none'}", flush=True)
    print(f"active nonzero charge sites: {active_charge_count}", flush=True)
    if frame_window is None:
        print("trajectory desired length: full trajectory", flush=True)
    else:
        print(
            f"trajectory desired length: range({frame_window[0]}, {frame_window[1]}, {frame_window[2]})",
            flush=True,
        )
    print(f"file sets used: {len(selected_file_indices)}", flush=True)
    if trajectory_selection:
        print(f"trajectory selection: {trajectory_indices}", flush=True)

    task_args = [
        (
            file_index,
            charge_values_path,
            rq_file,
            rp_file,
            mu_file,
            charge_values,
            k_magnitudes,
            box,
            k_max_primary,
            k_max_secondary,
            k_max_tertiary,
            k_resolution_primary,
            k_resolution_secondary,
            k_resolution_tertiary,
            cutoff_primary,
            cutoff_secondary,
            cutoff_tertiary,
            cell_size_primary,
            cell_size_secondary,
            cell_size_tertiary,
            charge_coords_stride,
            dipole_positions_stride,
            dipole_vectors_stride,
            delete_residue_index,
            frame_chunk,
            charge_chunk,
            dipole_chunk,
            k_chunk,
            frame_window,
            charge_coords_io_spec,
            dipole_positions_io_spec,
            dipole_vectors_io_spec,
            isotropic_output_path,
            isotropic_output_io_spec,
        )
        for file_index, rq_file, rp_file, mu_file in zip(
            selected_file_indices,
            charge_coordinate_files,
            dipole_position_files,
            dipole_vector_files,
            strict=False,
        )
    ]

    worker_results: list[dict[str, Any]] = []
    if int(max_workers) > 1 and len(task_args) > 1:
        with ProcessPoolExecutor(max_workers=min(int(max_workers), len(task_args))) as executor:
            futures = [executor.submit(_compute_single_charge_dipole_isotropic, args) for args in task_args]
            for future in as_completed(futures):
                worker_results.append(future.result())
    else:
        for args in task_args:
            worker_results.append(_compute_single_charge_dipole_isotropic(args))

    worker_results.sort(key=lambda item: int(item.get("trajectory_index", 0)))
    failed = [item for item in worker_results if item.get("error")]
    if failed:
        for item in failed:
            print(
                f"Failed isotropic charge-dipole file set {item.get('trajectory_index')}: {item.get('error')}",
                flush=True,
            )
        raise RuntimeError(
            "Charge-dipole isotropic calculation did not finish all selected trajectories; "
            "final total Sqp(k) was not written."
        )

    for item in worker_results:
        report_path = str(item["output_file"])
        frame_count = int(item["frame_count"])
        dipoles_in_cutoff_tier_sums += np.asarray(item["dipoles_in_cutoff_tier_sums"], dtype=np.float64)
        dipoles_in_cutoff_tier_frames += np.asarray(item["dipoles_in_cutoff_tier_frames"], dtype=np.int64)
        dipoles_count_files.append(str(item["dipoles_count_file"]))
        total_frames += int(frame_count)
        isotropic_reports.append(
            {
                "trajectory_index": int(item["trajectory_index"]),
                "file_path": str(item["charge_coordinate_file"]),
                "frame_count": int(frame_count),
                "coord_count": int(item["dipole_count"]),
                "elapsed_s": float(item["elapsed_s"]),
                "output_file": report_path,
            }
        )
        isotropic_single_trajectory_reports.append(report_path)
        per_file_stats.append(
            {
                "charge_coordinate_file": item["charge_coordinate_file"],
                "dipole_position_file": item["dipole_position_file"],
                "dipole_vector_file": item["dipole_vector_file"],
                "frame_count": int(frame_count),
                "dipole_count": int(item["dipole_count"]),
                "dipoles_count_file": str(item["dipoles_count_file"]),
                "dipoles_in_cutoff_tier_sums": np.asarray(item["dipoles_in_cutoff_tier_sums"], dtype=np.float64),
                "dipoles_in_cutoff_tier_frames": np.asarray(item["dipoles_in_cutoff_tier_frames"], dtype=np.int64),
                "elapsed_s": float(item["elapsed_s"]),
                "isotropic_output_file": report_path,
            }
        )

    if total_frames <= 0:
        raise ValueError("No frames were accumulated for the charge-dipole structure-factor calculation.")

    isotropic_raw_total = _sum_saved_trajectory_outputs(
        reports=isotropic_reports,
        value_columns=(1,),
        io_spec=isotropic_output_io_spec,
        default_mode="text",
        default_precision="double",
        expected_prefix=k_magnitudes.reshape(-1, 1),
        expected_prefix_columns=(0,),
    )[:, 0]
    cutoff_count_averages = np.full(3, np.nan, dtype=np.float64)
    np.divide(
        dipoles_in_cutoff_tier_sums,
        dipoles_in_cutoff_tier_frames,
        out=cutoff_count_averages,
        where=dipoles_in_cutoff_tier_frames > 0,
    )

    cutoff_count_columns = np.tile(cutoff_count_averages, (k_magnitudes.shape[0], 1))
    isotropic_output_array = np.column_stack((k_magnitudes, isotropic_raw_total, cutoff_count_columns))
    dipole_counts = sorted({int(item["dipole_count"]) for item in per_file_stats})
    dipole_count_header: int | str
    if len(dipole_counts) == 1:
        dipole_count_header = dipole_counts[0]
    else:
        dipole_count_header = ", ".join(str(value) for value in dipole_counts)
    output_header = _charge_dipole_isotropic_output_header(
        charge_values_path=charge_values_path,
        charge_coords_pattern=charge_coords_pattern,
        charge_coords_stride=int(charge_coords_stride),
        dipole_positions_pattern=dipole_positions_pattern,
        dipole_positions_stride=int(dipole_positions_stride),
        dipole_vectors_pattern=dipole_vectors_pattern,
        dipole_vectors_stride=int(dipole_vectors_stride),
        k_max_primary=float(k_max_primary),
        k_max_secondary=float(k_max_secondary),
        k_max_tertiary=float(k_max_tertiary),
        k_resolution_primary=float(k_resolution_primary),
        k_resolution_secondary=float(k_resolution_secondary),
        k_resolution_tertiary=float(k_resolution_tertiary),
        Lx=float(Lx),
        Ly=float(Ly),
        Lz=float(Lz),
        cutoff_primary=cutoff_primary,
        cutoff_secondary=cutoff_secondary,
        cutoff_tertiary=cutoff_tertiary,
        cell_size_primary=cell_size_primary,
        cell_size_secondary=cell_size_secondary,
        cell_size_tertiary=cell_size_tertiary,
        charge_count=active_charge_count,
        dipole_count=dipole_count_header,
        frame_count=int(total_frames),
        trajectory_count=len(per_file_stats),
        delete_residue_index=delete_residue_index,
        frame_window=frame_window,
        trajectory_selection=trajectory_selection,
    )
    save_numeric_array(
        isotropic_output_path,
        isotropic_output_array,
        isotropic_output_io_spec,
        default_mode="text",
        default_precision="double",
        header=output_header,
    )

    overall_elapsed = time.time() - overall_start
    print(f"\nSaved isotropic charge-dipole output: {isotropic_output_path}", flush=True)
    print(f"Saved dipoles count outputs: {_dipoles_count_directory_for_output(isotropic_output_path)}", flush=True)
    print(f"total file sets used: {len(per_file_stats)}", flush=True)
    print(f"total frames used: {total_frames}", flush=True)
    print(f"total elapsed time: {overall_elapsed:.2f} s", flush=True)

    return {
        "mode": "isotropic",
        "isotropic_k_count": int(k_magnitudes.shape[0]),
        "k_magnitudes": k_magnitudes,
        "isotropic_values": isotropic_raw_total,
        "dipoles_in_cutoff_averages": cutoff_count_averages,
        "total_frames": int(total_frames),
        "isotropic_output_file": isotropic_output_path,
        "dipoles_count_files": dipoles_count_files,
        "per_file_stats": per_file_stats,
        "isotropic_single_trajectory_reports": isotropic_single_trajectory_reports,
    }


def _compute_single_trajectory_sk_report(args: tuple[Any, ...]) -> dict[str, Any]:
    (
        trajectory_index,
        file_path,
        k_vals,
        tier_masks,
        tier_cutoffs,
        tier_cell_sizes,
        box,
        input_io_spec,
        output_path,
        output_io_spec,
        coords_stride,
        frame_chunk,
        coord_chunk,
        k_chunk,
        frame_window,
        status_path,
        status_start_epoch,
    ) = args

    if not os.path.exists(file_path):
        return {"trajectory_index": int(trajectory_index), "file_path": file_path, "error": f"Missing file: {file_path}"}

    try:
        start_time = time.time()
        print(f"\nProcessing coordinate file: {file_path}", flush=True)
        array = load_numeric_array(
            file_path,
            input_io_spec,
            default_mode="text",
            default_precision="double",
        )
        coords, _ = _prepare_coordinate_array(array)
        coords = _apply_frame_stride(coords, coords_stride, label="Coordinate Path")
        coords = _apply_frame_window(coords, frame_window)
        frame_count = int(coords.shape[0])
        if frame_count <= 0:
            raise ValueError("No frames found in coordinate file.")

        k_array = np.asarray(k_vals, dtype=np.float64)
        sk_total = np.zeros(k_array.size, dtype=np.float64)
        status_logger = None if status_path is None or status_start_epoch is None else _StatusLogger(
            status_path,
            status_start_epoch,
            trajectory_index=int(trajectory_index),
        )
        print(f"  shape of r: {tuple(int(value) for value in coords.shape)}", flush=True)
        print(f"  shape of k vectors: {tuple(int(value) for value in k_array.shape)}", flush=True)

        for mask, tier_cutoff, tier_cell_size in zip(tier_masks, tier_cutoffs, tier_cell_sizes, strict=False):
            mask_array = np.asarray(mask, dtype=bool)
            if not np.any(mask_array):
                continue
            tier_values = isotropic_structure_factor_db_density(
                coords,
                k_array[mask_array],
                box=box,
                cutoff=tier_cutoff,
                cell_size=tier_cell_size,
                frame_chunk=int(frame_chunk),
                i_chunk=int(coord_chunk),
                j_chunk=int(coord_chunk),
                k_chunk=int(k_chunk),
                normalize_by_frames=False,
                status_logger=status_logger,
            )
            sk_total[mask_array] = np.asarray(tier_values, dtype=np.float64)

        report_path = _write_single_trajectory_output(
            output_path,
            int(trajectory_index),
            np.column_stack((k_array, sk_total / float(frame_count))),
            output_io_spec,
            default_mode="text",
            default_precision="double",
        )
        elapsed_s = float(time.time() - start_time)
        print(f"  completed in {elapsed_s:.2f} s", flush=True)
        return {
            "trajectory_index": int(trajectory_index),
            "file_path": file_path,
            "frame_count": frame_count,
            "coord_count": int(coords.shape[1]),
            "output_file": report_path,
            "elapsed_s": elapsed_s,
            "error": "",
        }
    except Exception as exc:
        return {"trajectory_index": int(trajectory_index), "file_path": file_path, "error": str(exc)}


def _compute_single_trajectory_directional_sk_report(args: tuple[Any, ...]) -> dict[str, Any]:
    (
        trajectory_index,
        file_path,
        k_vectors_array,
        tier_masks,
        tier_cutoffs,
        tier_cell_sizes,
        box,
        input_io_spec,
        output_path,
        output_io_spec,
        coords_stride,
        frame_chunk,
        coord_chunk,
        k_chunk,
        frame_window,
        status_path,
        status_start_epoch,
    ) = args

    if not os.path.exists(file_path):
        return {"trajectory_index": int(trajectory_index), "file_path": file_path, "error": f"Missing file: {file_path}"}

    try:
        start_time = time.time()
        print(f"\nProcessing coordinate file: {file_path}", flush=True)
        array = load_numeric_array(
            file_path,
            input_io_spec,
            default_mode="text",
            default_precision="double",
        )
        coords, _ = _prepare_coordinate_array(array)
        coords = _apply_frame_stride(coords, coords_stride, label="Coordinate Path")
        coords = _apply_frame_window(coords, frame_window)
        frame_count = int(coords.shape[0])
        if frame_count <= 0:
            raise ValueError("No frames found in coordinate file.")

        k_vectors_use = np.asarray(k_vectors_array, dtype=np.float64)
        k_magnitudes = np.linalg.norm(k_vectors_use, axis=1)
        sk_total = np.zeros(int(k_vectors_use.shape[0]), dtype=np.float64)
        status_logger = None if status_path is None or status_start_epoch is None else _StatusLogger(
            status_path,
            status_start_epoch,
            trajectory_index=int(trajectory_index),
        )
        print(f"  shape of r: {tuple(int(value) for value in coords.shape)}", flush=True)
        print(f"  directional k-vector count: {int(k_vectors_use.shape[0])}", flush=True)

        for mask, tier_cutoff, tier_cell_size in zip(tier_masks, tier_cutoffs, tier_cell_sizes, strict=False):
            mask_array = np.asarray(mask, dtype=bool)
            if not np.any(mask_array):
                continue
            tier_values = directional_structure_factor(
                coords,
                k_vectors_use[mask_array],
                box=box,
                cutoff=tier_cutoff,
                cell_size=tier_cell_size,
                frame_chunk=int(frame_chunk),
                atom_chunk=int(coord_chunk),
                k_chunk=int(k_chunk),
                normalize_by_frames=False,
                status_logger=status_logger,
            )
            sk_total[mask_array] = np.asarray(tier_values, dtype=np.float64)

        report_path = _write_single_trajectory_output(
            output_path,
            int(trajectory_index),
            np.column_stack((k_vectors_use, k_magnitudes, sk_total / float(frame_count))),
            output_io_spec,
            default_mode="text",
            default_precision="double",
        )
        elapsed_s = float(time.time() - start_time)
        print(f"  completed in {elapsed_s:.2f} s", flush=True)
        return {
            "trajectory_index": int(trajectory_index),
            "file_path": file_path,
            "frame_count": frame_count,
            "coord_count": int(coords.shape[1]),
            "output_file": report_path,
            "elapsed_s": elapsed_s,
            "error": "",
        }
    except Exception as exc:
        return {"trajectory_index": int(trajectory_index), "file_path": file_path, "error": str(exc)}


def compute_static_structure_factor_from_files(
    *,
    baseDir: str,
    coords_pattern: str,
    coords_stride: int = 1,
    output_file: str,
    k_max_primary: float,
    k_max_secondary: float,
    k_max_tertiary: float,
    k_resolution_primary: float,
    k_resolution_secondary: float,
    k_resolution_tertiary: float,
    Lx: float,
    Ly: float,
    Lz: float,
    shell_width: float,
    cutoff_primary: float | None = None,
    cutoff_secondary: float | None = None,
    cutoff_tertiary: float | None = None,
    cell_size_primary: float | None = None,
    cell_size_secondary: float | None = None,
    cell_size_tertiary: float | None = None,
    cutoff: float | None = None,
    num_trajectories: int | None = None,
    trajectory_selection: str | None = None,
    max_workers: int = 1,
    frame_chunk: int = 10,
    coord_chunk: int = 256,
    k_chunk: int = 64,
    frame_window: tuple[int, int | None, int] | None = None,
    input_io_spec: dict[str, Any] | None = None,
    output_io_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overall_start = time.time()
    k_max_primary, k_max_secondary, k_max_tertiary, k_resolution_primary, k_resolution_secondary, k_resolution_tertiary = _resolve_three_tier_k_parameters(
        k_max_primary,
        k_max_secondary,
        k_max_tertiary,
        k_resolution_primary,
        k_resolution_secondary,
        k_resolution_tertiary,
    )
    if k_max_primary < 0 or k_max_secondary <= 0 or k_max_tertiary <= 0:
        raise ValueError("k max values must satisfy k_max_primary >= 0, k_max_secondary > 0, and k_max_tertiary > 0.")
    if k_max_primary > k_max_secondary or k_max_secondary > k_max_tertiary:
        raise ValueError("k max values must satisfy k_max_primary <= k_max_secondary <= k_max_tertiary.")
    if k_resolution_primary <= 0 or k_resolution_secondary <= 0 or k_resolution_tertiary <= 0:
        raise ValueError("k resolution values must be positive.")
    if int(coords_stride) <= 0:
        raise ValueError("Coordinate stride must be a positive integer.")
    if shell_width <= 0:
        raise ValueError("shell_width must be positive.")
    cutoff_primary, cutoff_secondary, cutoff_tertiary = _resolve_three_tier_cutoffs(
        cutoff_primary,
        cutoff_secondary,
        cutoff_tertiary,
        legacy_cutoff=cutoff,
    )
    cell_size_primary, cell_size_secondary, cell_size_tertiary = _resolve_three_tier_cell_sizes(
        cutoff_primary,
        cutoff_secondary,
        cutoff_tertiary,
        cell_size_primary,
        cell_size_secondary,
        cell_size_tertiary,
    )
    if max_workers <= 0:
        raise ValueError("max_workers must be positive.")
    if frame_chunk <= 0 or coord_chunk <= 0 or k_chunk <= 0:
        raise ValueError("Chunk sizes must be positive integers.")

    is_valid, error_message = validate_path_pattern(coords_pattern)
    if not is_valid:
        raise ValueError(f"Invalid Coordinate Path: {error_message}")
    coordinate_files = _discover_coordinate_files(baseDir, coords_pattern)
    trajectory_indices = _selected_trajectory_indices(len(coordinate_files), num_trajectories, trajectory_selection)

    box = np.array([float(Lx), float(Ly), float(Lz)], dtype=np.float64)
    if np.any(box <= 0):
        raise ValueError("Lx, Ly, and Lz must all be positive.")

    raw_k = density_k_magnitudes_by_resolution_three_tier(
        float(Lx),
        float(Ly),
        float(Lz),
        float(k_max_primary),
        float(k_max_secondary),
        float(k_max_tertiary),
        float(k_resolution_primary),
        float(k_resolution_secondary),
        float(k_resolution_tertiary),
    )
    if raw_k.size == 0:
        raise ValueError("No reciprocal-vector magnitudes were generated. Increase the k Max values or verify the box lengths.")
    k_vals = unique_with_tolerance(raw_k, float(shell_width))
    if k_vals.size == 0:
        raise ValueError("No unique k values remain after applying the shell width.")
    resolved_output_file = _resolve_output_path(baseDir, output_file)
    status_log_path = _status_log_path_for_output(resolved_output_file)

    print("=" * 60, flush=True)
    print("ISOTROPIC STATIC STRUCTURE FACTOR CALCULATION", flush=True)
    print("=" * 60, flush=True)
    print(f"Coordinate path pattern: {coords_pattern}", flush=True)
    print(f"Coordinate stride: {int(coords_stride)}", flush=True)
    print(f"Output path: {output_file}", flush=True)
    print(f"k max primary: {float(k_max_primary):.8f}", flush=True)
    print(f"k max secondary: {float(k_max_secondary):.8f}", flush=True)
    print(f"k max tertiary: {float(k_max_tertiary):.8f}", flush=True)
    print(f"k resolution primary: {float(k_resolution_primary):.8f}", flush=True)
    print(f"k resolution secondary: {float(k_resolution_secondary):.8f}", flush=True)
    print(f"k resolution tertiary: {float(k_resolution_tertiary):.8f}", flush=True)
    print(f"shell width: {float(shell_width):.8f}", flush=True)
    for label, value in [
        ("cutoff 1", cutoff_primary),
        ("cutoff 2", cutoff_secondary),
        ("cutoff 3", cutoff_tertiary),
    ]:
        print(f"{label}: {float(value):.8f}" if value is not None else f"{label}: none", flush=True)
    for label, value in [
        ("cell size 1", cell_size_primary),
        ("cell size 2", cell_size_secondary),
        ("cell size 3", cell_size_tertiary),
    ]:
        print(f"{label}: {float(value):.8f}" if value is not None else f"{label}: none", flush=True)
    print(f"shape of k vectors: {tuple(int(value) for value in np.asarray(k_vals).shape)}", flush=True)
    print(f"anticipated S(k) output shape: {(int(k_vals.size), 2)}", flush=True)
    if frame_window is None:
        print("trajectory desired length: full trajectory", flush=True)
    else:
        print(
            f"trajectory desired length: range({frame_window[0]}, {frame_window[1]}, {frame_window[2]})",
            flush=True,
        )
    print(f"max workers: {int(max_workers)}", flush=True)
    print(
        f"chunk sizes: frame={int(frame_chunk)}, coordinates={int(coord_chunk)}, k={int(k_chunk)}",
        flush=True,
    )
    print(f"coordinate files discovered: {len(coordinate_files)}", flush=True)
    print(f"coordinate files used: {len(trajectory_indices)}", flush=True)
    if trajectory_selection:
        print(f"trajectory selection: {trajectory_indices}", flush=True)
    tier_masks = _three_tier_masks_from_magnitudes(
        k_vals,
        k_max_primary,
        k_max_secondary,
        k_max_tertiary,
    )
    selected_pairs = [(index + 1, coordinate_files[index]) for index in trajectory_indices]
    task_args = [
        (
            index,
            file_path,
            k_vals,
            tier_masks,
            (cutoff_primary, cutoff_secondary, cutoff_tertiary),
            (cell_size_primary, cell_size_secondary, cell_size_tertiary),
            box,
            input_io_spec,
            resolved_output_file,
            output_io_spec,
            coords_stride,
            frame_chunk,
            coord_chunk,
            k_chunk,
            frame_window,
            status_log_path,
            overall_start,
        )
        for index, file_path in selected_pairs
    ]
    stats = _run_report_tasks(
        task_args=task_args,
        worker=_compute_single_trajectory_sk_report,
        max_workers=max_workers,
        dataset_label="the coordinate dataset",
    )
    sk_values = _average_saved_trajectory_outputs(
        reports=stats["reports"],
        value_columns=(1,),
        io_spec=output_io_spec,
        default_mode="text",
        default_precision="double",
        expected_prefix=k_vals.reshape(-1, 1),
        expected_prefix_columns=(0,),
    )[:, 0]
    output_array = np.column_stack((k_vals, sk_values))
    single_trajectory_reports = [str(item["output_file"]) for item in stats["reports"]]

    save_numeric_array(
        resolved_output_file,
        output_array,
        output_io_spec,
        default_mode="text",
        default_precision="double",
    )

    overall_elapsed = time.time() - overall_start
    print(f"\nSaved isotropic S(k) output: {resolved_output_file}", flush=True)
    print(f"successful files: {stats['success_count']}", flush=True)
    print(f"failed files: {stats['failure_count']}", flush=True)
    print(f"total frames used: {stats['total_frames']}", flush=True)
    print(f"total elapsed time: {overall_elapsed:.2f} s", flush=True)

    return {
        "k_values_count": int(k_vals.size),
        "k_values": k_vals,
        "output_file": resolved_output_file,
        "stats": stats,
        "single_trajectory_reports": single_trajectory_reports,
    }


def compute_directional_structure_factor_from_files(
    *,
    baseDir: str,
    coords_pattern: str,
    coords_stride: int = 1,
    output_file: str,
    k_max_primary: float,
    k_max_secondary: float,
    k_max_tertiary: float,
    k_resolution_primary: float,
    k_resolution_secondary: float,
    k_resolution_tertiary: float,
    Lx: float,
    Ly: float,
    Lz: float,
    shell_width: float,
    cutoff_primary: float | None = None,
    cutoff_secondary: float | None = None,
    cutoff_tertiary: float | None = None,
    cell_size_primary: float | None = None,
    cell_size_secondary: float | None = None,
    cell_size_tertiary: float | None = None,
    cutoff: float | None = None,
    num_trajectories: int | None = None,
    trajectory_selection: str | None = None,
    max_workers: int = 1,
    frame_chunk: int = 10,
    coord_chunk: int = 256,
    k_chunk: int = 64,
    frame_window: tuple[int, int | None, int] | None = None,
    input_io_spec: dict[str, Any] | None = None,
    output_io_spec: dict[str, Any] | None = None,
    active_axes: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    overall_start = time.time()
    k_max_primary, k_max_secondary, k_max_tertiary, k_resolution_primary, k_resolution_secondary, k_resolution_tertiary = _resolve_three_tier_k_parameters(
        k_max_primary,
        k_max_secondary,
        k_max_tertiary,
        k_resolution_primary,
        k_resolution_secondary,
        k_resolution_tertiary,
    )
    if k_max_primary < 0 or k_max_secondary <= 0 or k_max_tertiary <= 0:
        raise ValueError("k max values must satisfy k_max_primary >= 0, k_max_secondary > 0, and k_max_tertiary > 0.")
    if k_max_primary > k_max_secondary or k_max_secondary > k_max_tertiary:
        raise ValueError("k max values must satisfy k_max_primary <= k_max_secondary <= k_max_tertiary.")
    if k_resolution_primary <= 0 or k_resolution_secondary <= 0 or k_resolution_tertiary <= 0:
        raise ValueError("k resolution values must be positive.")
    if int(coords_stride) <= 0:
        raise ValueError("Coordinate stride must be a positive integer.")
    if shell_width <= 0:
        raise ValueError("shell_width must be positive.")
    cutoff_primary, cutoff_secondary, cutoff_tertiary = _resolve_three_tier_cutoffs(
        cutoff_primary,
        cutoff_secondary,
        cutoff_tertiary,
        legacy_cutoff=cutoff,
    )
    cell_size_primary, cell_size_secondary, cell_size_tertiary = _resolve_three_tier_cell_sizes(
        cutoff_primary,
        cutoff_secondary,
        cutoff_tertiary,
        cell_size_primary,
        cell_size_secondary,
        cell_size_tertiary,
    )
    if max_workers <= 0:
        raise ValueError("max_workers must be positive.")
    if frame_chunk <= 0 or coord_chunk <= 0 or k_chunk <= 0:
        raise ValueError("Chunk sizes must be positive integers.")

    is_valid, error_message = validate_path_pattern(coords_pattern)
    if not is_valid:
        raise ValueError(f"Invalid Coordinate Path: {error_message}")
    coordinate_files = _discover_coordinate_files(baseDir, coords_pattern)
    trajectory_indices = _selected_trajectory_indices(len(coordinate_files), num_trajectories, trajectory_selection)

    box = np.array([float(Lx), float(Ly), float(Lz)], dtype=np.float64)
    if np.any(box <= 0):
        raise ValueError("Lx, Ly, and Lz must all be positive.")

    k_vectors_array = directional_k_vectors_by_resolution_three_tier(
        float(Lx),
        float(Ly),
        float(Lz),
        float(k_max_primary),
        float(k_max_secondary),
        float(k_max_tertiary),
        float(k_resolution_primary),
        float(k_resolution_secondary),
        float(k_resolution_tertiary),
        active_axes=active_axes,
    )
    if k_vectors_array.shape[0] == 0:
        raise ValueError("No directional k vectors were generated. Increase the k Max values or verify the selected directions.")
    resolved_output_file = _resolve_output_path(baseDir, output_file)
    status_log_path = _status_log_path_for_output(resolved_output_file)

    print("=" * 60, flush=True)
    print("DIRECTIONAL STATIC STRUCTURE FACTOR CALCULATION", flush=True)
    print("=" * 60, flush=True)
    print(f"Coordinate path pattern: {coords_pattern}", flush=True)
    print(f"Coordinate stride: {int(coords_stride)}", flush=True)
    print(f"Output path: {output_file}", flush=True)
    print(f"k max primary: {float(k_max_primary):.8f}", flush=True)
    print(f"k max secondary: {float(k_max_secondary):.8f}", flush=True)
    print(f"k max tertiary: {float(k_max_tertiary):.8f}", flush=True)
    print(f"k resolution primary: {float(k_resolution_primary):.8f}", flush=True)
    print(f"k resolution secondary: {float(k_resolution_secondary):.8f}", flush=True)
    print(f"k resolution tertiary: {float(k_resolution_tertiary):.8f}", flush=True)
    print(f"shell width: {float(shell_width):.8f}", flush=True)
    for label, value in [
        ("cutoff 1", cutoff_primary),
        ("cutoff 2", cutoff_secondary),
        ("cutoff 3", cutoff_tertiary),
    ]:
        print(f"{label}: {float(value):.8f}" if value is not None else f"{label}: none", flush=True)
    for label, value in [
        ("cell size 1", cell_size_primary),
        ("cell size 2", cell_size_secondary),
        ("cell size 3", cell_size_tertiary),
    ]:
        print(f"{label}: {float(value):.8f}" if value is not None else f"{label}: none", flush=True)
    print(f"directional k vectors: {int(k_vectors_array.shape[0])}", flush=True)
    print(f"anticipated directional S(k) output shape: {(int(k_vectors_array.shape[0]), 5)}", flush=True)
    print(f"coordinate files discovered: {len(coordinate_files)}", flush=True)
    print(f"coordinate files used: {len(trajectory_indices)}", flush=True)
    if trajectory_selection:
        print(f"trajectory selection: {trajectory_indices}", flush=True)
    if frame_window is None:
        print("trajectory desired length: full trajectory", flush=True)
    else:
        print(
            f"trajectory desired length: range({frame_window[0]}, {frame_window[1]}, {frame_window[2]})",
            flush=True,
        )

    k_magnitudes = np.linalg.norm(k_vectors_array, axis=1)
    tier_masks = _three_tier_masks_from_magnitudes(
        k_magnitudes,
        k_max_primary,
        k_max_secondary,
        k_max_tertiary,
    )
    selected_pairs = [(index + 1, coordinate_files[index]) for index in trajectory_indices]
    task_args = [
        (
            index,
            file_path,
            k_vectors_array,
            tier_masks,
            (cutoff_primary, cutoff_secondary, cutoff_tertiary),
            (cell_size_primary, cell_size_secondary, cell_size_tertiary),
            box,
            input_io_spec,
            resolved_output_file,
            output_io_spec,
            coords_stride,
            frame_chunk,
            coord_chunk,
            k_chunk,
            frame_window,
            status_log_path,
            overall_start,
        )
        for index, file_path in selected_pairs
    ]
    stats = _run_report_tasks(
        task_args=task_args,
        worker=_compute_single_trajectory_directional_sk_report,
        max_workers=max_workers,
        dataset_label="the directional coordinate dataset",
    )
    expected_prefix = np.column_stack((k_vectors_array, k_magnitudes))
    sk_values = _average_saved_trajectory_outputs(
        reports=stats["reports"],
        value_columns=(4,),
        io_spec=output_io_spec,
        default_mode="text",
        default_precision="double",
        expected_prefix=expected_prefix,
        expected_prefix_columns=(0, 1, 2, 3),
    )[:, 0]
    output_array = np.column_stack((k_vectors_array, k_magnitudes, sk_values))
    single_trajectory_reports = [str(item["output_file"]) for item in stats["reports"]]
    save_numeric_array(
        resolved_output_file,
        output_array,
        output_io_spec,
        default_mode="text",
        default_precision="double",
    )

    overall_elapsed = time.time() - overall_start
    print(f"\nSaved directional S(k) output: {resolved_output_file}", flush=True)
    print(f"successful files: {stats['success_count']}", flush=True)
    print(f"failed files: {stats['failure_count']}", flush=True)
    print(f"total frames used: {stats['total_frames']}", flush=True)
    print(f"total elapsed time: {overall_elapsed:.2f} s", flush=True)

    return {
        "k_values_count": int(k_vectors_array.shape[0]),
        "k_vectors": k_vectors_array,
        "output_file": resolved_output_file,
        "stats": stats,
        "single_trajectory_reports": single_trajectory_reports,
    }


def _component_output_path(base_output_file: str, label: str, total_components: int) -> str:
    if total_components <= 1:
        return base_output_file
    root, ext = os.path.splitext(base_output_file)
    return f"{root}_{label}{ext}"


def compute_k_component_structure_factors_from_files(
    *,
    baseDir: str,
    coords_pattern: str,
    coords_stride: int = 1,
    output_file: str,
    components_selection: str,
    k_max_primary: float,
    k_max_secondary: float,
    k_max_tertiary: float,
    k_resolution_primary: float,
    k_resolution_secondary: float,
    k_resolution_tertiary: float,
    Lx: float,
    Ly: float,
    Lz: float,
    shell_width: float,
    cutoff_primary: float | None = None,
    cutoff_secondary: float | None = None,
    cutoff_tertiary: float | None = None,
    cell_size_primary: float | None = None,
    cell_size_secondary: float | None = None,
    cell_size_tertiary: float | None = None,
    cutoff: float | None = None,
    num_trajectories: int | None = None,
    trajectory_selection: str | None = None,
    max_workers: int = 1,
    frame_chunk: int = 10,
    coord_chunk: int = 256,
    k_chunk: int = 64,
    frame_window: tuple[int, int | None, int] | None = None,
    input_io_spec: dict[str, Any] | None = None,
    output_io_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overall_start = time.time()
    k_max_primary, k_max_secondary, k_max_tertiary, k_resolution_primary, k_resolution_secondary, k_resolution_tertiary = _resolve_three_tier_k_parameters(
        k_max_primary,
        k_max_secondary,
        k_max_tertiary,
        k_resolution_primary,
        k_resolution_secondary,
        k_resolution_tertiary,
    )
    components = parse_k_component_selection(components_selection)
    coordinate_files = _discover_coordinate_files(baseDir, coords_pattern)
    trajectory_indices = _selected_trajectory_indices(len(coordinate_files), num_trajectories, trajectory_selection)
    box = np.array([float(Lx), float(Ly), float(Lz)], dtype=np.float64)
    if np.any(box <= 0):
        raise ValueError("Lx, Ly, and Lz must all be positive.")
    if k_max_primary < 0 or k_max_secondary <= 0 or k_max_tertiary <= 0:
        raise ValueError("k max values must satisfy k_max_primary >= 0, k_max_secondary > 0, and k_max_tertiary > 0.")
    if k_max_primary > k_max_secondary or k_max_secondary > k_max_tertiary:
        raise ValueError("k max values must satisfy k_max_primary <= k_max_secondary <= k_max_tertiary.")
    if k_resolution_primary <= 0 or k_resolution_secondary <= 0 or k_resolution_tertiary <= 0:
        raise ValueError("k resolution values must be positive.")
    if int(coords_stride) <= 0:
        raise ValueError("Coordinate stride must be a positive integer.")
    if shell_width <= 0:
        raise ValueError("shell_width must be positive.")
    cutoff_primary, cutoff_secondary, cutoff_tertiary = _resolve_three_tier_cutoffs(
        cutoff_primary,
        cutoff_secondary,
        cutoff_tertiary,
        legacy_cutoff=cutoff,
    )
    cell_size_primary, cell_size_secondary, cell_size_tertiary = _resolve_three_tier_cell_sizes(
        cutoff_primary,
        cutoff_secondary,
        cutoff_tertiary,
        cell_size_primary,
        cell_size_secondary,
        cell_size_tertiary,
    )
    base_output_path = _resolve_output_path(baseDir, output_file)
    status_log_path = _status_log_path_for_output(base_output_path)

    results: list[dict[str, Any]] = []
    for axes in components:
        label = component_label(axes)
        component_output_file = _component_output_path(output_file, label, len(components))
        resolved_output_file = _resolve_output_path(baseDir, component_output_file)
        single_trajectory_reports: list[str] = []
        k_vectors_array = directional_k_vectors_by_resolution_three_tier(
            float(Lx),
            float(Ly),
            float(Lz),
            float(k_max_primary),
            float(k_max_secondary),
            float(k_max_tertiary),
            float(k_resolution_primary),
            float(k_resolution_secondary),
            float(k_resolution_tertiary),
            active_axes=axes,
        )
        if k_vectors_array.shape[0] == 0:
            raise ValueError(f"No k vectors were generated for component selection {label}.")

        print("=" * 60, flush=True)
        print(f"K-COMPONENT STRUCTURE FACTOR CALCULATION: {label}", flush=True)
        print("=" * 60, flush=True)

        k_magnitudes = np.linalg.norm(k_vectors_array, axis=1)
        tier_masks = _three_tier_masks_from_magnitudes(
            k_magnitudes,
            k_max_primary,
            k_max_secondary,
            k_max_tertiary,
        )
        selected_pairs = [(index + 1, coordinate_files[index]) for index in trajectory_indices]
        task_args = [
            (
                index,
                file_path,
                k_vectors_array,
                tier_masks,
                (cutoff_primary, cutoff_secondary, cutoff_tertiary),
                (cell_size_primary, cell_size_secondary, cell_size_tertiary),
                box,
                input_io_spec,
                resolved_output_file,
                output_io_spec,
                coords_stride,
                frame_chunk,
                coord_chunk,
                k_chunk,
                frame_window,
                status_log_path,
                overall_start,
            )
            for index, file_path in selected_pairs
        ]
        stats = _run_report_tasks(
            task_args=task_args,
            worker=_compute_single_trajectory_directional_sk_report,
            max_workers=max_workers,
            dataset_label=f"the {label} coordinate dataset",
        )
        expected_prefix = np.column_stack((k_vectors_array, k_magnitudes))
        sk_values = _average_saved_trajectory_outputs(
            reports=stats["reports"],
            value_columns=(4,),
            io_spec=output_io_spec,
            default_mode="text",
            default_precision="double",
            expected_prefix=expected_prefix,
            expected_prefix_columns=(0, 1, 2, 3),
        )[:, 0]

        if len(axes) == 1:
            axis_index = {"x": 0, "y": 1, "z": 2}[axes[0]]
            axis_k = np.abs(k_vectors_array[:, axis_index])
            output_array = np.column_stack((axis_k, sk_values))
        else:
            axis_k, sk_values = _aggregate_by_tolerance(k_magnitudes, sk_values, float(shell_width))
            output_array = np.column_stack((axis_k, sk_values))

        single_trajectory_reports = [str(item["output_file"]) for item in stats["reports"]]
        save_numeric_array(
            resolved_output_file,
            output_array,
            output_io_spec,
            default_mode="text",
            default_precision="double",
        )
        print(f"\nSaved component S(k) output ({label}): {resolved_output_file}", flush=True)

        results.append(
            {
                "label": label,
                "axes": axes,
                "output_file": resolved_output_file,
                "stats": stats,
                "k_values_count": int(output_array.shape[0]),
                "single_trajectory_reports": single_trajectory_reports,
            }
        )

    return {
        "results": results,
        "selection_count": len(results),
    }


def structure_factor_db_density(*args: Any, **kwargs: Any) -> np.ndarray:
    return isotropic_structure_factor_db_density(*args, **kwargs)


def k_vectors(Lx: float, Ly: float, Lz: float, kmax: float, dtype: Any = np.float64) -> np.ndarray:
    return isotropic_k_magnitudes(Lx, Ly, Lz, kmax, dtype=dtype)


def compute_isotropic_structure_factor_from_files(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return compute_static_structure_factor_from_files(*args, **kwargs)
