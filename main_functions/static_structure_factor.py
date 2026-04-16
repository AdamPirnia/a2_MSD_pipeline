from __future__ import annotations

import os
import re
import time
from glob import glob
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

try:
    from .numeric_io import load_numeric_array, save_numeric_array
    from .path_utils import expand_path_pattern, validate_path_pattern
except ImportError:
    from numeric_io import load_numeric_array, save_numeric_array
    from path_utils import expand_path_pattern, validate_path_pattern


def isotropic_structure_factor_db_density(
    r_flat: np.ndarray,
    k_vals: np.ndarray,
    box: np.ndarray | None = None,
    cutoff: float | None = None,
    frame_chunk: int = 10,
    i_chunk: int = 256,
    j_chunk: int = 256,
    k_chunk: int = 64,
    dtype: Any = np.float64,
    include_leading_one: bool = True,
    normalize_by_frames: bool = True,
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
    if cutoff is not None and float(cutoff) <= 0:
        raise ValueError("cutoff must be positive when provided.")

    inv_pi = dtype(1.0 / np.pi)
    s_accum = np.zeros(k_vals.size, dtype=dtype)
    cutoff_sq = None if cutoff is None else float(cutoff) ** 2

    for f0 in range(0, total_frames, frame_chunk):
        f1 = min(total_frames, f0 + frame_chunk)
        r_chunk = np.ascontiguousarray(r[f0:f1])

        for r_frame in r_chunk:
            pair_sum_k = np.zeros(k_vals.size, dtype=dtype)

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
    c1: int = 1,
    c2: int = 1,
    c3: int = 1,
    dtype: Any = np.float64,
) -> np.ndarray:
    if kmax1 < 0 or kmax2 <= 0 or kmax3 <= 0:
        raise ValueError("kmax values must satisfy kmax1 >= 0, kmax2 > 0, and kmax3 > 0.")
    if kmax1 > kmax2 or kmax2 > kmax3:
        raise ValueError("kmax values must satisfy kmax1 <= kmax2 <= kmax3.")
    if int(c1) <= 0 or int(c2) <= 0 or int(c3) <= 0:
        raise ValueError("k stride values must be positive integers.")

    two_pi = np.array(2.0 * np.pi, dtype=dtype)
    nxmax = int(np.ceil(kmax3 * Lx / two_pi))
    nymax = int(np.ceil(kmax3 * Ly / two_pi))
    nzmax = int(np.ceil(kmax3 * Lz / two_pi))

    kmags: list[float] = []
    seen: set[tuple[int, int, int]] = set()

    for kmin, kmax, stride in (
        (0.0, float(kmax1), int(c1)),
        (float(kmax1), float(kmax2), int(c2)),
        (float(kmax2), float(kmax3), int(c3)),
    ):
        for nx in range(1, nxmax + 1, stride):
            for ny in range(1, nymax + 1, stride):
                for nz in range(1, nzmax + 1, stride):
                    index_triplet = (int(nx), int(ny), int(nz))
                    if index_triplet in seen:
                        continue
                    kvec = two_pi * np.array([nx / Lx, ny / Ly, nz / Lz], dtype=dtype)
                    kmag = float(np.linalg.norm(kvec))
                    if kmin <= kmag <= kmax and (kmin == 0.0 or kmag > kmin):
                        kmags.append(kmag)
                        seen.add(index_triplet)

    k_mags = np.asarray(kmags, dtype=dtype)
    if k_mags.size > 1:
        k_mags = k_mags[np.argsort(k_mags)]
    return k_mags


def directional_k_vectors_three_tier(
    Lx: float,
    Ly: float,
    Lz: float,
    kmax1: float,
    kmax2: float,
    kmax3: float,
    c1: int = 1,
    c2: int = 1,
    c3: int = 1,
    active_axes: tuple[str, ...] | None = None,
    dtype: Any = np.float64,
) -> np.ndarray:
    if kmax1 < 0 or kmax2 <= 0 or kmax3 <= 0:
        raise ValueError("kmax values must satisfy kmax1 >= 0, kmax2 > 0, and kmax3 > 0.")
    if kmax1 > kmax2 or kmax2 > kmax3:
        raise ValueError("kmax values must satisfy kmax1 <= kmax2 <= kmax3.")
    if int(c1) <= 0 or int(c2) <= 0 or int(c3) <= 0:
        raise ValueError("k stride values must be positive integers.")

    axis_order = ("x", "y", "z")
    active = set(active_axes or axis_order)
    length_map = {"x": float(Lx), "y": float(Ly), "z": float(Lz)}
    limits = {
        axis: int(np.ceil(kmax3 * length_map[axis] / (2.0 * np.pi)))
        for axis in axis_order
    }

    vectors: list[tuple[float, float, float]] = []
    seen: set[tuple[int, int, int]] = set()
    for kmin, kmax, stride in (
        (0.0, float(kmax1), int(c1)),
        (float(kmax1), float(kmax2), int(c2)),
        (float(kmax2), float(kmax3), int(c3)),
    ):
        for nx in range(0, limits["x"] + 1 if "x" in active else 1, stride):
            for ny in range(0, limits["y"] + 1 if "y" in active else 1, stride):
                for nz in range(0, limits["z"] + 1 if "z" in active else 1, stride):
                    if nx == 0 and ny == 0 and nz == 0:
                        continue
                    index_triplet = (int(nx), int(ny), int(nz))
                    if index_triplet in seen:
                        continue
                    kvec = 2.0 * np.pi * np.array(
                        [nx / float(Lx), ny / float(Ly), nz / float(Lz)],
                        dtype=dtype,
                    )
                    mag = float(np.linalg.norm(kvec))
                    if kmin <= mag <= kmax and (kmin == 0.0 or mag > kmin):
                        vectors.append((float(kvec[0]), float(kvec[1]), float(kvec[2])))
                        seen.add(index_triplet)

    if not vectors:
        return np.empty((0, 3), dtype=dtype)

    k_vectors_array = np.asarray(vectors, dtype=dtype)
    sort_order = np.lexsort(
        (
            k_vectors_array[:, 2],
            k_vectors_array[:, 1],
            k_vectors_array[:, 0],
            np.linalg.norm(k_vectors_array, axis=1),
        )
    )
    return k_vectors_array[sort_order]


def isotropic_k_magnitudes(Lx: float, Ly: float, Lz: float, kmax: float, dtype: Any = np.float64) -> np.ndarray:
    return isotropic_k_magnitudes_three_tier(Lx, Ly, Lz, 0.0, kmax, kmax, c1=1, c2=1, c3=1, dtype=dtype)


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
        c1=1,
        c2=1,
        c3=1,
        active_axes=active_axes,
        dtype=dtype,
    )


def directional_structure_factor(
    r_flat: np.ndarray,
    k_vectors_array: np.ndarray,
    box: np.ndarray | None = None,
    cutoff: float | None = None,
    frame_chunk: int = 10,
    atom_chunk: int = 256,
    k_chunk: int = 64,
    dtype: Any = np.float64,
    normalize_by_frames: bool = True,
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
    if cutoff is not None and float(cutoff) <= 0:
        raise ValueError("cutoff must be positive when provided.")

    s_accum = np.zeros(k_vectors_array.shape[0], dtype=np.float64)
    cutoff_sq = None if cutoff is None else float(cutoff) ** 2

    for f0 in range(0, total_frames, frame_chunk):
        f1 = min(total_frames, f0 + frame_chunk)
        r_chunk = np.ascontiguousarray(r[f0:f1])

        for r_frame in r_chunk:
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


def component_label(axes: tuple[str, ...]) -> str:
    return "k" + "".join(axes)


def _aggregate_by_tolerance(
    k_magnitudes: np.ndarray,
    values: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
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
        charge_values = charge_array[:, 1]
    else:
        charge_values = charge_array.reshape(-1)
    if delete_index is not None:
        charge_values = np.delete(charge_values, int(delete_index), axis=0)
    if charge_values.size <= 0:
        raise ValueError("Charge values file produced an empty charge array.")
    return np.asarray(charge_values, dtype=np.float64)


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
    c1: int = 1,
    c2: int = 1,
    c3: int = 1,
    dtype: Any = np.float64,
    include_zero: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if kmax1 < 0 or kmax2 <= 0 or kmax3 <= 0:
        raise ValueError("kmax values must satisfy kmax1 >= 0, kmax2 > 0, and kmax3 > 0.")
    if kmax1 > kmax2 or kmax2 > kmax3:
        raise ValueError("kmax values must satisfy kmax1 <= kmax2 <= kmax3.")
    if int(c1) <= 0 or int(c2) <= 0 or int(c3) <= 0:
        raise ValueError("k stride values must be positive integers.")

    two_pi = np.array(2.0 * np.pi, dtype=dtype)
    nxmax = int(np.ceil(float(kmax3) * float(Lx) / float(two_pi)))
    nymax = int(np.ceil(float(kmax3) * float(Ly) / float(two_pi)))
    nzmax = int(np.ceil(float(kmax3) * float(Lz) / float(two_pi)))

    kmags: list[float] = []
    kvecs: list[tuple[float, float, float]] = []
    seen: set[tuple[int, int, int]] = set()

    for kmin, kmax, stride in (
        (0.0, float(kmax1), int(c1)),
        (float(kmax1), float(kmax2), int(c2)),
        (float(kmax2), float(kmax3), int(c3)),
    ):
        for nx in range(-nxmax, nxmax + 1, stride):
            for ny in range(-nymax, nymax + 1, stride):
                for nz in range(-nzmax, nzmax + 1, stride):
                    if not include_zero and nx == 0 and ny == 0 and nz == 0:
                        continue
                    index_triplet = (int(nx), int(ny), int(nz))
                    if index_triplet in seen:
                        continue

                    kvec = two_pi * np.array(
                        [nx / float(Lx), ny / float(Ly), nz / float(Lz)],
                        dtype=dtype,
                    )
                    kmag = float(np.linalg.norm(kvec))
                    if kmin <= kmag <= kmax and (kmin == 0.0 or kmag > kmin):
                        kmags.append(kmag)
                        kvecs.append((float(kvec[0]), float(kvec[1]), float(kvec[2])))
                        seen.add(index_triplet)

    if not kmags:
        return np.empty(0, dtype=dtype), np.empty((0, 3), dtype=dtype)

    k_mags = np.asarray(kmags, dtype=dtype)
    k_vectors_array = np.asarray(kvecs, dtype=dtype)
    sort_order = np.argsort(k_mags)
    return k_mags[sort_order], k_vectors_array[sort_order]


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


def charge_dipole_structure_factor_directional(
    k_vectors_array: np.ndarray,
    charge_positions: np.ndarray,
    charge_values: np.ndarray,
    dipole_positions: np.ndarray,
    dipole_directions: np.ndarray,
    cutoff: float | None = None,
    dtype: Any = np.float64,
    normalize_by_frames: bool = True,
) -> np.ndarray:
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

    khat = _build_khat(k_vectors_use)
    cutoff_sq = -1.0 if cutoff is None else float(cutoff) ** 2

    out_re = np.empty(k_vectors_use.shape[0], dtype=np.float64)
    out_im = np.empty(k_vectors_use.shape[0], dtype=np.float64)
    frame_count = int(rq.shape[0])
    dipole_count = int(rp.shape[1])
    inv_dipole_count = 1.0 / float(dipole_count)

    for k_index, (kvec, khat_vec) in enumerate(zip(k_vectors_use, khat, strict=False)):
        kx, ky, kz = (float(kvec[0]), float(kvec[1]), float(kvec[2]))
        khx, khy, khz = (float(khat_vec[0]), float(khat_vec[1]), float(khat_vec[2]))
        acc_re = 0.0
        acc_im = 0.0

        for frame_index in range(frame_count):
            if cutoff_sq > 0.0:
                for charge_index in range(int(rq.shape[1])):
                    zq = float(z[charge_index])
                    charge_phase = (
                        kx * float(rq[frame_index, charge_index, 0])
                        + ky * float(rq[frame_index, charge_index, 1])
                        + kz * float(rq[frame_index, charge_index, 2])
                    )
                    cos_q = float(np.cos(charge_phase))
                    sin_q = float(np.sin(charge_phase))

                    for dipole_index in range(dipole_count):
                        dx = float(rq[frame_index, charge_index, 0] - rp[frame_index, dipole_index, 0])
                        dy = float(rq[frame_index, charge_index, 1] - rp[frame_index, dipole_index, 1])
                        dz = float(rq[frame_index, charge_index, 2] - rp[frame_index, dipole_index, 2])
                        dist_sq = dx * dx + dy * dy + dz * dz
                        if dist_sq > cutoff_sq:
                            continue

                        dipole_phase = (
                            kx * float(rp[frame_index, dipole_index, 0])
                            + ky * float(rp[frame_index, dipole_index, 1])
                            + kz * float(rp[frame_index, dipole_index, 2])
                        )
                        cos_p = float(np.cos(dipole_phase))
                        sin_p = float(np.sin(dipole_phase))
                        weight = (
                            khx * float(ehat[frame_index, dipole_index, 0])
                            + khy * float(ehat[frame_index, dipole_index, 1])
                            + khz * float(ehat[frame_index, dipole_index, 2])
                        )
                        cos_diff = cos_p * cos_q + sin_p * sin_q
                        sin_diff = sin_p * cos_q - cos_p * sin_q
                        acc_re -= zq * weight * sin_diff
                        acc_im += zq * weight * cos_diff
            else:
                q_cos = 0.0
                q_sin = 0.0
                p_cos = 0.0
                p_sin = 0.0

                for charge_index in range(int(rq.shape[1])):
                    charge_phase = (
                        kx * float(rq[frame_index, charge_index, 0])
                        + ky * float(rq[frame_index, charge_index, 1])
                        + kz * float(rq[frame_index, charge_index, 2])
                    )
                    zq = float(z[charge_index])
                    q_cos += zq * float(np.cos(charge_phase))
                    q_sin += zq * float(np.sin(charge_phase))

                for dipole_index in range(dipole_count):
                    dipole_phase = (
                        kx * float(rp[frame_index, dipole_index, 0])
                        + ky * float(rp[frame_index, dipole_index, 1])
                        + kz * float(rp[frame_index, dipole_index, 2])
                    )
                    weight = (
                        khx * float(ehat[frame_index, dipole_index, 0])
                        + khy * float(ehat[frame_index, dipole_index, 1])
                        + khz * float(ehat[frame_index, dipole_index, 2])
                    )
                    p_cos += weight * float(np.cos(dipole_phase))
                    p_sin += weight * float(np.sin(dipole_phase))

                acc_re += q_sin * p_cos - q_cos * p_sin
                acc_im += q_cos * p_cos + q_sin * p_sin

        scale = inv_dipole_count / float(frame_count) if normalize_by_frames else inv_dipole_count
        out_re[k_index] = acc_re * scale
        out_im[k_index] = acc_im * scale

    return out_re + 1j * out_im


def compute_charge_dipole_structure_factor_from_files(
    *,
    baseDir: str,
    charge_values_path: str,
    charge_coords_pattern: str,
    dipole_positions_pattern: str,
    dipole_vectors_pattern: str,
    isotropic_output_file: str,
    directional_output_file: str,
    k_max_primary: float,
    k_max_secondary: float,
    k_max_tertiary: float,
    k_stride_primary: int,
    k_stride_secondary: int,
    k_stride_tertiary: int,
    Lx: float,
    Ly: float,
    Lz: float,
    shell_width: float,
    cutoff: float | None = None,
    num_trajectories: int | None = None,
    delete_residue_index: int | None = None,
    frame_window: tuple[int, int | None, int] | None = None,
    input_io_spec: dict[str, Any] | None = None,
    output_io_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overall_start = time.time()
    box = np.array([float(Lx), float(Ly), float(Lz)], dtype=np.float64)
    if np.any(box <= 0):
        raise ValueError("Lx, Ly, and Lz must all be positive.")
    if float(k_max_primary) < 0 or float(k_max_secondary) <= 0 or float(k_max_tertiary) <= 0:
        raise ValueError("k max values must satisfy k_max_primary >= 0, k_max_secondary > 0, and k_max_tertiary > 0.")
    if float(k_max_primary) > float(k_max_secondary) or float(k_max_secondary) > float(k_max_tertiary):
        raise ValueError("k max values must satisfy k_max_primary <= k_max_secondary <= k_max_tertiary.")
    if int(k_stride_primary) <= 0 or int(k_stride_secondary) <= 0 or int(k_stride_tertiary) <= 0:
        raise ValueError("k stride values must be positive integers.")
    if float(shell_width) <= 0:
        raise ValueError("shell_width must be positive.")
    if cutoff is not None and float(cutoff) <= 0:
        raise ValueError("cutoff must be positive when provided.")

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
    if num_trajectories is not None and int(num_trajectories) <= 0:
        raise ValueError("num_trajectories must be positive when provided.")
    if num_trajectories is not None:
        limit = int(num_trajectories)
        charge_coordinate_files = charge_coordinate_files[:limit]
        dipole_position_files = dipole_position_files[:limit]
        dipole_vector_files = dipole_vector_files[:limit]
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

    k_magnitudes, k_vectors_array = charge_dipole_k_vectors_three_tier(
        float(Lx),
        float(Ly),
        float(Lz),
        float(k_max_primary),
        float(k_max_secondary),
        float(k_max_tertiary),
        c1=int(k_stride_primary),
        c2=int(k_stride_secondary),
        c3=int(k_stride_tertiary),
    )
    if k_vectors_array.shape[0] == 0:
        raise ValueError("No charge-dipole k vectors were generated. Increase k max values or verify the box lengths.")

    charge_values = _load_charge_values(charge_values_file, input_io_spec, delete_residue_index)
    sqp_sum = np.zeros(k_vectors_array.shape[0], dtype=np.complex128)
    total_frames = 0
    per_file_stats: list[dict[str, Any]] = []

    print("=" * 60, flush=True)
    print("CHARGE-DIPOLE STRUCTURE FACTOR CALCULATION", flush=True)
    print("=" * 60, flush=True)
    print(f"Charge values path: {charge_values_path}", flush=True)
    print(f"Charge coordinates path: {charge_coords_pattern}", flush=True)
    print(f"Dipole positions path: {dipole_positions_pattern}", flush=True)
    print(f"Dipole vectors path: {dipole_vectors_pattern}", flush=True)
    print(f"Isotropic output path: {isotropic_output_file}", flush=True)
    print(f"Directional output path: {directional_output_file}", flush=True)
    print(f"k max primary: {float(k_max_primary):.8f}", flush=True)
    print(f"k max secondary: {float(k_max_secondary):.8f}", flush=True)
    print(f"k max tertiary: {float(k_max_tertiary):.8f}", flush=True)
    print(f"k stride primary: {int(k_stride_primary)}", flush=True)
    print(f"k stride secondary: {int(k_stride_secondary)}", flush=True)
    print(f"k stride tertiary: {int(k_stride_tertiary)}", flush=True)
    print(f"directional k vectors: {int(k_vectors_array.shape[0])}", flush=True)
    print(f"shell width: {float(shell_width):.8f}", flush=True)
    print(f"delete residue index: {delete_residue_index if delete_residue_index is not None else 'none'}", flush=True)
    if frame_window is None:
        print("trajectory desired length: full trajectory", flush=True)
    else:
        print(
            f"trajectory desired length: range({frame_window[0]}, {frame_window[1]}, {frame_window[2]})",
            flush=True,
        )

    for file_index, (rq_file, rp_file, mu_file) in enumerate(
        zip(charge_coordinate_files, dipole_position_files, dipole_vector_files, strict=False),
        start=1,
    ):
        print(f"\nProcessing charge-dipole file set {file_index}/{len(charge_coordinate_files)}", flush=True)
        print(f"  charge coordinates: {rq_file}", flush=True)
        print(f"  dipole positions:   {rp_file}", flush=True)
        print(f"  dipole vectors:     {mu_file}", flush=True)

        charge_position_array = load_numeric_array(
            rq_file,
            input_io_spec,
            default_mode="text",
            default_precision="double",
        )
        dipole_position_array = load_numeric_array(
            rp_file,
            input_io_spec,
            default_mode="text",
            default_precision="double",
        )
        dipole_vector_array = load_numeric_array(
            mu_file,
            input_io_spec,
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

        directional_values = charge_dipole_structure_factor_directional(
            k_vectors_array,
            charge_positions,
            charge_values,
            dipole_positions,
            dipole_directions,
            cutoff=cutoff,
            normalize_by_frames=False,
        )
        sqp_sum += directional_values
        total_frames += int(frame_count)
        per_file_stats.append(
            {
                "charge_coordinate_file": rq_file,
                "dipole_position_file": rp_file,
                "dipole_vector_file": mu_file,
                "frame_count": int(frame_count),
            }
        )
        print(f"  accumulated frames: {frame_count}", flush=True)

    if total_frames <= 0:
        raise ValueError("No frames were accumulated for the charge-dipole structure-factor calculation.")

    directional_average = sqp_sum / float(total_frames)
    k_shell, isotropic_shell, shell_counts = _shell_average_complex(k_magnitudes, directional_average, float(shell_width))

    isotropic_output_path = _resolve_output_path(baseDir, isotropic_output_file)
    directional_output_path = _resolve_output_path(baseDir, directional_output_file)
    isotropic_output_array = np.column_stack((k_shell, isotropic_shell.real, isotropic_shell.imag, shell_counts))
    directional_output_array = np.column_stack((k_vectors_array, k_magnitudes, directional_average.real, directional_average.imag))

    save_numeric_array(
        isotropic_output_path,
        isotropic_output_array,
        output_io_spec,
        default_mode="text",
        default_precision="double",
    )
    save_numeric_array(
        directional_output_path,
        directional_output_array,
        output_io_spec,
        default_mode="text",
        default_precision="double",
    )

    overall_elapsed = time.time() - overall_start
    print(f"\nSaved isotropic charge-dipole output: {isotropic_output_path}", flush=True)
    print(f"Saved directional charge-dipole output: {directional_output_path}", flush=True)
    print(f"total file sets used: {len(per_file_stats)}", flush=True)
    print(f"total frames used: {total_frames}", flush=True)
    print(f"total elapsed time: {overall_elapsed:.2f} s", flush=True)

    return {
        "directional_k_count": int(k_vectors_array.shape[0]),
        "isotropic_shell_count": int(k_shell.shape[0]),
        "k_vectors": k_vectors_array,
        "k_magnitudes": k_magnitudes,
        "directional_values": directional_average,
        "k_shell": k_shell,
        "isotropic_values": isotropic_shell,
        "shell_counts": shell_counts,
        "total_frames": int(total_frames),
        "isotropic_output_file": isotropic_output_path,
        "directional_output_file": directional_output_path,
        "per_file_stats": per_file_stats,
    }


def _compute_dataset_average(
    *,
    coordinate_files: list[str],
    k_vals: np.ndarray,
    box: np.ndarray,
    cutoff: float | None,
    num_trajectories: int | None,
    input_io_spec: dict[str, Any] | None,
    dataset_label: str,
    max_workers: int,
    frame_chunk: int,
    coord_chunk: int,
    k_chunk: int,
    frame_window: tuple[int, int | None, int] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    weighted_sum = np.zeros(k_vals.size, dtype=np.float64)
    total_frames = 0
    success_indices: list[int] = []
    failed: list[dict[str, Any]] = []

    selected_files = coordinate_files[: int(num_trajectories)] if num_trajectories is not None else coordinate_files

    task_args = [
        (
            file_path,
            k_vals,
            box,
            cutoff,
            input_io_spec,
            frame_chunk,
            coord_chunk,
            k_chunk,
            frame_window,
        )
        for file_path in selected_files
    ]

    if int(max_workers) > 1 and len(task_args) > 1:
        with ProcessPoolExecutor(max_workers=min(int(max_workers), len(task_args))) as executor:
            task_results = list(executor.map(_compute_single_trajectory_sk, task_args))
    else:
        task_results = [_compute_single_trajectory_sk(args) for args in task_args]

    for item in task_results:
        if item["error"]:
            failed.append({"file_path": item["file_path"], "error": item["error"]})
            continue
        weighted_sum += np.asarray(item["sk_values"], dtype=np.float64)
        total_frames += int(item["frame_count"])
        success_indices.append(str(item["file_path"]))

    if total_frames <= 0:
        raise ValueError(f"No usable trajectories were found for {dataset_label}.")

    return weighted_sum / float(total_frames), {
        "success_indices": success_indices,
        "failed": failed,
        "total_frames": total_frames,
        "success_count": len(success_indices),
        "failure_count": len(failed),
    }


def _compute_dataset_average_directional(
    *,
    coordinate_files: list[str],
    k_vectors_array: np.ndarray,
    box: np.ndarray,
    cutoff: float | None,
    num_trajectories: int | None,
    input_io_spec: dict[str, Any] | None,
    dataset_label: str,
    max_workers: int,
    frame_chunk: int,
    coord_chunk: int,
    k_chunk: int,
    frame_window: tuple[int, int | None, int] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    weighted_sum = np.zeros(int(k_vectors_array.shape[0]), dtype=np.float64)
    total_frames = 0
    success_indices: list[str] = []
    failed: list[dict[str, Any]] = []

    selected_files = coordinate_files[: int(num_trajectories)] if num_trajectories is not None else coordinate_files

    task_args = [
        (
            file_path,
            k_vectors_array,
            box,
            cutoff,
            input_io_spec,
            frame_chunk,
            coord_chunk,
            k_chunk,
            frame_window,
        )
        for file_path in selected_files
    ]

    if int(max_workers) > 1 and len(task_args) > 1:
        with ProcessPoolExecutor(max_workers=min(int(max_workers), len(task_args))) as executor:
            task_results = list(executor.map(_compute_single_trajectory_directional_sk, task_args))
    else:
        task_results = [_compute_single_trajectory_directional_sk(args) for args in task_args]

    for item in task_results:
        if item["error"]:
            failed.append({"file_path": item["file_path"], "error": item["error"]})
            continue
        weighted_sum += np.asarray(item["sk_values"], dtype=np.float64)
        total_frames += int(item["frame_count"])
        success_indices.append(str(item["file_path"]))

    if total_frames <= 0:
        raise ValueError(f"No usable trajectories were found for {dataset_label}.")

    return weighted_sum / float(total_frames), {
        "success_indices": success_indices,
        "failed": failed,
        "total_frames": total_frames,
        "success_count": len(success_indices),
        "failure_count": len(failed),
    }


def _compute_single_trajectory_sk(args: tuple[Any, ...]) -> dict[str, Any]:
    (
        file_path,
        k_vals,
        box,
        cutoff,
        input_io_spec,
        frame_chunk,
        coord_chunk,
        k_chunk,
        frame_window,
    ) = args

    if not os.path.exists(file_path):
        return {"file_path": file_path, "error": f"Missing file: {file_path}"}

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
        coords = _apply_frame_window(coords, frame_window)
        frame_count = int(coords.shape[0])
        if frame_count <= 0:
            raise ValueError("No frames found in coordinate file.")

        k_array = np.asarray(k_vals, dtype=np.float64)
        anticipated_output_shape = (int(k_array.size), 2)
        print(f"  shape of r: {tuple(int(value) for value in coords.shape)}", flush=True)
        print(f"  k max: {float(np.max(k_array)) if k_array.size else 0.0:.8f}", flush=True)
        print(f"  shape of k vectors: {tuple(int(value) for value in k_array.shape)}", flush=True)
        print(f"  anticipated S(k) output shape: {anticipated_output_shape}", flush=True)

        sk_values = isotropic_structure_factor_db_density(
            coords,
            k_vals,
            box=box,
            cutoff=cutoff,
            frame_chunk=int(frame_chunk),
            i_chunk=int(coord_chunk),
            j_chunk=int(coord_chunk),
            k_chunk=int(k_chunk),
            normalize_by_frames=False,
        )
        elapsed_s = float(time.time() - start_time)
        print(f"  completed in {elapsed_s:.2f} s", flush=True)
        return {
            "file_path": file_path,
            "frame_count": frame_count,
            "sk_values": np.asarray(sk_values, dtype=np.float64),
            "elapsed_s": elapsed_s,
            "error": "",
        }
    except Exception as exc:
        return {"file_path": file_path, "error": str(exc)}


def _compute_single_trajectory_directional_sk(args: tuple[Any, ...]) -> dict[str, Any]:
    (
        file_path,
        k_vectors_array,
        box,
        cutoff,
        input_io_spec,
        frame_chunk,
        coord_chunk,
        k_chunk,
        frame_window,
    ) = args

    if not os.path.exists(file_path):
        return {"file_path": file_path, "error": f"Missing file: {file_path}"}

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
        coords = _apply_frame_window(coords, frame_window)
        frame_count = int(coords.shape[0])
        if frame_count <= 0:
            raise ValueError("No frames found in coordinate file.")

        k_vectors_use = np.asarray(k_vectors_array, dtype=np.float64)
        anticipated_output_shape = (int(k_vectors_use.shape[0]), 5)
        print(f"  shape of r: {tuple(int(value) for value in coords.shape)}", flush=True)
        print(f"  directional k-vector count: {int(k_vectors_use.shape[0])}", flush=True)
        print(f"  anticipated directional S(k) output shape: {anticipated_output_shape}", flush=True)

        sk_values = directional_structure_factor(
            coords,
            k_vectors_use,
            box=box,
            cutoff=cutoff,
            frame_chunk=int(frame_chunk),
            atom_chunk=int(coord_chunk),
            k_chunk=int(k_chunk),
            normalize_by_frames=False,
        )
        elapsed_s = float(time.time() - start_time)
        print(f"  completed in {elapsed_s:.2f} s", flush=True)
        return {
            "file_path": file_path,
            "frame_count": frame_count,
            "sk_values": np.asarray(sk_values, dtype=np.float64),
            "elapsed_s": elapsed_s,
            "error": "",
        }
    except Exception as exc:
        return {"file_path": file_path, "error": str(exc)}


def compute_static_structure_factor_from_files(
    *,
    baseDir: str,
    coords_pattern: str,
    output_file: str,
    k_max_primary: float,
    k_max_secondary: float,
    k_max_tertiary: float,
    k_stride_primary: int,
    k_stride_secondary: int,
    k_stride_tertiary: int,
    Lx: float,
    Ly: float,
    Lz: float,
    shell_width: float,
    cutoff: float | None = None,
    num_trajectories: int | None = None,
    max_workers: int = 1,
    frame_chunk: int = 10,
    coord_chunk: int = 256,
    k_chunk: int = 64,
    frame_window: tuple[int, int | None, int] | None = None,
    input_io_spec: dict[str, Any] | None = None,
    output_io_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overall_start = time.time()
    if k_max_primary < 0 or k_max_secondary <= 0 or k_max_tertiary <= 0:
        raise ValueError("k max values must satisfy k_max_primary >= 0, k_max_secondary > 0, and k_max_tertiary > 0.")
    if k_max_primary > k_max_secondary or k_max_secondary > k_max_tertiary:
        raise ValueError("k max values must satisfy k_max_primary <= k_max_secondary <= k_max_tertiary.")
    if k_stride_primary <= 0 or k_stride_secondary <= 0 or k_stride_tertiary <= 0:
        raise ValueError("k stride values must be positive integers.")
    if shell_width <= 0:
        raise ValueError("shell_width must be positive.")
    if cutoff is not None and float(cutoff) <= 0:
        raise ValueError("cutoff must be positive when provided.")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive.")
    if frame_chunk <= 0 or coord_chunk <= 0 or k_chunk <= 0:
        raise ValueError("Chunk sizes must be positive integers.")

    is_valid, error_message = validate_path_pattern(coords_pattern)
    if not is_valid:
        raise ValueError(f"Invalid Coordinate Path: {error_message}")
    coordinate_files = _discover_coordinate_files(baseDir, coords_pattern)
    if num_trajectories is not None and int(num_trajectories) <= 0:
        raise ValueError("num_trajectories must be positive when provided.")

    box = np.array([float(Lx), float(Ly), float(Lz)], dtype=np.float64)
    if np.any(box <= 0):
        raise ValueError("Lx, Ly, and Lz must all be positive.")

    raw_k = isotropic_k_magnitudes_three_tier(
        float(Lx),
        float(Ly),
        float(Lz),
        float(k_max_primary),
        float(k_max_secondary),
        float(k_max_tertiary),
        c1=int(k_stride_primary),
        c2=int(k_stride_secondary),
        c3=int(k_stride_tertiary),
    )
    if raw_k.size == 0:
        raise ValueError("No reciprocal-vector magnitudes were generated. Increase the k Max values or verify the box lengths.")
    k_vals = unique_with_tolerance(raw_k, float(shell_width))
    if k_vals.size == 0:
        raise ValueError("No unique k values remain after applying the shell width.")

    print("=" * 60, flush=True)
    print("ISOTROPIC STATIC STRUCTURE FACTOR CALCULATION", flush=True)
    print("=" * 60, flush=True)
    print(f"Coordinate path pattern: {coords_pattern}", flush=True)
    print(f"Output path: {output_file}", flush=True)
    print(f"k max primary: {float(k_max_primary):.8f}", flush=True)
    print(f"k max secondary: {float(k_max_secondary):.8f}", flush=True)
    print(f"k max tertiary: {float(k_max_tertiary):.8f}", flush=True)
    print(f"k stride primary: {int(k_stride_primary)}", flush=True)
    print(f"k stride secondary: {int(k_stride_secondary)}", flush=True)
    print(f"k stride tertiary: {int(k_stride_tertiary)}", flush=True)
    print(f"shell width: {float(shell_width):.8f}", flush=True)
    print(f"cutoff: {float(cutoff):.8f}" if cutoff is not None else "cutoff: none", flush=True)
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
    used_file_count = min(len(coordinate_files), int(num_trajectories)) if num_trajectories is not None else len(coordinate_files)
    print(f"coordinate files discovered: {len(coordinate_files)}", flush=True)
    print(f"coordinate files used: {used_file_count}", flush=True)

    sk_values, stats = _compute_dataset_average(
        coordinate_files=coordinate_files,
        k_vals=k_vals,
        box=box,
        cutoff=cutoff,
        num_trajectories=num_trajectories,
        input_io_spec=input_io_spec,
        dataset_label="the coordinate dataset",
        max_workers=max_workers,
        frame_chunk=frame_chunk,
        coord_chunk=coord_chunk,
        k_chunk=k_chunk,
        frame_window=frame_window,
    )
    output_array = np.column_stack((k_vals, sk_values))

    resolved_output_file = _resolve_output_path(baseDir, output_file)

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
    }


def compute_directional_structure_factor_from_files(
    *,
    baseDir: str,
    coords_pattern: str,
    output_file: str,
    k_max_primary: float,
    k_max_secondary: float,
    k_max_tertiary: float,
    k_stride_primary: int,
    k_stride_secondary: int,
    k_stride_tertiary: int,
    Lx: float,
    Ly: float,
    Lz: float,
    shell_width: float,
    cutoff: float | None = None,
    num_trajectories: int | None = None,
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
    if k_max_primary < 0 or k_max_secondary <= 0 or k_max_tertiary <= 0:
        raise ValueError("k max values must satisfy k_max_primary >= 0, k_max_secondary > 0, and k_max_tertiary > 0.")
    if k_max_primary > k_max_secondary or k_max_secondary > k_max_tertiary:
        raise ValueError("k max values must satisfy k_max_primary <= k_max_secondary <= k_max_tertiary.")
    if k_stride_primary <= 0 or k_stride_secondary <= 0 or k_stride_tertiary <= 0:
        raise ValueError("k stride values must be positive integers.")
    if shell_width <= 0:
        raise ValueError("shell_width must be positive.")
    if cutoff is not None and float(cutoff) <= 0:
        raise ValueError("cutoff must be positive when provided.")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive.")
    if frame_chunk <= 0 or coord_chunk <= 0 or k_chunk <= 0:
        raise ValueError("Chunk sizes must be positive integers.")

    is_valid, error_message = validate_path_pattern(coords_pattern)
    if not is_valid:
        raise ValueError(f"Invalid Coordinate Path: {error_message}")
    coordinate_files = _discover_coordinate_files(baseDir, coords_pattern)
    if num_trajectories is not None and int(num_trajectories) <= 0:
        raise ValueError("num_trajectories must be positive when provided.")

    box = np.array([float(Lx), float(Ly), float(Lz)], dtype=np.float64)
    if np.any(box <= 0):
        raise ValueError("Lx, Ly, and Lz must all be positive.")

    k_vectors_array = directional_k_vectors_three_tier(
        float(Lx),
        float(Ly),
        float(Lz),
        float(k_max_primary),
        float(k_max_secondary),
        float(k_max_tertiary),
        c1=int(k_stride_primary),
        c2=int(k_stride_secondary),
        c3=int(k_stride_tertiary),
        active_axes=active_axes,
    )
    if k_vectors_array.shape[0] == 0:
        raise ValueError("No directional k vectors were generated. Increase the k Max values or verify the selected directions.")

    print("=" * 60, flush=True)
    print("DIRECTIONAL STATIC STRUCTURE FACTOR CALCULATION", flush=True)
    print("=" * 60, flush=True)
    print(f"Coordinate path pattern: {coords_pattern}", flush=True)
    print(f"Output path: {output_file}", flush=True)
    print(f"k max primary: {float(k_max_primary):.8f}", flush=True)
    print(f"k max secondary: {float(k_max_secondary):.8f}", flush=True)
    print(f"k max tertiary: {float(k_max_tertiary):.8f}", flush=True)
    print(f"k stride primary: {int(k_stride_primary)}", flush=True)
    print(f"k stride secondary: {int(k_stride_secondary)}", flush=True)
    print(f"k stride tertiary: {int(k_stride_tertiary)}", flush=True)
    print(f"shell width: {float(shell_width):.8f}", flush=True)
    print(f"cutoff: {float(cutoff):.8f}" if cutoff is not None else "cutoff: none", flush=True)
    print(f"directional k vectors: {int(k_vectors_array.shape[0])}", flush=True)
    print(f"anticipated directional S(k) output shape: {(int(k_vectors_array.shape[0]), 5)}", flush=True)
    used_file_count = min(len(coordinate_files), int(num_trajectories)) if num_trajectories is not None else len(coordinate_files)
    print(f"coordinate files discovered: {len(coordinate_files)}", flush=True)
    print(f"coordinate files used: {used_file_count}", flush=True)
    if frame_window is None:
        print("trajectory desired length: full trajectory", flush=True)
    else:
        print(
            f"trajectory desired length: range({frame_window[0]}, {frame_window[1]}, {frame_window[2]})",
            flush=True,
        )

    sk_values, stats = _compute_dataset_average_directional(
        coordinate_files=coordinate_files,
        k_vectors_array=k_vectors_array,
        box=box,
        cutoff=cutoff,
        num_trajectories=num_trajectories,
        input_io_spec=input_io_spec,
        dataset_label="the directional coordinate dataset",
        max_workers=max_workers,
        frame_chunk=frame_chunk,
        coord_chunk=coord_chunk,
        k_chunk=k_chunk,
        frame_window=frame_window,
    )
    k_magnitudes = np.linalg.norm(k_vectors_array, axis=1)
    output_array = np.column_stack((k_vectors_array, k_magnitudes, sk_values))

    resolved_output_file = _resolve_output_path(baseDir, output_file)
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
    output_file: str,
    components_selection: str,
    k_max_primary: float,
    k_max_secondary: float,
    k_max_tertiary: float,
    k_stride_primary: int,
    k_stride_secondary: int,
    k_stride_tertiary: int,
    Lx: float,
    Ly: float,
    Lz: float,
    shell_width: float,
    cutoff: float | None = None,
    num_trajectories: int | None = None,
    max_workers: int = 1,
    frame_chunk: int = 10,
    coord_chunk: int = 256,
    k_chunk: int = 64,
    frame_window: tuple[int, int | None, int] | None = None,
    input_io_spec: dict[str, Any] | None = None,
    output_io_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    components = parse_k_component_selection(components_selection)
    coordinate_files = _discover_coordinate_files(baseDir, coords_pattern)
    if num_trajectories is not None and int(num_trajectories) <= 0:
        raise ValueError("num_trajectories must be positive when provided.")
    box = np.array([float(Lx), float(Ly), float(Lz)], dtype=np.float64)
    if np.any(box <= 0):
        raise ValueError("Lx, Ly, and Lz must all be positive.")
    if k_max_primary < 0 or k_max_secondary <= 0 or k_max_tertiary <= 0:
        raise ValueError("k max values must satisfy k_max_primary >= 0, k_max_secondary > 0, and k_max_tertiary > 0.")
    if k_max_primary > k_max_secondary or k_max_secondary > k_max_tertiary:
        raise ValueError("k max values must satisfy k_max_primary <= k_max_secondary <= k_max_tertiary.")
    if k_stride_primary <= 0 or k_stride_secondary <= 0 or k_stride_tertiary <= 0:
        raise ValueError("k stride values must be positive integers.")
    if shell_width <= 0:
        raise ValueError("shell_width must be positive.")
    if cutoff is not None and float(cutoff) <= 0:
        raise ValueError("cutoff must be positive when provided.")

    results: list[dict[str, Any]] = []
    for axes in components:
        label = component_label(axes)
        component_output_file = _component_output_path(output_file, label, len(components))
        k_vectors_array = directional_k_vectors_three_tier(
            float(Lx),
            float(Ly),
            float(Lz),
            float(k_max_primary),
            float(k_max_secondary),
            float(k_max_tertiary),
            c1=int(k_stride_primary),
            c2=int(k_stride_secondary),
            c3=int(k_stride_tertiary),
            active_axes=axes,
        )
        if k_vectors_array.shape[0] == 0:
            raise ValueError(f"No k vectors were generated for component selection {label}.")

        print("=" * 60, flush=True)
        print(f"K-COMPONENT STRUCTURE FACTOR CALCULATION: {label}", flush=True)
        print("=" * 60, flush=True)

        sk_values, stats = _compute_dataset_average_directional(
            coordinate_files=coordinate_files,
            k_vectors_array=k_vectors_array,
            box=box,
            cutoff=cutoff,
            num_trajectories=num_trajectories,
            input_io_spec=input_io_spec,
            dataset_label=f"the {label} coordinate dataset",
            max_workers=max_workers,
            frame_chunk=frame_chunk,
            coord_chunk=coord_chunk,
            k_chunk=k_chunk,
            frame_window=frame_window,
        )

        if len(axes) == 1:
            axis_index = {"x": 0, "y": 1, "z": 2}[axes[0]]
            axis_k = np.abs(k_vectors_array[:, axis_index])
            output_array = np.column_stack((axis_k, sk_values))
        else:
            k_magnitudes = np.linalg.norm(k_vectors_array, axis=1)
            axis_k, sk_values = _aggregate_by_tolerance(k_magnitudes, sk_values, float(shell_width))
            output_array = np.column_stack((axis_k, sk_values))

        resolved_output_file = _resolve_output_path(baseDir, component_output_file)
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
