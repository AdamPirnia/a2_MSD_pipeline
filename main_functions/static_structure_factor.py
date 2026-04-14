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
    frame_chunk: int = 10,
    i_chunk: int = 256,
    j_chunk: int = 256,
    k_chunk: int = 64,
    dtype: Any = np.float64,
    include_leading_one: bool = True,
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

    inv_pi = dtype(1.0 / np.pi)
    s_accum = np.zeros(k_vals.size, dtype=dtype)

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
                    dist = np.sqrt(np.sum(dr * dr, axis=2)).ravel()

                    for k0 in range(0, k_vals.size, k_chunk):
                        k1 = min(k_vals.size, k0 + k_chunk)
                        x = np.outer(k_vals[k0:k1], dist)
                        pair_sum_k[k0:k1] += np.sinc(x * inv_pi).sum(axis=1)

            if include_leading_one:
                s_accum += 1.0 + (2.0 / natms) * pair_sum_k
            else:
                s_accum += (2.0 / natms) * pair_sum_k

    return s_accum / float(total_frames)


def isotropic_k_magnitudes(Lx: float, Ly: float, Lz: float, kmax: float, dtype: Any = np.float64) -> np.ndarray:
    two_pi = np.array(2.0 * np.pi, dtype=dtype)
    nxmax = int(np.ceil(kmax * Lx / two_pi))
    nymax = int(np.ceil(kmax * Ly / two_pi))
    nzmax = int(np.ceil(kmax * Lz / two_pi))

    km: list[float] = []
    for nx in range(1, nxmax + 1):
        for ny in range(1, nymax + 1):
            for nz in range(1, nzmax + 1):
                kvec = two_pi * np.array([nx / Lx, ny / Ly, nz / Lz], dtype=dtype)
                mag = float(np.linalg.norm(kvec))
                if mag <= kmax:
                    km.append(mag)

    k_mags = np.asarray(km, dtype=dtype)
    if k_mags.size > 1:
        k_mags = k_mags[np.argsort(k_mags)]
    return k_mags


def directional_k_vectors(
    Lx: float,
    Ly: float,
    Lz: float,
    kmax: float,
    active_axes: tuple[str, ...] | None = None,
    dtype: Any = np.float64,
) -> np.ndarray:
    axis_order = ("x", "y", "z")
    active = set(active_axes or axis_order)
    length_map = {"x": float(Lx), "y": float(Ly), "z": float(Lz)}
    limits = {
        axis: int(np.ceil(kmax * length_map[axis] / (2.0 * np.pi)))
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
                if mag <= float(kmax):
                    vectors.append((float(kvec[0]), float(kvec[1]), float(kvec[2])))

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


def directional_structure_factor(
    r_flat: np.ndarray,
    k_vectors_array: np.ndarray,
    box: np.ndarray | None = None,
    frame_chunk: int = 10,
    atom_chunk: int = 256,
    k_chunk: int = 64,
    dtype: Any = np.float64,
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

    s_accum = np.zeros(k_vectors_array.shape[0], dtype=np.float64)

    for f0 in range(0, total_frames, frame_chunk):
        f1 = min(total_frames, f0 + frame_chunk)
        r_chunk = np.ascontiguousarray(r[f0:f1])

        for r_frame in r_chunk:
            if box is not None:
                r_frame_use = np.mod(r_frame, box)
            else:
                r_frame_use = r_frame

            for k0 in range(0, k_vectors_array.shape[0], k_chunk):
                k1 = min(k_vectors_array.shape[0], k0 + k_chunk)
                rho = np.zeros(k1 - k0, dtype=np.complex128)
                for a0 in range(0, natms, atom_chunk):
                    a1 = min(natms, a0 + atom_chunk)
                    phase = np.matmul(r_frame_use[a0:a1], k_vectors_array[k0:k1].T)
                    rho += np.exp(1j * phase).sum(axis=0)
                s_accum[k0:k1] += (rho * np.conjugate(rho)).real / float(natms)

    return s_accum / float(total_frames)


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


def _compute_dataset_average(
    *,
    coordinate_files: list[str],
    k_vals: np.ndarray,
    box: np.ndarray,
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

    task_args = [
        (
            file_path,
            k_vals,
            box,
            input_io_spec,
            frame_chunk,
            coord_chunk,
            k_chunk,
            frame_window,
        )
        for file_path in coordinate_files
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
        weighted_sum += np.asarray(item["sk_values"], dtype=np.float64) * int(item["frame_count"])
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

    task_args = [
        (
            file_path,
            k_vectors_array,
            box,
            input_io_spec,
            frame_chunk,
            coord_chunk,
            k_chunk,
            frame_window,
        )
        for file_path in coordinate_files
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
        weighted_sum += np.asarray(item["sk_values"], dtype=np.float64) * int(item["frame_count"])
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
            frame_chunk=int(frame_chunk),
            i_chunk=int(coord_chunk),
            j_chunk=int(coord_chunk),
            k_chunk=int(k_chunk),
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
            frame_chunk=int(frame_chunk),
            atom_chunk=int(coord_chunk),
            k_chunk=int(k_chunk),
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
    k_max: float,
    Lx: float,
    Ly: float,
    Lz: float,
    tolerance: float,
    max_workers: int = 1,
    frame_chunk: int = 10,
    coord_chunk: int = 256,
    k_chunk: int = 64,
    frame_window: tuple[int, int | None, int] | None = None,
    input_io_spec: dict[str, Any] | None = None,
    output_io_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overall_start = time.time()
    if k_max <= 0:
        raise ValueError("k_max must be positive.")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive.")
    if frame_chunk <= 0 or coord_chunk <= 0 or k_chunk <= 0:
        raise ValueError("Chunk sizes must be positive integers.")

    is_valid, error_message = validate_path_pattern(coords_pattern)
    if not is_valid:
        raise ValueError(f"Invalid Coordinate Path: {error_message}")
    coordinate_files = _discover_coordinate_files(baseDir, coords_pattern)

    box = np.array([float(Lx), float(Ly), float(Lz)], dtype=np.float64)
    if np.any(box <= 0):
        raise ValueError("Lx, Ly, and Lz must all be positive.")

    raw_k = isotropic_k_magnitudes(float(Lx), float(Ly), float(Lz), float(k_max))
    if raw_k.size == 0:
        raise ValueError("No reciprocal-vector magnitudes were generated. Increase k Max or verify the box lengths.")
    k_vals = unique_with_tolerance(raw_k, float(tolerance))
    if k_vals.size == 0:
        raise ValueError("No unique k values remain after applying the tolerance.")

    print("=" * 60, flush=True)
    print("ISOTROPIC STATIC STRUCTURE FACTOR CALCULATION", flush=True)
    print("=" * 60, flush=True)
    print(f"Coordinate path pattern: {coords_pattern}", flush=True)
    print(f"Output path: {output_file}", flush=True)
    print(f"k max: {float(k_max):.8f}", flush=True)
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

    sk_values, stats = _compute_dataset_average(
        coordinate_files=coordinate_files,
        k_vals=k_vals,
        box=box,
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
    k_max: float,
    Lx: float,
    Ly: float,
    Lz: float,
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
    if k_max <= 0:
        raise ValueError("k_max must be positive.")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive.")
    if frame_chunk <= 0 or coord_chunk <= 0 or k_chunk <= 0:
        raise ValueError("Chunk sizes must be positive integers.")

    is_valid, error_message = validate_path_pattern(coords_pattern)
    if not is_valid:
        raise ValueError(f"Invalid Coordinate Path: {error_message}")
    coordinate_files = _discover_coordinate_files(baseDir, coords_pattern)

    box = np.array([float(Lx), float(Ly), float(Lz)], dtype=np.float64)
    if np.any(box <= 0):
        raise ValueError("Lx, Ly, and Lz must all be positive.")

    k_vectors_array = directional_k_vectors(
        float(Lx),
        float(Ly),
        float(Lz),
        float(k_max),
        active_axes=active_axes,
    )
    if k_vectors_array.shape[0] == 0:
        raise ValueError("No directional k vectors were generated. Increase k Max or verify the selected directions.")

    print("=" * 60, flush=True)
    print("DIRECTIONAL STATIC STRUCTURE FACTOR CALCULATION", flush=True)
    print("=" * 60, flush=True)
    print(f"Coordinate path pattern: {coords_pattern}", flush=True)
    print(f"Output path: {output_file}", flush=True)
    print(f"k max: {float(k_max):.8f}", flush=True)
    print(f"directional k vectors: {int(k_vectors_array.shape[0])}", flush=True)
    print(f"anticipated directional S(k) output shape: {(int(k_vectors_array.shape[0]), 5)}", flush=True)
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
    k_max: float,
    Lx: float,
    Ly: float,
    Lz: float,
    tolerance: float,
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
    box = np.array([float(Lx), float(Ly), float(Lz)], dtype=np.float64)
    if np.any(box <= 0):
        raise ValueError("Lx, Ly, and Lz must all be positive.")

    results: list[dict[str, Any]] = []
    for axes in components:
        label = component_label(axes)
        component_output_file = _component_output_path(output_file, label, len(components))
        k_vectors_array = directional_k_vectors(
            float(Lx),
            float(Ly),
            float(Lz),
            float(k_max),
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
            axis_k, sk_values = _aggregate_by_tolerance(k_magnitudes, sk_values, float(tolerance))
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
