from __future__ import annotations

"""Radial distribution function utilities for saved coordinate arrays."""

import ast
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np

try:
    from .numeric_io import load_numeric_array, save_numeric_array
    from .path_utils import expand_path_pattern
except ImportError:
    from numeric_io import load_numeric_array, save_numeric_array
    from path_utils import expand_path_pattern


def _eval_index_selection_expr(expr: str) -> list[int]:
    tree = ast.parse(expr, mode="eval")
    allowed_nodes = (
        ast.Expression,
        ast.List,
        ast.Tuple,
        ast.Set,
        ast.Name,
        ast.Load,
        ast.Call,
        ast.Constant,
        ast.UnaryOp,
        ast.USub,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsupported selection expression element: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id != "range":
                raise ValueError("Only range() function calls are allowed in selections")

    value = eval(compile(tree, "<rdf_selection>", "eval"), {"__builtins__": {}}, {"range": range})
    if isinstance(value, range):
        return list(value)
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]
    return [int(value)]


def parse_index_selection(selection: Any, count: int, *, label: str) -> list[int]:
    text = str(selection or "").strip()
    count = int(count)
    if count <= 0:
        raise ValueError(f"{label} requires at least one available particle.")
    if not text:
        return list(range(count))

    if text.startswith(("[", "(", "{", "range(")):
        indices = _eval_index_selection_expr(text)
    elif "," in text:
        indices = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = [int(item.strip()) for item in part.split("-", 1)]
                indices.extend(range(start, end + 1))
            else:
                indices.append(int(part))
    elif "-" in text:
        start, end = [int(item.strip()) for item in text.split("-", 1)]
        indices = list(range(start, end + 1))
    else:
        indices = [int(text)]

    if not indices:
        raise ValueError(f"{label} did not resolve to any particles.")
    if any(index < 0 for index in indices):
        raise ValueError(f"{label} must contain zero-based, non-negative particle indices.")
    if max(indices) >= count:
        raise ValueError(
            f"{label} requests particle index {max(indices)}, but the input contains "
            f"indices 0-{count - 1}."
        )
    return sorted(dict.fromkeys(indices))


def parse_trajectory_selection(selection: Any, count: int) -> list[int]:
    return parse_index_selection(selection, count, label="Trajectory Selection")


def _parse_frame_selection(value: Any, frame_count: int) -> np.ndarray:
    text = str(value or "").strip()
    if not text:
        return np.arange(frame_count, dtype=np.int64)
    if text.startswith("range("):
        indices = _eval_index_selection_expr(text)
    else:
        desired = int(text)
        if desired < 0:
            raise ValueError("Trajectory desired length cannot be negative.")
        indices = list(range(min(desired, frame_count)))
    if any(index < 0 or index >= frame_count for index in indices):
        raise ValueError(f"Frame selection is outside the valid range 0-{frame_count - 1}.")
    return np.asarray(indices, dtype=np.int64)


def _prepare_coordinate_frames(data: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(data)
    if values.ndim == 3 and values.shape[2] == 3:
        return np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] % 3 != 0:
        raise ValueError(
            f"{label} must have shape (frames, 3 * particles) or "
            "(frames, particles, 3)."
        )
    return np.asarray(values.reshape(values.shape[0], values.shape[1] // 3, 3), dtype=np.float64)


def _histogram_for_frames(
    coords: np.ndarray,
    *,
    box: np.ndarray,
    edges: np.ndarray,
    selection1: list[int],
    selection2: list[int],
    exclude_self_pairs: bool,
    selection1_block_size: int | None,
    selection2_block_size: int | None,
) -> np.ndarray:
    hist = np.zeros(len(edges) - 1, dtype=np.float64)
    sel1 = np.asarray(selection1, dtype=np.int64)
    sel2 = np.asarray(selection2, dtype=np.int64)

    def _block_size(value: int | None, total: int, label: str) -> int:
        requested = 1 if value is None else int(value)
        if requested <= 0:
            raise ValueError(f"{label} must be a positive integer when provided.")
        return max(1, min(requested, total))

    block_size1 = _block_size(selection1_block_size, len(sel1), "selection1_block_size")
    block_size2 = _block_size(selection2_block_size, len(sel2), "selection2_block_size")

    for frame in coords:
        for start1 in range(0, len(sel1), block_size1):
            sel1_block = sel1[start1 : start1 + block_size1]
            coords1_indices = sel1_block[:, None]
            points1 = frame[sel1_block]
            for start2 in range(0, len(sel2), block_size2):
                sel2_block = sel2[start2 : start2 + block_size2]
                delta = points1[:, None, :] - frame[sel2_block][None, :, :]
                delta -= box * np.rint(delta / box)
                distances = np.linalg.norm(delta, axis=2).reshape(-1)
                if exclude_self_pairs:
                    keep = (coords1_indices != sel2_block[None, :]).reshape(-1)
                    distances = distances[keep]
                if distances.size:
                    frame_hist, _ = np.histogram(distances, bins=edges)
                    hist += frame_hist
    return hist


def _rdf_file_task(params: dict[str, Any]) -> dict[str, Any]:
    file_index = int(params["file_index"])
    base_dir = params["base_dir"]
    input_pattern = params["input_pattern"]
    common_term = params["common_term"]
    input_io_spec = params["input_io_spec"]
    input_file = os.path.join(base_dir, expand_path_pattern(input_pattern, common_term, file_index))
    try:
        data = load_numeric_array(
            input_file,
            input_io_spec,
            default_mode="text",
            default_precision="double",
        )
        coords = _prepare_coordinate_frames(data, label=f"Coordinate file {file_index}")
        coords = coords[:: int(params["coords_stride"])]
        frame_indices = _parse_frame_selection(params.get("trajectory_desired_length"), coords.shape[0])
        coords = coords[frame_indices]
        if coords.shape[0] == 0:
            raise ValueError("No frames remain after stride/frame selection.")

        n_particles = int(coords.shape[1])
        selection1 = parse_index_selection(params["selection1"], n_particles, label="Selection 1")
        selection2 = parse_index_selection(params["selection2"], n_particles, label="Selection 2")
        hist = _histogram_for_frames(
            coords,
            box=params["box"],
            edges=params["edges"],
            selection1=selection1,
            selection2=selection2,
            exclude_self_pairs=bool(params["exclude_self_pairs"]),
            selection1_block_size=params.get("selection1_block_size"),
            selection2_block_size=params.get("selection2_block_size"),
        )
        return {
            "file_index": file_index,
            "input_file": input_file,
            "hist": hist,
            "frame_count": int(coords.shape[0]),
            "n_particles": n_particles,
            "n_selection1": len(selection1),
            "n_selection2": len(selection2),
            "error": None,
        }
    except Exception as exc:
        return {
            "file_index": file_index,
            "input_file": input_file,
            "error": str(exc),
        }


def compute_radial_distribution_function_from_files(
    baseDir: str,
    input_pattern: str,
    output_path: str,
    num_trajectories: int,
    box: Any,
    *,
    coords_stride: int = 1,
    selection1: Any = "",
    selection2: Any = "",
    exclude_self_pairs: bool = False,
    r_max: float | None = None,
    dr: float = 0.1,
    trajectory_selection: Any = "",
    trajectory_desired_length: Any = "",
    common_term: str = "",
    max_workers: int = 1,
    selection1_block_size: int | None = 1,
    selection2_block_size: int | None = 1,
    input_io_spec: dict[str, Any] | None = None,
    output_io_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not baseDir:
        raise ValueError("baseDir is required")
    if not input_pattern:
        raise ValueError("input_pattern is required")
    if not output_path:
        raise ValueError("output_path is required")

    num_trajectories = int(num_trajectories)
    if num_trajectories <= 0:
        raise ValueError("num_trajectories must be positive")
    coords_stride = int(coords_stride)
    if coords_stride <= 0:
        raise ValueError("coords_stride must be a positive integer")
    max_workers = int(max_workers)
    if max_workers <= 0:
        raise ValueError("max_workers must be a positive integer")

    box_array = np.asarray(box, dtype=np.float64)
    if box_array.shape != (3,):
        raise ValueError("box must contain three dimensions: Lx, Ly, Lz.")
    if np.any(box_array <= 0):
        raise ValueError("box dimensions must be positive.")
    volume = float(np.prod(box_array))

    dr = float(dr)
    if dr <= 0:
        raise ValueError("dr must be positive")
    if r_max is None or str(r_max).strip() == "":
        resolved_r_max = 0.5 * float(np.min(box_array))
    else:
        resolved_r_max = float(r_max)
    if resolved_r_max <= 0:
        raise ValueError("r_max must be positive when provided")

    edges = np.arange(0.0, resolved_r_max + dr, dr, dtype=np.float64)
    if edges.size < 2:
        raise ValueError("r_max and dr do not define any RDF bins.")
    r = 0.5 * (edges[:-1] + edges[1:])
    shell_volumes = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)

    trajectory_indices = parse_trajectory_selection(trajectory_selection, num_trajectories)
    tasks = [
        {
            "file_index": index,
            "base_dir": baseDir,
            "input_pattern": input_pattern,
            "common_term": common_term,
            "input_io_spec": input_io_spec,
            "coords_stride": coords_stride,
            "trajectory_desired_length": trajectory_desired_length,
            "selection1": selection1,
            "selection2": selection2,
            "exclude_self_pairs": exclude_self_pairs,
            "selection1_block_size": selection1_block_size,
            "selection2_block_size": selection2_block_size,
            "box": box_array,
            "edges": edges,
        }
        for index in trajectory_indices
    ]

    if max_workers == 1 or len(tasks) <= 1:
        results = [_rdf_file_task(task) for task in tasks]
    else:
        worker_count = min(max_workers, len(tasks))
        results = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(_rdf_file_task, task): task["file_index"] for task in tasks}
            for future in as_completed(future_map):
                index = future_map[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append({"file_index": index, "error": str(exc)})

    failures = [result for result in results if result.get("error")]
    if failures:
        details = "\n".join(
            f"- {item.get('file_index')}: {item.get('error')}" for item in sorted(failures, key=lambda x: int(x.get("file_index", 0)))
        )
        raise RuntimeError(f"RDF calculation failed for {len(failures)} file(s):\n{details}")

    ordered_results = sorted(results, key=lambda item: int(item["file_index"]))
    n_particles_values = {int(item["n_particles"]) for item in ordered_results}
    n_selection1_values = {int(item["n_selection1"]) for item in ordered_results}
    n_selection2_values = {int(item["n_selection2"]) for item in ordered_results}
    if len(n_particles_values) != 1 or len(n_selection1_values) != 1 or len(n_selection2_values) != 1:
        raise ValueError("All RDF coordinate files must have consistent particle and selection counts.")

    hist = np.sum([np.asarray(item["hist"], dtype=np.float64) for item in ordered_results], axis=0)
    total_frames = int(sum(int(item["frame_count"]) for item in ordered_results))
    if total_frames <= 0:
        raise ValueError("No frames were read.")

    n_selection1 = n_selection1_values.pop()
    n_selection2 = n_selection2_values.pop()
    rho2 = float(n_selection2) / volume
    normalization = float(total_frames) * float(n_selection1) * rho2 * shell_volumes
    g_r = np.divide(hist, normalization, out=np.zeros_like(hist, dtype=np.float64), where=normalization != 0.0)
    coordination_number = np.cumsum(g_r * shell_volumes * rho2)

    output_file = os.path.join(baseDir, expand_path_pattern(output_path, common_term, None))
    output_data = np.column_stack((r, g_r, coordination_number, hist))
    header = "\n".join(
        [
            "Radial Distribution Function",
            "columns: r g(r) coordination_number hist",
            f"input_pattern: {input_pattern}",
            f"num_trajectories: {num_trajectories}",
            f"trajectory_indices: {trajectory_indices}",
            f"total_frames: {total_frames}",
            f"box: {box_array.tolist()}",
            f"selection1: {selection1 if str(selection1).strip() else 'all'}",
            f"selection2: {selection2 if str(selection2).strip() else 'all'}",
            f"exclude_self_pairs: {bool(exclude_self_pairs)}",
            f"selection1_block_size: {selection1_block_size if selection1_block_size is not None else 1}",
            f"selection2_block_size: {selection2_block_size if selection2_block_size is not None else 1}",
            f"dr: {dr}",
            f"r_max: {resolved_r_max}",
            f"rho_selection2: {rho2:.15g}",
            "hist is the raw pair-count histogram before RDF normalization.",
        ]
    )
    save_numeric_array(
        output_file,
        output_data,
        output_io_spec,
        default_mode="text",
        default_precision="double",
        header=header,
    )

    return {
        "output_file": output_file,
        "total_frames": total_frames,
        "n_files": len(ordered_results),
        "n_particles": n_particles_values.pop(),
        "n_selection1": n_selection1,
        "n_selection2": n_selection2,
        "rho_selection2": rho2,
        "r_max": resolved_r_max,
        "dr": dr,
        "hist_sum": float(np.sum(hist)),
    }
