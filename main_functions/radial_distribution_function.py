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


def _slice_from_stride(value: Any) -> str:
    stride = int(value)
    if stride <= 0:
        raise ValueError("coords_stride must be a positive integer")
    return f"[::{stride}]"


def _path_with_suffix(path: str, suffix: str) -> str:
    root, extension = os.path.splitext(path)
    return f"{root}{suffix}{extension}" if extension else f"{path}{suffix}"


def _slice_int_value(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant):
        if node.value is None:
            return None
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _slice_int_value(node.operand)
        if value is None:
            raise ValueError("Negative coordinate-slice values must be integers.")
        return -value
    raise ValueError("Coordinate slicing accepts only integers, blanks, ':', and ','.")


def _parse_slice_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Slice):
        return slice(
            _slice_int_value(node.lower) if node.lower is not None else None,
            _slice_int_value(node.upper) if node.upper is not None else None,
            _slice_int_value(node.step) if node.step is not None else None,
        )
    if isinstance(node, ast.Tuple):
        return tuple(_parse_slice_node(item) for item in node.elts)
    if isinstance(node, ast.Constant) and node.value is Ellipsis:
        return Ellipsis
    value = _slice_int_value(node)
    if value is None:
        raise ValueError("Coordinate slicing cannot use None as an index.")
    return value


def _split_bracketed_axis_slices(text: str) -> list[str] | None:
    parts: list[str] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index] in " \t,":
            index += 1
        if index >= len(text):
            break
        if text[index] != "[":
            return None
        start = index + 1
        depth = 1
        index += 1
        while index < len(text) and depth:
            char = text[index]
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
            index += 1
        if depth != 0:
            return None
        part = text[start : index - 1].strip()
        if not part:
            return None
        parts.append(part)
        while index < len(text) and text[index] in " \t":
            index += 1
        if index < len(text):
            if text[index] != ",":
                return None
            index += 1
    return parts if parts else None


def _normalize_coordinate_slice_text(text: str) -> str:
    candidates = [text]
    if text.startswith("[[") and text.endswith("]]"):
        candidates.insert(0, text[1:-1].strip())

    for candidate in candidates:
        parts = _split_bracketed_axis_slices(candidate)
        if parts and len(parts) > 1:
            return "[" + ", ".join(parts) + "]"
    return text


def _parse_coordinate_slice(value: Any, *, label: str) -> Any:
    text = _normalize_coordinate_slice_text(str(value or "").strip())
    if not text:
        return (slice(None), slice(None), slice(None))

    expression = f"_coords{text}" if text.startswith("[") else f"_coords[{text}]"
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"{label} is not valid Python slicing syntax: {text!r}") from exc
    if not isinstance(tree.body, ast.Subscript):
        raise ValueError(f"{label} must be a Python slicing expression such as [::10].")
    parsed = _parse_slice_node(tree.body.slice)
    if parsed is Ellipsis:
        return parsed
    if not isinstance(parsed, tuple):
        parsed = (parsed,)
    if len(parsed) > 3:
        raise ValueError(f"{label} can slice at most frame, particle, and coordinate axes.")
    return parsed + (slice(None),) * (3 - len(parsed))


def _apply_coordinate_slice(coords: np.ndarray, slice_value: Any, *, label: str) -> np.ndarray:
    index = _parse_coordinate_slice(slice_value, label=label)
    try:
        sliced = coords[index]
    except Exception as exc:
        raise ValueError(f"{label} failed for shape {coords.shape}: {slice_value!r}") from exc
    if sliced.ndim != 3 or sliced.shape[2] != 3:
        raise ValueError(
            f"{label} must preserve coordinate shape (frames, particles, 3). "
            "Use ranges such as [:, 67:68] instead of integer particle indexing."
        )
    if sliced.shape[0] == 0:
        raise ValueError(f"{label} leaves zero frames.")
    if sliced.shape[1] == 0:
        raise ValueError(f"{label} leaves zero particles.")
    return np.asarray(sliced, dtype=np.float64)


def _histogram_for_frames(
    coords1: np.ndarray,
    coords2: np.ndarray,
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

    for frame1, frame2 in zip(coords1, coords2):
        for start1 in range(0, len(sel1), block_size1):
            sel1_block = sel1[start1 : start1 + block_size1]
            coords1_indices = sel1_block[:, None]
            points1 = frame1[sel1_block]
            for start2 in range(0, len(sel2), block_size2):
                sel2_block = sel2[start2 : start2 + block_size2]
                delta = points1[:, None, :] - frame2[sel2_block][None, :, :]
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
    input_pattern1 = params.get("input_pattern1") or params["input_pattern"]
    input_pattern2 = params.get("input_pattern2") or input_pattern1
    common_term = params["common_term"]
    input_io_spec = params["input_io_spec"]
    input_file1 = os.path.join(base_dir, expand_path_pattern(input_pattern1, common_term, file_index))
    input_file2 = os.path.join(base_dir, expand_path_pattern(input_pattern2, common_term, file_index))
    coords_slice1 = params.get("coords_slice1") or params.get("coords_slice") or _slice_from_stride(params["coords_stride"])
    coords_slice2 = params.get("coords_slice2") or coords_slice1
    try:
        data1 = load_numeric_array(
            input_file1,
            input_io_spec,
            default_mode="text",
            default_precision="double",
        )
        if os.path.abspath(input_file2) == os.path.abspath(input_file1):
            data2 = data1
        else:
            data2 = load_numeric_array(
                input_file2,
                input_io_spec,
                default_mode="text",
                default_precision="double",
            )
        coords1 = _prepare_coordinate_frames(data1, label=f"Coordinate Path 1 file {file_index}")
        coords2 = _prepare_coordinate_frames(data2, label=f"Coordinate Path 2 file {file_index}")
        coords1 = _apply_coordinate_slice(coords1, coords_slice1, label="Coordinate Path 1 slicing")
        coords2 = _apply_coordinate_slice(coords2, coords_slice2, label="Coordinate Path 2 slicing")

        frame_indices1 = _parse_frame_selection(params.get("trajectory_desired_length"), coords1.shape[0])
        frame_indices2 = _parse_frame_selection(params.get("trajectory_desired_length"), coords2.shape[0])
        coords1 = coords1[frame_indices1]
        coords2 = coords2[frame_indices2]
        if coords1.shape[0] == 0 or coords2.shape[0] == 0:
            raise ValueError("No frames remain after slicing/frame selection.")
        if coords1.shape[0] != coords2.shape[0]:
            raise ValueError(
                "Coordinate Path 1 and Coordinate Path 2 must have the same frame count "
                f"after slicing/windowing; got {coords1.shape[0]} and {coords2.shape[0]}."
            )

        n_particles1 = int(coords1.shape[1])
        n_particles2 = int(coords2.shape[1])
        selection1 = parse_index_selection(params["selection1"], n_particles1, label="Selection 1")
        selection2 = parse_index_selection(params["selection2"], n_particles2, label="Selection 2")
        same_coordinate_source = (
            os.path.abspath(input_file1) == os.path.abspath(input_file2)
            and str(coords_slice1).strip() == str(coords_slice2).strip()
        )
        hist = _histogram_for_frames(
            coords1,
            coords2,
            box=params["box"],
            edges=params["edges"],
            selection1=selection1,
            selection2=selection2,
            exclude_self_pairs=bool(params["exclude_self_pairs"]) and same_coordinate_source,
            selection1_block_size=params.get("selection1_block_size"),
            selection2_block_size=params.get("selection2_block_size"),
        )
        return {
            "file_index": file_index,
            "input_file": input_file1,
            "input_file1": input_file1,
            "input_file2": input_file2,
            "hist": hist,
            "frame_count": int(coords1.shape[0]),
            "n_particles1": n_particles1,
            "n_particles2": n_particles2,
            "n_selection1": len(selection1),
            "n_selection2": len(selection2),
            "error": None,
        }
    except Exception as exc:
        return {
            "file_index": file_index,
            "input_file": input_file1,
            "input_file1": input_file1,
            "input_file2": input_file2,
            "error": str(exc),
        }


def compute_radial_distribution_function_from_files(
    baseDir: str,
    input_pattern: str,
    output_path: str,
    num_trajectories: int,
    box: Any,
    *,
    input_pattern1: str | None = None,
    input_pattern2: str | None = None,
    coords_stride: int = 1,
    coords_slice: Any = "",
    coords_slice1: Any = "",
    coords_slice2: Any = "",
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
    density_normalize: bool = True,
    input_io_spec: dict[str, Any] | None = None,
    output_io_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not baseDir:
        raise ValueError("baseDir is required")
    input_pattern1 = str(input_pattern1 or input_pattern or "").strip()
    input_pattern2 = str(input_pattern2 or "").strip()
    if not input_pattern1:
        raise ValueError("input_pattern1 is required")
    if not output_path:
        raise ValueError("output_path is required")

    num_trajectories = int(num_trajectories)
    if num_trajectories <= 0:
        raise ValueError("num_trajectories must be positive")
    coords_slice1 = str(coords_slice1 or coords_slice or _slice_from_stride(coords_stride)).strip()
    coords_slice2 = str(coords_slice2 or "").strip()
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
            "input_pattern": input_pattern1,
            "input_pattern1": input_pattern1,
            "input_pattern2": input_pattern2,
            "common_term": common_term,
            "input_io_spec": input_io_spec,
            "coords_stride": coords_stride,
            "coords_slice1": coords_slice1,
            "coords_slice2": coords_slice2,
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
    n_particles1_values = {int(item["n_particles1"]) for item in ordered_results}
    n_particles2_values = {int(item["n_particles2"]) for item in ordered_results}
    n_selection1_values = {int(item["n_selection1"]) for item in ordered_results}
    n_selection2_values = {int(item["n_selection2"]) for item in ordered_results}
    if (
        len(n_particles1_values) != 1
        or len(n_particles2_values) != 1
        or len(n_selection1_values) != 1
        or len(n_selection2_values) != 1
    ):
        raise ValueError("All RDF coordinate files must have consistent particle and selection counts.")

    hist = np.sum([np.asarray(item["hist"], dtype=np.float64) for item in ordered_results], axis=0)
    total_frames = int(sum(int(item["frame_count"]) for item in ordered_results))
    if total_frames <= 0:
        raise ValueError("No frames were read.")

    n_selection1 = n_selection1_values.pop()
    n_selection2 = n_selection2_values.pop()
    rho2 = float(n_selection2) / volume
    base_normalization = float(total_frames) * float(n_selection1) * shell_volumes
    radial_number_density = np.divide(
        hist,
        base_normalization,
        out=np.zeros_like(hist, dtype=np.float64),
        where=base_normalization != 0.0,
    )
    density_normalization = base_normalization * rho2
    g_r = np.divide(
        hist,
        density_normalization,
        out=np.zeros_like(hist, dtype=np.float64),
        where=density_normalization != 0.0,
    )
    coordination_number = np.cumsum(radial_number_density * shell_volumes)

    output_file = os.path.join(baseDir, expand_path_pattern(output_path, common_term, None))
    norm_output_file = _path_with_suffix(output_file, "_norm")
    not_norm_output_file = _path_with_suffix(output_file, "_notNorm")
    common_header_lines = [
        "Radial Distribution Function",
        f"input_pattern1: {input_pattern1}",
        f"input_pattern2: {input_pattern2 or input_pattern1}",
        f"coords_slice1: {coords_slice1 or '[:]'}",
        f"coords_slice2: {coords_slice2 or coords_slice1 or '[:]'}",
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
    norm_header = "\n".join(
        [
            *common_header_lines,
            "density_normalized: True",
            "columns: r g_r coordination_number hist",
        ]
    )
    not_norm_header = "\n".join(
        [
            *common_header_lines,
            "density_normalized: False",
            "columns: r radial_number_density coordination_number hist",
        ]
    )
    norm_output_data = np.column_stack((r, g_r, coordination_number, hist))
    not_norm_output_data = np.column_stack((r, radial_number_density, coordination_number, hist))
    save_numeric_array(
        norm_output_file,
        norm_output_data,
        output_io_spec,
        default_mode="text",
        default_precision="double",
        header=norm_header,
        column_names=("r", "g_r", "coordination_number", "hist"),
        compact_columns=(3,),
        result_format=True,
    )
    save_numeric_array(
        not_norm_output_file,
        not_norm_output_data,
        output_io_spec,
        default_mode="text",
        default_precision="double",
        header=not_norm_header,
        column_names=("r", "radial_number_density", "coordination_number", "hist"),
        compact_columns=(3,),
        result_format=True,
    )

    return {
        "output_file": norm_output_file,
        "output_file_norm": norm_output_file,
        "output_file_not_norm": not_norm_output_file,
        "total_frames": total_frames,
        "n_files": len(ordered_results),
        "n_particles": n_particles1_values.copy().pop(),
        "n_particles1": n_particles1_values.pop(),
        "n_particles2": n_particles2_values.pop(),
        "n_selection1": n_selection1,
        "n_selection2": n_selection2,
        "rho_selection2": rho2,
        "density_normalize": True,
        "column2_label": "g(r)",
        "r_max": resolved_r_max,
        "dr": dr,
        "hist_sum": float(np.sum(hist)),
    }
