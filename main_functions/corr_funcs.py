from __future__ import annotations

"""Correlation-function utilities for scalar and vector time series."""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np

try:
    from .numeric_io import load_numeric_array, save_numeric_array
    from .path_utils import expand_path_pattern
except ImportError:
    from numeric_io import load_numeric_array, save_numeric_array
    from path_utils import expand_path_pattern


def _validate_sampling(delta: int, max_lag: int, length: int) -> tuple[int, int]:
    delta = int(delta)
    max_lag = int(max_lag)
    if delta <= 0:
        raise ValueError("delta must be a positive integer")
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    if length <= 0:
        raise ValueError("input arrays must contain at least one frame")
    if max_lag >= length:
        max_lag = length - 1
    return delta, max_lag


def _center_array(array: np.ndarray, subtract_mean: bool) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if subtract_mean:
        return values - np.mean(values, axis=0)
    return values


def _prepare_centered_pair(array1: np.ndarray, array2: np.ndarray, prepare, subtract_mean: bool) -> tuple[np.ndarray, np.ndarray]:
    values1 = prepare(array1)
    values2 = values1 if array2 is array1 else prepare(array2)
    arr1 = _center_array(values1, subtract_mean)
    arr2 = arr1 if values2 is values1 else _center_array(values2, subtract_mean)
    return arr1, arr2


def _mean_product(array1: np.ndarray, array2: np.ndarray, axes: str) -> float:
    if array1.shape != array2.shape:
        raise ValueError("correlation slices must have identical shape")
    if array1.size == 0:
        raise ValueError("correlation slice is empty")
    return float(np.einsum(f"{axes},{axes}->", array1, array2, optimize=True) / array1.size)


def _prepare_scalar_single(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 0:
        raise ValueError("scalar input must contain at least one frame")
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]
    raise ValueError("single scalar input must have shape (n_frames,) or (n_frames, 1)")


def _prepare_scalar_multiple(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    if values.ndim == 2:
        return values
    raise ValueError("multiple scalar input must have shape (n_frames, n_variables)")


def _prepare_vector_single(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("single vector input must have shape (n_frames, 3)")
    return values


def _prepare_vector_multiple(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 3 and values.shape[2] == 3:
        return values
    if values.ndim == 2:
        if values.shape[1] % 3 != 0:
            raise ValueError(
                "multiple vector input must have shape (n_frames, n_vectors, 3) "
                "or flattened shape (n_frames, 3 * n_vectors)"
            )
        return values.reshape(values.shape[0], values.shape[1] // 3, 3)
    raise ValueError(
        "multiple vector input must have shape (n_frames, n_vectors, 3) "
        "or flattened shape (n_frames, 3 * n_vectors)"
    )


def _scalar_single_correlation(
    array1: np.ndarray,
    array2: np.ndarray,
    delta: int,
    max_lag: int,
    t1: float,
    subtract_mean: bool,
) -> tuple[float, np.ndarray]:
    arr1, arr2 = _prepare_centered_pair(array1, array2, _prepare_scalar_single, subtract_mean)
    if arr1.shape[0] != arr2.shape[0]:
        raise ValueError("scalar arrays must have the same number of frames")

    delta, max_lag = _validate_sampling(delta, max_lag, arr1.shape[0])

    def corr_at_lag(lag: int) -> float:
        return _mean_product(arr1[: arr1.shape[0] - lag : delta], arr2[lag::delta], "i")

    raw_values = np.array([corr_at_lag(lag) for lag in range(max_lag + 1)], dtype=np.float64)
    variance = float(raw_values[0])
    if variance == 0.0:
        raise ValueError("C(0) is zero; cannot normalize correlation function")
    values = raw_values / variance
    times = np.arange(max_lag + 1, dtype=np.float64) * float(t1)
    return variance, np.column_stack((times, values))


def _scalar_multiple_correlation(
    array1: np.ndarray,
    array2: np.ndarray,
    delta: int,
    max_lag: int,
    t1: float,
    subtract_mean: bool,
) -> tuple[float, np.ndarray]:
    arr1, arr2 = _prepare_centered_pair(array1, array2, _prepare_scalar_multiple, subtract_mean)
    if arr1.shape != arr2.shape:
        raise ValueError("multiple scalar arrays must have identical shape")

    delta, max_lag = _validate_sampling(delta, max_lag, arr1.shape[0])

    def corr_at_lag(lag: int) -> float:
        return _mean_product(arr1[: arr1.shape[0] - lag : delta, :], arr2[lag::delta, :], "ij")

    raw_values = np.array(
        [corr_at_lag(lag) for lag in range(max_lag + 1)],
        dtype=np.float64,
    )
    variance = float(raw_values[0])
    if variance == 0.0:
        raise ValueError("C(0) is zero; cannot normalize correlation function")
    values = raw_values / variance
    times = np.arange(max_lag + 1, dtype=np.float64) * float(t1)
    return variance, np.column_stack((times, values))


def _vector_single_correlation(
    array1: np.ndarray,
    array2: np.ndarray,
    delta: int,
    max_lag: int,
    t1: float,
    subtract_mean: bool,
    coef: float,
) -> tuple[float, np.ndarray]:
    arr1, arr2 = _prepare_centered_pair(array1, array2, _prepare_vector_single, subtract_mean)
    if arr1.shape != arr2.shape:
        raise ValueError("single vector arrays must have identical shape")

    delta, max_lag = _validate_sampling(delta, max_lag, arr1.shape[0])

    def corr_at_lag(lag: int) -> float:
        a = arr1[: arr1.shape[0] - lag : delta]
        b = arr2[lag::delta]
        return float(np.einsum("ij,ij->", a, b, optimize=True) / a.shape[0])

    raw_values = np.array([corr_at_lag(lag) for lag in range(max_lag + 1)], dtype=np.float64)
    variance = float(raw_values[0])
    if variance == 0.0:
        raise ValueError("C(0) is zero; cannot normalize correlation function")
    values = raw_values / variance
    times = np.arange(max_lag + 1, dtype=np.float64) * float(t1)
    return variance, np.column_stack((times, values))


def _vector_multiple_correlation(
    array1: np.ndarray,
    array2: np.ndarray,
    delta: int,
    max_lag: int,
    t1: float,
    subtract_mean: bool,
    coef: float,
) -> tuple[float, np.ndarray]:
    arr1, arr2 = _prepare_centered_pair(array1, array2, _prepare_vector_multiple, subtract_mean)
    if arr1.shape != arr2.shape:
        raise ValueError("multiple vector arrays must have identical shape")

    delta, max_lag = _validate_sampling(delta, max_lag, arr1.shape[0])

    def corr_at_lag(lag: int) -> float:
        a = arr1[: arr1.shape[0] - lag : delta]
        b = arr2[lag::delta]
        return float(np.einsum("ijk,ijk->", a, b, optimize=True) / (a.shape[0] * a.shape[1]))

    raw_values = np.array(
        [corr_at_lag(lag) for lag in range(max_lag + 1)],
        dtype=np.float64,
    )
    variance = float(raw_values[0])
    if variance == 0.0:
        raise ValueError("C(0) is zero; cannot normalize correlation function")
    values = raw_values / variance
    times = np.arange(max_lag + 1, dtype=np.float64) * float(t1)
    return variance, np.column_stack((times, values))


def calculate_correlation(
    array1: np.ndarray,
    array2: np.ndarray,
    delta: int,
    max_lag: int,
    t1: float,
    *,
    array1_kind: str = "scalar",
    array2_kind: str = "scalar",
    variable_mode: str = "single",
    subtract_mean: bool = True,
    coef: float = 3.0,
) -> tuple[float, np.ndarray]:
    kind1 = str(array1_kind).strip().lower()
    kind2 = str(array2_kind).strip().lower()
    mode = str(variable_mode).strip().lower()

    if kind1 not in {"scalar", "vector"} or kind2 not in {"scalar", "vector"}:
        raise ValueError("array kinds must be 'scalar' or 'vector'")
    if mode not in {"single", "multiple"}:
        raise ValueError("variable_mode must be 'single' or 'multiple'")
    if kind1 != kind2:
        raise ValueError("mixed scalar/vector correlations are not supported")

    if kind1 == "scalar" and mode == "single":
        return _scalar_single_correlation(array1, array2, delta, max_lag, t1, subtract_mean)
    if kind1 == "scalar":
        return _scalar_multiple_correlation(array1, array2, delta, max_lag, t1, subtract_mean)
    if mode == "single":
        return _vector_single_correlation(array1, array2, delta, max_lag, t1, subtract_mean, coef)
    return _vector_multiple_correlation(array1, array2, delta, max_lag, t1, subtract_mean, coef)


def corrF(array1, array2, delta, max_lag, t1):
    return _scalar_single_correlation(array1, array2, delta, max_lag, t1, subtract_mean=True)


def corrVV_2vec_single(array1, array2, delta, max_lag, t1, coef=3):
    return _vector_single_correlation(array1, array2, delta, max_lag, t1, subtract_mean=True, coef=coef)


def corrVV_2vec_multiple(array1, array2, delta, max_lag, t1, coef=3):
    return _vector_multiple_correlation(array1, array2, delta, max_lag, t1, subtract_mean=True, coef=coef)


def corrF_2vec(array1, array2, delta, max_lag, t1, coef=3):
    return _vector_multiple_correlation(array1, array2, delta, max_lag, t1, subtract_mean=True, coef=coef)


def nonNormal_corrF_vec(array, delta, max_lag, t1):
    _, series = _vector_single_correlation(array, array, delta, max_lag, t1, subtract_mean=False, coef=3.0)
    return series


def _expand_pattern(pattern: str, file_index: int, common_term: str = "") -> str:
    return expand_path_pattern(pattern, common_term, file_index)


def _select_particle_columns(
    array: np.ndarray,
    *,
    array_kind: str,
    num_particles: int | None,
    particle_indices: list[int] | None,
    label: str,
) -> np.ndarray:
    kind = str(array_kind).strip().lower()
    values = np.asarray(array)
    total_particles = int(num_particles or 0)

    if particle_indices is None:
        if total_particles <= 0:
            return values
        indices = list(range(total_particles))
    else:
        indices = [int(index) for index in particle_indices]

    if not indices:
        return values
    if min(indices) < 0:
        raise ValueError(f"{label}: particle indices must be non-negative")

    required_particles = max(indices) + 1
    if total_particles > 0 and required_particles > total_particles:
        raise ValueError(
            f"{label}: particle selection needs {required_particles} particle(s), "
            f"but Number of Particles is {total_particles}"
        )

    if kind == "scalar":
        if values.ndim != 2:
            return values
        if values.shape[1] < required_particles:
            raise ValueError(
                f"{label}: scalar input has {values.shape[1]} column(s), "
                f"but at least {required_particles} are required"
            )
        return values[:, indices]

    if kind == "vector":
        if values.ndim == 3:
            if values.shape[1] < required_particles:
                raise ValueError(
                    f"{label}: vector input has {values.shape[1]} particle vector(s), "
                    f"but at least {required_particles} are required"
                )
            return values[:, indices, :]

        if values.ndim != 2:
            return values
        required_columns = required_particles * 3
        if values.shape[1] < required_columns:
            raise ValueError(
                f"{label}: vector input has {values.shape[1]} column(s), "
                f"but at least {required_columns} are required"
            )
        columns: list[int] = []
        for index in indices:
            start = index * 3
            columns.extend([start, start + 1, start + 2])
        return values[:, columns]

    return values


def _calculate_correlation_file_task(params: dict[str, Any]) -> dict[str, Any]:
    index = params["index"]
    baseDir = params["baseDir"]
    array1_pattern = params["array1_pattern"]
    array2_pattern = params["array2_pattern"]
    output_pattern = params["output_pattern"]
    variance_output_pattern = params["variance_output_pattern"]
    common_term = params["common_term"]
    array1_io_spec = params["array1_io_spec"]
    array2_io_spec = params["array2_io_spec"]
    output_io_spec = params["output_io_spec"]

    array1_file = os.path.join(baseDir, _expand_pattern(array1_pattern, index, common_term))
    array2_file = os.path.join(baseDir, _expand_pattern(array2_pattern, index, common_term))
    output_file = os.path.join(baseDir, _expand_pattern(output_pattern, index, common_term))
    variance_output_file = os.path.join(baseDir, _expand_pattern(variance_output_pattern, index, common_term))

    try:
        array1 = load_numeric_array(
            array1_file,
            array1_io_spec,
            default_mode="text",
            default_precision="double",
        )
        if (
            os.path.abspath(array1_file) == os.path.abspath(array2_file)
            and array1_io_spec == array2_io_spec
        ):
            array2 = array1
        else:
            array2 = load_numeric_array(
                array2_file,
                array2_io_spec,
                default_mode="text",
                default_precision="double",
            )
        array1 = array1[:: params["array1_stride"]]
        if array2 is array1 and params["array2_stride"] == params["array1_stride"]:
            array2 = array1
        else:
            array2 = array2[:: params["array2_stride"]]
        array1 = _select_particle_columns(
            array1,
            array_kind=params["array1_kind"],
            num_particles=params["num_particles"],
            particle_indices=params["particle_indices"],
            label="Array 1",
        )
        array2 = array1 if array2 is array1 else _select_particle_columns(
            array2,
            array_kind=params["array2_kind"],
            num_particles=params["num_particles"],
            particle_indices=params["particle_indices"],
            label="Array 2",
        )
        variance, series = calculate_correlation(
            array1,
            array2,
            params["delta"],
            params["max_lag"],
            params["t1"],
            array1_kind=params["array1_kind"],
            array2_kind=params["array2_kind"],
            variable_mode=params["variable_mode"],
            subtract_mean=params["subtract_mean"],
            coef=params["coef"],
        )
        save_numeric_array(
            output_file,
            series,
            output_io_spec,
            default_mode="text",
            default_precision="double",
        )
        scaled_variance = float(variance) * float(params["coef"]) ** 2
        save_numeric_array(
            variance_output_file,
            np.array([[scaled_variance]], dtype=np.float64),
            output_io_spec,
            default_mode="text",
            default_precision="double",
        )
        return {
            "index": index,
            "output_file": output_file,
            "variance": scaled_variance,
            "error": None,
        }
    except Exception as exc:
        return {
            "index": index,
            "error": str(exc),
            "array1_file": array1_file,
            "array2_file": array2_file,
        }


def calculate_correlation_from_files(
    baseDir: str,
    array1_pattern: str,
    array2_pattern: str,
    array1_stride: int,
    array2_stride: int,
    output_pattern: str,
    variance_output_pattern: str,
    num_dcd: int,
    delta: int,
    max_lag: int,
    t1: float,
    *,
    array1_kind: str = "scalar",
    array2_kind: str = "scalar",
    variable_mode: str = "single",
    acf_mode: str = "acf",
    coef: float = 3.0,
    dcd_indices: list[int] | None = None,
    num_particles: int | None = None,
    particle_indices: list[int] | None = None,
    common_term: str = "",
    max_workers: int = 1,
    array1_io_spec: dict[str, Any] | None = None,
    array2_io_spec: dict[str, Any] | None = None,
    output_io_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not baseDir:
        raise ValueError("baseDir is required")
    if not array1_pattern or not array2_pattern or not output_pattern or not variance_output_pattern:
        raise ValueError("array1_pattern, array2_pattern, output_pattern, and variance_output_pattern are required")
    array1_stride = int(array1_stride)
    array2_stride = int(array2_stride)
    if array1_stride <= 0 or array2_stride <= 0:
        raise ValueError("array1_stride and array2_stride must be positive integers")

    indices = list(range(int(num_dcd))) if dcd_indices is None else [int(i) for i in dcd_indices]
    subtract_mean = str(acf_mode).strip().lower() == "acf"
    max_workers = int(max_workers)
    if max_workers <= 0:
        raise ValueError("max_workers must be a positive integer")

    successful: list[str] = []
    failed: list[dict[str, Any]] = []
    variances: dict[int, float] = {}

    tasks = [
        {
            "index": index,
            "baseDir": baseDir,
            "array1_pattern": array1_pattern,
            "array2_pattern": array2_pattern,
            "array1_stride": array1_stride,
            "array2_stride": array2_stride,
            "output_pattern": output_pattern,
            "variance_output_pattern": variance_output_pattern,
            "delta": delta,
            "max_lag": max_lag,
            "t1": t1,
            "array1_kind": array1_kind,
            "array2_kind": array2_kind,
            "variable_mode": variable_mode,
            "subtract_mean": subtract_mean,
            "coef": coef,
            "num_particles": num_particles,
            "particle_indices": particle_indices,
            "common_term": common_term,
            "array1_io_spec": array1_io_spec,
            "array2_io_spec": array2_io_spec,
            "output_io_spec": output_io_spec,
        }
        for index in indices
    ]

    if max_workers == 1 or len(tasks) <= 1:
        results = [_calculate_correlation_file_task(task) for task in tasks]
    else:
        worker_count = min(max_workers, len(tasks))
        results = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(_calculate_correlation_file_task, task): task["index"]
                for task in tasks
            }
            for future in as_completed(future_map):
                index = future_map[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append({"index": index, "error": str(exc)})

    for result in sorted(results, key=lambda item: item["index"]):
        if result.get("error"):
            failed.append({
                "index": result["index"],
                "error": result["error"],
                "array1_file": result.get("array1_file"),
                "array2_file": result.get("array2_file"),
            })
        else:
            successful.append(result["output_file"])
            variances[result["index"]] = result["variance"]

    return {
        "successful": successful,
        "failed": failed,
        "variances": variances,
        "success_count": len(successful),
        "failure_count": len(failed),
    }
