from __future__ import annotations

"""Correlation-function utilities for scalar and vector time series."""

import os
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
    arr1 = _center_array(_prepare_scalar_single(array1), subtract_mean)
    arr2 = _center_array(_prepare_scalar_single(array2), subtract_mean)
    if arr1.shape[0] != arr2.shape[0]:
        raise ValueError("scalar arrays must have the same number of frames")

    delta, max_lag = _validate_sampling(delta, max_lag, arr1.shape[0])

    def corr_at_lag(lag: int) -> float:
        return float(np.mean(arr1[: arr1.shape[0] - lag : delta] * arr2[lag::delta]))

    variance = corr_at_lag(0)
    values = np.array([corr_at_lag(lag) for lag in range(max_lag + 1)], dtype=np.float64)
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
    arr1 = _center_array(_prepare_scalar_multiple(array1), subtract_mean)
    arr2 = _center_array(_prepare_scalar_multiple(array2), subtract_mean)
    if arr1.shape != arr2.shape:
        raise ValueError("multiple scalar arrays must have identical shape")

    delta, max_lag = _validate_sampling(delta, max_lag, arr1.shape[0])

    def corr_at_lag(lag: int) -> np.ndarray:
        return np.mean(arr1[: arr1.shape[0] - lag : delta, :] * arr2[lag::delta, :], axis=0)

    variance_per_variable = corr_at_lag(0)
    variance = float(np.mean(variance_per_variable))
    values = np.array(
        [np.mean(corr_at_lag(lag)) for lag in range(max_lag + 1)],
        dtype=np.float64,
    )
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
    arr1 = _center_array(_prepare_vector_single(array1), subtract_mean)
    arr2 = _center_array(_prepare_vector_single(array2), subtract_mean)
    if arr1.shape != arr2.shape:
        raise ValueError("single vector arrays must have identical shape")

    delta, max_lag = _validate_sampling(delta, max_lag, arr1.shape[0])

    def corr_at_lag(lag: int) -> float:
        products = np.einsum("ij,ij->i", arr1[: arr1.shape[0] - lag : delta], arr2[lag::delta])
        return float(np.mean(products))

    variance = corr_at_lag(0)
    scale = float(coef) / 3.0
    values = np.array([scale * corr_at_lag(lag) for lag in range(max_lag + 1)], dtype=np.float64)
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
    arr1 = _center_array(_prepare_vector_multiple(array1), subtract_mean)
    arr2 = _center_array(_prepare_vector_multiple(array2), subtract_mean)
    if arr1.shape != arr2.shape:
        raise ValueError("multiple vector arrays must have identical shape")

    delta, max_lag = _validate_sampling(delta, max_lag, arr1.shape[0])

    def corr_at_lag(lag: int) -> np.ndarray:
        products = np.einsum("ijk,ijk->ij", arr1[: arr1.shape[0] - lag : delta], arr2[lag::delta])
        return np.mean(products, axis=0)

    variance_per_vector = corr_at_lag(0)
    variance = float(np.mean(variance_per_vector))
    scale = float(coef) / 3.0
    values = np.array(
        [scale * np.mean(corr_at_lag(lag)) for lag in range(max_lag + 1)],
        dtype=np.float64,
    )
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


def calculate_correlation_from_files(
    baseDir: str,
    array1_pattern: str,
    array2_pattern: str,
    output_pattern: str,
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
    common_term: str = "",
    array1_io_spec: dict[str, Any] | None = None,
    array2_io_spec: dict[str, Any] | None = None,
    output_io_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not baseDir:
        raise ValueError("baseDir is required")
    if not array1_pattern or not array2_pattern or not output_pattern:
        raise ValueError("array1_pattern, array2_pattern, and output_pattern are required")

    indices = list(range(int(num_dcd))) if dcd_indices is None else [int(i) for i in dcd_indices]
    subtract_mean = str(acf_mode).strip().lower() == "acf"

    successful: list[str] = []
    failed: list[dict[str, Any]] = []
    variances: dict[int, float] = {}

    for index in indices:
        array1_file = os.path.join(baseDir, _expand_pattern(array1_pattern, index, common_term))
        array2_file = os.path.join(baseDir, _expand_pattern(array2_pattern, index, common_term))
        output_file = os.path.join(baseDir, _expand_pattern(output_pattern, index, common_term))

        try:
            array1 = load_numeric_array(
                array1_file,
                array1_io_spec,
                default_mode="text",
                default_precision="double",
            )
            array2 = load_numeric_array(
                array2_file,
                array2_io_spec,
                default_mode="text",
                default_precision="double",
            )
            variance, series = calculate_correlation(
                array1,
                array2,
                delta,
                max_lag,
                t1,
                array1_kind=array1_kind,
                array2_kind=array2_kind,
                variable_mode=variable_mode,
                subtract_mean=subtract_mean,
                coef=coef,
            )
            save_numeric_array(
                output_file,
                series,
                output_io_spec,
                default_mode="text",
                default_precision="double",
            )
            successful.append(output_file)
            variances[index] = variance
        except Exception as exc:
            failed.append({"index": index, "error": str(exc), "array1_file": array1_file, "array2_file": array2_file})

    return {
        "successful": successful,
        "failed": failed,
        "variances": variances,
        "success_count": len(successful),
        "failure_count": len(failed),
    }
