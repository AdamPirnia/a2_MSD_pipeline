import os
from typing import Any, Sequence

import numpy as np


DEFAULT_SINGLE_DECIMALS = 6
DEFAULT_DOUBLE_DECIMALS = 15


def _load_ragged_text_array(input_file: str, dtype: np.dtype) -> np.ndarray:
    """Load whitespace-delimited text and trim ragged rows to the shared minimum width."""
    rows: list[np.ndarray] = []
    min_columns: int | None = None

    with open(input_file, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            try:
                row = np.asarray(stripped.split(), dtype=dtype)
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric data encountered in {input_file} on line {line_number}"
                ) from exc

            if row.size == 0:
                continue

            rows.append(row)
            min_columns = row.size if min_columns is None else min(min_columns, row.size)

    if not rows:
        raise ValueError(f"No numeric data found in text file: {input_file}")
    if min_columns is None or min_columns <= 0:
        raise ValueError(f"Unable to determine usable columns for text file: {input_file}")

    if any(row.size != min_columns for row in rows):
        rows = [row[:min_columns] for row in rows]

    return np.vstack(rows)


def normalize_io_spec(
    spec: dict[str, Any] | None,
    *,
    default_mode: str,
    default_precision: str,
    default_decimals: int | None = None,
) -> dict[str, Any]:
    spec = dict(spec or {})
    mode = str(spec.get("mode") or default_mode).strip().lower()
    precision = str(spec.get("precision") or default_precision).strip().lower()
    decimals = spec.get("decimals", default_decimals)

    if mode not in {"binary", "text"}:
        mode = default_mode
    if precision not in {"single", "double", "custom"}:
        precision = default_precision
    if precision == "custom":
        try:
            decimals = int(decimals)
        except Exception:
            decimals = DEFAULT_SINGLE_DECIMALS
        decimals = max(0, decimals)
    else:
        decimals = None

    return {
        "mode": mode,
        "precision": precision,
        "decimals": decimals,
    }


def dtype_for_spec(spec: dict[str, Any]) -> np.dtype:
    precision = str(spec["precision"])
    if precision == "single":
        return np.float32
    return np.float64


def text_format_for_spec(spec: dict[str, Any]) -> str:
    precision = str(spec["precision"])
    if precision == "single":
        decimals = DEFAULT_SINGLE_DECIMALS
    elif precision == "double":
        decimals = DEFAULT_DOUBLE_DECIMALS
    else:
        decimals = int(spec["decimals"])
    return f"%.{decimals}f"


def decimals_for_spec(spec: dict[str, Any]) -> int:
    precision = str(spec["precision"])
    if precision == "single":
        return DEFAULT_SINGLE_DECIMALS
    if precision == "double":
        return DEFAULT_DOUBLE_DECIMALS
    return int(spec["decimals"])


def _round_if_needed(data: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    if str(spec["precision"]) == "custom":
        return np.round(np.asarray(data, dtype=np.float64), int(spec["decimals"]))
    return data


def _format_fixed_width_value(value: Any, decimals: int, *, compact: bool) -> str:
    number = float(value)
    if not np.isfinite(number):
        return str(number)
    text = f"{number:.{decimals}f}"
    if compact:
        text = text.rstrip("0").rstrip(".")
        if text in {"", "-0"}:
            text = "0"
    return text


def _write_fixed_width_text_array(
    output_file: str,
    array: np.ndarray,
    *,
    decimals: int,
    header: str | None = None,
    column_names: Sequence[str] | None = None,
    compact_columns: Sequence[int] | None = None,
) -> None:
    data = np.asarray(array)
    if data.ndim == 0:
        data = data.reshape(1, 1)
    elif data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.ndim > 2:
        data = data.reshape(data.shape[0], -1)

    compact_set = {int(index) for index in (compact_columns or [])}
    rows: list[list[str]] = []
    for row in data:
        rows.append(
            [
                _format_fixed_width_value(value, decimals, compact=column_index in compact_set)
                for column_index, value in enumerate(row)
            ]
        )

    column_count = int(data.shape[1]) if data.ndim == 2 else 1
    names = [str(name) for name in (column_names or [])]
    if len(names) < column_count:
        names.extend(f"col_{index + 1}" for index in range(len(names), column_count))
    names = names[:column_count]
    widths = [len(names[index]) for index in range(column_count)]
    for row in rows:
        for index, text in enumerate(row):
            widths[index] = max(widths[index], len(text))

    with open(output_file, "w", encoding="utf-8") as handle:
        if header:
            for line in str(header).splitlines():
                handle.write(f"# {line}\n")
        handle.write("# " + " ".join(f"{names[index]:>{widths[index]}}" for index in range(column_count)) + "\n")
        for row in rows:
            handle.write(" ".join(f"{row[index]:>{widths[index]}}" for index in range(column_count)) + "\n")


def load_numeric_array(
    input_file: str,
    spec: dict[str, Any] | None,
    *,
    default_mode: str,
    default_precision: str,
    default_decimals: int | None = None,
    mmap_mode: str | None = None,
) -> np.ndarray:
    io_spec = normalize_io_spec(
        spec,
        default_mode=default_mode,
        default_precision=default_precision,
        default_decimals=default_decimals,
    )
    if io_spec["mode"] == "binary":
        try:
            if mmap_mode:
                data = np.load(input_file, allow_pickle=False, mmap_mode=mmap_mode)
            else:
                with open(input_file, "rb") as fh:
                    data = np.load(fh, allow_pickle=False)
        except Exception as exc:
            raise ValueError(f"Input file must be binary NumPy format: {input_file}") from exc
    else:
        try:
            data = _load_ragged_text_array(input_file, dtype_for_spec(io_spec))
        except Exception as exc:
            raise ValueError(f"Input file must be plain text numeric data: {input_file}") from exc

    data = np.asarray(data, dtype=dtype_for_spec(io_spec))
    data = _round_if_needed(data, io_spec)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def save_numeric_array(
    output_file: str,
    data: np.ndarray,
    spec: dict[str, Any] | None,
    *,
    default_mode: str,
    default_precision: str,
    default_decimals: int | None = None,
    delimiter: str = " ",
    header: str | None = None,
    column_names: Sequence[str] | None = None,
    compact_columns: Sequence[int] | None = None,
    result_format: bool = False,
) -> dict[str, Any]:
    io_spec = normalize_io_spec(
        spec,
        default_mode=default_mode,
        default_precision=default_precision,
        default_decimals=default_decimals,
    )
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    array = np.asarray(data, dtype=dtype_for_spec(io_spec))
    array = _round_if_needed(array, io_spec)

    if io_spec["mode"] == "binary":
        with open(output_file, "wb") as fh:
            np.save(fh, array, allow_pickle=False)
    elif result_format or column_names is not None:
        _write_fixed_width_text_array(
            output_file,
            array,
            decimals=decimals_for_spec(io_spec),
            header=header,
            column_names=column_names,
            compact_columns=compact_columns,
        )
    else:
        np.savetxt(
            output_file,
            array,
            fmt=text_format_for_spec(io_spec),
            delimiter=delimiter,
            header="" if header is None else str(header),
            comments="# ",
        )

    return io_spec
