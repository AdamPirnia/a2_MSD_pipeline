import os
import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import struct
from difflib import get_close_matches
import numpy as np
try:
    from .numeric_io import (
        DEFAULT_DOUBLE_DECIMALS,
        DEFAULT_SINGLE_DECIMALS,
    )
except ImportError:
    from numeric_io import (
        DEFAULT_DOUBLE_DECIMALS,
        DEFAULT_SINGLE_DECIMALS,
    )
import re
from collections import deque
# Inlined path_utils functions to keep generated/runtime scripts self-contained
_SAFE_INDEX_EXPR = re.compile(r"^i(?:\s*[+\-*/]\s*\d+)?$")

def expand_path_pattern(pattern, common_param="", file_index=None):
    """Expand a path pattern with common parameter and file index."""
    if pattern is None:
        return ""
    
    result = str(pattern)
    
    # Replace * with common parameter
    if common_param is not None:
        result = result.replace('*', str(common_param))
    
    # Replace {i} expressions with file index if provided
    if file_index is not None:
        import re
        
        def replace_expression(match):
            expr = match.group(1).strip()
            try:
                namespace = {'i': file_index}
                if _SAFE_INDEX_EXPR.match(expr):
                    return str(eval(expr, {"__builtins__": {}}, namespace))
                else:
                    return match.group(0)
            except:
                return match.group(0)
        
        result = re.sub(r'\{([^}]+)\}', replace_expression, result)
    
    return result

def validate_path_pattern(pattern):
    """Validate a path pattern for correct syntax."""
    if not pattern:
        return True, ""
    
    # Check for unmatched braces
    if pattern.count('{') != pattern.count('}'):
        return False, "Unmatched braces in pattern"

    placeholders = re.findall(r'\{([^}]+)\}', pattern)
    invalid = [f"{{{expr}}}" for expr in placeholders if not _SAFE_INDEX_EXPR.match(expr.strip())]
    if invalid:
        return False, (
            f"Unsupported format placeholders: {', '.join(invalid)}. "
            "Supported forms are {i}, {i+N}, {i-N}, {i*N}, and {i/N}."
        )
    
    return True, ""


def _tcl_quote(value):
    """Return a Tcl-safe double-quoted string literal body."""
    text = str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace("\"", "\\\"")
    text = text.replace("$", "\\$")
    text = text.replace("[", "\\[")
    text = text.replace("]", "\\]")
    text = text.replace("\n", "\\n")
    text = text.replace("\r", "\\r")
    return text


def _format_missing_file_message(label, path):
    directory = os.path.dirname(path)
    basename = os.path.basename(path)
    details = [f"{label} file not found: {path}"]

    if os.path.isdir(directory):
        try:
            siblings = [name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]
        except OSError:
            siblings = []
        matches = get_close_matches(basename, siblings, n=3, cutoff=0.5)
        if matches:
            details.append(f"Closest match(es) in {directory}: {', '.join(matches)}")
    else:
        details.append(f"Parent directory does not exist: {directory}")

    return ". ".join(details)


def _format_unreadable_file_message(label, path):
    details = [f"{label} file exists but is empty or unavailable locally: {path}"]
    details.append(
        "If this file is in Dropbox, iCloud, OneDrive, or another cloud-synced folder, "
        "make it available offline before running the analysis."
    )
    return " ".join(details)


def _normalize_output_io_spec(output_io_spec):
    spec = dict(output_io_spec or {})
    mode = str(spec.get("mode") or "binary").strip().lower()
    precision = str(spec.get("precision") or "single").strip().lower()
    decimals = spec.get("decimals")

    if mode not in {"binary", "text"}:
        mode = "binary"
    if precision not in {"single", "double", "custom"}:
        precision = "single"
    if precision == "custom":
        try:
            decimals = max(0, int(decimals))
        except Exception:
            decimals = DEFAULT_SINGLE_DECIMALS
    else:
        decimals = None

    return {"mode": mode, "precision": precision, "decimals": decimals}


def _raw_binary_output_path(output_file):
    return f"{output_file}.rawf32"


def _raw_binary_shape_path(output_file):
    return f"{output_file}.shape"


def _vmd_coordinate_temp_path(output_file):
    return f"{output_file}.vmdtmp.bin"


def _npy_writing_path(output_file):
    return f"{output_file}.writing.npy"


def _coords_status_path(common_term, index):
    suffix = f"{common_term}_{index}" if common_term else str(index)
    return os.path.join("logs", f"coords_status_{suffix}.log")


def _cleanup_binary_coordinate_temp_files(output_file):
    for temp_file in (
        _vmd_coordinate_temp_path(output_file),
        _raw_binary_output_path(output_file),
        _raw_binary_shape_path(output_file),
        _npy_writing_path(output_file),
        f"{output_file}.writing",
    ):
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except OSError:
            pass


def _vmd_temp_dtype_for_spec(output_io_spec):
    precision = str(output_io_spec.get("precision") or "single").lower()
    if precision == "single":
        return {"label": "f4", "tcl_format": "r3", "dtype": np.float32}
    return {"label": "f8", "tcl_format": "d3", "dtype": np.float64}


def _output_dtype_for_spec(output_io_spec):
    precision = str(output_io_spec.get("precision") or "single").lower()
    if precision == "single":
        return np.float32
    return np.float64


def _text_format_string(io_spec):
    if io_spec["precision"] == "double":
        decimals = DEFAULT_DOUBLE_DECIMALS
    elif io_spec["precision"] == "custom":
        decimals = int(io_spec["decimals"])
    else:
        decimals = DEFAULT_SINGLE_DECIMALS
    return f"%.{decimals}f"


def _read_dcd_frame_count(dcd_file):
    """Read the frame count from a standard DCD header without loading coordinates."""
    with open(dcd_file, "rb") as handle:
        first_record = handle.read(4)
        if len(first_record) != 4:
            raise ValueError(f"DCD header is too short: {dcd_file}")

        record_size = struct.unpack("<i", first_record)[0]
        endian = "<"
        if record_size != 84:
            record_size = struct.unpack(">i", first_record)[0]
            endian = ">"
        if record_size != 84:
            raise ValueError(f"Unsupported DCD header record size in {dcd_file}: {record_size}")

        magic = handle.read(4)
        if magic not in {b"CORD", b"VELD"}:
            raise ValueError(f"Unsupported DCD magic {magic!r} in {dcd_file}")

        nset_bytes = handle.read(4)
        if len(nset_bytes) != 4:
            raise ValueError(f"DCD frame count is missing from header: {dcd_file}")

    n_frames = struct.unpack(f"{endian}i", nset_bytes)[0]
    if n_frames <= 0:
        raise ValueError(f"DCD header reports an invalid frame count ({n_frames}): {dcd_file}")
    return n_frames


def _read_status_tail(status_path, max_lines=20):
    if not os.path.exists(status_path):
        return ""
    try:
        with open(status_path, "r", encoding="utf-8", errors="replace") as handle:
            lines = deque(handle, maxlen=max_lines)
    except OSError:
        return ""
    return "".join(lines).strip()


def _format_status_tail(status_path):
    tail = _read_status_tail(status_path)
    if not tail:
        return f"No VMD status file was written at {status_path}."
    return f"Last VMD status lines from {status_path}:\n{tail}"


def _read_vmd_temp_coordinate_header(handle, temp_file):
    header_start = handle.tell()
    header_line = handle.readline(256)
    header_end = handle.tell()
    if not header_line.endswith(b"\n"):
        raise ValueError(f"Invalid or truncated VMD coordinate temp header in {temp_file}")
    header = header_line.decode("ascii", errors="replace").strip().split()
    if len(header) != 4 or header[0] != "ADMDYNANLZ_COORD_BIN_V1":
        raise ValueError(f"Invalid VMD coordinate temp header in {temp_file}")

    dtype_token = header[1]
    try:
        n_frames = int(header[2])
        n_columns = int(header[3])
    except ValueError as exc:
        raise ValueError(f"Invalid VMD coordinate temp shape in {temp_file}") from exc

    if dtype_token == "f4":
        dtype = np.float32
    elif dtype_token == "f8":
        dtype = np.float64
    else:
        raise ValueError(f"Unsupported VMD coordinate temp dtype {dtype_token!r} in {temp_file}")

    if n_frames <= 0 or n_columns <= 0:
        raise ValueError(f"VMD coordinate temp has invalid shape ({n_frames}, {n_columns}): {temp_file}")

    return dtype, n_frames, n_columns, header_end - header_start


def _stream_vmd_temp_coordinate_output(temp_file, output_file, output_io_spec, conversion_chunk_rows=None):
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(temp_file, "rb") as handle:
        temp_dtype, n_frames, n_columns, header_bytes = _read_vmd_temp_coordinate_header(handle, temp_file)
        payload_bytes = os.path.getsize(temp_file) - header_bytes
        expected_values = n_frames * n_columns
        expected_bytes = expected_values * np.dtype(temp_dtype).itemsize
        if payload_bytes != expected_bytes:
            found_values = payload_bytes // np.dtype(temp_dtype).itemsize
            raise ValueError(
                f"VMD coordinate temp size mismatch for {temp_file}: "
                f"expected {expected_values} values, found {found_values}"
            )

        output_dtype = _output_dtype_for_spec(output_io_spec)
        chunk_rows = os.environ.get("ADMDYNANLZ_CONVERT_CHUNK_ROWS", "8") if conversion_chunk_rows is None else conversion_chunk_rows
        try:
            chunk_rows = max(1, int(chunk_rows))
        except (TypeError, ValueError):
            chunk_rows = 8

        if output_io_spec["mode"] == "binary":
            temp_output_file = _npy_writing_path(output_file)
            if os.path.exists(temp_output_file):
                os.remove(temp_output_file)
            array = np.lib.format.open_memmap(
                temp_output_file,
                mode="w+",
                dtype=output_dtype,
                shape=(n_frames, n_columns),
            )
            for row_start in range(0, n_frames, chunk_rows):
                rows = min(chunk_rows, n_frames - row_start)
                chunk = np.fromfile(handle, dtype=temp_dtype, count=rows * n_columns)
                if chunk.size != rows * n_columns:
                    raise ValueError(f"Unexpected EOF while reading VMD coordinate temp output: {temp_file}")
                chunk = chunk.reshape(rows, n_columns)
                if output_io_spec["precision"] == "custom":
                    chunk = np.round(chunk.astype(output_dtype, copy=False), int(output_io_spec["decimals"]))
                elif chunk.dtype != output_dtype:
                    chunk = chunk.astype(output_dtype, copy=False)
                array[row_start:row_start + rows, :] = chunk
            array.flush()
            del array
            os.replace(temp_output_file, output_file)
            return

        temp_output_file = f"{output_file}.writing"
        fmt = _text_format_string(output_io_spec)
        with open(temp_output_file, "w", encoding="utf-8") as out_handle:
            for row_start in range(0, n_frames, chunk_rows):
                rows = min(chunk_rows, n_frames - row_start)
                chunk = np.fromfile(handle, dtype=temp_dtype, count=rows * n_columns)
                if chunk.size != rows * n_columns:
                    raise ValueError(f"Unexpected EOF while reading VMD coordinate temp output: {temp_file}")
                chunk = chunk.reshape(rows, n_columns)
                if output_io_spec["precision"] == "custom":
                    chunk = np.round(chunk.astype(output_dtype, copy=False), int(output_io_spec["decimals"]))
                elif chunk.dtype != output_dtype:
                    chunk = chunk.astype(output_dtype, copy=False)
                np.savetxt(out_handle, chunk, fmt=fmt, delimiter=" ")
        os.replace(temp_output_file, output_file)


def _finalize_vmd_temp_coordinate_output(temp_file, output_file, output_io_spec, min_temp_mtime=None, status_path=None, wait_seconds=60, conversion_chunk_rows=None):
    deadline = time.time() + max(0, wait_seconds)
    last_error = None
    while True:
        try:
            if not os.path.exists(temp_file):
                raise FileNotFoundError(f"VMD coordinate temp output not found: {temp_file}")
            if min_temp_mtime is not None and os.path.getmtime(temp_file) < min_temp_mtime:
                raise FileNotFoundError(f"VMD did not update coordinate temp output: {temp_file}")
            if os.path.getsize(temp_file) <= 0:
                raise ValueError(f"VMD wrote an empty coordinate temp output file: {temp_file}")
            _stream_vmd_temp_coordinate_output(temp_file, output_file, output_io_spec, conversion_chunk_rows)
            os.remove(temp_file)
            return
        except Exception as exc:
            last_error = exc
            if time.time() >= deadline:
                details = _format_status_tail(status_path) if status_path else ""
                if details:
                    raise type(last_error)(f"{last_error}\n{details}") from last_error
                raise last_error
            time.sleep(1)


def raw_coords(baseDir, psf_pattern, dcd_pattern, output_pattern=None, num_dcd=None, target_selection=None, save_com=False, grouping_unit="residue", vmd=None, max_workers=None, dcd_indices=None, common_term="", stride=1, wrap_settings=None, output_io_spec=None, vmd_frame_batch_size=None, conversion_chunk_rows=None):
    """
    Extracts raw XYZ coordinates from a series of DCD trajectories using VMD in parallel,
    by generating per-segment Tcl scripts, running them in batch, and saving
    the results directly as NumPy `.npy` files.

    For each trajectory chunk, this function:
      1. Creates output directories based on output pattern.
      2. Writes a VMD Tcl script that:
         - Loads the PSF and DCD for that chunk using the expanded patterns.
         - Iterates over every frame.
         - Selects atoms using the target_selection criteria.
         - Extracts their x,y,z coordinates efficiently.
         - Writes results using the output pattern.
      3. Invokes VMD in text mode (`-dispdev text`) in parallel processes.
      4. Captures stdout/stderr to `logs/log_{i}.log`.

    Parameters
    ----------
    baseDir : str
        Root directory for the simulation.
    psf_pattern : str
        Path pattern for PSF files. Can contain * (common term) and {i} (file index).
        Example: "trajectories_*/run_{i}/system.psf"
    dcd_pattern : str
        Path pattern for DCD files. Can contain * (common term) and {i} (file index).
        Example: "trajectories_*/run_{i}/traj.dcd"
    output_pattern : str
        Path pattern for output files. Can contain * (common term) and {i} (file index).
        Example: "path/to/xyz_{i}.npy"
    num_dcd : int
        Number of trajectory chunks to process.  Generates scripts for
        indices `0` through `num_dcd-1`.
    target_selection : str
        VMD atom selection string (e.g., "resname WAT and residue 0 to 999", "water", "protein").
    save_com : bool, optional
        If True, save one center of mass per grouping unit instead of raw atom coordinates.
    grouping_unit : str, optional
        Grouping unit used when save_com=True. One of residue, chain, segname.
    vmd : str
        The path to the VMD executable.
    max_workers : int, optional
        Maximum number of parallel workers. Defaults to min(num_dcd, CPU count).
    dcd_indices : list, optional
        List of DCD indices to process (e.g., [0, 1, 4, 5] to process only DCDs 0, 1, 4, 5).
        If None, processes all DCDs from 0 to num_dcd-1. Default is None.
    common_term : str, optional
        Value to replace * placeholders in patterns. Default is "".
    stride : int, optional
        Frame stride for extraction. Use 1 to save every frame, 2 to save every
        other frame, etc. Default is 1.
    wrap_settings : dict, optional
        Optional pbctools wrapping configuration. When enabled, a `pbc wrap`
        command is inserted before coordinates are extracted.
    vmd_frame_batch_size : int, optional
        Number of saved frames loaded into VMD in each DCD batch.
    conversion_chunk_rows : int, optional
        Number of frame rows converted from the VMD temporary binary output at once.
    
    Side Effects
    ------------
    - Creates directories:
        - `{baseDir}/{output_pattern}`
        - `writenCodes`
        - `logs`
    - Writes Tcl scripts to `writenCodes/coords_{i}.tcl`.
    - Runs VMD on each script in parallel, logging to `logs/log_{i}.log`.
    - Produces coordinate files `xyz_{i}.npy` in `{baseDir}/{output_pattern}`.

    Returns
    -------
    dict
        Summary of processing results with success/failure counts and timing.

    Example
    -------
    >>> results = raw_coords(
    ...     baseDir="/home/user/sim",
    ...     psf_pattern="trajectories_*/run_{i}/system.psf",
    ...     dcd_pattern="trajectories_*/run_{i}/traj.dcd",
    ...     output_pattern="path/to/xyz_{i}.npy",
    ...     num_dcd=6,
    ...     target_selection="resname WAT and residue 0 to 999",
    ...     vmd="/usr/local/bin/vmd",
    ...     common_term="240"
    ... )
    # Creates:
    #   /home/user/sim/path/to/xyz_0.npy, …, path/to/xyz_5.npy
    #   writenCodes/coords_0.tcl, …, coords_5.tcl
    #   logs/log_0.log, …, log_5.log
    """
    
    start_time = time.time()
    
    # Validate path patterns
    is_valid, error_msg = validate_path_pattern(psf_pattern)
    if not is_valid:
        raise ValueError(f"Invalid PSF pattern: {error_msg}")
        
    is_valid, error_msg = validate_path_pattern(dcd_pattern)
    if not is_valid:
        raise ValueError(f"Invalid DCD pattern: {error_msg}")
        
    is_valid, error_msg = validate_path_pattern(output_pattern)
    if not is_valid:
        raise ValueError(f"Invalid output pattern: {error_msg}")
    
    print(f"{'='*50}")
    print(f"COORDINATE EXTRACTION")
    print(f"{'='*50}")
    print(f"Base directory: {baseDir}")
    print(f"PSF pattern: {psf_pattern}")
    print(f"DCD pattern: {dcd_pattern}")
    print(f"Output pattern: {output_pattern}")
    print(f"Common term: {common_term}")
    print(f"Number of DCDs: {num_dcd}")
    print(f"Target selection: {target_selection}")
    print(f"Save COM: {save_com}")
    if save_com:
        print(f"COM grouping unit: {grouping_unit}")
    print(f"VMD executable: {vmd}")
    print(f"Stride: {stride}")
    print(f"Wrap enabled: {bool((wrap_settings or {}).get('enabled'))}")

    if save_com and grouping_unit not in {"residue", "chain", "segname"}:
        raise ValueError("grouping_unit must be one of residue, chain, segname when save_com=True")

    # Validate stride
    try:
        stride = int(stride)
    except (TypeError, ValueError):
        raise ValueError("stride must be an integer")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    def normalize_optional_positive_int(value, label):
        if value is None or str(value).strip() == "":
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be an integer >= 1") from exc
        if parsed < 1:
            raise ValueError(f"{label} must be >= 1")
        return parsed

    vmd_frame_batch_size = normalize_optional_positive_int(vmd_frame_batch_size, "VMD frame batch size")
    conversion_chunk_rows = normalize_optional_positive_int(conversion_chunk_rows, "Python conversion chunk rows")
    
    # Determine indices to process
    if dcd_indices is None:
        dcd_list = list(range(num_dcd))
    else:
        dcd_list = dcd_indices
        print(f"Processing selected DCDs: {dcd_list}")
    
    if max_workers is None:
        max_workers = min(len(dcd_list), mp.cpu_count())
    
    print(f"Using {max_workers} parallel workers")
    
    # Create necessary directories
    # Create output directory from first output file pattern
    if dcd_list:
        first_output = expand_path_pattern(output_pattern, common_term, dcd_list[0])
        output_full_path = os.path.join(baseDir, first_output)
        output_dir = os.path.dirname(output_full_path)
        
        print(f"Output pattern: {output_pattern}")
        print(f"Expanded pattern: {first_output}")
        print(f"Full output path: {output_full_path}")
        print(f"Output directory: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
    os.makedirs("writenCodes", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Note: With flexible target_selection, we can't easily estimate output size
    # The actual number of selected atoms will be determined by VMD at runtime
    
    # Validate that input files exist for first DCD (sanity check)
    if dcd_list:
        test_psf = os.path.join(baseDir, expand_path_pattern(psf_pattern, common_term, dcd_list[0]))
        test_dcd = os.path.join(baseDir, expand_path_pattern(dcd_pattern, common_term, dcd_list[0]))
        
        if not os.path.exists(test_psf):
            raise FileNotFoundError(_format_missing_file_message("PSF", test_psf))
        if not os.path.exists(test_dcd):
            raise FileNotFoundError(_format_missing_file_message("DCD", test_dcd))
        if os.path.getsize(test_psf) <= 0:
            raise ValueError(_format_unreadable_file_message("PSF", test_psf))
        if os.path.getsize(test_dcd) <= 0:
            raise ValueError(_format_unreadable_file_message("DCD", test_dcd))
        
        print(f"✓ Input files validated for index {dcd_list[0]}")
    
    # Results tracking
    results = {
        'successful': [],
        'failed': [],
        'total_time': 0,
        'num_processed': len(dcd_list)
    }
    
    # Generate all TCL scripts first with pattern validation
    for i in dcd_list:
        success = _write_tcl_script(
            i,
            baseDir,
            psf_pattern,
            dcd_pattern,
            output_pattern,
            target_selection,
            save_com,
            grouping_unit,
            common_term,
            stride,
            wrap_settings,
            output_io_spec,
            vmd_frame_batch_size,
        )
        if not success:
            print(f"ERROR: Failed to generate TCL script for coordinates chunk {i} due to pattern validation failure.")
            print("Please fix the corrupted patterns in your GUI input fields and try again.")
            return {
                'successful': [],
                'failed': dcd_list,
                'total_time': 0,
                'num_processed': len(dcd_list),
                'missing_files': []
            }
    
    # Process VMD scripts in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_index = {
            executor.submit(
                _run_vmd_script, i, vmd, baseDir, output_pattern, common_term, output_io_spec, conversion_chunk_rows
            ): i 
            for i in dcd_list
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                success, stdout, stderr = future.result()
                if success:
                    results['successful'].append(index)
                    print(f"✓ Completed trajectory chunk {index}")
                else:
                    results['failed'].append(index)
                    print(f"✗ Failed trajectory chunk {index}")
                    if stderr:
                        print(f"  Error: {stderr}")
                        
            except Exception as exc:
                results['failed'].append(index)
                print(f"✗ Exception in trajectory chunk {index}: {exc}")
    
    results['total_time'] = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"COORDINATE EXTRACTION SUMMARY")
    print(f"{'='*50}")
    print(f"Total chunks: {num_dcd}")
    print(f"Successful: {len(results['successful'])}")
    print(f"Failed: {len(results['failed'])}")
    if results['failed']:
        print(f"Failed indices: {results['failed']}")
    print(f"Total time: {results['total_time']:.2f} seconds")
    print(f"Parallel efficiency: {max_workers} workers")
    print(f"{'='*50}\n")
    
    # Note: Output shape validation removed since expected count depends on VMD selection
    
    return results



def _write_tcl_script(
    i,
    baseDir,
    psf_pattern,
    dcd_pattern,
    output_pattern,
    target_selection,
    save_com=False,
    grouping_unit="residue",
    common_term="",
    stride=1,
    wrap_settings=None,
    output_io_spec=None,
    vmd_frame_batch_size=None,
):
    """Write optimized TCL script for a single trajectory chunk."""
    
    # Expand patterns to get actual file paths
    psf_path = expand_path_pattern(psf_pattern, common_term, i)
    dcd_path = expand_path_pattern(dcd_pattern, common_term, i)
    output_path = expand_path_pattern(output_pattern, common_term, i)
    
    # Validate patterns to ensure they match user input expectations
    def validate_coords_pattern_consistency(pattern_name, pattern, expanded_path, common_term, index):
        """Validate that the expanded pattern matches what should be expected from user input"""
        if not pattern or not common_term:
            return True  # Skip validation if no pattern or common term
        
        # Create the expected expanded pattern by manually expanding the user's input
        expected_pattern = pattern.replace('*', common_term)
        # Replace {i} with the actual index
        import re
        def replace_index_expressions(match):
            expr = match.group(1)
            try:
                namespace = {'i': index}
                if re.match(r'^i[+\-*/]\d+$|^i$', expr):
                    return str(eval(expr, {"__builtins__": {}}, namespace))
                else:
                    return match.group(0)
            except:
                return match.group(0)
        
        expected_pattern = re.sub(r'\{([^}]+)\}', replace_index_expressions, expected_pattern)
        
        # Compare the actual expanded path with the expected pattern
        if expanded_path != expected_pattern:
            print(f"ERROR: {pattern_name} pattern expansion mismatch!")
            print(f"  User input pattern: {pattern}")
            print(f"  Common term: {common_term}")
            print(f"  Index: {index}")
            print(f"  Expected expanded: {expected_pattern}")
            print(f"  Actual expanded:   {expanded_path}")
            print(f"  This indicates the pattern may contain corrupted data or unexpected terms.")
            print(f"  Please verify your GUI input fields contain clean patterns with only * and {{i}} placeholders.")
            return False
        
        return True
    
    # Validate PSF, DCD, and output patterns
    psf_valid = validate_coords_pattern_consistency("PSF", psf_pattern, psf_path, common_term, i)
    dcd_valid = validate_coords_pattern_consistency("DCD", dcd_pattern, dcd_path, common_term, i)
    output_valid = validate_coords_pattern_consistency("Output", output_pattern, output_path, common_term, i)
    
    if not psf_valid or not dcd_valid or not output_valid:
        print(f"ERROR: Pattern validation failed for coordinates chunk {i}")
        print(f"Please fix the corrupted patterns in your GUI input fields before proceeding.")
        return False
    
    # Create absolute paths
    psf_full_path = os.path.join(baseDir, psf_path)
    dcd_full_path = os.path.join(baseDir, dcd_path)
    output_full_path = os.path.join(baseDir, output_path)
    dcd_frame_count = _read_dcd_frame_count(dcd_full_path)
    
    print(f"DEBUG TCL Script {i}:")
    print(f"  Output pattern: {output_pattern}")
    print(f"  Expanded output: {output_path}")
    print(f"  Full output path: {output_full_path}")
    print(f"  Output directory: {os.path.dirname(output_full_path)}")
    print(f"  DCD frames: {dcd_frame_count}")
    
    # Ensure output directory exists before creating TCL script
    output_dir = os.path.dirname(output_full_path)
    if output_dir and output_dir != baseDir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create more specific TCL filename with common term
    tcl_filename = f"coords_{common_term}_{i}.tcl" if common_term else f"coords_{i}.tcl"
    status_path = _coords_status_path(common_term, i)

    wrap_settings = wrap_settings or {}
    normalized_output = _normalize_output_io_spec(output_io_spec)
    wrap_enabled = bool(wrap_settings.get("enabled"))
    atomselection = wrap_settings.get("atomselection", "").strip() or target_selection or "all"
    wrap_option_flags = wrap_settings.get("options", {})

    wrap_lines = []
    if wrap_enabled:
        wrap_shape = wrap_settings.get("shape") or "parallelepiped"
        wrap_compound = wrap_settings.get("compound")
        wrap_center = wrap_settings.get("center") or "unitcell"

        wrap_lines.extend([
            "",
            "# Load pbctools and wrap coordinates before extraction",
            "if {[catch {package require pbctools} pbctools_err]} {",
            '    abort_with_error "Failed to load pbctools package: $pbctools_err"',
            "}",
            "set wrap_args [list -molid $molid -now]",
        ])

        if wrap_shape in {"parallelepiped", "orthorhombic"}:
            wrap_lines.append(f'lappend wrap_args -{wrap_shape}')
        if wrap_option_flags.get("sel"):
            wrap_lines.append(f'lappend wrap_args -sel "{atomselection}"')
        if wrap_compound in {"res", "segid", "chain", "fragment"}:
            wrap_lines.append(f'lappend wrap_args -compound {wrap_compound}')
        if wrap_option_flags.get("compoundref"):
            wrap_lines.append(f'lappend wrap_args -compoundref "{atomselection}"')
        if wrap_center in {"origin", "unitcell", "com", "bb"}:
            wrap_lines.append(f'lappend wrap_args -center {wrap_center}')
        if wrap_option_flags.get("centersel"):
            wrap_lines.append(f'lappend wrap_args -centersel "{atomselection}"')
        if wrap_option_flags.get("verbose"):
            wrap_lines.append("lappend wrap_args -verbose")
        if wrap_option_flags.get("draw"):
            wrap_lines.append("lappend wrap_args -draw")

        wrap_lines.extend([
            'puts "Configured per-frame wrapping with: pbc wrap [join $wrap_args { }]"',
        ])

    wrap_block = "\n".join(wrap_lines)
    temp_output_full_path = _vmd_coordinate_temp_path(output_full_path)
    temp_dtype = _vmd_temp_dtype_for_spec(normalized_output)
    frame_batch_value = os.environ.get("ADMDYNANLZ_VMD_FRAME_BATCH_SIZE", "25") if vmd_frame_batch_size is None else vmd_frame_batch_size
    try:
        frame_batch_size = max(1, int(frame_batch_value))
    except (TypeError, ValueError):
        frame_batch_size = 25
    output_open_block = "\n".join([
        f'set outfile [open "{_tcl_quote(temp_output_full_path)}" w]',
        "fconfigure $outfile -translation binary -encoding binary",
    ])
    if save_com:
        column_expr = "$num_groups * 3"
    else:
        column_expr = "$num_atoms * 3"
    output_header_block = f"""
# VMD writes a simple temporary .bin file; Python converts it to the requested final format.
set vmd_payload_format "{_tcl_quote(temp_dtype["tcl_format"])}"
set output_frames [expr {{($num_frames + $stride - 1) / $stride}}]
set num_columns [expr {{{column_expr}}}]
puts $outfile "ADMDYNANLZ_COORD_BIN_V1 {_tcl_quote(temp_dtype["label"])} $output_frames $num_columns"
status "opened temp output {_tcl_quote(temp_output_full_path)} with $output_frames frame(s), $num_columns column(s)"
"""
    if save_com:
        output_frame_block = """
    set binary_chunk ""
    set chunk_groups 0
    foreach group_sel $group_sels {
        $group_sel frame $frame
        set center [measure center $group_sel weight mass]
        append binary_chunk [binary format $vmd_payload_format $center]
        incr chunk_groups
        if {$chunk_groups >= 4096} {
            puts -nonewline $outfile $binary_chunk
            set binary_chunk ""
            set chunk_groups 0
        }
    }
    if {$chunk_groups > 0} {
        puts -nonewline $outfile $binary_chunk
    }
"""
        output_summary_label = "VMD temp binary COM output written to"
    else:
        output_frame_block = """
    $sel frame $frame
    set coords [$sel get {x y z}]
    
    # Stream binary output in chunks to avoid building enormous Tcl objects
    set binary_chunk ""
    set chunk_atoms 0
    foreach coord $coords {
        append binary_chunk [binary format $vmd_payload_format $coord]
        incr chunk_atoms
        if {$chunk_atoms >= 4096} {
            puts -nonewline $outfile $binary_chunk
            set binary_chunk ""
            set chunk_atoms 0
        }
    }
    if {$chunk_atoms > 0} {
        puts -nonewline $outfile $binary_chunk
    }
"""
        output_summary_label = "VMD temp binary coordinate output written to"

    if save_com:
        group_setup_block = f"""
set grouping_unit "{_tcl_quote(grouping_unit)}"
set atom_indices [$sel get index]
if {{$grouping_unit eq "residue"}} {{
    set atom_group_segids [$sel get segid]
    set atom_group_resids [$sel get resid]
    set atom_groups [list]
    foreach group_segid $atom_group_segids group_resid $atom_group_resids {{
        lappend atom_groups [list $group_segid $group_resid]
    }}
}} else {{
    set atom_group_values [$sel get $grouping_unit]
    set atom_groups [list]
    foreach group_value $atom_group_values {{
        lappend atom_groups [list $group_value]
    }}
}}
set group_order [list]
set grouped_indices [dict create]
foreach atom_index $atom_indices group_key $atom_groups {{
    if {{![dict exists $grouped_indices $group_key]}} {{
        lappend group_order $group_key
        dict set grouped_indices $group_key [list]
    }}
    dict lappend grouped_indices $group_key $atom_index
}}
set group_sels [list]
foreach group_key $group_order {{
    set group_indices [dict get $grouped_indices $group_key]
    set group_sel [atomselect $molid "index [join $group_indices {{ }}]"]
    lappend group_sels $group_sel
}}
set num_groups [llength $group_sels]
puts "Resolved $num_groups COM group(s) using $grouping_unit"
if {{$num_groups == 0}} {{
    abort_with_error "No COM groups were resolved from the selected atoms"
}}
"""
    else:
        group_setup_block = ""
    
    with open(f"writenCodes/{tcl_filename}", "w") as f:
        f.write(f"""# Optimized coordinate extraction script for chunk {i}
# Generated by coordinates_extract.py

puts "Starting coordinate extraction for chunk {i}"
puts "Timestamp: [clock format [clock seconds]]"

set ::coords_statusfile [open "{_tcl_quote(status_path)}" w]
proc status {{message}} {{
    global coords_statusfile
    puts $coords_statusfile "[clock format [clock seconds]] $message"
    flush $coords_statusfile
}}
proc abort_with_error {{message}} {{
    status "ERROR: $message"
    puts "ERROR: $message"
    error $message
}}
status "script started for chunk {i}"

set __coords_status [catch {{
# Set paths
set baseDir "{_tcl_quote(baseDir)}"

# Load trajectory
set PSF "{_tcl_quote(psf_full_path)}"
set DCD "{_tcl_quote(dcd_full_path)}"

puts "Loading PSF: $PSF"
puts "Loading DCD: $DCD"

# Debug output - check file existence
puts "About to load: PSF=$PSF, DCD=$DCD"
if {{![file exists $PSF]}} {{
    abort_with_error "PSF file does not exist: $PSF"
}}
if {{![file exists $DCD]}} {{
    abort_with_error "DCD file does not exist: $DCD"
}}

set molid [mol new $PSF type psf waitfor all]
mol top $molid
puts "Molecule topology loaded with ID: $molid"
status "molecule topology loaded with ID $molid"

# Get number of frames
set num_frames {dcd_frame_count}
puts "Total frames: $num_frames"
status "trajectory has $num_frames frame(s)"
set stride {stride}
set frame_batch_size {frame_batch_size}
{wrap_block}

# Create selection
set sel [atomselect $molid "{_tcl_quote(target_selection)}"]
set num_atoms [$sel num]
puts "Selected $num_atoms atoms"
status "selected $num_atoms atom(s)"

if {{$num_atoms == 0}} {{
    abort_with_error "No atoms selected with criteria '{_tcl_quote(target_selection)}'"
}}

{group_setup_block}
{output_open_block}
{output_header_block}

# Extract coordinates for all frames
puts "Extracting coordinates..."
catch {{animate delete all}}
set frame_count 0
for {{set batch_start 0}} {{$batch_start < $num_frames}} {{incr batch_start [expr {{$stride * $frame_batch_size}}]}} {{
    set batch_end [expr {{$batch_start + (($frame_batch_size - 1) * $stride)}}]
    if {{$batch_end >= $num_frames}} {{
        set batch_end [expr {{$num_frames - 1}}]
    }}
    puts "Reading DCD frames $batch_start to $batch_end with stride $stride"
    status "reading DCD frames $batch_start to $batch_end with stride $stride"
    if {{[catch {{mol addfile $DCD type dcd first $batch_start last $batch_end step $stride waitfor all molid $molid}} read_err]}} {{
        abort_with_error "failed to read DCD frames $batch_start to $batch_end: $read_err"
    }}
    set loaded_frames [molinfo $molid get numframes]
    set expected_batch_frames [expr {{(($batch_end - $batch_start) / $stride) + 1}}]
    if {{$loaded_frames <= 0}} {{
        abort_with_error "VMD loaded zero frames from DCD batch $batch_start to $batch_end"
    }}
    if {{$loaded_frames != $expected_batch_frames}} {{
        abort_with_error "VMD loaded $loaded_frames frame(s) from DCD batch $batch_start to $batch_end, expected $expected_batch_frames"
    }}
    for {{set local_frame 0}} {{$local_frame < $loaded_frames}} {{incr local_frame}} {{
    set frame $local_frame
    set actual_frame [expr {{$batch_start + ($local_frame * $stride)}}]
    animate goto $frame
""")
        if wrap_enabled:
            f.write(f"""
    if {{[catch {{eval pbc wrap $wrap_args}} wrap_err]}} {{
        abort_with_error "pbc wrap failed at frame $actual_frame: $wrap_err"
    }}
""")
        f.write(f"""
{output_frame_block}
    
    incr frame_count
    if {{$frame_count % 100 == 0}} {{
        puts "Processed $frame_count frames..."
    }}
    }}
    catch {{animate delete all}}
}}

$sel delete
if {{[info exists group_sels]}} {{
    foreach group_sel $group_sels {{
        $group_sel delete
    }}
}}
close $outfile
status "closed output file"
mol delete $molid

puts "Extraction complete: $frame_count frames processed"
puts "{output_summary_label}: {output_full_path}"
status "extraction complete: $frame_count frame(s) processed"
puts "Timestamp: [clock format [clock seconds]]"
}} __coords_error]

if {{$__coords_status != 0}} {{
    status "ERROR: $__coords_error"
    catch {{close $outfile}}
    catch {{mol delete $molid}}
    catch {{close $::coords_statusfile}}
    exit 1
}}

catch {{close $::coords_statusfile}}
exit 0
""")
    
    return True


def _run_vmd_script(index, vmd_path, baseDir, output_pattern, common_term="", output_io_spec=None, conversion_chunk_rows=None):
    """Run a single VMD script and return results."""
    
    # Use the same naming convention as in _write_tcl_script
    tcl_filename = f"coords_{common_term}_{index}.tcl" if common_term else f"coords_{index}.tcl"
    script_path = f"writenCodes/{tcl_filename}"
    log_path = f"logs/log_{index}.log"
    output_file = os.path.join(baseDir, expand_path_pattern(output_pattern, common_term, index))
    temp_output_file = _vmd_coordinate_temp_path(output_file)
    status_path = _coords_status_path(common_term, index)
    normalized_output = _normalize_output_io_spec(output_io_spec)
    _cleanup_binary_coordinate_temp_files(output_file)
    try:
        if os.path.exists(status_path):
            os.remove(status_path)
    except OSError:
        pass
    
    # Build VMD command - ensure proper quoting for paths with spaces.
    # Some VMD builds do not support "-nt" and treat the following value as an
    # input molecule filename, so keep the batch invocation to portable flags.
    command = [vmd_path, "-dispdev", "text", "-e", script_path]
    
    print(f"Command: {' '.join(command)}")
    print(f"Script path: {script_path}")
    print(f"Log path: {log_path}")
    
    try:
        # Run VMD process
        run_started_at = time.time()
        process = subprocess.run(
            command, 
            shell=False,
            capture_output=True, 
            text=True,
            timeout=None  # No timeout for large trajectories
        )
        
        def write_vmd_log(extra_error=None, extra_warning=None):
            with open(log_path, 'w') as log_file:
                log_file.write(f"Command: {' '.join(command)}\n")
                log_file.write(f"Return code: {process.returncode}\n")
                if extra_error:
                    log_file.write(f"Detected error: {extra_error}\n")
                if extra_warning:
                    log_file.write(f"Warning: {extra_warning}\n")
                log_file.write(f"STDOUT:\n{process.stdout}\n")
                log_file.write(f"STDERR:\n{process.stderr}\n")

        # Save log file
        write_vmd_log()

        combined_output = process.stdout + "\n" + process.stderr
        tcl_errors = [
            line.strip()
            for line in combined_output.splitlines()
            if line.strip().startswith("ERROR:")
        ]
        if tcl_errors:
            error_msg = "\n".join(tcl_errors)
            write_vmd_log(error_msg)
            return False, process.stdout, error_msg

        success = process.returncode == 0
        completion_warning = None
        if success and "Extraction complete:" not in combined_output:
            completion_warning = (
                "VMD returned code 0, but the coordinate Tcl completion message was not captured. "
                "Continuing with output-file validation."
            )

        if success:
            try:
                _finalize_vmd_temp_coordinate_output(
                    temp_output_file,
                    output_file,
                normalized_output,
                min_temp_mtime=run_started_at,
                status_path=status_path,
                conversion_chunk_rows=conversion_chunk_rows,
            )
            except Exception as e:
                error_msg = f"Converting VMD coordinate temp output failed: {e}"
                write_vmd_log(error_msg)
                return False, process.stdout, error_msg
        if completion_warning:
            write_vmd_log(extra_warning=completion_warning)
        return success, process.stdout, process.stderr
        
    except subprocess.TimeoutExpired:
        error_msg = f"VMD process for chunk {index} timed out (timeout removed for large trajectories)"
        with open(log_path, 'w') as log_file:
            log_file.write(f"ERROR: {error_msg}\n")
        return False, "", error_msg
        
    except Exception as e:
        error_msg = f"Exception running VMD for chunk {index}: {str(e)}"
        with open(log_path, 'w') as log_file:
            log_file.write(f"ERROR: {error_msg}\n")
        return False, "", error_msg



    
