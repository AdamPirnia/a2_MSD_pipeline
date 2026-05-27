import numpy as np
import os
import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import warnings
try:
    from .numeric_io import load_numeric_array, save_numeric_array
except ImportError:
    from numeric_io import load_numeric_array, save_numeric_array
import re
# Inlined path_utils functions to keep generated/runtime scripts self-contained
_SAFE_INDEX_EXPR = re.compile(r"^i(?:\s*[+\-*/]\s*\d+)?$")


def _format_unreadable_file_message(label, path):
    return (
        f"{label} file exists but is empty or unavailable locally: {path}. "
        "If this file is in Dropbox, iCloud, OneDrive, or another cloud-synced folder, "
        "make it available offline before running the analysis."
    )

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


def _load_coordinate_array(input_file, dtype=np.float32, io_spec=None):
    """Load coordinate data using the configured storage mode and precision."""
    data = load_numeric_array(
        input_file,
        io_spec,
        default_mode="binary",
        default_precision="single",
    )
    return np.asarray(data, dtype=dtype)


def _save_coordinate_array(output_file, data, io_spec=None):
    """Save coordinates using the configured storage mode and precision."""
    save_numeric_array(
        output_file,
        data,
        io_spec,
        default_mode="binary",
        default_precision="single",
    )


def _resolve_usable_atom_count(num_atoms, n_cols):
    """Return the number of full xyz triplets that can be processed from a row width."""
    available_atoms = n_cols // 3
    if available_atoms <= 0:
        raise ValueError(f"Input data has only {n_cols} columns; at least 3 are required")

    usable_atoms = available_atoms if num_atoms is None else min(int(num_atoms), available_atoms)
    if usable_atoms <= 0:
        raise ValueError("No complete atomic coordinates are available for unwrapping")

    if n_cols % 3 != 0:
        warnings.warn(
            f"Input has {n_cols} columns; trimming to {usable_atoms * 3} columns to keep full xyz triplets."
        )

    return usable_atoms

    
def unwrapper(
    baseDir,
    input_pattern=None,
    output_pattern=None,
    xsc_pattern=None,
    num_dcd=None,
    max_workers=None,
    chunk_size=None,
    dcd_indices=None,
    common_term="",
    input_io_spec=None,
    output_io_spec=None,
    psf_pattern=None,
    target_selection=None,
    vmd=None,
    grouping_unit="residue",
    repair_first_frame=False,
):
    """
    Read a series of coordinates directly extracted from MD trajectories, unwrap 
    periodic boundary crossings, and write out unwrapped coordinate files with
    optimized memory usage and optional parallel processing.

    This function:
      1. Reads the last line of the `.xsc` file to determine the simulation box vectors.
      2. For each frame in the input pattern (optionally in parallel):
         a. Loads the wrapped coordinates from input files.
         b. Uses vectorized numpy operations to compute frame-to-frame displacements.
         c. Applies minimum-image corrections using vectorized operations
            (disp -= box * round(disp / box)).
         d. Cumulatively sums the corrected displacements to produce unwrapped
            trajectories using np.cumsum for maximum efficiency.
         e. Saves the unwrapped coordinates to the output pattern.

    Parameters
    ----------
    baseDir : str
        Root directory for the simulation analysis.
    input_pattern : str
        Path pattern for input coordinate files. Can contain * (common term) and {i} (file index).
        Example: "anlz/NVT_*/wrapped/xyz_{i}.dat"
    output_pattern : str
        Path pattern for output unwrapped files. Can contain * (common term) and {i} (file index).
        Example: "anlz/NVT_*/unwrapped/unwrapped_xyz_{i}.dat"
    xsc_pattern : str
        Path pattern for XSC files. Can contain * (common term) and {i} (file index).
        Example: "anlz/NVT_*/restart_equil.xsc" (if same for all) or "anlz/NVT_*/restart_{i}.xsc"
    num_dcd : int
        Number of frames to process (i.e. number of input files).
    max_workers : int, optional
        Maximum number of parallel workers for processing files. 
        Defaults to min(num_dcd, CPU count) for parallel processing, 
        or None for sequential processing.
    chunk_size : int, optional
        Number of frames to process at once for memory efficiency.
        Automatically determined if not specified.
    dcd_indices : list, optional
        List of DCD indices to process (e.g., [0, 1, 4, 5] to process only DCDs 0, 1, 4, 5).
        If None, processes all DCDs from 0 to num_dcd-1. Default is None.
    common_term : str, optional
        Value to replace * placeholders in patterns. Default is "".

    Notes
    -----
    - The box lengths are taken from indices [1], [5], [9] of the last XSC line.
    - Uses vectorized numpy operations for maximum efficiency: disp -= box * round(disp / box).
    - The function reshapes each flat XYZ array into shape (n_frames, num_atoms, 3)
      before unwrapping, and flattens it back when saving.
    - Outputs are written as float32 binary NumPy arrays.
    - Vectorized algorithm is much faster than frame-by-frame loops, especially for large systems.

    Returns
    -------
    dict
        Processing results including timing and success/failure counts.

    Examples
    --------
    >>> results = unwrapper(
    ...     baseDir="/home/user/sim",
    ...     input_pattern="anlz/NVT_*/wrapped/xyz_{i}.dat",
    ...     output_pattern="anlz/NVT_*/unwrapped/unwrapped_xyz_{i}.dat",
    ...     xsc_pattern="anlz/NVT_*/restart_equil.xsc",
    ...     num_dcd=6,
    ...     common_term="240"
    ... )
    """
    
    start_time = time.time()
    
    # Validate path patterns
    is_valid, error_msg = validate_path_pattern(input_pattern)
    if not is_valid:
        raise ValueError(f"Invalid input pattern: {error_msg}")
        
    is_valid, error_msg = validate_path_pattern(output_pattern)
    if not is_valid:
        raise ValueError(f"Invalid output pattern: {error_msg}")
        
    is_valid, error_msg = validate_path_pattern(xsc_pattern)
    if not is_valid:
        raise ValueError(f"Invalid XSC pattern: {error_msg}")

    first_frame_repair_enabled = bool(
        repair_first_frame
        and psf_pattern
        and target_selection
        and vmd
        and grouping_unit in {"residue", "chain", "segname"}
    )
    if first_frame_repair_enabled:
        is_valid, error_msg = validate_path_pattern(psf_pattern)
        if not is_valid:
            raise ValueError(f"Invalid PSF pattern for first-frame repair: {error_msg}")
    
    print(f"{'='*50}")
    print(f"COORDINATE UNWRAPPING")
    print(f"{'='*50}")
    print(f"Base directory: {baseDir}")
    print(f"Input pattern: {input_pattern}")
    print(f"Output pattern: {output_pattern}")
    print(f"XSC pattern: {xsc_pattern}")
    print(f"Common term: {common_term}")
    print(f"Number of DCDs: {num_dcd}")
    print(f"Repair first frame by {grouping_unit}: {first_frame_repair_enabled}")
    
    # Determine indices to process
    if dcd_indices is None:
        dcd_list = list(range(num_dcd))
    else:
        dcd_list = dcd_indices
        print(f"Processing selected DCDs: {dcd_list}")
    
    # Create output directory (extract directory from first output file)
    if dcd_list:
        first_output = expand_path_pattern(output_pattern, common_term, dcd_list[0])
        output_dir = os.path.dirname(os.path.join(baseDir, first_output))
        os.makedirs(output_dir, exist_ok=True)
    
    # Read box size from XSC file
    try:
        # For XSC, use the first file's pattern (often XSC is the same for all)
        xsc_file = expand_path_pattern(xsc_pattern, common_term, dcd_list[0] if dcd_list else 0)
        xsc_path = os.path.join(baseDir, xsc_file)
        if os.path.exists(xsc_path) and os.path.getsize(xsc_path) <= 0:
            raise ValueError(_format_unreadable_file_message("XSC", xsc_path))
        
        with open(xsc_path, 'r') as fr:
            lines = fr.readlines()
        box_size = np.array([
            float(lines[-1].split()[1]),   # x-dimension
            float(lines[-1].split()[5]),   # y-dimension  
            float(lines[-1].split()[9])    # z-dimension
        ])
        print(f"✓ Box dimensions from {xsc_path}: {box_size}")
        
    except (FileNotFoundError, IndexError, ValueError) as e:
        raise ValueError(f"Error reading XSC file {xsc_path}: {e}")
    
    # Validate first input file exists
    if dcd_list:
        first_input = expand_path_pattern(input_pattern, common_term, dcd_list[0])
        first_input_path = os.path.join(baseDir, first_input)
        if not os.path.exists(first_input_path):
            raise FileNotFoundError(f"First input file not found: {first_input_path}")
        if os.path.getsize(first_input_path) <= 0:
            raise ValueError(_format_unreadable_file_message("Input coordinate", first_input_path))
        print(f"✓ Input files validated for index {dcd_list[0]}")
    
    # Determine optimal chunk size for memory efficiency
    if chunk_size is None:
        chunk_size = 1000
    
    # Processing setup
    if max_workers is None:
        max_workers = min(len(dcd_list), mp.cpu_count())
    
    print(f"Using {max_workers} workers, chunk size: {chunk_size}")
    
    expected_columns = None
    print("Expected output shape: inferred from Step 1 coordinate output width")
    
    results = {'success': 0, 'failed': [], 'total_time': 0}
    
    if max_workers == 1:
        # Sequential processing
        for i in dcd_list:
            try:
                _unwrap_single_file(
                    i, baseDir, input_pattern, output_pattern, xsc_pattern,
                    box_size, chunk_size, common_term, input_io_spec, output_io_spec,
                    psf_pattern, target_selection, vmd, grouping_unit, first_frame_repair_enabled
                )
                results['success'] += 1
                print(f"✓ Completed file {i}")
            except Exception as e:
                results['failed'].append(i)
                print(f"✗ Failed file {i}: {e}")
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_index = {
                executor.submit(
                    _unwrap_single_file, 
                    i, baseDir, input_pattern, output_pattern, xsc_pattern,
                    box_size, chunk_size, common_term, input_io_spec, output_io_spec,
                    psf_pattern, target_selection, vmd, grouping_unit, first_frame_repair_enabled
                ): i 
                for i in dcd_list
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    future.result()  # Raises exception if failed
                    results['success'] += 1
                    print(f"✓ Completed file {index}")
                except Exception as exc:
                    results['failed'].append(index)
                    print(f"✗ Failed file {index}: {exc}")
    
    results['total_time'] = time.time() - start_time
    
    # Summary
    print(f"\n{'='*50}")
    print(f"COORDINATE UNWRAPPING SUMMARY")
    print(f"{'='*50}")
    print(f"Total files: {num_dcd}")
    print(f"Successful: {results['success']}")
    print(f"Failed: {len(results['failed'])}")
    if results['failed']:
        print(f"Failed indices: {results['failed']}")
    print(f"Total time: {results['total_time']:.2f} seconds")
    print(f"Average time per file: {results['total_time']/num_dcd:.2f} seconds")
    print(f"{'='*50}\n")
    
    # Validate output shapes
    if results['success'] > 0:
        print(f"Validating output file shapes...")
        _validate_unwrapped_shapes(baseDir, output_pattern, dcd_list[:min(3, len(dcd_list))], expected_columns, common_term, output_io_spec)
    
    return results


def _validate_unwrapped_shapes(baseDir, output_pattern, sample_indices, expected_columns, common_term="", output_io_spec=None):
    """Validate that unwrapped output files have the expected shape (F, M*N*3)"""
    # expand_path_pattern already defined above
    
    validation_results = []
    for idx in sample_indices:
        try:
            output_file = expand_path_pattern(output_pattern, common_term, idx)
            output_path = os.path.join(baseDir, output_file)
            
            if os.path.exists(output_path):
                data = _load_coordinate_array(output_path, dtype=np.float32, io_spec=output_io_spec)
                lines, actual_columns = data.shape

                if expected_columns is None:
                    validation_results.append(f"  File {idx}: ✓ Shape ({lines}, {actual_columns})")
                elif actual_columns == expected_columns:
                    validation_results.append(f"  File {idx}: ✓ Correct shape ({lines}, {actual_columns})")
                else:
                    validation_results.append(
                        f"  File {idx}: ⚠️  Shape trimmed or mismatched - expected up to {expected_columns} columns, got {actual_columns}"
                    )
        except Exception as e:
            validation_results.append(f"  File {idx}: ❌ Validation failed - {e}")
    
    if validation_results:
        print("Shape validation results:")
        for result in validation_results:
            print(result)


def _resolve_first_frame_group_indices(file_index, baseDir, psf_pattern, target_selection, vmd_path, grouping_unit, common_term=""):
    """Ask VMD for selected atom grouping, preserving the Step 1 selection order."""

    psf_file_rel = expand_path_pattern(psf_pattern, common_term, file_index)
    psf_file = os.path.join(baseDir, psf_file_rel)
    if not os.path.exists(psf_file):
        raise FileNotFoundError(f"PSF file for first-frame repair not found: {psf_file}")
    if os.path.getsize(psf_file) <= 0:
        raise ValueError(_format_unreadable_file_message("PSF", psf_file))

    os.makedirs("writenCodes", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    tcl_filename = f"unwrap_groups_{common_term}_{file_index}.tcl" if common_term else f"unwrap_groups_{file_index}.tcl"
    script_path = os.path.join("writenCodes", tcl_filename)
    log_path = os.path.join("logs", f"unwrap_groups_{file_index}.log")

    with open(script_path, "w") as handle:
        handle.write(f"""# Resolve selected atom groups for Step 2 first-frame repair
set PSF "{_tcl_quote(psf_file)}"
set grouping_unit "{_tcl_quote(grouping_unit)}"

if {{![file exists $PSF]}} {{
    puts "ERROR: PSF file does not exist: $PSF"
    exit 1
}}

set molid [mol new $PSF type psf waitfor all]
set sel [atomselect $molid "{_tcl_quote(target_selection)}"]
set num_atoms [$sel num]
if {{$num_atoms == 0}} {{
    puts "ERROR: No atoms selected with criteria '{_tcl_quote(target_selection)}'"
    exit 1
}}

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
set grouped_positions [dict create]
set selected_position 0
foreach group_key $atom_groups {{
    if {{![dict exists $grouped_positions $group_key]}} {{
        lappend group_order $group_key
        dict set grouped_positions $group_key [list]
    }}
    dict lappend grouped_positions $group_key $selected_position
    incr selected_position
}}

set group_position_parts [list]
foreach group_key $group_order {{
    lappend group_position_parts [join [dict get $grouped_positions $group_key] ","]
}}

puts "__ADMDYN_GROUP_POSITIONS__ [join $group_position_parts {{;}}]"
$sel delete
mol delete $molid
exit 0
""")

    command = [vmd_path, "-dispdev", "text", "-nt", "1", "-e", script_path]
    process = subprocess.run(command, shell=False, capture_output=True, text=True, timeout=None)
    with open(log_path, "w") as handle:
        handle.write(f"Command: {' '.join(command)}\n")
        handle.write(f"Return code: {process.returncode}\n")
        handle.write(f"STDOUT:\n{process.stdout}\n")
        handle.write(f"STDERR:\n{process.stderr}\n")

    if process.returncode != 0:
        raise RuntimeError(f"VMD failed while resolving Step 2 first-frame groups for index {file_index}. See {log_path}")

    group_indices = None
    for line in process.stdout.splitlines():
        marker = "__ADMDYN_GROUP_POSITIONS__"
        if line.startswith(marker):
            payload = line[len(marker):].strip()
            group_indices = [
                [int(item) for item in group_payload.split(",") if item]
                for group_payload in payload.split(";")
                if group_payload
            ]
            break

    if not group_indices:
        raise ValueError(f"No {grouping_unit} groups were resolved for Step 2 first-frame repair. See {log_path}")

    return group_indices


def _periodic_group_center(axis_coords, length):
    """Return a wrapped center for one periodic coordinate axis."""

    wrapped = np.mod(axis_coords, length)
    angles = wrapped * (2.0 * np.pi / length)
    sin_mean = float(np.mean(np.sin(angles)))
    cos_mean = float(np.mean(np.cos(angles)))
    if np.hypot(sin_mean, cos_mean) < 1e-12:
        return float(wrapped[0])
    angle = np.arctan2(sin_mean, cos_mean)
    return float(np.mod(angle * length / (2.0 * np.pi), length))


def _make_group_whole_and_centered(group_coords, box_size):
    """Make a group whole across PBC and place its geometric center in the primary cell."""

    repaired = np.array(group_coords, copy=True)
    for axis in range(3):
        length = float(box_size[axis])
        if length <= 0.0:
            raise ValueError(f"Invalid box length for axis {axis}: {length}")

        wrapped_axis = np.mod(group_coords[:, axis], length)
        center = _periodic_group_center(group_coords[:, axis], length)
        delta = wrapped_axis - center
        delta -= length * np.round(delta / length)
        repaired[:, axis] = center + delta

        group_center = float(np.mean(repaired[:, axis]))
        repaired[:, axis] -= np.floor(group_center / length) * length
    return repaired


def _repair_first_frame_by_group(first_frame, box_size, group_indices):
    """Make each selected group whole in frame 0 and return fixed coords plus atom shifts."""

    repaired = np.array(first_frame, copy=True)
    for indices in group_indices:
        if len(indices) <= 1:
            repaired[indices] = np.mod(repaired[indices], box_size)
            continue
        repaired[indices] = _make_group_whole_and_centered(repaired[indices], box_size)
    return repaired, repaired - first_frame


def _unwrap_single_file(file_index, baseDir, input_pattern, output_pattern, xsc_pattern,
                       box_size, chunk_size, common_term, input_io_spec=None, output_io_spec=None,
                       psf_pattern=None, target_selection=None, vmd=None, grouping_unit="residue",
                       repair_first_frame=False):
    """Process a single coordinate file with vectorized unwrapping algorithm."""
    
    input_file_rel = expand_path_pattern(input_pattern, common_term, file_index)
    output_file_rel = expand_path_pattern(output_pattern, common_term, file_index)
    
    input_file = os.path.join(baseDir, input_file_rel)
    output_file = os.path.join(baseDir, output_file_rel)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Load coordinates
        print(f"Processing {input_file}...")
        coords = _load_coordinate_array(input_file, dtype=np.float32, io_spec=input_io_spec)
        
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        n_frames, n_cols = coords.shape
        
        # Validate dimensions
        usable_atoms = _resolve_usable_atom_count(None, n_cols)
        usable_cols = usable_atoms * 3
        if n_cols != usable_cols:
            print(f"Trimming {input_file_rel} from {n_cols} to {usable_cols} columns ({usable_atoms} atoms)")
            coords = coords[:, :usable_cols]
            n_cols = usable_cols

        # Reshape to (frames, atoms, 3)
        coords = coords.reshape(n_frames, usable_atoms, 3)

        if repair_first_frame:
            group_indices = _resolve_first_frame_group_indices(
                file_index, baseDir, psf_pattern, target_selection, vmd, grouping_unit, common_term
            )
            grouped_positions = sorted(position for group in group_indices for position in group)
            if grouped_positions != list(range(usable_atoms)):
                raise ValueError(
                    f"Step 2 first-frame repair metadata does not match coordinate width for index {file_index}: "
                    f"VMD selection has {len(grouped_positions)} atoms, coordinate file has {usable_atoms} atoms. "
                    "Make sure Step 1 target selection and Step 2 input files match."
                )
            repaired_first_frame, atom_shifts = _repair_first_frame_by_group(coords[0], box_size, group_indices)
            shifted_atoms = int(np.count_nonzero(np.any(np.abs(atom_shifts) > 1e-6, axis=1)))
            coords += atom_shifts.reshape(1, usable_atoms, 3)
            coords[0] = repaired_first_frame
            print(
                f"✓ Fixed first frame using {len(group_indices)} {grouping_unit} group(s); "
                f"applied shifts to {shifted_atoms} atom(s) across all frames"
            )
        
        # Vectorized unwrapping algorithm - much more efficient than loops
        if n_frames == 1:
            # Single frame case - first-frame group repair may still have been applied.
            unwrapped_flat = coords.reshape(n_frames, -1)
        else:
            # Displacements between consecutive frames: shape (T-1, N, 3)
            disp = np.diff(coords, axis=0)
            
            # Apply minimum image convention in one vectorized step:
            # disp_MI = disp - box * round(disp / box)
            disp -= box_size * np.round(disp / box_size)
            
            # Now integrate displacements to get unwrapped positions
            unwrapped = np.empty_like(coords)
            unwrapped[0] = coords[0]                               # reference frame
            unwrapped[1:] = coords[0] + np.cumsum(disp, axis=0)    # cumulative sum
            
            # Flatten back to (n_frames, n_cols)
            unwrapped_flat = unwrapped.reshape(n_frames, -1)
        _save_coordinate_array(output_file, unwrapped_flat, io_spec=output_io_spec)
        
        print(f"✓ Completed {input_file_rel} -> {output_file_rel}")
        
    except Exception as e:
        print(f"✗ Failed {input_file_rel}: {e}")
        raise
