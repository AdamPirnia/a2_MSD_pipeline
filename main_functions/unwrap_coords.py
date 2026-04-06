import numpy as np
import os
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

    
def unwrapper(baseDir, input_pattern=None, output_pattern=None, xsc_pattern=None, num_dcd=None, num_atoms=None, interval=slice(None), stride=1, max_workers=None, chunk_size=None, dcd_indices=None, common_term="", input_io_spec=None, output_io_spec=None):
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
    num_atoms : int
        Number of all atoms for which α₂(t) is calculated. Each input file 
        is expected to have `num_atoms*3` columns when flattened.
    interval : slice function, optional
        Slices the time length of the trajectory to be analyzed.
    stride : int, optional
        Frame stride for processing. Use 1 to process every frame, 2 to process
        every other frame, etc.
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
    ...     num_atoms=3000,
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
    
    print(f"{'='*50}")
    print(f"COORDINATE UNWRAPPING")
    print(f"{'='*50}")
    print(f"Base directory: {baseDir}")
    print(f"Input pattern: {input_pattern}")
    print(f"Output pattern: {output_pattern}")
    print(f"XSC pattern: {xsc_pattern}")
    print(f"Common term: {common_term}")
    print(f"Number of DCDs: {num_dcd}")
    print(f"Number of atoms: {num_atoms}")
    print(f"Stride: {stride}")

    try:
        stride = int(stride)
    except (TypeError, ValueError):
        raise ValueError("stride must be an integer")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    
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
        print(f"✓ Input files validated for index {dcd_list[0]}")
    
    # Determine optimal chunk size for memory efficiency
    if chunk_size is None:
        chunk_size = 1000 if num_atoms is None else min(1000, num_atoms)
    
    # Processing setup
    if max_workers is None:
        max_workers = min(len(dcd_list), mp.cpu_count())
    
    print(f"Using {max_workers} workers, chunk size: {chunk_size}")
    
    # Calculate expected output shape for validation
    # Expected output: (F, M*N*3) where F=frames, M=molecules, N=atoms per molecule
    expected_columns = None if num_atoms is None else num_atoms * 3
    if expected_columns is None:
        print("Expected output shape: inferred per file from available xyz triplets")
    else:
        print(f"Expected output shape: (frames, up to {expected_columns}) for {num_atoms} atoms")
    
    results = {'success': 0, 'failed': [], 'total_time': 0}
    
    if max_workers == 1:
        # Sequential processing
        for i in dcd_list:
            try:
                _unwrap_single_file(
                    i, baseDir, input_pattern, output_pattern, xsc_pattern, num_atoms, 
                    box_size, interval, stride, chunk_size, common_term, input_io_spec, output_io_spec
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
                    i, baseDir, input_pattern, output_pattern, xsc_pattern, num_atoms,
                    box_size, interval, stride, chunk_size, common_term, input_io_spec, output_io_spec
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


def _unwrap_single_file(file_index, baseDir, input_pattern, output_pattern, xsc_pattern, num_atoms, 
                       box_size, interval, stride, chunk_size, common_term, input_io_spec=None, output_io_spec=None):
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
        
        # Apply interval slicing first, then stride to match the generated pipeline order.
        coords = coords[interval]
        if stride > 1:
            coords = coords[::stride]
        n_frames, n_cols = coords.shape
        
        # Validate dimensions
        usable_atoms = _resolve_usable_atom_count(num_atoms, n_cols)
        usable_cols = usable_atoms * 3
        if n_cols != usable_cols:
            print(f"Trimming {input_file_rel} from {n_cols} to {usable_cols} columns ({usable_atoms} atoms)")
            coords = coords[:, :usable_cols]
            n_cols = usable_cols

        # Reshape to (frames, atoms, 3)
        coords = coords.reshape(n_frames, usable_atoms, 3)
        
        # Vectorized unwrapping algorithm - much more efficient than loops
        if n_frames == 1:
            # Single frame case - no unwrapping needed
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
