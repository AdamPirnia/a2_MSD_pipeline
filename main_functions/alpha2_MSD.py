import numpy as np
import os
import time
import warnings
try:
    from .numeric_io import load_numeric_array, load_numeric_array_trimmed, save_numeric_array
except ImportError:
    from numeric_io import load_numeric_array, load_numeric_array_trimmed, save_numeric_array
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

def _derive_default_msd_output_pattern(output_pattern):
    output_dir = os.path.dirname(output_pattern)
    output_name = os.path.basename(output_pattern)
    stem, ext = os.path.splitext(output_name)
    if not ext:
        ext = ".dat"
    if not stem:
        stem = "analysis"
    return os.path.join(output_dir, f"{stem}_MSD{ext}") if output_dir else f"{stem}_MSD{ext}"

######################################################    Parameters    

def a2_MSD(
    baseDir,
    input_pattern,
    output_pattern,
    num_dcd,
    partcl_num,
    numFrames=None,
    input_stride=1,
    validate_data=True,
    common_term="",
    dcd_indices=None,
    input_io_spec=None,
    output_io_spec=None,
    msd_output_pattern=None,
    alpha2_output_pattern=None,
):
    """
    Compute ensemble-averaged mean square displacements (MSD) and non-Gaussian
    parameter α₂(t) for a set of particles from a series of center-of-mass (COM)
    trajectory files. Each trajectory represents 1 ns of data, and displacements
    are calculated within each trajectory relative to its first frame.

    Parameters
    ----------
    baseDir : str
        Path to the top-level working directory.
    input_pattern : str
        Path pattern for input COM files. Can contain * (common term) and {i} (file index).
        Example: "anlz/NVT_*/com_data/com_{i}.dat"
    output_pattern : str
        Backward-compatible output path used for α₂(t) results when
        alpha2_output_pattern is not provided. Can contain * (common term).
        Example: "anlz/NVT_*/analysis/ngp_results_a2.dat"
    num_dcd : int
        Number of trajectory (DCD) frames to process.
    partcl_num : int
        Number of particles whose COM position is stored in each file.
        Each .dat file is expected to have `partcl_num*3` columns: x, y, z
        for each molecule.
    numFrames : int, optional
        Minimum number of time-points in each .dat file to include it in the
        average. If omitted, all readable files are included and the common
        frame count is inferred from the shortest successful file.
    validate_data : bool, optional
        Perform data validation checks during processing.
    common_term : str, optional
        Value to replace * placeholders in patterns. Default is "".
    dcd_indices : list[int], optional
        Specific trajectory indices to process. If None, process range(num_dcd).
    msd_output_pattern : str, optional
        Independent output path for MSD results. If omitted, a *_MSD filename is
        derived from output_pattern.
    alpha2_output_pattern : str, optional
        Independent output path for α₂(t) results. If omitted, output_pattern is used.

    Returns
    -------
    dict
        Processing results including timing, success/failure counts, and data quality metrics.

    Notes
    -----
    - Displacements are calculated within each trajectory relative to its first frame.
    - Results are ensemble-averaged over all successful trajectories.
    - Numerical stability improvements prevent division by very small numbers.
    """
    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    start_time = time.time()
    
    # Validate path patterns
    is_valid, error_msg = validate_path_pattern(input_pattern)
    if not is_valid:
        raise ValueError(f"Invalid input pattern: {error_msg}")
        
    alpha2_output_pattern = alpha2_output_pattern or output_pattern
    msd_output_pattern = msd_output_pattern or _derive_default_msd_output_pattern(output_pattern)
    for label, pattern in [("MSD output", msd_output_pattern), ("α₂(t) output", alpha2_output_pattern)]:
        is_valid, error_msg = validate_path_pattern(pattern)
        if not is_valid:
            raise ValueError(f"Invalid {label} pattern: {error_msg}")
    
    # Determine which DCDs to process
    if dcd_indices is None:
        dcd_list = list(range(num_dcd))
    else:
        dcd_list = list(dcd_indices)

    print(f"{'='*50}")
    print(f"α₂(t) AND MSD CALCULATION")
    print(f"{'='*50}")
    print(f"Base directory: {baseDir}")
    print(f"Input pattern: {input_pattern}")
    print(f"MSD output pattern: {msd_output_pattern}")
    print(f"α₂(t) output pattern: {alpha2_output_pattern}")
    print(f"Common term: {common_term}")
    print(f"Number of DCDs: {len(dcd_list)}")
    if dcd_indices is not None:
        print(f"Processing selected DCD indices: {dcd_list}")
    if numFrames in ("", None):
        min_frames = None
        print(f"Particles: {partcl_num}, Min frames: not set")
    else:
        min_frames = int(numFrames)
        if min_frames <= 0:
            raise ValueError("numFrames must be a positive integer when provided.")
        print(f"Particles: {partcl_num}, Min frames: {min_frames}")
    input_stride = int(input_stride)
    if input_stride <= 0:
        raise ValueError("input_stride must be a positive integer.")
    print(f"Input stride: {input_stride}")
    
    msd_file = os.path.join(baseDir, expand_path_pattern(msd_output_pattern, common_term))
    alpha2_file = os.path.join(baseDir, expand_path_pattern(alpha2_output_pattern, common_term))
    os.makedirs(os.path.dirname(msd_file) or baseDir, exist_ok=True)
    os.makedirs(os.path.dirname(alpha2_file) or baseDir, exist_ok=True)
    print(f"MSD output path prepared: {msd_file}")
    print(f"α₂(t) output path prepared: {alpha2_file}")
    
    # Validate first input file exists
    if not dcd_list:
        raise ValueError("No trajectory indices specified for processing")
    first_input = expand_path_pattern(input_pattern, common_term, dcd_list[0])
    first_input_path = os.path.join(baseDir, first_input)
    if not os.path.exists(first_input_path):
        raise FileNotFoundError(f"First input file not found: {first_input_path}")
    print(f"✓ Input files validated for index {dcd_list[0]}")
    
    # Initialize processing
    successful_files = 0
    total_frames = 0
    frame_counts = []

    results = {
        'success': 0,
        'failed': [],
        'skipped_files': [],
        'total_time': 0,
        'data_quality': {},
        'numerical_warnings': []
    }
    
    print(f"Starting α₂(t) and MSD calculation...")
    print(f"Processing {len(dcd_list)} trajectory files with {partcl_num} particles")
    print(f"Minimum frames required: {min_frames if min_frames is not None else 'not set'}")
    
    try:
        # Initialize accumulators - will be set after processing first successful file
        msd_accumulator = None
        msd4_accumulator = None
        
        # Process each trajectory file (including com_0.dat)
        total_files = len(dcd_list)
        for idx, f in enumerate(dcd_list, start=1):
            try:
                file_path_rel = expand_path_pattern(input_pattern, common_term, f)
                file_path = os.path.join(baseDir, file_path_rel)
                
                if not os.path.exists(file_path):
                    print(f"⚠ File not found: {file_path_rel}")
                    results['failed'].append(f)
                    continue
                
                # Load trajectory data
                data = load_numeric_array_trimmed(
                    file_path,
                    input_io_spec,
                    default_mode="text",
                    default_precision="double",
                    context=f"MSD input file {f}",
                )
                if input_stride > 1:
                    data = data[::input_stride]
                
                # Validate frame length
                if min_frames is not None and len(data) < min_frames:
                    print(f"⚠ File {f}: {len(data)} frames < {min_frames} required")
                    results['skipped_files'].append(f)
                    continue
                if min_frames is not None:
                    data = data[:min_frames]
                
                # Reshape data to (frames, molecules, 3)
                data = data.reshape(len(data), partcl_num, 3)
                
                # Calculate displacements relative to first frame of this trajectory
                data = data - data[0, :, :]  # Subtract first frame
                
                # Calculate squared displacements using einsum for efficiency
                data2 = np.einsum('ijk,ijk->ij', data, data)  # Sum over x,y,z dimensions
                data4 = data2**2
                
                # Initialize accumulators with first successful file
                if msd_accumulator is None:
                    msd_accumulator = np.zeros_like(data2, dtype=np.float64)
                    msd4_accumulator = np.zeros_like(data4, dtype=np.float64)
                elif len(data2) != len(msd_accumulator):
                    common_frames = min(len(data2), len(msd_accumulator))
                    if common_frames == 0:
                        print(f"⚠ File {f}: no frames available after stride")
                        results['skipped_files'].append(f)
                        continue
                    if len(msd_accumulator) != common_frames:
                        msd_accumulator = msd_accumulator[:common_frames]
                        msd4_accumulator = msd4_accumulator[:common_frames]
                    data2 = data2[:common_frames]
                    data4 = data4[:common_frames]
                
                # Accumulate MSD and MSD⁴
                msd_accumulator += data2
                msd4_accumulator += data4
                
                successful_files += 1
                
                # Progress reporting
                if idx % 10 == 0 or idx == total_files:
                    print(f"✓ Processed {idx}/{total_files} files ({successful_files} successful)")
                    
            except Exception as e:
                print(f"✗ Error processing file {f}: {e}")
                results['failed'].append(f)
                continue
        
        if successful_files == 0 or msd_accumulator is None:
            raise ValueError("No trajectory files were successfully processed")
        
        # Compute ensemble averages
        print(f"Computing ensemble averages from {successful_files} successful files...")
        
        # Type assertions for the linter
        assert msd_accumulator is not None and msd4_accumulator is not None
        msd_avg = msd_accumulator / successful_files
        msd4_avg = msd4_accumulator / successful_files
        
        # Average over particles if more than 1 molecule
        if partcl_num > 1:
            msd_avg = np.mean(msd_avg, axis=1)  # Average over molecules
            msd4_avg = np.mean(msd4_avg, axis=1)
        
        # Compute α₂(t) with numerical stability
        epsilon = 1e-12
        msd_squared = msd_avg**2
        
        # Avoid division by very small numbers
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            alpha2 = 3 * msd4_avg / (5 * msd_squared) - 1
        
        # Handle problematic values
        small_msd_mask = msd_squared < epsilon
        alpha2[small_msd_mask] = 0.0
        alpha2[~np.isfinite(alpha2)] = 0.0
        
        print(f"Saving MSD results to: {msd_file}")
        print(f"Saving α₂(t) results to: {alpha2_file}")
        
        save_numeric_array(
            msd_file,
            msd_avg,
            output_io_spec,
            default_mode="text",
            default_precision="double",
            column_names=("msd",),
            result_format=True,
        )
        save_numeric_array(
            alpha2_file,
            alpha2,
            output_io_spec,
            default_mode="text",
            default_precision="double",
            column_names=("alpha2",),
            result_format=True,
        )
        
        print(f"✓ Results saved:")
        print(f"  MSD: {msd_file}")
        print(f"  α₂(t): {alpha2_file}")
        
        results['success'] = successful_files
        results['total_time'] = time.time() - start_time
        
        # Data quality metrics
        results['data_quality'] = {
            'mean_msd': float(np.mean(msd_avg)),
            'max_msd': float(np.max(msd_avg)),
            'mean_alpha2': float(np.mean(alpha2)),
            'alpha2_range': [float(np.min(alpha2)), float(np.max(alpha2))]
        }
        
    except Exception as e:
        results['failed'].append('general_error')
        results['total_time'] = time.time() - start_time
        raise RuntimeError(f"Critical error in α₂/MSD calculation: {e}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"α₂(t) AND MSD CALCULATION SUMMARY") 
    print(f"{'='*50}")
    print(f"Total files processed: {len(dcd_list)}")
    print(f"Successful: {results['success']}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Skipped (insufficient frames): {len(results['skipped_files'])}")
    print(f"Processing time: {results['total_time']:.2f} seconds")
    print(f"Data quality:")
    for key, value in results['data_quality'].items():
        print(f"  {key}: {value}")
    print(f"{'='*50}\n")
    
    return results
