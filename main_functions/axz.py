import numpy as np
import os
import time
import warnings
try:
    from .path_utils import expand_path_pattern, validate_path_pattern
except ImportError:
    from path_utils import expand_path_pattern, validate_path_pattern
try:
    from .numeric_io import load_numeric_array, save_numeric_array
except ImportError:
    from numeric_io import load_numeric_array, save_numeric_array

######################################################    Parameters    

def alpha_xz(baseDir, input_pattern, output_pattern, num_dcd, partcl_num, numFrames, validate_data=True, common_term="", dcd_indices=None, input_io_spec=None, output_io_spec=None):
    """
    Compute directional non-Gaussian parameter α_xz(t) for a set of particles
    from a series of center-of-mass (COM) trajectory files. This parameter
    measures correlations between x and z displacements.

    Parameters
    ----------
    baseDir : str
        Path to the top-level working directory.
    input_pattern : str
        Path pattern for input COM files. Can contain * (common term) and {i} (file index).
        Example: "anlz/NVT_*/com_data/com_{i}.dat"
    output_pattern : str
        Path pattern for output directory. Can contain * (common term).
        Directory `alpha_xz` will be created inside.
        Example: "anlz/NVT_*/analysis"
    num_dcd : int
        Number of trajectory (DCD) frames to process.
    partcl_num : int
        Number of particles whose COM position is stored in each file.
        Each .dat file is expected to have `partcl_num*3` columns: x, y, z
        for each molecule.
    numFrames : int
        Minimum number of time-points in each .dat file to include it in the
        average. If a file has fewer lines than `numFrames`, it is skipped
        (and counted as a failure).
    validate_data : bool, optional
        Perform data validation checks during processing.
    common_term : str, optional
        Value to replace * placeholders in patterns. Default is "".

    Returns
    -------
    dict
        Processing results including timing, success/failure counts, and data quality metrics.

    Notes
    -----
    - Calculates α_xz(t) = ⟨Δx² · Δz²⟩/(⟨Δx²⟩ · ⟨Δz²⟩) - 1
    - Displacements are calculated within each trajectory relative to its first frame.
    - Results are ensemble-averaged over all successful trajectories.
    """
    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    start_time = time.time()
    
    # Validate path patterns
    is_valid, error_msg = validate_path_pattern(input_pattern)
    if not is_valid:
        raise ValueError(f"Invalid input pattern: {error_msg}")
        
    is_valid, error_msg = validate_path_pattern(output_pattern)
    if not is_valid:
        raise ValueError(f"Invalid output pattern: {error_msg}")
    
    print(f"{'='*50}")
    print(f"α_xz(t) CALCULATION")
    print(f"{'='*50}")
    print(f"Base directory: {baseDir}")
    print(f"Input pattern: {input_pattern}")
    print(f"Output pattern: {output_pattern}")
    print(f"Common term: {common_term}")
    print(f"Number of DCDs: {num_dcd}")
    print(f"Particles: {partcl_num}, Min frames: {numFrames}")
    
    output_target_rel = expand_path_pattern(output_pattern, common_term)
    output_target = os.path.join(baseDir, output_target_rel)
    output_dir = os.path.dirname(output_target) or baseDir
    os.makedirs(output_dir, exist_ok=True)
    output_name = os.path.basename(output_target)
    stem, ext = os.path.splitext(output_name)
    if not ext:
        ext = ".dat"
    if not stem:
        stem = f"anisotropy_{partcl_num}mols"
    print(f"Output base path prepared: {output_target}")
    
    # Determine which DCDs to process
    if dcd_indices is None:
        dcd_list = list(range(num_dcd))
    else:
        dcd_list = dcd_indices
        print(f"Processing selected DCDs: {dcd_list}")
    
    # Validate first input file exists
    if not dcd_list:
        raise ValueError("No trajectory indices specified for processing")
    first_input = expand_path_pattern(input_pattern, common_term, dcd_list[0])
    first_input_path = os.path.join(baseDir, first_input)
    if not os.path.exists(first_input_path):
        raise FileNotFoundError(f"First input file not found: {first_input_path}")
    print(f"✓ Input files validated for index {dcd_list[0]}")

    results = {
        'success': 0,
        'failed': [],
        'skipped_files': [],
        'total_time': 0,
        'data_quality': {},
        'numerical_warnings': []
    }
    
    print(f"Starting α_xz(t) calculation...")
    print(f"Processing {len(dcd_list)} trajectory files with {partcl_num} particles")
    print(f"Minimum frames required: {numFrames}")
    
    try:
        # Initialize accumulators - will be set after processing first successful file
        x2_accumulator = None
        y2_accumulator = None
        z2_accumulator = None
        x2y2_accumulator = None
        x2z2_accumulator = None
        y2z2_accumulator = None
        successful_files = 0
        
        # Process each trajectory file
        for f in dcd_list:
            try:
                file_path = expand_path_pattern(input_pattern, common_term, f)
                file_path = os.path.join(baseDir, file_path)
                
                if not os.path.exists(file_path):
                    print(f"⚠ File not found: {file_path}")
                    results['failed'].append(f)
                    continue
                
                # Load trajectory data
                data = load_numeric_array(
                    file_path,
                    input_io_spec,
                    default_mode="text",
                    default_precision="double",
                )
                
                # Validate frame length
                if len(data) < numFrames:
                    print(f"⚠ File {f}: {len(data)} frames < {numFrames} required")
                    results['skipped_files'].append(f)
                    continue
                
                # Reshape data to (frames, molecules, 3)
                data = data.reshape(len(data), partcl_num, 3)
                
                # Calculate displacements relative to first frame of this trajectory
                displacements = data - data[0, :, :]  # Subtract first frame
                
                # Extract x and z displacements
                dx = displacements[:, :, 0]  # x displacements
                dz = displacements[:, :, 2]  # z displacements
                dy = displacements[:, :, 1]  # y displacements
                
                # Calculate squared displacements
                dx2 = dx**2  # Δx²
                dz2 = dz**2  # Δz²
                dy2 = dy**2  # Δy²
                dx2_dz2 = dx2 * dz2  # Δx² · Δz²
                dx2_dy2 = dx2 * dy2  # Δx² · Δy²
                dy2_dz2 = dy2 * dz2  # Δy² · Δz²
                
                # Initialize accumulators with first successful file
                if x2_accumulator is None:
                    x2_accumulator = np.zeros_like(dx2, dtype=np.float64)
                    z2_accumulator = np.zeros_like(dz2, dtype=np.float64)
                    y2_accumulator = np.zeros_like(dy2, dtype=np.float64)
                    x2z2_accumulator = np.zeros_like(dx2_dz2, dtype=np.float64)
                    x2y2_accumulator = np.zeros_like(dx2_dy2, dtype=np.float64)
                    y2z2_accumulator = np.zeros_like(dy2_dz2, dtype=np.float64)
                
                # Type assertions for the linter
                assert x2_accumulator is not None and z2_accumulator is not None and x2z2_accumulator is not None and y2_accumulator is not None and x2y2_accumulator is not None and y2z2_accumulator is not None
                
                # Accumulate squared displacements
                x2_accumulator += dx2
                y2_accumulator += dy2
                z2_accumulator += dz2
                x2z2_accumulator += dx2_dz2
                x2y2_accumulator += dx2_dy2
                y2z2_accumulator += dy2_dz2
                
                successful_files += 1
                    
            except Exception as e:
                print(f"✗ Error processing file {f}: {e}")
                results['failed'].append(f)
                continue
        
        if successful_files == 0 or x2_accumulator is None:
            raise ValueError("No trajectory files were successfully processed")
        
        # Compute ensemble averages
        print(f"Computing ensemble averages from {successful_files} successful files...")
        
        # Average over 1-nanosecond trajectories
        x2_avg = x2_accumulator / successful_files  # ⟨Δx²⟩
        y2_avg = y2_accumulator / successful_files  # ⟨Δy²⟩
        z2_avg = z2_accumulator / successful_files  # ⟨Δz²⟩
        x2z2_avg = x2z2_accumulator / successful_files  # ⟨Δx² · Δz²⟩
        x2y2_avg = x2y2_accumulator / successful_files  # ⟨Δx² · Δy²⟩
        y2z2_avg = y2z2_accumulator / successful_files  # ⟨Δy² · Δz²⟩
        
        # Average over particles if more than 1 molecule
        if partcl_num > 1:
            if np.shape(x2_avg)[1]==partcl_num:
                x2_avg = np.mean(x2_avg, axis=1)  # Average over molecules
                y2_avg = np.mean(y2_avg, axis=1)  # Average over molecules
                z2_avg = np.mean(z2_avg, axis=1)  # Average over molecules
                x2z2_avg = np.mean(x2z2_avg, axis=1)  # Average over molecules
                x2y2_avg = np.mean(x2y2_avg, axis=1)  # Average over molecules
                y2z2_avg = np.mean(y2z2_avg, axis=1)  # Average over molecules
            else:
                print(f"Particle number ({partcl_num}) is not consistent with data shape ({np.shape(x2_avg)[1]})")
        
        # Compute α_xz(t) with numerical stability
        epsilon = 1e-12
        denominator_xz = x2_avg * z2_avg
        denominator_xy = x2_avg * y2_avg
        denominator_yz = y2_avg * z2_avg
        
        # Avoid division by very small numbers
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            alpha_xz = np.divide(x2z2_avg, denominator_xz, out=np.zeros_like(x2z2_avg), where=(np.abs(denominator_xz) > epsilon)) - 1
            alpha_xy = np.divide(x2y2_avg, denominator_xy, out=np.zeros_like(x2y2_avg), where=(np.abs(denominator_xy) > epsilon)) - 1
            alpha_yz = np.divide(y2z2_avg, denominator_yz, out=np.zeros_like(y2z2_avg), where=(np.abs(denominator_yz) > epsilon)) - 1
        
        # Handle problematic values
        small_denom_mask = np.abs(denominator_xz) < epsilon
        alpha_xz[small_denom_mask] = 0.0
        alpha_xz[~np.isfinite(alpha_xz)] = 0.0

        small_denom_mask = np.abs(denominator_xy) < epsilon
        alpha_xy[small_denom_mask] = 0.0
        alpha_xy[~np.isfinite(alpha_xy)] = 0.0

        small_denom_mask = np.abs(denominator_yz) < epsilon
        alpha_yz[small_denom_mask] = 0.0
        alpha_yz[~np.isfinite(alpha_yz)] = 0.0
        
        # Calculate anisotropy
        anisotropy = (alpha_xy + alpha_yz + alpha_xz) / 3

        # Save results
        alpha_xz_file = os.path.join(output_dir, f"{stem}_axz{ext}")
        alpha_xy_file = os.path.join(output_dir, f"{stem}_axy{ext}")
        alpha_yz_file = os.path.join(output_dir, f"{stem}_ayz{ext}")
        alpha_anisotropy_file = os.path.join(output_dir, f"{stem}_anisotropy{ext}")
        
        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(alpha_xz_file), exist_ok=True)
        os.makedirs(os.path.dirname(alpha_xy_file), exist_ok=True)
        os.makedirs(os.path.dirname(alpha_yz_file), exist_ok=True)
        os.makedirs(os.path.dirname(alpha_anisotropy_file), exist_ok=True)

        print(f"Saving α_xz(t) results to: {alpha_xz_file}")
        print(f"Saving α_xy(t) results to: {alpha_xy_file}")
        print(f"Saving α_yz(t) results to: {alpha_yz_file}")
        print(f"Saving anisotropy results to: {alpha_anisotropy_file}")
        
        save_numeric_array(alpha_xz_file, alpha_xz, output_io_spec, default_mode="text", default_precision="double")
        save_numeric_array(alpha_xy_file, alpha_xy, output_io_spec, default_mode="text", default_precision="double")
        save_numeric_array(alpha_yz_file, alpha_yz, output_io_spec, default_mode="text", default_precision="double")
        save_numeric_array(alpha_anisotropy_file, anisotropy, output_io_spec, default_mode="text", default_precision="double")
        
        print(f"✓ Results saved:")
        print(f"  α_xz(t): {alpha_xz_file}, α_xy(t): {alpha_xy_file}, α_yz(t): {alpha_yz_file}, anisotropy: {alpha_anisotropy_file}")
        
        results['success'] = successful_files
        results['total_time'] = time.time() - start_time
        
        # Data quality metrics
        results['data_quality'] = {
            'mean alpha_xz, xy, yz, anisotropy': [float(np.mean(alpha_xz)), float(np.mean(alpha_xy)), float(np.mean(alpha_yz)), float(np.mean(anisotropy))],
            'mean_x2': float(np.mean(x2_avg)),
            'mean_y2': float(np.mean(y2_avg)),
            'mean_z2': float(np.mean(z2_avg)),
            'mean_x2z2': float(np.mean(x2z2_avg)),
            'mean_x2y2': float(np.mean(x2y2_avg)),
            'mean_y2z2': float(np.mean(y2z2_avg)),
            'mean_anisotropy': float(np.mean(anisotropy))
        }
        
    except Exception as e:
        results['failed'].append('general_error')
        results['total_time'] = time.time() - start_time
        raise RuntimeError(f"Critical error in α_xz calculation: {e}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"α_xz, xy, yz, anisotropy CALCULATION SUMMARY") 
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
