#!/usr/bin/env python3
"""
Optimized Dipole Moment Calculation Module

Calculates molecular dipole moments from coordinate and COM data with enhanced
performance, parallel processing, and robust error handling.
"""
import numpy as np
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
try:
    from .numeric_io import load_numeric_array, save_numeric_array
except ImportError:
    from numeric_io import load_numeric_array, save_numeric_array

# Debye conversion constant  
DEBYE_CONVERSION = 0.2081943  # e*Å to Debye


def _load_array(input_file, dtype=np.float64, io_spec=None):
    """Load numeric data according to the selected storage mode and precision."""
    data = load_numeric_array(
        input_file,
        io_spec,
        default_mode="binary",
        default_precision="single",
    )
    return np.asarray(data, dtype=dtype)


def dipoleM(coords, charges, com, atoms_per_mol, num_molecules_to_calc, chunk_size=None, neutral=False):
    """
    Calculate dipole moments for a set of molecular coordinates.
    
    Parameters:
    -----------
    coords : numpy.ndarray
        Coordinate array of shape (n_frames, n_particles * atoms_per_mol, 3)
    charges : numpy.ndarray
        Charge array of shape (atoms_per_mol,)
    com : numpy.ndarray or None
        Center of mass array of shape (n_frames, n_particles, 3). Not needed when neutral=True.
    atoms_per_mol : int
        Number of atoms per molecule
    num_molecules_to_calc : int
        Number of molecules to calculate (first N molecules)
    chunk_size : int, optional
        Ignored - kept for compatibility
    neutral : bool, optional
        If True, skip COM subtraction and use raw coordinates directly.
        
    Returns:
    --------
    tuple : (dipole_vectors, dipole_magnitudes)
    """
    n_frames = coords.shape[0]
    
    # Limit coordinates to only the first num_molecules_to_calc molecules
    num_coord_cols = num_molecules_to_calc * atoms_per_mol * 3
    coords_limited = coords[:, :num_coord_cols]
    
    # Reshape coordinates to (n_frames, num_molecules_to_calc, atoms_per_mol, 3)
    coords_reshaped = coords_limited.reshape(n_frames, num_molecules_to_calc, atoms_per_mol, 3)

    if neutral:
        relative_coords = coords_reshaped
    else:
        com_limited = com[:, :num_molecules_to_calc, :]
        # Reshape COM to (n_frames, num_molecules_to_calc, 1, 3) for broadcasting
        com_reshaped = com_limited.reshape(n_frames, num_molecules_to_calc, 1, 3)
        # Calculate relative coordinates (center of charge)
        relative_coords = coords_reshaped - com_reshaped
    
    # Calculate dipole moments: sum over atoms in each molecule
    # dipole = sum(charge_i * r_i) for each molecule
    dipole_vectors = np.sum(
        relative_coords * charges[np.newaxis, np.newaxis, :, np.newaxis], 
        axis=2
    ) / DEBYE_CONVERSION
    
    # Calculate dipole magnitudes
    dipole_magnitudes = np.linalg.norm(dipole_vectors, axis=2)
    
    return dipole_vectors, dipole_magnitudes


def _process_single_dipole_file(file_idx, baseDir, coords_pattern, com_pattern, output_pattern, 
                                magnitudes_pattern, charges_array, atoms_per_particle, 
                                effective_molecules, stride, common_term, neutral=False,
                                coords_input_io_spec=None, com_input_io_spec=None,
                                vectors_output_io_spec=None, magnitudes_output_io_spec=None):
    """Process a single trajectory file for dipole calculation - at module level for multiprocessing."""
    try:
        try:
            from .path_utils import expand_path_pattern
        except ImportError:
            from path_utils import expand_path_pattern
        
        # File paths
        coord_file_rel = expand_path_pattern(coords_pattern, common_term, file_idx)
        coord_file = os.path.join(baseDir, coord_file_rel)
        com_file = None
        if not neutral:
            com_file_rel = expand_path_pattern(com_pattern, common_term, file_idx)
            com_file = os.path.join(baseDir, com_file_rel)
        
        print(f"Processing file {file_idx}:")
        print(f"  Coord file: {coord_file}")
        if neutral:
            print("  COM file: skipped (neutral molecules)")
        else:
            print(f"  COM file: {com_file}")
        
        # Check if files exist
        if not os.path.exists(coord_file):
            error_msg = f'Coordinate file not found: {coord_file}'
            print(f"  ERROR: {error_msg}")
            return {'success': False, 'error': error_msg, 'file_idx': file_idx}
        if not neutral and not os.path.exists(com_file):
            error_msg = f'COM file not found: {com_file}'
            print(f"  ERROR: {error_msg}")
            return {'success': False, 'error': error_msg, 'file_idx': file_idx}
            
        # Load data
        print(f"  Loading data from files...")
        coord_data = _load_array(coord_file, dtype=np.float32, io_spec=coords_input_io_spec)
        if neutral:
            com_data = None
            print(f"  Loaded coords: {coord_data.shape}, COM: skipped (neutral molecules)")
        else:
            com_data = _load_array(com_file, dtype=np.float64, io_spec=com_input_io_spec)
            print(f"  Loaded coords: {coord_data.shape}, COM: {com_data.shape}")
        
        # Apply stride if specified
        if stride > 1:
            coord_data = coord_data[::stride]
            if not neutral:
                com_data = com_data[::stride]

        expected_coord_cols = effective_molecules * atoms_per_particle * 3
        actual_coord_cols = int(coord_data.shape[1]) if coord_data.ndim > 1 else 0
        if actual_coord_cols < expected_coord_cols:
            hint = ""
            if actual_coord_cols == effective_molecules * 3:
                hint = " This looks like center-of-mass data rather than atomic coordinates; use raw/unwrapped atomic coordinates instead of com_xyz."
            raise ValueError(
                f"Coordinate input has {actual_coord_cols} columns, but dipole calculation requires "
                f"{expected_coord_cols} columns for {effective_molecules} molecules x {atoms_per_particle} atoms/molecule.{hint}"
            )
        if actual_coord_cols > expected_coord_cols:
            coord_data = coord_data[:, :expected_coord_cols]
        
        # Reshape COM data to (n_frames, n_particles, 3) for dipoleM function
        n_frames = coord_data.shape[0]
        com_reshaped = None
        num_particles = effective_molecules
        if not neutral:
            num_particles = com_data.shape[1] // 3  # Total particles in system
            com_reshaped = com_data.reshape(n_frames, num_particles, 3)
        
        print(f"  Processing {n_frames} frames, {num_particles} particles, calculating {effective_molecules} molecules")
        
        # Use the dipoleM function to calculate dipole moments
        if neutral:
            print(f"  Calling dipoleM with shapes: coords {coord_data.shape}, charges {charges_array.shape}, com skipped (neutral)")
        else:
            print(f"  Calling dipoleM with shapes: coords {coord_data.shape}, charges {charges_array.shape}, com {com_reshaped.shape}")
        dipole_vectors, dipole_magnitudes = dipoleM(
            coord_data, charges_array, com_reshaped, atoms_per_particle, effective_molecules, neutral=neutral
        )
        print(f"  dipoleM returned: vectors {dipole_vectors.shape}, magnitudes {dipole_magnitudes.shape}")
        
        # Save results
        if magnitudes_pattern:
            dipole_file = expand_path_pattern(output_pattern, common_term, file_idx)
            magnitude_file = expand_path_pattern(magnitudes_pattern, common_term, file_idx)
            
            dipole_path = os.path.join(baseDir, dipole_file)
            magnitude_path = os.path.join(baseDir, magnitude_file)
            
            os.makedirs(os.path.dirname(dipole_path), exist_ok=True)
            if magnitude_path != dipole_path:
                os.makedirs(os.path.dirname(magnitude_path), exist_ok=True)
        else:
            output_path = os.path.join(baseDir, expand_path_pattern(output_pattern, common_term, file_idx))
            dipole_path = os.path.join(output_path, f'dipoles_{file_idx}.dat')
            magnitude_path = os.path.join(output_path, f'dipole_magnitudes_{file_idx}.dat')
            os.makedirs(output_path, exist_ok=True)
        
        # Reshape dipole vectors to match expected output format
        dipole_vectors_flat = dipole_vectors.reshape(n_frames, -1)
        
        print(f"  Saving to: {dipole_path}")
        print(f"  Saving to: {magnitude_path}")
        save_numeric_array(
            dipole_path,
            dipole_vectors_flat,
            vectors_output_io_spec,
            default_mode="text",
            default_precision="double",
        )
        save_numeric_array(
            magnitude_path,
            dipole_magnitudes,
            magnitudes_output_io_spec,
            default_mode="text",
            default_precision="double",
        )
        
        print(f"  SUCCESS: File {file_idx} processed, {n_frames} frames")
        return {
            'success': True, 
            'file_idx': file_idx, 
            'frames_processed': n_frames,
            'mean_magnitude': np.mean(dipole_magnitudes),
            'std_magnitude': np.std(dipole_magnitudes)
        }
        
    except Exception as e:
        import traceback
        error_details = f"File {file_idx} failed: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(f"  ERROR: {error_details}")
        return {'success': False, 'error': error_details, 'file_idx': file_idx}

def dipole_functions(baseDir, coords_pattern, com_pattern, output_pattern, Charges, num_dcds, num_particles, 
                    atoms_per_particle=3, stride=1, max_workers=1, chunk_processing=True, 
                    validate_data=True, progress_callback=None, molecules_to_process=None, common_term="",
                    magnitudes_pattern=None, dcd_indices=None, neutral=False,
                    coords_input_io_spec=None, com_input_io_spec=None,
                    vectors_output_io_spec=None, magnitudes_output_io_spec=None):
    """
    Calculate molecular dipole moments from trajectory data.
    
    Parameters:
    -----------
    baseDir : str
        Base directory path
    coords_pattern : str  
        Path pattern for coordinate files. Can contain * (common term) and {i} (file index).
        Example: "anlz/NVT_*/unwrapped/continued_xyz_{i}.dat"
    com_pattern : str
        Path pattern for COM files. Can contain * (common term) and {i} (file index).
        Example: "anlz/NVT_*/com_data/com_{i}.dat"
    output_pattern : str
        Path pattern for output vector files OR output directory (backward compatibility).
        If magnitudes_pattern is None, treated as directory. Otherwise, treated as vector file pattern.
        Example: "anlz/NVT_*/dipole/vectors_{i}.dat"
    magnitudes_pattern : str, optional
        Path pattern for magnitude output files. If provided, output_pattern is treated as vector pattern.
        Example: "anlz/NVT_*/dipole/magnitudes_{i}.dat"
    dcd_indices : list, optional
        List of specific DCD indices to process. If None, processes all DCDs.
    Charges : list
        List of atomic charges for each atom in a molecule
    num_dcds : int
        Number of trajectory files to process
    num_particles : int
        Total number of molecules/particles in each file
    atoms_per_particle : int, optional
        Number of atoms per molecule (default: 3)
    stride : int, optional
        Frame stride for processing (default: 1)
    max_workers : int, optional
        Number of parallel workers (default: 1)
    chunk_processing : bool, optional
        Enable chunked processing for memory efficiency (default: True)
    validate_data : bool, optional
        Perform data validation checks (default: True)
    progress_callback : callable, optional
        Callback function for progress updates
    molecules_to_process : int, optional
        Number of molecules to process (default: all)
    common_term : str, optional
        Common term for path expansion (default: "")
    neutral : bool, optional
        If True, do not require COM files and do not subtract COM.
    """
    
    start_time = time.time()
    
    # Validate path patterns
    try:
        from .path_utils import expand_path_pattern, validate_path_pattern
    except ImportError:
        from path_utils import expand_path_pattern, validate_path_pattern
    
    is_valid, error_msg = validate_path_pattern(coords_pattern)
    if not is_valid:
        raise ValueError(f"Invalid coords pattern: {error_msg}")
        
    if not neutral:
        is_valid, error_msg = validate_path_pattern(com_pattern)
        if not is_valid:
            raise ValueError(f"Invalid COM pattern: {error_msg}")
        
    is_valid, error_msg = validate_path_pattern(output_pattern)
    if not is_valid:
        raise ValueError(f"Invalid output pattern: {error_msg}")
    
    # Determine effective number of molecules to process
    if molecules_to_process is None:
        effective_molecules = num_particles
    else:
        effective_molecules = min(molecules_to_process, num_particles)
        if molecules_to_process > num_particles:
            print(f"Warning: molecules_to_process ({molecules_to_process}) > num_particles ({num_particles}). Processing all {num_particles} molecules.")
    
    # Input validation
    if not isinstance(Charges, (list, tuple, np.ndarray)):
        raise ValueError("Charges must be a list, tuple, or numpy array")
    
    charges_array = np.array(Charges, dtype=float)
    if len(charges_array) != atoms_per_particle:
        raise ValueError(f"Number of charges ({len(charges_array)}) must match atoms_per_particle ({atoms_per_particle})")
    
    # Convert to debye units (charge * distance conversion factor)
    DEBYE_CONVERSION = 0.2081943  # e*Å to Debye
    
    # Create full paths and determine output mode
    vectors_as_files = magnitudes_pattern is not None
    
    if vectors_as_files:
        # New mode: separate file patterns for vectors and magnitudes
        print(f"Output mode: Separate file patterns")
        print(f"Vectors pattern: {output_pattern}")
        print(f"Magnitudes pattern: {magnitudes_pattern}")
    else:
        # Legacy mode: directory pattern with hardcoded filenames
        output_dir_rel = expand_path_pattern(output_pattern, common_term)
        output_path = os.path.join(baseDir, output_dir_rel)
        os.makedirs(output_path, exist_ok=True)
        print(f"Output mode: Directory pattern (legacy)")
        print(f"Output directory: {output_path}")
    
    print(f"{'='*50}")
    print(f"DIPOLE MOMENT CALCULATION")
    print(f"{'='*50}")
    print(f"Base directory: {baseDir}")
    print(f"Coords pattern: {coords_pattern}")
    if neutral:
        print("COM pattern: skipped (neutral molecules)")
    else:
        print(f"COM pattern: {com_pattern}")
    print(f"Common term: {common_term}")
    print(f"Processing {effective_molecules} out of {num_particles} total molecules")
    print(f"Using {max_workers} workers, atoms per particle: {atoms_per_particle}")
    print(f"Charges: {charges_array.tolist()}")
    
    # Handle DCD selection
    if dcd_indices is not None:
        actual_dcd_list = dcd_indices
        actual_num_dcds = len(dcd_indices)
        print(f"Using DCD selection: {dcd_indices}")
    else:
        actual_dcd_list = list(range(num_dcds))
        actual_num_dcds = num_dcds
        print(f"Processing all DCDs: 0 to {num_dcds-1}")
    




    # Process files
    results = []
    successful_files = 0
    successful_indices = []
    failed_results = []
    total_frames = 0
    quality_metrics = {'mean_magnitudes': [], 'std_magnitudes': []}
    
    print(f"Processing {actual_num_dcds} trajectory files for dipole moment calculation...")
    print(f"Using {max_workers} workers, atoms per particle: {atoms_per_particle}")
    print(f"Processing {effective_molecules} out of {num_particles} total molecules")
    print(f"Charges: {charges_array.tolist()}")
    
    # Use all requested workers - no memory limitations on supercomputer
    print(f"Using {max_workers} workers for parallel processing")
    
    if max_workers > 1:
        # Try parallel processing with fallback to single-threaded
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_idx = {executor.submit(_process_single_dipole_file, i, baseDir, coords_pattern, 
                                                 com_pattern, output_pattern, magnitudes_pattern, 
                                                 charges_array, atoms_per_particle, effective_molecules, 
                                                 stride, common_term, neutral, coords_input_io_spec,
                                                 com_input_io_spec, vectors_output_io_spec, magnitudes_output_io_spec): i for i in actual_dcd_list}
                
                # Process completed tasks
                for future in as_completed(future_to_idx):
                    result = future.result()
                    results.append(result)
                    if result['success']:
                        successful_files += 1
                        successful_indices.append(result['file_idx'])
                        total_frames += result['frames_processed']
                        quality_metrics['mean_magnitudes'].append(result['mean_magnitude'])
                        quality_metrics['std_magnitudes'].append(result['std_magnitude'])
                        print(f"✓ File {result['file_idx']}: {result['frames_processed']} frames, avg magnitude: {result['mean_magnitude']:.3f} D")
                    else:
                        failed_results.append(result)
                        print(f"✗ File {result['file_idx']}: {result['error']}")
                    if progress_callback:
                        progress_callback(len(results), actual_num_dcds)
        except (TypeError, AttributeError) as e:
            if 'pickle' in str(e).lower() or 'local object' in str(e):
                print(f"⚠️  Multiprocessing failed due to function pickling: {e}")
                print("   Falling back to single-threaded processing...")
                max_workers = 1  # Force single-threaded fallback
                # Process files sequentially
                for i in actual_dcd_list:
                    result = _process_single_dipole_file(i, baseDir, coords_pattern, com_pattern, 
                                                       output_pattern, magnitudes_pattern, charges_array, 
                                                       atoms_per_particle, effective_molecules, stride, common_term, neutral,
                                                       coords_input_io_spec, com_input_io_spec, vectors_output_io_spec, magnitudes_output_io_spec)
                    results.append(result)
                    if result['success']:
                        successful_files += 1
                        successful_indices.append(result['file_idx'])
                        total_frames += result['frames_processed']
                        quality_metrics['mean_magnitudes'].append(result['mean_magnitude'])
                        quality_metrics['std_magnitudes'].append(result['std_magnitude'])
                        print(f"✓ File {result['file_idx']}: {result['frames_processed']} frames, avg magnitude: {result['mean_magnitude']:.3f} D")
                    else:
                        failed_results.append(result)
                        print(f"✗ File {result['file_idx']}: {result['error']}")
                    if progress_callback:
                        progress_callback(len(results), actual_num_dcds)
            else:
                raise e  # Re-raise other errors
    else:
        # Sequential processing
        for i in actual_dcd_list:
            result = _process_single_dipole_file(i, baseDir, coords_pattern, com_pattern, 
                                               output_pattern, magnitudes_pattern, charges_array, 
                                               atoms_per_particle, effective_molecules, stride, common_term, neutral,
                                               coords_input_io_spec, com_input_io_spec, vectors_output_io_spec, magnitudes_output_io_spec)
            results.append(result)
            
            if result['success']:
                successful_files += 1
                successful_indices.append(result['file_idx'])
                total_frames += result['frames_processed']
                quality_metrics['mean_magnitudes'].append(result['mean_magnitude'])
                quality_metrics['std_magnitudes'].append(result['std_magnitude'])
                print(f"✓ File {result['file_idx']}: {result['frames_processed']} frames, avg magnitude: {result['mean_magnitude']:.3f} D")
            else:
                failed_results.append(result)
                print(f"✗ File {result['file_idx']}: {result['error']}")
            
            if progress_callback:
                progress_callback(i + 1, actual_num_dcds)
    
    # Calculate summary statistics
    total_time = time.time() - start_time
    
    summary = {
        'success': successful_files,
        'successful': successful_indices,
        'failed': failed_results,
        'total': actual_num_dcds,
        'total_time': total_time,
        'total_frames': total_frames,
        'avg_time_per_file': total_time / actual_num_dcds if actual_num_dcds > 0 else 0,
        'data_quality': 'Good' if successful_files == actual_num_dcds else f'Partial ({successful_files}/{actual_num_dcds})',
        'output_directory': output_path if not vectors_as_files else None # Only show if not vectors_as_files
    }
    
    if quality_metrics['mean_magnitudes']:
        summary.update({
            'overall_mean_magnitude': np.mean(quality_metrics['mean_magnitudes']),
            'overall_std_magnitude': np.mean(quality_metrics['std_magnitudes']),
            'magnitude_range': [np.min(quality_metrics['mean_magnitudes']), np.max(quality_metrics['mean_magnitudes'])]
        })
    
    print(f"\nDipole calculation completed:")
    print(f"  Successful files: {successful_files}/{actual_num_dcds}")
    print(f"  Total frames processed: {total_frames}")
    print(f"  Total time: {total_time:.2f}s")
    if quality_metrics['mean_magnitudes']:
        print(f"  Average dipole magnitude: {summary['overall_mean_magnitude']:.3f} ± {summary['overall_std_magnitude']:.3f} D")
    
    return summary

# Alias for compatibility
dipole_calculation = dipole_functions
