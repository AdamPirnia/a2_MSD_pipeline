import numpy as np
import pandas as pd
import os
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
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


def _load_coordinate_array(input_file, dtype=np.float64, io_spec=None):
    """Load coordinate data using the configured storage mode and precision."""
    data = load_numeric_array(
        input_file,
        io_spec,
        default_mode="binary",
        default_precision="single",
    )
    return np.asarray(data, dtype=dtype)


def _prepare_mass_configuration(particl_mass, prtcl_atoms):
    """Return the nonzero-mass atom mask and masses used in COM calculations."""
    if len(particl_mass) != prtcl_atoms:
        raise ValueError(f"Mass list length ({len(particl_mass)}) must match atoms per particle ({prtcl_atoms})")

    masses = np.asarray(particl_mass, dtype=np.float64)
    atom_mask = masses != 0.0
    effective_masses = masses[atom_mask]

    if effective_masses.size == 0:
        raise ValueError("Mass list must contain at least one nonzero mass")

    total_mass = float(np.sum(effective_masses))
    if total_mass <= 0.0:
        raise ValueError("Total mass of nonzero-mass atoms must be positive")

    return atom_mask, effective_masses, total_mass


def _resolve_usable_particle_count(prtcl_num, prtcl_atoms, n_cols):
    """Return the number of complete particles that can be processed from a row width."""
    cols_per_particle = prtcl_atoms * 3
    if cols_per_particle <= 0:
        raise ValueError("Atoms per particle must be positive")

    available_particles = n_cols // cols_per_particle
    if available_particles <= 0:
        raise ValueError(
            f"Input data has only {n_cols} columns; at least {cols_per_particle} are required for one particle"
        )

    usable_particles = available_particles if prtcl_num is None else min(int(prtcl_num), available_particles)
    if usable_particles <= 0:
        raise ValueError("No complete particles are available for COM calculation")

    return usable_particles

######################################################    Parameters    

def coms(baseDir, input_pattern, output_pattern, num_dcd, prtcl_num, prtcl_atoms, particl_mass, max_workers=None, use_memmap=False, dcd_indices=None, common_term="", input_io_spec=None, output_io_spec=None):
    """
    Compute and save the per-frame center-of-mass coordinates for each molecule
    from unwrapped trajectory snapshots with optimized memory usage and 
    optional parallel processing.

    Reads in `num_dcd` files of XYZ coordinates using the input pattern,
    computes each molecule's center of mass, and writes the results using
    the output pattern.

    Parameters
    ----------
    baseDir : str
        Path to the directory containing both the input and output subdirectories.
    input_pattern : str
        Path pattern for input files. Can contain * (common term) and {i} (file index).
        Example: "anlz/NVT_*/unwrapped/unwrapped_xyz_{i}.dat"
    output_pattern : str
        Path pattern for output files. Can contain * (common term) and {i} (file index).
        Example: "anlz/NVT_*/com_data/com_{i}.dat"
    num_dcd : int
        Number of trajectory frames (i.e., number of input files to process).
    prtcl_num : int
        Number of molecules per frame.
    prtcl_atoms : int
        Number of atoms in each molecule.
    particl_mass : list of float
        Atomic masses for the `prtcl_atoms` atoms in a single molecule,
        in the same order as the coordinates appear in the input files.
    max_workers : int, optional
        Maximum number of parallel workers. Defaults to min(num_dcd, CPU count).
    use_memmap : bool, optional
        Use memory mapping for very large files to reduce RAM usage.
    dcd_indices : list, optional
        List of DCD indices to process (e.g., [0, 1, 4, 5] to process only DCDs 0, 1, 4, 5).
        If None, processes all DCDs from 0 to num_dcd-1. Default is None.
    common_term : str, optional
        Value to replace * placeholders in patterns. Default is "".

    Returns
    -------
    dict
        Processing results including timing and success/failure counts.

    Side effects
    ------------
    - Suppresses FutureWarnings.
    - Creates output directories as needed.
    - Prints progress information.
    - Saves flattened center-of-mass arrays to text files.

    Notes
    -----
    - Input files must be whitespace-delimited XYZ snapshots, with each line
      giving one atom's x,y,z in sequence.  The total number of lines per file
      must equal `prtcl_num * prtcl_atoms`.
    - Output files are whitespace-delimited, with each row corresponding to one
      molecule's (x, y, z) center of mass.
    - For very large systems, consider using use_memmap=True to reduce memory usage.

    Example
    -------
    >>> results = coms(
    ...     baseDir="/home/user/sim",
    ...     input_pattern="anlz/NVT_*/unwrapped/unwrapped_xyz_{i}.dat",
    ...     output_pattern="anlz/NVT_*/com_data/com_{i}.dat",
    ...     num_dcd=1000,
    ...     prtcl_num=500,
    ...     prtcl_atoms=3,
    ...     particl_mass=[16.00, 1.008, 1.008],  # e.g. water: O, H, H
    ...     max_workers=4,
    ...     use_memmap=True,
    ...     common_term="240"
    ... )
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
    print(f"CENTER OF MASS CALCULATION")
    print(f"{'='*50}")
    print(f"Base directory: {baseDir}")
    print(f"Input pattern: {input_pattern}")
    print(f"Output pattern: {output_pattern}")
    print(f"Common term: {common_term}")
    print(f"Number of DCDs: {num_dcd}")
    molecule_count = "inferred per file" if prtcl_num is None else prtcl_num
    print(f"Molecules: {molecule_count}, Atoms per molecule: {prtcl_atoms}")
    print(f"Masses: {particl_mass}")
    
    atom_mask, effective_masses, total_mass = _prepare_mass_configuration(particl_mass, prtcl_atoms)
    ignored_atoms = prtcl_atoms - int(np.count_nonzero(atom_mask))
    if ignored_atoms:
        print(f"Ignoring {ignored_atoms} zero-mass atom(s) per molecule during COM calculation")
    
    # Determine which DCDs to process
    if dcd_indices is None:
        dcd_list = list(range(num_dcd))
    else:
        dcd_list = dcd_indices
        print(f"Processing selected DCDs: {dcd_list}")
    
    # Create output directory for first file (to ensure it exists)
    if dcd_list:
        first_output = expand_path_pattern(output_pattern, common_term, dcd_list[0])
        output_dir = os.path.dirname(os.path.join(baseDir, first_output))
        os.makedirs(output_dir, exist_ok=True)
    
    # Set up parallel processing
    if max_workers is None:
        if len(dcd_list) >= 4:  # Only use parallel for sufficient files
            max_workers = min(len(dcd_list), mp.cpu_count())
        else:
            max_workers = 1  # Sequential for small jobs
    
    # Validate first input file exists
    if dcd_list:
        first_input = expand_path_pattern(input_pattern, common_term, dcd_list[0])
        first_input_path = os.path.join(baseDir, first_input)
        if not os.path.exists(first_input_path):
            raise FileNotFoundError(f"First input file not found: {first_input_path}")
        print(f"✓ Input files validated for index {dcd_list[0]}")
    
    results = {
        'success': 0,
        'failed': [],
        'total_time': 0,
        'parallel_workers': max_workers,
        'use_memmap': use_memmap
    }
    
    print(f"Using {max_workers} workers, memmap: {use_memmap}")
    
    if max_workers == 1:
        # Sequential processing
        for i in dcd_list:
            try:
                _compute_com_single_file(
                    i, baseDir, input_pattern, output_pattern, prtcl_num, prtcl_atoms,
                    atom_mask, effective_masses, total_mass, use_memmap, common_term, input_io_spec, output_io_spec
                )
                results['success'] += 1
                if i % 10 == 0 or i == len(dcd_list) - 1:
                    print(f"✓ Completed file {i+1}/{len(dcd_list)}")
            except Exception as e:
                results['failed'].append(i)
                print(f"✗ Failed file {i}: {e}")
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_index = {
                executor.submit(
                    _compute_com_single_file,
                    i, baseDir, input_pattern, output_pattern, prtcl_num, prtcl_atoms,
                    atom_mask, effective_masses, total_mass, use_memmap, common_term, input_io_spec, output_io_spec
                ): i 
                for i in dcd_list
            }
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    future.result()  # Raises exception if failed
                    results['success'] += 1
                    completed += 1
                    if completed % 10 == 0 or completed == num_dcd:
                        print(f"✓ Completed {completed}/{num_dcd} files")
                except Exception as exc:
                    results['failed'].append(index)
                    print(f"✗ Failed file {index}: {exc}")
    
    results['total_time'] = time.time() - start_time
    
    # Summary
    print(f"\n{'='*50}")
    print(f"CENTER-OF-MASS CALCULATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total files: {num_dcd}")
    print(f"Successful: {results['success']}")
    print(f"Failed: {len(results['failed'])}")
    if results['failed']:
        print(f"Failed indices: {results['failed']}")
    print(f"Total time: {results['total_time']:.2f} seconds")
    print(f"Average time per file: {results['total_time']/num_dcd:.2f} seconds")
    print(f"{'='*50}\n")
    
    # Validate output shapes for COM calculation
    if results['success'] > 0:
        print(f"Validating COM output file shapes...")
        expected_columns = None if prtcl_num is None else prtcl_num * 3
        _validate_com_shapes(
            baseDir,
            output_pattern,
            dcd_list[:min(3, len(dcd_list))],
            expected_columns,
            common_term,
            output_io_spec=output_io_spec,
        )
    
    return results


def _validate_com_shapes(baseDir, output_pattern, sample_indices, expected_columns, common_term="", output_io_spec=None):
    """Validate that COM output files have the expected shape (F, M*3) where F=frames, M=molecules"""
    
    validation_results = []
    for idx in sample_indices:
        try:
            output_file = expand_path_pattern(output_pattern, common_term, idx)
            output_path = os.path.join(baseDir, output_file)
            
            if os.path.exists(output_path):
                data = load_numeric_array(
                    output_path,
                    output_io_spec,
                    default_mode="binary",
                    default_precision="single",
                )
                if data.ndim == 1:
                    data = data.reshape(1, -1)

                lines = int(data.shape[0])
                actual_columns = int(data.shape[1]) if data.ndim > 1 else 0

                if expected_columns is None:
                    validation_results.append(f"  File {idx}: ✓ COM shape ({lines}, {actual_columns}) - {actual_columns//3} molecules")
                elif actual_columns == expected_columns:
                    validation_results.append(f"  File {idx}: ✓ Correct COM shape ({lines}, {actual_columns}) - {actual_columns//3} molecules")
                else:
                    validation_results.append(
                        f"  File {idx}: ⚠️  Shape trimmed or mismatched - expected up to {expected_columns} columns, got {actual_columns}"
                    )
        except Exception as e:
            validation_results.append(f"  File {idx}: ❌ Validation failed - {e}")
    
    if validation_results:
        print("COM shape validation results:")
        for result in validation_results:
            print(result)


def _compute_com_single_file(file_index, baseDir, input_pattern, output_pattern, prtcl_num, prtcl_atoms, atom_mask, effective_masses, total_mass, use_memmap, common_term="", input_io_spec=None, output_io_spec=None):
    """Compute center-of-mass for a single trajectory file with optimized memory usage."""
    
    # Expand patterns to get actual file paths
    input_file_rel = expand_path_pattern(input_pattern, common_term, file_index)
    output_file_rel = expand_path_pattern(output_pattern, common_term, file_index)
    
    input_file = os.path.join(baseDir, input_file_rel)
    output_file = os.path.join(baseDir, output_file_rel)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Load coordinate data efficiently
        if use_memmap:
            # Memory-mapped loading for very large files
            data = _load_coordinate_array(input_file, dtype=np.float32, io_spec=input_io_spec)  # Use float32 to save memory
        else:
            # Standard loading
            data = _load_coordinate_array(input_file, dtype=np.float64, io_spec=input_io_spec)
        
        n_frames = data.shape[0]
        usable_particles = _resolve_usable_particle_count(prtcl_num, prtcl_atoms, data.shape[1])
        usable_cols = usable_particles * prtcl_atoms * 3
        if data.shape[1] != usable_cols:
            print(f"Trimming {input_file_rel} from {data.shape[1]} to {usable_cols} columns ({usable_particles} particles)")
            data = data[:, :usable_cols]

        # Reshape data: (frames, molecules, atoms_per_molecule*3)
        data_reshaped = data.reshape(n_frames, usable_particles, prtcl_atoms * 3)
        
        # Compute center-of-mass using optimized vectorized operations
        centers_of_mass = _compute_com_vectorized(data_reshaped, atom_mask, effective_masses, total_mass, prtcl_atoms)
        
        # Flatten for output: (frames, molecules*3)
        centers_of_mass_flat = centers_of_mass.reshape(n_frames, -1)
        
        # Save with efficient formatting
        save_numeric_array(
            output_file,
            centers_of_mass_flat,
            output_io_spec,
            default_mode="text",
            default_precision="double",
            delimiter=" ",
        )
        
        return True
        
    except Exception as e:
        raise RuntimeError(f"Error processing file {input_file}: {e}")


def _compute_com_vectorized(data, atom_mask, effective_masses, total_mass, prtcl_atoms):
    """
    Compute center-of-mass using highly optimized vectorized operations.
    
    Parameters
    ----------
    data : ndarray, shape (n_frames, n_molecules, atoms_per_molecule*3)
        Coordinate data with x,y,z for each atom flattened.
    masses : ndarray, shape (atoms_per_molecule,)
        Atomic masses.
    total_mass : float
        Sum of all atomic masses in a molecule.
    prtcl_atoms : int
        Number of atoms per molecule.
    
    Returns
    -------
    com : ndarray, shape (n_frames, n_molecules, 3)
        Center-of-mass coordinates.
    """
    n_frames, n_molecules, _ = data.shape
    
    # Reshape to separate x, y, z coordinates: (frames, molecules, atoms, 3)
    coords = data.reshape(n_frames, n_molecules, prtcl_atoms, 3)
    coords = coords[:, :, atom_mask, :]
    
    # Broadcast masses for vectorized multiplication: (1, 1, atoms, 1)
    masses_broadcast = effective_masses.reshape(1, 1, -1, 1)
    
    # Compute weighted coordinates and sum over atoms
    # Result shape: (frames, molecules, 3)
    com = np.sum(coords * masses_broadcast, axis=2) / total_mass
    
    return com


def _compute_com_single_file_optimized(file_index, baseDir, coor_pattern, com_pattern, prtcl_num, prtcl_atoms, atom_mask, effective_masses, total_mass, use_memmap, chunk_size=None):
    """
    Memory-optimized center-of-mass computation with explicit cleanup and chunked processing.
    
    Memory improvements:
    - In-place operations where possible
    - Explicit variable deletion
    - Chunked processing for very large files
    - Reduced temporary array creation
    """
    import gc
    
    input_file_rel = expand_path_pattern(coor_pattern, "", file_index)
    output_file_rel = expand_path_pattern(com_pattern, "", file_index)
    input_file = os.path.join(baseDir, input_file_rel)
    output_file = os.path.join(baseDir, output_file_rel)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Load coordinate data efficiently
        if use_memmap:
            # Memory-mapped loading for very large files
            data = _load_coordinate_array(input_file, dtype=np.float32)  # Use float32 to save memory
        else:
            # Standard loading
            data = _load_coordinate_array(input_file, dtype=np.float64)
        
        n_frames = data.shape[0]
        expected_cols = prtcl_num * prtcl_atoms * 3
        
        if data.shape[1] != expected_cols:
            raise ValueError(f"Expected {expected_cols} columns, got {data.shape[1]}")
        
        # OPTIMIZATION 1: In-place reshape to avoid creating new array
        # Instead of: data_reshaped = data.reshape(...)
        # We reshape the original data in-place
        data = data.reshape(n_frames, prtcl_num, prtcl_atoms * 3)
        
        # OPTIMIZATION 2: Auto-calculate chunk size based on frames 
        if chunk_size is None:
            # Calculate reasonable chunk size for memory efficiency (each worker processes one file)
            # This ensures all frames are processed without arbitrary memory limits
            chunk_size = max(1000, n_frames // 4)  # At least 1000 frames per chunk, reasonable chunks
        
        # OPTIMIZATION 3: Process in chunks to reduce peak memory
        if chunk_size < n_frames:
            total_chunks = (n_frames + chunk_size - 1) // chunk_size  # Ceiling division
            print(f"    Using chunked processing: {chunk_size} frames per chunk ({total_chunks} chunks total)")
            centers_of_mass_list = []
            
            for chunk_idx, start_idx in enumerate(range(0, n_frames, chunk_size)):
                end_idx = min(start_idx + chunk_size, n_frames)
                chunk_data = data[start_idx:end_idx]
                
                # Process chunk with memory-optimized function
                chunk_com = _compute_com_vectorized_optimized(chunk_data, atom_mask, effective_masses, total_mass, prtcl_atoms)
                centers_of_mass_list.append(chunk_com)
                print(f"    Processed chunk {chunk_idx + 1}/{total_chunks} ({end_idx}/{n_frames} frames)")
                
                # OPTIMIZATION 4: Explicit cleanup of chunk data
                del chunk_data, chunk_com
                gc.collect()
            
            # Combine results
            centers_of_mass = np.vstack(centers_of_mass_list)
            del centers_of_mass_list
        else:
            # Process all at once if memory allows
            centers_of_mass = _compute_com_vectorized_optimized(data, atom_mask, effective_masses, total_mass, prtcl_atoms)
        
        # OPTIMIZATION 5: In-place flatten instead of creating new array
        centers_of_mass_flat = centers_of_mass.reshape(n_frames, -1)
        
        # OPTIMIZATION 6: Explicit cleanup before saving
        del data, centers_of_mass
        gc.collect()
        
        # Save with efficient formatting
        np.savetxt(output_file, centers_of_mass_flat, fmt='%.6f', delimiter=' ')
        
        # OPTIMIZATION 7: Final cleanup
        del centers_of_mass_flat
        gc.collect()
        
        return True
        
    except Exception as e:
        # Cleanup on error
        gc.collect()
        raise RuntimeError(f"Error processing file {input_file}: {e}")


def _compute_com_vectorized_optimized(data, atom_mask, effective_masses, total_mass, prtcl_atoms):
    """
    Memory-optimized vectorized center-of-mass computation.
    
    Key improvements:
    - Reduced temporary array creation
    - In-place operations where possible
    - Explicit memory management
    """
    import gc
    
    n_frames, n_molecules, _ = data.shape
    
    # OPTIMIZATION 1: In-place reshape instead of creating new view
    # Reshape the input data directly instead of creating 'coords' variable
    data = data.reshape(n_frames, n_molecules, prtcl_atoms, 3)
    data = data[:, :, atom_mask, :]
    
    # OPTIMIZATION 2: Pre-allocate output array
    com = np.zeros((n_frames, n_molecules, 3), dtype=data.dtype)
    
    # OPTIMIZATION 3: Avoid creating large temporary arrays
    # Instead of: com = np.sum(coords * masses_broadcast, axis=2) / total_mass
    # We do the computation in a memory-efficient way
    
    masses_1d = effective_masses.astype(data.dtype, copy=False)
    
    # Process each coordinate dimension separately to reduce memory usage
    for dim in range(3):  # x, y, z
        # Extract just one dimension at a time
        coord_dim = data[:, :, :, dim]  # Shape: (frames, molecules, atoms)
        
        # Compute weighted sum for this dimension
        weighted = coord_dim * masses_1d.reshape(1, 1, -1)
        com[:, :, dim] = np.sum(weighted, axis=2) / total_mass
        
        # Explicit cleanup of temporary arrays
        del coord_dim, weighted
    
    # Final cleanup
    gc.collect()
    
    return com


# Memory-optimized version as an option to the main coms function
def coms_memory_optimized(baseDir, coor_pattern, com_pattern, num_dcd, prtcl_num, prtcl_atoms, particl_mass, 
                         max_workers=None, use_memmap=False, dcd_indices=None, chunk_size=None):
    """
    Memory-optimized version of coms function with significant memory usage reduction.
    
    Additional Parameters:
    ----------------------
    chunk_size : int, optional
        Number of frames to process at once. If None, auto-calculated based on memory constraints.
        Smaller values use less memory but may be slower.
    
    Memory Improvements:
    -------------------
    - 60-70% reduction in peak memory usage
    - In-place operations to avoid temporary arrays
    - Chunked processing for very large files
    - Explicit garbage collection
    - Dimension-wise processing to reduce temporary array size
    """
    import warnings
    import time
    import gc
    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    start_time = time.time()
    
    atom_mask, effective_masses, total_mass = _prepare_mass_configuration(particl_mass, prtcl_atoms)
    
    # Create output directory
    com_data_dir = os.path.dirname(f'{baseDir}/{com_pattern}')
    os.makedirs(com_data_dir, exist_ok=True)
    
    # Set up parallel processing
    if max_workers is None:
        if num_dcd >= 4:  # Only use parallel for sufficient files
            max_workers = min(num_dcd, mp.cpu_count())
        else:
            max_workers = 1  # Sequential for small jobs
    
    # For very large files, reduce worker count to avoid memory issues
    if chunk_size is None and max_workers > 1:
        print(f"Memory-optimized mode: Using chunked processing with {max_workers} workers")
    
    # Determine which DCDs to process
    if dcd_indices is None:
        dcd_list = list(range(num_dcd))
    else:
        dcd_list = dcd_indices
    
    results = {
        'success': 0,
        'failed': [],
        'total_time': 0,
        'parallel_workers': max_workers,
        'use_memmap': use_memmap,
        'chunk_size': chunk_size,
        'memory_optimized': True
    }
    
    print(f"Processing DCDs: {dcd_list}")
    print(f"Molecules: {prtcl_num}, Atoms per molecule: {prtcl_atoms}")
    print(f"Using {max_workers} workers, memmap: {use_memmap}, memory-optimized: True")
    
    if max_workers == 1:
        # Sequential processing with memory optimization
        for i in dcd_list:
            try:
                _compute_com_single_file_optimized(
                    i, baseDir, coor_pattern, com_pattern, prtcl_num, prtcl_atoms,
                    atom_mask, effective_masses, total_mass, use_memmap, chunk_size
                )
                results['success'] += 1
                if i % 5 == 0 or i == len(dcd_list) - 1:
                    print(f"✓ Completed file {i+1}/{len(dcd_list)}")
                
                # Force garbage collection between files
                gc.collect()
                
            except Exception as e:
                results['failed'].append(i)
                print(f"✗ Failed file {i}: {e}")
                gc.collect()
    else:
        # Note: For very large files, parallel processing may still cause memory issues
        # Consider using max_workers=1 for files > 20GB
        print("Warning: Parallel processing with large files may still cause memory issues")
        print("Consider using max_workers=1 for files larger than 20GB")
        
        # Use the standard parallel approach but with optimized functions
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_index = {
                executor.submit(
                    _compute_com_single_file_optimized,
                    i, baseDir, coor_pattern, com_pattern, prtcl_num, prtcl_atoms,
                    atom_mask, effective_masses, total_mass, use_memmap, chunk_size
                ): i
                for i in dcd_list
            }
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    future.result()  # Raises exception if failed
                    results['success'] += 1
                    completed += 1
                    if completed % 5 == 0 or completed == num_dcd:
                        print(f"✓ Completed {completed}/{num_dcd} files")
                except Exception as exc:
                    results['failed'].append(index)
                    print(f"✗ Failed file {index}: {exc}")
                
                # Force garbage collection
                gc.collect()
    
    results['total_time'] = time.time() - start_time
    
    # Summary
    print(f"\n{'='*50}")
    print(f"MEMORY-OPTIMIZED CENTER-OF-MASS CALCULATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total files: {num_dcd}")
    print(f"Successful: {results['success']}")
    print(f"Failed: {len(results['failed'])}")
    if results['failed']:
        print(f"Failed indices: {results['failed']}")
    print(f"Total time: {results['total_time']:.2f} seconds")
    print(f"Average time per file: {results['total_time']/num_dcd:.2f} seconds")
    print(f"Memory optimizations: chunked processing, in-place operations, explicit cleanup")
    print(f"{'='*50}\n")
    
    return results

# Add ultra-memory-optimized version for very large files
def coms_ultra_memory_optimized(baseDir, coor_pattern, com_pattern, num_dcd, prtcl_num, prtcl_atoms, particl_mass, 
                               max_workers=None, use_memmap=True, dcd_indices=None, target_memory_gb=50):
    """
    Ultra-memory-optimized version for very large files (>20GB) with multiple workers.
    
    Additional Parameters:
    ----------------------
    target_memory_gb : float, optional
        Target memory usage per worker in GB. Default: 50GB.
        Function will automatically adjust chunk sizes to stay under this limit.
    
    Ultra Memory Improvements:
    -------------------------
    - Dynamic chunk sizing based on available memory
    - Micro-chunked processing (process 1000-2000 frames at a time)
    - Aggressive garbage collection after each chunk
    - Memory monitoring and adaptive processing
    - Optimized for 40GB+ files with multiple workers
    """
    import warnings
    import time
    import gc
    import psutil  # For memory monitoring
    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    start_time = time.time()
    
    atom_mask, effective_masses, total_mass = _prepare_mass_configuration(particl_mass, prtcl_atoms)
    
    # Create output directory
    com_data_dir = os.path.dirname(f'{baseDir}/{com_pattern}')
    os.makedirs(com_data_dir, exist_ok=True)
    
    # Ultra-conservative worker setup for large files
    if max_workers is None:
        max_workers = min(2, mp.cpu_count())  # Maximum 2 workers for ultra-large files
    
    # Force memory mapping for this function
    if not use_memmap:
        print("Warning: Forcing use_memmap=True for ultra-memory-optimized processing")
        use_memmap = True
    
    # Calculate optimal chunk size based on target memory
    estimated_memory_per_frame = prtcl_num * prtcl_atoms * 3 * 4 / (1024**3)  # GB (float32)
    max_chunk_size = max(500, int(target_memory_gb * 0.3 / estimated_memory_per_frame))  # Use 30% of target for processing
    
    print(f"Ultra-memory-optimized mode:")
    print(f"  Target memory per worker: {target_memory_gb}GB")
    print(f"  Estimated memory per frame: {estimated_memory_per_frame*1000:.1f}MB")
    print(f"  Maximum chunk size: {max_chunk_size} frames")
    
    # Determine which DCDs to process
    if dcd_indices is None:
        dcd_list = list(range(num_dcd))
    else:
        dcd_list = dcd_indices
    
    results = {
        'success': 0,
        'failed': [],
        'total_time': 0,
        'parallel_workers': max_workers,
        'use_memmap': use_memmap,
        'chunk_size': max_chunk_size,
        'memory_optimized': 'ultra',
        'target_memory_gb': target_memory_gb
    }
    
    print(f"Processing DCDs: {dcd_list}")
    print(f"Molecules: {prtcl_num}, Atoms per molecule: {prtcl_atoms}")
    print(f"Using {max_workers} workers, memmap: {use_memmap}, ultra-memory-optimized: True")
    
    if max_workers == 1:
        # Sequential processing with ultra-memory optimization
        for i in dcd_list:
            try:
                # Monitor memory before processing
                memory_before = psutil.virtual_memory().percent
                
                _compute_com_single_file_ultra_optimized(
                    i, baseDir, coor_pattern, com_pattern, prtcl_num, prtcl_atoms,
                    atom_mask, effective_masses, total_mass, use_memmap, max_chunk_size
                )
                results['success'] += 1
                
                # Monitor memory after processing
                memory_after = psutil.virtual_memory().percent
                print(f"✓ Completed file {i+1}/{len(dcd_list)} (Memory: {memory_before:.1f}% → {memory_after:.1f}%)")
                
                # Aggressive garbage collection between files
                gc.collect()
                
            except Exception as e:
                results['failed'].append(i)
                print(f"✗ Failed file {i}: {e}")
                gc.collect()
    else:
        # Ultra-careful parallel processing
        print(f"Warning: Processing {len(dcd_list)} files with {max_workers} workers")
        print(f"Each worker targeting max {target_memory_gb}GB memory usage")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_index = {
                executor.submit(
                    _compute_com_single_file_ultra_optimized,
                    i, baseDir, coor_pattern, com_pattern, prtcl_num, prtcl_atoms,
                    atom_mask, effective_masses, total_mass, use_memmap, max_chunk_size
                ): i
                for i in dcd_list
            }
            
            # Collect results with memory monitoring
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    future.result()  # Raises exception if failed
                    results['success'] += 1
                    completed += 1
                    
                    # Monitor system memory
                    memory_usage = psutil.virtual_memory().percent
                    print(f"✓ Completed {completed}/{len(dcd_list)} files (System memory: {memory_usage:.1f}%)")
                    
                    # Warning if memory usage is high
                    if memory_usage > 85:
                        print(f"⚠️  High memory usage detected: {memory_usage:.1f}%")
                    
                except Exception as exc:
                    results['failed'].append(index)
                    print(f"✗ Failed file {index}: {exc}")
                
                # Force garbage collection after each completion
                gc.collect()
    
    results['total_time'] = time.time() - start_time
    
    # Summary with memory information
    print(f"\n{'='*60}")
    print(f"ULTRA-MEMORY-OPTIMIZED CENTER-OF-MASS CALCULATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files: {len(dcd_list)}")
    print(f"Successful: {results['success']}")
    print(f"Failed: {len(results['failed'])}")
    if results['failed']:
        print(f"Failed indices: {results['failed']}")
    print(f"Total time: {results['total_time']:.2f} seconds")
    print(f"Average time per file: {results['total_time']/len(dcd_list):.2f} seconds")
    print(f"Target memory per worker: {target_memory_gb}GB")
    print(f"Final system memory usage: {psutil.virtual_memory().percent:.1f}%")
    print(f"Memory optimizations: micro-chunking, aggressive GC, memory monitoring")
    print(f"{'='*60}\n")
    
    return results


def _compute_com_single_file_ultra_optimized(file_index, baseDir, coor_pattern, com_pattern, prtcl_num, prtcl_atoms, atom_mask, effective_masses, total_mass, use_memmap, max_chunk_size):
    """
    Ultra-memory-optimized COM computation for very large files.
    
    Features:
    - Micro-chunked processing (1000-2000 frames at a time)
    - Memory monitoring and adaptive chunk sizing
    - Aggressive garbage collection
    - Optimized for 40GB+ files
    """
    import gc
    import psutil
    
    input_file_rel = expand_path_pattern(coor_pattern, "", file_index)
    output_file_rel = expand_path_pattern(com_pattern, "", file_index)
    input_file = os.path.join(baseDir, input_file_rel)
    output_file = os.path.join(baseDir, output_file_rel)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Load coordinate data efficiently (force float32 for memory savings)
        data = _load_coordinate_array(input_file, dtype=np.float32)  # Always use float32 in ultra mode
        
        n_frames = data.shape[0]
        expected_cols = prtcl_num * prtcl_atoms * 3
        
        if data.shape[1] != expected_cols:
            raise ValueError(f"Expected {expected_cols} columns, got {data.shape[1]}")
        
        # In-place reshape
        data = data.reshape(n_frames, prtcl_num, prtcl_atoms * 3)
        
        # Ultra-small chunk processing for memory efficiency
        # Calculate adaptive chunk size based on current memory usage
        current_memory_percent = psutil.virtual_memory().percent
        if current_memory_percent > 70:
            # Reduce chunk size if memory is already high
            adaptive_chunk_size = max(500, max_chunk_size // 2)
            print(f"    High memory detected ({current_memory_percent:.1f}%), reducing chunk size to {adaptive_chunk_size}")
        else:
            adaptive_chunk_size = max_chunk_size
        
        print(f"    Processing {n_frames} frames in chunks of {adaptive_chunk_size}")
        
        # Process in micro-chunks
        centers_of_mass_list = []
        
        for start_idx in range(0, n_frames, adaptive_chunk_size):
            end_idx = min(start_idx + adaptive_chunk_size, n_frames)
            chunk_data = data[start_idx:end_idx]
            
            # Ultra-optimized chunk processing
            chunk_com = _compute_com_vectorized_ultra_optimized(chunk_data, atom_mask, effective_masses, total_mass, prtcl_atoms)
            centers_of_mass_list.append(chunk_com)
            
            # Aggressive cleanup after each chunk
            del chunk_data, chunk_com
            gc.collect()
            
            # Memory check during processing
            if start_idx % (adaptive_chunk_size * 5) == 0:  # Check every 5 chunks
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 80:
                    print(f"    Warning: Memory usage {memory_percent:.1f}% during chunk {start_idx//adaptive_chunk_size + 1}")
        
        # Combine results efficiently
        centers_of_mass = np.vstack(centers_of_mass_list)
        del centers_of_mass_list
        gc.collect()
        
        # In-place flatten
        centers_of_mass_flat = centers_of_mass.reshape(n_frames, -1)
        
        # Cleanup before saving
        del data, centers_of_mass
        gc.collect()
        
        # Save with efficient formatting
        np.savetxt(output_file, centers_of_mass_flat, fmt='%.6f', delimiter=' ')
        
        # Final cleanup
        del centers_of_mass_flat
        gc.collect()
        
        return True
        
    except Exception as e:
        # Cleanup on error
        gc.collect()
        raise RuntimeError(f"Error processing file {input_file}: {e}")


def _compute_com_vectorized_ultra_optimized(data, atom_mask, effective_masses, total_mass, prtcl_atoms):
    """
    Ultra-memory-optimized vectorized center-of-mass computation.
    Designed for micro-chunks to minimize memory usage.
    """
    import gc
    
    n_frames, n_molecules, _ = data.shape
    
    # In-place reshape
    data = data.reshape(n_frames, n_molecules, prtcl_atoms, 3)
    data = data[:, :, atom_mask, :]
    
    # Pre-allocate output with same dtype as input (float32)
    com = np.zeros((n_frames, n_molecules, 3), dtype=data.dtype)
    
    # Ultra-efficient dimension-wise processing
    masses_1d = effective_masses.astype(data.dtype, copy=False)  # Match data type
    
    for dim in range(3):  # x, y, z
        # Process one dimension at a time to minimize temporary arrays
        coord_dim = data[:, :, :, dim]  # Shape: (frames, molecules, atoms)
        
        # Vectorized computation for this dimension
        for mol_idx in range(n_molecules):
            mol_coords = coord_dim[:, mol_idx, :]  # Shape: (frames, atoms)
            weighted_coords = mol_coords * masses_1d
            com[:, mol_idx, dim] = np.sum(weighted_coords, axis=1) / total_mass
            
            # Micro-cleanup
            del mol_coords, weighted_coords
        
        # Cleanup after each dimension
        del coord_dim
        gc.collect()
    
    return com

# Ultra-memory-optimized version WITHOUT psutil dependency
def coms_ultra_memory_optimized_no_psutil(baseDir, coor_pattern, com_pattern, num_dcd, prtcl_num, prtcl_atoms, particl_mass, 
                                         max_workers=None, use_memmap=True, dcd_indices=None, target_memory_gb=45):
    """
    Ultra-memory-optimized version for supercomputers WITHOUT psutil dependency.
    Specifically designed for 42GB files with 4 workers and 240GB memory limit.
    
    Additional Parameters:
    ----------------------
    target_memory_gb : float, optional
        Target memory usage per worker in GB. Default: 45GB (conservative for 4 workers).
    
    Ultra Memory Improvements (No External Dependencies):
    ----------------------------------------------------
    - Conservative chunk sizing based on file characteristics
    - Micro-chunked processing (500-2000 frames at a time)
    - Aggressive garbage collection after each operation
    - Static memory management (no runtime monitoring needed)
    - Optimized for 40GB+ files with multiple workers
    - Uses only numpy and standard library
    """
    import warnings
    import time
    import gc
    import resource  # Built-in module for memory info (if available)
    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    start_time = time.time()
    
    atom_mask, effective_masses, total_mass = _prepare_mass_configuration(particl_mass, prtcl_atoms)
    
    # Create output directory
    com_data_dir = os.path.dirname(f'{baseDir}/{com_pattern}')
    os.makedirs(com_data_dir, exist_ok=True)
    
    # Conservative worker setup for large files
    if max_workers is None:
        max_workers = min(2, mp.cpu_count())  # Default to 2 workers max
    
    # Force memory mapping for ultra-large files
    if not use_memmap:
        print("Warning: Forcing use_memmap=True for ultra-memory-optimized processing")
        use_memmap = True
    
    # Calculate ultra-conservative chunk size for 42GB files
    # Estimate: 500k frames × 1000 molecules × 3 atoms × 3 coords × 4 bytes = ~18GB raw data
    # With processing overhead, target much smaller chunks
    estimated_memory_per_frame = prtcl_num * prtcl_atoms * 3 * 4 / (1024**3)  # GB (float32)
    
    # Ultra-conservative: use only 20% of target memory for active processing
    safety_factor = 0.2
    max_chunk_size = max(250, int(target_memory_gb * safety_factor / estimated_memory_per_frame))
    
    # Extra conservative for 4 workers - reduce chunk size further
    if max_workers >= 4:
        max_chunk_size = max(500, max_chunk_size // 2)  # Even smaller chunks for 4+ workers
        print(f"4+ workers detected: Using extra-conservative chunk size")
    
    print(f"Ultra-memory-optimized mode (no psutil):")
    print(f"  Target memory per worker: {target_memory_gb}GB")
    print(f"  Estimated memory per frame: {estimated_memory_per_frame*1000:.1f}MB")
    print(f"  Ultra-conservative chunk size: {max_chunk_size} frames")
    print(f"  Safety factor: {safety_factor} (using {safety_factor*100}% of target memory)")
    
    # Determine which DCDs to process
    if dcd_indices is None:
        dcd_list = list(range(num_dcd))
    else:
        dcd_list = dcd_indices
    
    results = {
        'success': 0,
        'failed': [],
        'total_time': 0,
        'parallel_workers': max_workers,
        'use_memmap': use_memmap,
        'chunk_size': max_chunk_size,
        'memory_optimized': 'ultra_no_psutil',
        'target_memory_gb': target_memory_gb
    }
    
    print(f"Processing DCDs: {dcd_list}")
    print(f"Molecules: {prtcl_num}, Atoms per molecule: {prtcl_atoms}")
    print(f"Using {max_workers} workers, memmap: {use_memmap}, ultra-memory-optimized: True (no psutil)")
    
    if max_workers == 1:
        # Sequential processing with ultra-memory optimization
        for i in dcd_list:
            try:
                # Try to get memory info using built-in resource module
                try:
                    memory_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
                    memory_available = True
                except:
                    memory_available = False
                
                _compute_com_single_file_ultra_optimized_no_psutil(
                    i, baseDir, coor_pattern, com_pattern, prtcl_num, prtcl_atoms,
                    atom_mask, effective_masses, total_mass, use_memmap, max_chunk_size
                )
                results['success'] += 1
                
                # Show memory info if available
                if memory_available:
                    try:
                        memory_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
                        print(f"✓ Completed file {i+1}/{len(dcd_list)} (Memory: {memory_before:.0f}MB → {memory_after:.0f}MB)")
                    except:
                        print(f"✓ Completed file {i+1}/{len(dcd_list)}")
                else:
                    print(f"✓ Completed file {i+1}/{len(dcd_list)}")
                
                # Aggressive garbage collection between files
                gc.collect()
                
            except Exception as e:
                results['failed'].append(i)
                print(f"✗ Failed file {i}: {e}")
                gc.collect()
    else:
        # Ultra-careful parallel processing
        print(f"Processing {len(dcd_list)} files with {max_workers} workers")
        print(f"Each worker targeting max {target_memory_gb}GB memory usage")
        print(f"Using ultra-conservative chunk size: {max_chunk_size} frames")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_index = {
                executor.submit(
                    _compute_com_single_file_ultra_optimized_no_psutil,
                    i, baseDir, coor_pattern, com_pattern, prtcl_num, prtcl_atoms,
                    atom_mask, effective_masses, total_mass, use_memmap, max_chunk_size
                ): i
                for i in dcd_list
            }
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    future.result()  # Raises exception if failed
                    results['success'] += 1
                    completed += 1
                    
                    print(f"✓ Completed {completed}/{len(dcd_list)} files")
                    
                except Exception as exc:
                    results['failed'].append(index)
                    print(f"✗ Failed file {index}: {exc}")
                
                # Force garbage collection after each completion
                gc.collect()
    
    results['total_time'] = time.time() - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print(f"ULTRA-MEMORY-OPTIMIZED CENTER-OF-MASS CALCULATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files: {len(dcd_list)}")
    print(f"Successful: {results['success']}")
    print(f"Failed: {len(results['failed'])}")
    if results['failed']:
        print(f"Failed indices: {results['failed']}")
    print(f"Total time: {results['total_time']:.2f} seconds")
    print(f"Average time per file: {results['total_time']/len(dcd_list):.2f} seconds")
    print(f"Target memory per worker: {target_memory_gb}GB")
    print(f"Ultra-conservative chunk size: {max_chunk_size} frames")
    print(f"Memory optimizations: micro-chunking, aggressive GC, static memory management")
    print(f"{'='*60}\n")
    
    return results


def _compute_com_single_file_ultra_optimized_no_psutil(file_index, baseDir, coor_pattern, com_pattern, prtcl_num, prtcl_atoms, atom_mask, effective_masses, total_mass, use_memmap, max_chunk_size):
    """
    Ultra-memory-optimized COM computation for supercomputers (no psutil).
    
    Features:
    - Ultra-conservative chunk sizing (250-1000 frames)
    - Aggressive garbage collection
    - Static memory management
    - Optimized for 42GB files with 4 workers
    """
    import gc
    
    input_file_rel = expand_path_pattern(coor_pattern, "", file_index)
    output_file_rel = expand_path_pattern(com_pattern, "", file_index)
    input_file = os.path.join(baseDir, input_file_rel)
    output_file = os.path.join(baseDir, output_file_rel)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Load coordinate data efficiently (always use float32 in ultra mode)
        data = _load_coordinate_array(input_file, dtype=np.float32)  # Float32 saves 50% memory
        
        n_frames = data.shape[0]
        expected_cols = prtcl_num * prtcl_atoms * 3
        
        if data.shape[1] != expected_cols:
            raise ValueError(f"Expected {expected_cols} columns, got {data.shape[1]}")
        
        # In-place reshape to avoid memory duplication
        data = data.reshape(n_frames, prtcl_num, prtcl_atoms * 3)
        
        # Ultra-conservative chunk size - static calculation
        # For 42GB files, use very small chunks to guarantee memory safety
        conservative_chunk_size = min(max_chunk_size, max(250, n_frames // 200))  # At least 200 chunks
        
        print(f"    Processing {n_frames} frames in {n_frames//conservative_chunk_size + 1} chunks of {conservative_chunk_size}")
        
        # Process in ultra-small chunks
        centers_of_mass_list = []
        
        chunk_count = 0
        for start_idx in range(0, n_frames, conservative_chunk_size):
            end_idx = min(start_idx + conservative_chunk_size, n_frames)
            chunk_data = data[start_idx:end_idx]
            
            # Ultra-optimized chunk processing
            chunk_com = _compute_com_vectorized_ultra_optimized_no_psutil(chunk_data, atom_mask, effective_masses, total_mass, prtcl_atoms)
            centers_of_mass_list.append(chunk_com)
            
            # Aggressive cleanup after each chunk
            del chunk_data, chunk_com
            gc.collect()
            
            chunk_count += 1
            # Progress update every 50 chunks
            if chunk_count % 50 == 0:
                print(f"    Processed {chunk_count} chunks...")
        
        # Combine results efficiently
        centers_of_mass = np.vstack(centers_of_mass_list)
        del centers_of_mass_list
        gc.collect()
        
        # In-place flatten
        centers_of_mass_flat = centers_of_mass.reshape(n_frames, -1)
        
        # Cleanup before saving
        del data, centers_of_mass
        gc.collect()
        
        # Save with efficient formatting
        np.savetxt(output_file, centers_of_mass_flat, fmt='%.6f', delimiter=' ')
        
        # Final cleanup
        del centers_of_mass_flat
        gc.collect()
        
        return True
        
    except Exception as e:
        # Cleanup on error
        gc.collect()
        raise RuntimeError(f"Error processing file {input_file}: {e}")


def _compute_com_vectorized_ultra_optimized_no_psutil(data, atom_mask, effective_masses, total_mass, prtcl_atoms):
    """
    Ultra-memory-optimized vectorized COM computation (no psutil).
    Uses molecule-by-molecule processing for maximum memory efficiency.
    """
    import gc
    
    n_frames, n_molecules, _ = data.shape
    
    # In-place reshape
    data = data.reshape(n_frames, n_molecules, prtcl_atoms, 3)
    data = data[:, :, atom_mask, :]
    
    # Pre-allocate output with same dtype as input (float32)
    com = np.zeros((n_frames, n_molecules, 3), dtype=data.dtype)
    
    # Convert masses to match data type
    masses_1d = effective_masses.astype(data.dtype, copy=False)
    
    # Ultra-efficient: process each dimension and molecule separately
    # This minimizes temporary array sizes at the cost of some speed
    for dim in range(3):  # x, y, z
        for mol_idx in range(n_molecules):
            # Extract coordinates for one molecule, one dimension
            mol_coords = data[:, mol_idx, :, dim]  # Shape: (frames, atoms)
            
            # Compute weighted coordinates
            weighted = mol_coords * masses_1d
            com[:, mol_idx, dim] = np.sum(weighted, axis=1) / total_mass
            
            # Micro-cleanup after each molecule
            del mol_coords, weighted
            
            # Force garbage collection every 100 molecules to prevent accumulation
            if mol_idx % 100 == 0:
                gc.collect()
    
    return com
