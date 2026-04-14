#!/usr/bin/env python3
"""
Optimized Dipole Moment Calculation Module

Calculates molecular dipole moments from coordinate and COM data with enhanced
performance, parallel processing, and robust error handling.
"""
import numpy as np
import os
import time
import subprocess
import tempfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
try:
    from .numeric_io import load_numeric_array, save_numeric_array
except ImportError:
    from numeric_io import load_numeric_array, save_numeric_array

# 1 Debye = 0.2081943 e*Å
DEBYE_CONVERSION = 0.2081943


def _tcl_quote(value):
    text = str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace("\"", "\\\"")
    text = text.replace("$", "\\$")
    text = text.replace("[", "\\[")
    text = text.replace("]", "\\]")
    text = text.replace("\n", "\\n")
    text = text.replace("\r", "\\r")
    return text


def _load_array(input_file, dtype=np.float64, io_spec=None):
    """Load numeric data according to the selected storage mode and precision."""
    data = load_numeric_array(
        input_file,
        io_spec,
        default_mode="binary",
        default_precision="single",
    )
    return np.asarray(data, dtype=dtype)


def _extract_selection_metadata(baseDir, psf_pattern, file_index, target_selection, grouping_unit, vmd_path, common_term=""):
    """Use VMD to resolve selected atoms, charges, masses, and grouping metadata for dipole calculations."""
    if grouping_unit not in {"residue", "chain", "segname"}:
        raise ValueError(f"Unsupported grouping unit: {grouping_unit}")
    if not str(vmd_path or "").strip():
        raise ValueError("VMD path is required for individual dipole metadata extraction")
    if not str(target_selection or "").strip():
        raise ValueError("Target selection is required for individual dipole calculation")

    try:
        from .path_utils import expand_path_pattern
    except ImportError:
        from path_utils import expand_path_pattern

    psf_rel = expand_path_pattern(psf_pattern, common_term, file_index)
    psf_full = os.path.join(baseDir, psf_rel)
    if not os.path.exists(psf_full):
        raise FileNotFoundError(f"PSF file not found: {psf_full}")

    script_content = f"""set PSF "{_tcl_quote(psf_full)}"
set molid [mol new $PSF type psf waitfor all]
set sel [atomselect $molid "{_tcl_quote(target_selection)}"]
set groups [$sel get {grouping_unit}]
set masses [$sel get mass]
set charges [$sel get charge]
puts "__META_BEGIN__"
foreach group $groups mass $masses charge $charges {{
    puts "ATOM\\t$group\\t$mass\\t$charge"
}}
puts "__META_END__"
$sel delete
mol delete $molid
quit
"""

    script_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".tcl", delete=False) as handle:
            handle.write(script_content)
            script_path = handle.name

        process = subprocess.run(
            [vmd_path, "-dispdev", "text", "-nt", "1", "-e", script_path],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        if script_path and os.path.exists(script_path):
            os.unlink(script_path)

    if process.returncode != 0:
        raise RuntimeError(
            "VMD failed while extracting dipole metadata from the PSF.\n"
            f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
        )

    lines = process.stdout.splitlines()
    try:
        start = lines.index("__META_BEGIN__") + 1
        end = lines.index("__META_END__")
    except ValueError as exc:
        raise RuntimeError(
            "Could not find dipole metadata markers in VMD output.\n"
            f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
        ) from exc

    atom_groups = []
    atom_masses = []
    atom_charges = []
    for line in lines[start:end]:
        if not line.startswith("ATOM\t"):
            continue
        _tag, group_value, mass_value, charge_value = line.split("\t", 3)
        atom_groups.append(str(group_value))
        atom_masses.append(float(mass_value))
        atom_charges.append(float(charge_value))

    if not atom_groups:
        raise ValueError(
            "The dipole target selection did not resolve to any atoms. "
            "Check the PSF pattern and target selection."
        )

    group_labels = []
    group_lookup = {}
    group_indices = []
    for group_value in atom_groups:
        if group_value not in group_lookup:
            group_lookup[group_value] = len(group_labels)
            group_labels.append(group_value)
        group_indices.append(group_lookup[group_value])

    return {
        "atom_count": len(atom_groups),
        "atom_masses": np.asarray(atom_masses, dtype=np.float64),
        "atom_charges": np.asarray(atom_charges, dtype=np.float64),
        "group_indices": np.asarray(group_indices, dtype=np.int32),
        "group_labels": group_labels,
        "psf_path": psf_full,
    }


def _prepare_dipole_metadata(metadata):
    atom_masses = np.asarray(metadata["atom_masses"], dtype=np.float64)
    atom_charges = np.asarray(metadata["atom_charges"], dtype=np.float64)
    group_indices_all = np.asarray(metadata["group_indices"], dtype=np.int32)

    com_atom_mask = atom_masses != 0.0
    if not np.any(com_atom_mask):
        raise ValueError("Selected atoms do not contain any nonzero masses")

    group_labels = list(metadata["group_labels"])
    group_masses = np.zeros(len(group_labels), dtype=np.float64)
    for group_index, mass in zip(group_indices_all[com_atom_mask], atom_masses[com_atom_mask]):
        group_masses[int(group_index)] += float(mass)
    if np.any(group_masses <= 0.0):
        raise ValueError("At least one dipole group has non-positive total mass")

    return {
        "atom_count": int(metadata["atom_count"]),
        "group_indices_all": group_indices_all,
        "group_labels": group_labels,
        "group_count": len(group_labels),
        "atom_charges": atom_charges,
        "com_atom_mask": com_atom_mask,
        "com_group_indices": group_indices_all[com_atom_mask],
        "com_masses": atom_masses[com_atom_mask],
        "group_masses": group_masses,
    }


def _compute_group_com_vectorized(atom_coords, metadata):
    """Compute one center of mass per group for each frame."""
    coords_for_com = atom_coords[:, metadata["com_atom_mask"], :]
    n_frames = atom_coords.shape[0]
    n_groups = metadata["group_count"]
    centers_of_mass = np.zeros((n_frames, n_groups, 3), dtype=np.float64)

    for group_index in range(n_groups):
        atom_mask = metadata["com_group_indices"] == group_index
        group_coords = coords_for_com[:, atom_mask, :]
        group_masses = metadata["com_masses"][atom_mask]
        weighted = group_coords * group_masses.reshape(1, -1, 1)
        centers_of_mass[:, group_index, :] = np.sum(weighted, axis=1) / metadata["group_masses"][group_index]

    return centers_of_mass


def _compute_group_dipoles(atom_coords, metadata, dipole_unit="Debye", calculate_magnitudes=True):
    """Compute dipole vectors and magnitudes for variable-size groups."""
    centers_of_mass = _compute_group_com_vectorized(atom_coords, metadata)
    com_per_atom = centers_of_mass[:, metadata["group_indices_all"], :]
    relative_coords = atom_coords - com_per_atom
    weighted = relative_coords * metadata["atom_charges"].reshape(1, -1, 1)

    dipole_vectors = np.zeros((atom_coords.shape[0], metadata["group_count"], 3), dtype=np.float64)
    for group_index in range(metadata["group_count"]):
        atom_mask = metadata["group_indices_all"] == group_index
        dipole_vectors[:, group_index, :] = np.sum(weighted[:, atom_mask, :], axis=1)

    if dipole_unit == "Debye":
        dipole_vectors = dipole_vectors / DEBYE_CONVERSION
    elif dipole_unit != "e·Å":
        raise ValueError(f"Unsupported dipole unit: {dipole_unit}")

    dipole_magnitudes = None
    if calculate_magnitudes:
        dipole_magnitudes = np.linalg.norm(dipole_vectors, axis=2)

    return dipole_vectors, dipole_magnitudes


def _process_single_dipole_file(file_idx, baseDir, coords_pattern, output_pattern,
                                magnitudes_pattern, dipole_metadata,
                                stride, common_term, dipole_unit,
                                coords_input_io_spec=None,
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
        print(f"Processing file {file_idx}:")
        print(f"  Coord file: {coord_file}")
        
        # Check if files exist
        if not os.path.exists(coord_file):
            error_msg = f'Coordinate file not found: {coord_file}'
            print(f"  ERROR: {error_msg}")
            return {'success': False, 'error': error_msg, 'file_idx': file_idx}
            
        # Load data
        print(f"  Loading data from files...")
        coord_data = _load_array(coord_file, dtype=np.float64, io_spec=coords_input_io_spec)
        print(f"  Loaded coords: {coord_data.shape}")
        
        # Apply stride if specified
        if stride > 1:
            coord_data = coord_data[::stride]

        expected_coord_cols = dipole_metadata["atom_count"] * 3
        actual_coord_cols = int(coord_data.shape[1]) if coord_data.ndim > 1 else 0
        if actual_coord_cols < expected_coord_cols:
            raise ValueError(
                f"Coordinate input has {actual_coord_cols} columns, but dipole calculation requires "
                f"{expected_coord_cols} columns for {dipole_metadata['atom_count']} selected atoms."
            )
        if actual_coord_cols > expected_coord_cols:
            coord_data = coord_data[:, :expected_coord_cols]

        n_frames = coord_data.shape[0]
        atom_coords = coord_data.reshape(n_frames, dipole_metadata["atom_count"], 3)
        print(f"  Processing {n_frames} frames across {dipole_metadata['group_count']} dipole group(s)")

        calculate_magnitudes = bool(magnitudes_pattern)
        dipole_vectors, dipole_magnitudes = _compute_group_dipoles(
            atom_coords,
            dipole_metadata,
            dipole_unit=dipole_unit,
            calculate_magnitudes=calculate_magnitudes,
        )
        if calculate_magnitudes:
            print(f"  Dipole calculation returned: vectors {dipole_vectors.shape}, magnitudes {dipole_magnitudes.shape}")
        else:
            print(f"  Dipole calculation returned: vectors {dipole_vectors.shape}, magnitudes skipped")
        
        # Save vector results, and magnitude results only when requested.
        dipole_file = expand_path_pattern(output_pattern, common_term, file_idx)
        dipole_path = os.path.join(baseDir, dipole_file)
        os.makedirs(os.path.dirname(dipole_path), exist_ok=True)
        magnitude_path = None
        if magnitudes_pattern:
            magnitude_file = expand_path_pattern(magnitudes_pattern, common_term, file_idx)
            magnitude_path = os.path.join(baseDir, magnitude_file)
            if magnitude_path != dipole_path:
                os.makedirs(os.path.dirname(magnitude_path), exist_ok=True)
        
        # Reshape dipole vectors to match expected output format
        dipole_vectors_flat = dipole_vectors.reshape(n_frames, -1)
        
        print(f"  Saving to: {dipole_path}")
        save_numeric_array(
            dipole_path,
            dipole_vectors_flat,
            vectors_output_io_spec,
            default_mode="text",
            default_precision="double",
        )
        if magnitude_path is not None:
            print(f"  Saving to: {magnitude_path}")
            save_numeric_array(
                magnitude_path,
                dipole_magnitudes,
                magnitudes_output_io_spec,
                default_mode="text",
                default_precision="double",
            )

        # Verify outputs immediately so binary/text mismatches fail here instead of later modules.
        load_numeric_array(
            dipole_path,
            vectors_output_io_spec,
            default_mode="text",
            default_precision="double",
        )
        if magnitude_path is not None:
            load_numeric_array(
                magnitude_path,
                magnitudes_output_io_spec,
                default_mode="text",
                default_precision="double",
            )
        
        print(f"  SUCCESS: File {file_idx} processed, {n_frames} frames")
        result = {
            'success': True, 
            'file_idx': file_idx, 
            'frames_processed': n_frames,
        }
        if dipole_magnitudes is not None:
            result['mean_magnitude'] = np.mean(dipole_magnitudes)
            result['std_magnitude'] = np.std(dipole_magnitudes)
        return result
        
    except Exception as e:
        import traceback
        error_details = f"File {file_idx} failed: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(f"  ERROR: {error_details}")
        return {'success': False, 'error': error_details, 'file_idx': file_idx}

def dipole_functions(baseDir, coords_pattern, psf_pattern, target_selection, vmd_path, grouping_unit,
                    dipole_unit,
                    output_pattern, num_dcds, stride=1, max_workers=1, chunk_processing=True,
                    validate_data=True, progress_callback=None, common_term="",
                    magnitudes_pattern=None, dcd_indices=None,
                    coords_input_io_spec=None,
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
    psf_pattern : str
        Path pattern for PSF files used to resolve charges, masses, and grouping.
    target_selection : str
        VMD atom selection string used to define the atoms included in dipole calculations.
    vmd_path : str
        Path to the VMD executable used to resolve PSF metadata.
    grouping_unit : str
        Grouping unit used to define individual dipoles. One of residue, chain, segname.
    dipole_unit : str
        Output dipole unit. One of Debye or e·Å.
    output_pattern : str
        Path pattern for output vector files OR output directory (backward compatibility).
        If magnitudes_pattern is None, treated as directory. Otherwise, treated as vector file pattern.
        Example: "anlz/NVT_*/dipole/vectors_{i}.dat"
    magnitudes_pattern : str, optional
        Path pattern for magnitude output files. If provided, output_pattern is treated as vector pattern.
        Example: "anlz/NVT_*/dipole/magnitudes_{i}.dat"
    dcd_indices : list, optional
        List of specific DCD indices to process. If None, processes all DCDs.
    num_dcds : int
        Number of trajectory files to process
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
    common_term : str, optional
        Common term for path expansion (default: "")
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

    is_valid, error_msg = validate_path_pattern(psf_pattern)
    if not is_valid:
        raise ValueError(f"Invalid PSF pattern: {error_msg}")

    is_valid, error_msg = validate_path_pattern(output_pattern)
    if not is_valid:
        raise ValueError(f"Invalid output pattern: {error_msg}")
    if grouping_unit not in {"residue", "chain", "segname"}:
        raise ValueError("grouping_unit must be one of: residue, chain, segname")
    if dipole_unit not in {"Debye", "e·Å"}:
        raise ValueError("dipole_unit must be one of: Debye, e·Å")
    if not str(target_selection or "").strip():
        raise ValueError("target_selection is required for individual dipole calculation")
    if not str(vmd_path or "").strip():
        raise ValueError("vmd_path is required for individual dipole calculation")

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
    print(f"PSF pattern: {psf_pattern}")
    print(f"Target selection: {target_selection}")
    print(f"Grouping unit: {grouping_unit}")
    print(f"Dipole unit: {dipole_unit}")
    print(f"Common term: {common_term}")
    
    # Handle DCD selection
    if dcd_indices is not None:
        actual_dcd_list = dcd_indices
        actual_num_dcds = len(dcd_indices)
        print(f"Using DCD selection: {dcd_indices}")
    else:
        actual_dcd_list = list(range(num_dcds))
        actual_num_dcds = num_dcds
        print(f"Processing all DCDs: 0 to {num_dcds-1}")
    metadata = _extract_selection_metadata(
        baseDir,
        psf_pattern,
        actual_dcd_list[0] if actual_dcd_list else 0,
        target_selection,
        grouping_unit,
        vmd_path,
        common_term,
    )
    dipole_metadata = _prepare_dipole_metadata(metadata)
    print(f"Resolved {dipole_metadata['atom_count']} atom(s) from PSF selection {metadata['psf_path']}")
    print(f"Resolved {dipole_metadata['group_count']} dipole group(s)")

    # Process files
    results = []
    successful_files = 0
    successful_indices = []
    failed_results = []
    total_frames = 0
    quality_metrics = {'mean_magnitudes': [], 'std_magnitudes': []}
    
    print(f"Processing {actual_num_dcds} trajectory files for dipole moment calculation...")
    print(f"Using {max_workers} workers")
    
    # Use all requested workers - no memory limitations on supercomputer
    print(f"Using {max_workers} workers for parallel processing")
    
    if max_workers > 1:
        # Try parallel processing with fallback to single-threaded
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_idx = {executor.submit(_process_single_dipole_file, i, baseDir, coords_pattern,
                                                 output_pattern, magnitudes_pattern, dipole_metadata,
                                                 stride, common_term, dipole_unit, coords_input_io_spec,
                                                 vectors_output_io_spec, magnitudes_output_io_spec): i for i in actual_dcd_list}
                
                # Process completed tasks
                for future in as_completed(future_to_idx):
                    result = future.result()
                    results.append(result)
                    if result['success']:
                        successful_files += 1
                        successful_indices.append(result['file_idx'])
                        total_frames += result['frames_processed']
                        if 'mean_magnitude' in result:
                            quality_metrics['mean_magnitudes'].append(result['mean_magnitude'])
                            quality_metrics['std_magnitudes'].append(result['std_magnitude'])
                            print(f"✓ File {result['file_idx']}: {result['frames_processed']} frames, avg magnitude: {result['mean_magnitude']:.3f} D")
                        else:
                            print(f"✓ File {result['file_idx']}: {result['frames_processed']} frames")
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
                    result = _process_single_dipole_file(i, baseDir, coords_pattern,
                                                       output_pattern, magnitudes_pattern, dipole_metadata,
                                                       stride, common_term, dipole_unit, coords_input_io_spec, vectors_output_io_spec, magnitudes_output_io_spec)
                    results.append(result)
                    if result['success']:
                        successful_files += 1
                        successful_indices.append(result['file_idx'])
                        total_frames += result['frames_processed']
                        if 'mean_magnitude' in result:
                            quality_metrics['mean_magnitudes'].append(result['mean_magnitude'])
                            quality_metrics['std_magnitudes'].append(result['std_magnitude'])
                            print(f"✓ File {result['file_idx']}: {result['frames_processed']} frames, avg magnitude: {result['mean_magnitude']:.3f} D")
                        else:
                            print(f"✓ File {result['file_idx']}: {result['frames_processed']} frames")
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
            result = _process_single_dipole_file(i, baseDir, coords_pattern,
                                               output_pattern, magnitudes_pattern, dipole_metadata,
                                               stride, common_term, dipole_unit, coords_input_io_spec, vectors_output_io_spec, magnitudes_output_io_spec)
            results.append(result)
            
            if result['success']:
                successful_files += 1
                successful_indices.append(result['file_idx'])
                total_frames += result['frames_processed']
                if 'mean_magnitude' in result:
                    quality_metrics['mean_magnitudes'].append(result['mean_magnitude'])
                    quality_metrics['std_magnitudes'].append(result['std_magnitude'])
                    print(f"✓ File {result['file_idx']}: {result['frames_processed']} frames, avg magnitude: {result['mean_magnitude']:.3f} D")
                else:
                    print(f"✓ File {result['file_idx']}: {result['frames_processed']} frames")
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
