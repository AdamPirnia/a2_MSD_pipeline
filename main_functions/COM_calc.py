import numpy as np
import os
import subprocess
import tempfile
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
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


def _load_coordinate_array(input_file, dtype=np.float64, io_spec=None, mmap_mode=None):
    """Load coordinate data using the configured storage mode and precision."""
    data = load_numeric_array_trimmed(
        input_file,
        io_spec,
        default_mode="binary",
        default_precision="single",
        mmap_mode=mmap_mode,
        context=f"COM input {os.path.basename(input_file)}",
    )
    if mmap_mode:
        return np.asarray(data)
    return np.asarray(data, dtype=dtype)


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


def _extract_selection_metadata(baseDir, psf_pattern, file_index, target_selection, grouping_unit, vmd_path, common_term=""):
    """Use VMD's selection engine to resolve atom masses and group membership from the PSF."""
    if grouping_unit not in {"residue", "chain", "segname"}:
        raise ValueError(f"Unsupported grouping unit: {grouping_unit}")
    if not str(vmd_path or "").strip():
        raise ValueError("VMD path is required for COM metadata extraction")
    if not str(target_selection or "").strip():
        raise ValueError("Target selection is required for COM calculation")

    psf_rel = expand_path_pattern(psf_pattern, common_term, file_index)
    psf_full = os.path.join(baseDir, psf_rel)
    if not os.path.exists(psf_full):
        raise FileNotFoundError(f"PSF file not found: {psf_full}")

    if grouping_unit == "residue":
        group_tcl = """set group_segids [$sel get segid]
set group_resids [$sel get resid]
set groups [list]
foreach group_segid $group_segids group_resid $group_resids {
    lappend groups "$group_segid:$group_resid"
}
"""
    else:
        group_tcl = f"set groups [$sel get {grouping_unit}]\n"

    script_content = f"""set PSF "{_tcl_quote(psf_full)}"
set molid [mol new $PSF type psf waitfor all]
set sel [atomselect $molid "{_tcl_quote(target_selection)}"]
{group_tcl.rstrip()}
set masses [$sel get mass]
puts "__META_BEGIN__"
foreach group $groups mass $masses {{
    puts "ATOM\\t$group\\t$mass"
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
            "VMD failed while extracting COM metadata from the PSF.\n"
            f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
        )

    lines = process.stdout.splitlines()
    try:
        start = lines.index("__META_BEGIN__") + 1
        end = lines.index("__META_END__")
    except ValueError as exc:
        raise RuntimeError(
            "Could not find COM metadata markers in VMD output.\n"
            f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
        ) from exc

    atom_groups = []
    atom_masses = []
    for line in lines[start:end]:
        if not line.startswith("ATOM\t"):
            continue
        _tag, group_value, mass_value = line.split("\t", 2)
        atom_groups.append(str(group_value))
        atom_masses.append(float(mass_value))

    if not atom_groups:
        raise ValueError(
            "The COM target selection did not resolve to any atoms. "
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
        "group_indices": np.asarray(group_indices, dtype=np.int32),
        "group_labels": group_labels,
        "psf_path": psf_full,
    }


def _prepare_com_metadata(metadata, calc_mode):
    atom_masses = np.asarray(metadata["atom_masses"], dtype=np.float64)
    atom_mask = atom_masses != 0.0
    if not np.any(atom_mask):
        raise ValueError("Selected atoms do not contain any nonzero masses")

    if calc_mode == "collective":
        effective_group_indices = np.zeros(int(np.count_nonzero(atom_mask)), dtype=np.int32)
        group_labels = ["collective"]
    else:
        original_group_indices = np.asarray(metadata["group_indices"], dtype=np.int32)[atom_mask]
        original_group_labels = list(metadata["group_labels"])
        unique_indices = []
        seen = set()
        for index in original_group_indices:
            index_value = int(index)
            if index_value not in seen:
                seen.add(index_value)
                unique_indices.append(index_value)
        remap = {old_index: new_index for new_index, old_index in enumerate(unique_indices)}
        effective_group_indices = np.asarray([remap[int(index)] for index in original_group_indices], dtype=np.int32)
        group_labels = [original_group_labels[index] for index in unique_indices]

    effective_masses = atom_masses[atom_mask]
    group_masses = np.zeros(len(group_labels), dtype=np.float64)
    for group_index, mass in zip(effective_group_indices, effective_masses):
        group_masses[int(group_index)] += float(mass)
    if np.any(group_masses <= 0.0):
        raise ValueError("At least one inferred COM group has non-positive total mass")

    active_atom_indices = np.flatnonzero(atom_mask).astype(np.intp, copy=False)
    sort_order = np.argsort(effective_group_indices, kind="stable")
    sorted_group_indices = effective_group_indices[sort_order]
    present_group_indices, group_start_indices = np.unique(sorted_group_indices, return_index=True)
    expected_group_indices = np.arange(len(group_labels), dtype=present_group_indices.dtype)
    if present_group_indices.shape != expected_group_indices.shape or not np.array_equal(present_group_indices, expected_group_indices):
        raise ValueError("COM group metadata is not contiguous after zero-mass atoms are removed")

    return {
        "atom_count": int(metadata["atom_count"]),
        "atom_mask": atom_mask,
        "effective_masses": effective_masses,
        "group_indices": effective_group_indices,
        "group_labels": group_labels,
        "group_masses": group_masses,
        "sorted_atom_indices": active_atom_indices[sort_order],
        "sorted_effective_masses": effective_masses[sort_order],
        "group_start_indices": group_start_indices.astype(np.intp, copy=False),
    }


def _prepare_mass_configuration(particl_mass, prtcl_atoms):
    """Compatibility helper for legacy fixed-size COM workflows."""
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


def _build_legacy_com_metadata(prtcl_num, prtcl_atoms, particl_mass):
    """Build COM metadata from fixed molecule count/atom masses when VMD metadata is unavailable."""
    if prtcl_num is None or prtcl_atoms is None or particl_mass is None:
        raise ValueError("Legacy COM metadata requires prtcl_num, prtcl_atoms, and particl_mass")

    molecule_count = int(prtcl_num)
    atoms_per_molecule = int(prtcl_atoms)
    if molecule_count <= 0:
        raise ValueError("prtcl_num must be a positive integer")
    if atoms_per_molecule <= 0:
        raise ValueError("prtcl_atoms must be a positive integer")

    masses = np.asarray(particl_mass, dtype=np.float64)
    if masses.size != atoms_per_molecule:
        raise ValueError(
            f"Mass list length ({masses.size}) must match prtcl_atoms ({atoms_per_molecule})"
        )

    return {
        "atom_count": molecule_count * atoms_per_molecule,
        "atom_masses": np.tile(masses, molecule_count),
        "group_indices": np.repeat(np.arange(molecule_count, dtype=np.int32), atoms_per_molecule),
        "group_labels": [str(i) for i in range(molecule_count)],
        "psf_path": "<legacy-fixed-metadata>",
    }

######################################################    Parameters    

def coms(baseDir, input_pattern, output_pattern, num_dcd, psf_pattern=None, target_selection=None, vmd_path=None, calc_mode="individual", grouping_unit="residue", max_workers=None, use_memmap=False, chunk_size=None, dcd_indices=None, common_term="", input_io_spec=None, output_io_spec=None, prtcl_num=None, prtcl_atoms=None, particl_mass=None):
    """
    Compute and save per-frame center-of-mass coordinates from extracted atomic coordinates.

    The current workflow infers atom masses and group membership from the PSF plus
    VMD atom selection text, so heterogeneous groups with different atom counts are
    supported. Individual mode returns one COM per selected group, while collective
    mode returns exactly one COM per frame.

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
    psf_pattern : str
        Path pattern for PSF files used to infer masses and group membership.
    target_selection : str
        VMD atom selection used to choose atoms from the PSF.
    vmd_path : str
        Full path to the VMD executable.
    calc_mode : str
        Either "individual" or "collective".
    grouping_unit : str
        Grouping unit for individual COM calculation. One of residue, chain, segname.
    max_workers : int, optional
        Maximum number of parallel workers. Defaults to min(num_dcd, CPU count).
    use_memmap : bool, optional
        Use memory mapping for very large files to reduce RAM usage.
    chunk_size : int, optional
        Number of frames to process at once inside each worker. If None, an
        automatic chunk size is selected from the input frame count.
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
    - Input files must store the selected atoms in the same order used by the VMD
      selection on the PSF. Step 1 coordinate extraction already preserves that order.
    - In collective mode the output shape is (frames, 3).
    - For very large systems, consider using use_memmap=True to reduce memory usage.
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
    print(f"Calculation mode: {calc_mode}")
    if calc_mode == "individual":
        print(f"Grouping unit: {grouping_unit}")
    print(f"PSF pattern: {psf_pattern}")
    print(f"Target selection: {target_selection}")
    
    # Determine which DCDs to process
    if dcd_indices is None:
        dcd_list = list(range(num_dcd))
    else:
        dcd_list = dcd_indices
        print(f"Processing selected DCDs: {dcd_list}")
    if not dcd_list:
        raise ValueError("No DCD indices were selected for COM calculation")
    
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

    if prtcl_num is not None and prtcl_atoms is not None and particl_mass is not None:
        metadata = _build_legacy_com_metadata(prtcl_num, prtcl_atoms, particl_mass)
        print("Using legacy fixed-size COM metadata from prtcl_num/prtcl_atoms/particl_mass")
    else:
        metadata = _extract_selection_metadata(
            baseDir,
            psf_pattern,
            dcd_list[0],
            target_selection,
            grouping_unit,
            vmd_path,
            common_term,
        )
    com_metadata = _prepare_com_metadata(metadata, calc_mode)
    ignored_atoms = metadata["atom_count"] - int(np.count_nonzero(com_metadata["atom_mask"]))
    print(f"Resolved {metadata['atom_count']} atom(s) from PSF selection {metadata['psf_path']}")
    print(f"Resolved {len(com_metadata['group_labels'])} COM group(s)")
    if ignored_atoms:
        print(f"Ignoring {ignored_atoms} zero-mass atom(s) from the selected atoms during COM calculation")
    chunk_size = _apply_runtime_com_chunk_cap(chunk_size, com_metadata, int(max_workers))

    results = {
        'success': 0,
        'failed': [],
        'total_time': 0,
        'parallel_workers': max_workers,
        'use_memmap': use_memmap,
        'chunk_size': chunk_size
    }
    
    print(f"Using {max_workers} workers, memmap: {use_memmap}, chunk size: {chunk_size or 'auto'}")
    
    if max_workers == 1:
        # Sequential processing
        for i in dcd_list:
            try:
                _compute_com_single_file(
                    i, baseDir, input_pattern, output_pattern, com_metadata,
                    use_memmap, chunk_size, common_term, input_io_spec, output_io_spec
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
                    i, baseDir, input_pattern, output_pattern, com_metadata,
                    use_memmap, chunk_size, common_term, input_io_spec, output_io_spec
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
        expected_columns = len(com_metadata["group_labels"]) * 3
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


def _normalize_chunk_size(chunk_size, n_frames):
    if chunk_size is None or str(chunk_size).strip().lower() in {"", "auto"}:
        if n_frames <= 1000:
            return n_frames
        return min(n_frames, max(1000, n_frames // 4))
    try:
        value = int(chunk_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("Step 3 chunk size must be an integer >= 1 or 'auto'") from exc
    if value < 1:
        raise ValueError("Step 3 chunk size must be >= 1")
    return min(value, n_frames)


def _recommended_runtime_com_chunk_cap(com_metadata, max_workers):
    if max_workers <= 1:
        return None
    active_atoms = int(len(com_metadata["sorted_atom_indices"]))
    num_groups = int(len(com_metadata["group_masses"]))
    if active_atoms <= 0 or num_groups <= 0:
        return None

    chunk_workspace_per_frame_mb = (
        active_atoms * 8
        + num_groups * 8
        + num_groups * 3 * 8
    ) / (1024**2)
    if chunk_workspace_per_frame_mb <= 0:
        return None

    if max_workers >= 16:
        target_chunk_workspace_mb = 512.0
    elif max_workers >= 8:
        target_chunk_workspace_mb = 768.0
    else:
        target_chunk_workspace_mb = 1024.0
    return max(1, int(target_chunk_workspace_mb / chunk_workspace_per_frame_mb))


def _apply_runtime_com_chunk_cap(chunk_size, com_metadata, max_workers):
    cap = _recommended_runtime_com_chunk_cap(com_metadata, max_workers)
    if cap is None:
        return chunk_size
    if chunk_size is None or str(chunk_size).strip().lower() in {"", "auto"}:
        print(f"Auto Step 3 chunk size capped at {cap} frames for {max_workers} parallel COM workers")
        return cap
    try:
        requested = int(chunk_size)
    except (TypeError, ValueError):
        return chunk_size
    if requested > cap:
        print(
            f"Reducing Step 3 chunk size from {requested} to {cap} frames "
            f"for {max_workers} parallel COM workers"
        )
        return cap
    return chunk_size


def _compute_com_single_file(file_index, baseDir, input_pattern, output_pattern, com_metadata, use_memmap, chunk_size=None, common_term="", input_io_spec=None, output_io_spec=None):
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
            data = _load_coordinate_array(input_file, dtype=np.float32, io_spec=input_io_spec, mmap_mode="r")
        else:
            data = _load_coordinate_array(input_file, dtype=np.float64, io_spec=input_io_spec)
        
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_frames = data.shape[0]
        expected_cols = com_metadata["atom_count"] * 3
        if int(data.shape[1]) != expected_cols:
            raise ValueError(
                f"Coordinate input has {data.shape[1]} columns, but COM calculation requires "
                f"{expected_cols} columns for {com_metadata['atom_count']} selected atoms."
            )

        atom_coords = data.reshape(n_frames, com_metadata["atom_count"], 3)
        effective_chunk_size = _normalize_chunk_size(chunk_size, n_frames)
        num_groups = len(com_metadata["group_masses"])
        centers_of_mass_flat = np.empty((n_frames, num_groups * 3), dtype=np.float64)
        if effective_chunk_size < n_frames:
            total_chunks = (n_frames + effective_chunk_size - 1) // effective_chunk_size
            print(f"    Using Step 3 chunked processing: {effective_chunk_size} frames per chunk ({total_chunks} chunks)")
        for start_idx in range(0, n_frames, effective_chunk_size):
            end_idx = min(start_idx + effective_chunk_size, n_frames)
            centers_chunk = _compute_com_vectorized(
                atom_coords[start_idx:end_idx],
                com_metadata,
            )
            centers_of_mass_flat[start_idx:end_idx] = centers_chunk.reshape(end_idx - start_idx, -1)
        
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


def _compute_com_vectorized(data, com_metadata):
    """
    Compute center-of-mass using highly optimized vectorized operations.
    
    Parameters
    ----------
    data : ndarray, shape (n_frames, n_atoms, 3)
        Coordinate data for the selected atoms.
    
    Returns
    -------
    com : ndarray, shape (n_frames, n_groups, 3)
        Center-of-mass coordinates.
    """
    sorted_atom_indices = com_metadata["sorted_atom_indices"]
    sorted_masses = com_metadata["sorted_effective_masses"]
    group_start_indices = com_metadata["group_start_indices"]
    group_masses = com_metadata["group_masses"]
    num_groups = int(len(group_masses))
    centers = np.empty((data.shape[0], num_groups, 3), dtype=np.float64)
    mass_row = sorted_masses.reshape(1, -1)
    group_mass_row = group_masses.reshape(1, -1)

    for dim in range(3):
        weighted_component = np.asarray(data[:, sorted_atom_indices, dim], dtype=np.float64)
        weighted_component *= mass_row
        group_sums = np.add.reduceat(weighted_component, group_start_indices, axis=1)
        centers[:, :, dim] = group_sums / group_mass_row

    return centers


