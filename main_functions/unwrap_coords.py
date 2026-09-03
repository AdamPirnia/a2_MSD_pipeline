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


def _read_xsc_box_trajectory(xsc_path):
    """Read every data row of an XSC file as one (x, y, z) box-dimension frame.

    Used for constant-pressure (NPT) ensembles, where box dimensions fluctuate
    frame to frame and a single XSC snapshot is not representative of the
    whole trajectory.
    """
    with open(xsc_path, 'r') as fr:
        lines = fr.readlines()
    box_rows = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        parts = stripped.split()
        box_rows.append([float(parts[1]), float(parts[5]), float(parts[9])])
    if not box_rows:
        raise ValueError(f"No box-dimension rows found in XSC file: {xsc_path}")
    return np.array(box_rows, dtype=np.float64)


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
    wrap_groups_into_cell=False,
    wrap_center="cell",
    ensemble="nvt",
    mode=None,
    grouping_source="psf",
    atoms_per_molecule=None,
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
         e. Optionally wraps whole selected groups back into the primary cell.
         f. Saves the output coordinates to the output pattern.

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
        For ensemble="npt", each resolved XSC file must contain a full trajectory of
        box-dimension rows (one row per coordinate frame in the corresponding input
        file for that index), not a single restart snapshot.
    ensemble : str, optional
        "nvt" (constant volume, default) reads one box-size frame from the XSC file
        for the first DCD index and applies it uniformly to every frame of every
        file, which is exact for NVT and was the only behavior before this option
        existed. "npt" (constant pressure) reads a full per-frame box-dimension
        trajectory from each index's own XSC file and uses it frame-by-frame during
        unwrapping, first-frame repair, and group wrapping.
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

    ensemble = str(ensemble or "nvt").strip().lower()
    if ensemble not in {"nvt", "npt"}:
        raise ValueError("ensemble must be 'nvt' (constant volume) or 'npt' (constant pressure).")

    # Back-compat: the legacy ``wrap_groups_into_cell`` flag selects the new
    # whole-molecule mode when an explicit ``mode`` is not given.
    mode = str(mode or "").strip().lower()
    if not mode:
        mode = "whole_in_cell" if wrap_groups_into_cell else "continuous"
    if mode not in {"continuous", "whole_in_cell"}:
        raise ValueError("mode must be 'continuous' or 'whole_in_cell'.")

    grouping_source = str(grouping_source or "psf").strip().lower()
    if grouping_source in {"psf", "vmd"}:
        grouping_source = "psf"
    elif grouping_source in {"blocks", "block", "fixed", "no_psf", "atoms_per_molecule"}:
        grouping_source = "blocks"
    else:
        raise ValueError("grouping_source must be 'psf' or 'blocks'.")

    if mode == "continuous" and wrap_groups_into_cell:
        raise ValueError("wrap_groups_into_cell is incompatible with mode='continuous'.")

    groups_needed = (mode == "whole_in_cell") or repair_first_frame
    if groups_needed:
        if grouping_source == "psf":
            if not (psf_pattern and target_selection and vmd
                    and grouping_unit in {"residue", "chain", "segname"}):
                raise ValueError(
                    "PSF grouping requires psf_pattern, target_selection, vmd, and "
                    "grouping_unit in {residue, chain, segname}."
                )
            is_valid, error_msg = validate_path_pattern(psf_pattern)
            if not is_valid:
                raise ValueError(f"Invalid PSF pattern for Step 2 group metadata: {error_msg}")
        elif not atoms_per_molecule:
            raise ValueError(
                "Whole-molecule grouping without a PSF requires atoms_per_molecule "
                "(e.g. '3', '3x1000', or '3,3,10')."
            )

    print(f"{'='*50}")
    print(f"COORDINATE UNWRAPPING")
    print(f"{'='*50}")
    print(f"Base directory: {baseDir}")
    print(f"Input pattern: {input_pattern}")
    print(f"Output pattern: {output_pattern}")
    print(f"XSC pattern: {xsc_pattern}")
    print(f"Common term: {common_term}")
    print(f"Number of DCDs: {num_dcd}")
    print(f"Mode: {mode}")
    if groups_needed:
        print(f"Grouping source: {grouping_source}"
              + (f" ({grouping_unit})" if grouping_source == "psf" else f" ({atoms_per_molecule})"))
    if mode == "continuous":
        print(f"Repair first frame: {bool(repair_first_frame)}")
    
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
    box_size = None
    if ensemble == "nvt":
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
    else:
        print(
            "✓ Constant Pressure (NPT) ensemble selected: each input file will read its own "
            "per-frame box-dimension trajectory from its XSC file during processing."
        )
    
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
                    psf_pattern, target_selection, vmd, grouping_unit,
                    bool(repair_first_frame), False, wrap_center, ensemble,
                    mode, grouping_source, atoms_per_molecule,
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
                    psf_pattern, target_selection, vmd, grouping_unit,
                    bool(repair_first_frame), False, wrap_center, ensemble,
                    mode, grouping_source, atoms_per_molecule,
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


def _resolve_group_metadata_via_vmd(file_index, baseDir, psf_pattern, target_selection, vmd_path, grouping_unit, common_term=""):
    """Ask VMD for selected atom grouping, masses, and bonds, in Step 1 selection order.

    Returns a dict with:
      ``group_indices`` : list[list[int]]  -- selection-local column positions per molecule
      ``masses``        : np.ndarray | None -- per-atom mass in selection order
      ``bond_pairs``    : list[tuple[int, int]] -- intra-selection bonds (local positions)
    """

    psf_file_rel = expand_path_pattern(psf_pattern, common_term, file_index)
    psf_file = os.path.join(baseDir, psf_file_rel)
    if not os.path.exists(psf_file):
        raise FileNotFoundError(f"PSF file for Step 2 group metadata not found: {psf_file}")
    if os.path.getsize(psf_file) <= 0:
        raise ValueError(_format_unreadable_file_message("PSF", psf_file))

    os.makedirs("writenCodes", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    tcl_filename = f"unwrap_groups_{common_term}_{file_index}.tcl" if common_term else f"unwrap_groups_{file_index}.tcl"
    script_path = os.path.join("writenCodes", tcl_filename)
    log_path = os.path.join("logs", f"unwrap_groups_{file_index}.log")

    with open(script_path, "w") as handle:
        handle.write(f"""# Resolve selected atom groups for Step 2 group operations
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

# Per-atom masses in selection order (used for mass-weighted molecule centering).
set selected_masses [$sel get mass]
puts "__ADMDYN_ATOM_MASSES__ [join $selected_masses {{ }}]"

# Intra-selection bond list, expressed in selection-local atom positions so the
# Python side can make each molecule whole by walking real chemical bonds
# (no bond is ever longer than half the box, so the walk is unambiguous).
set selected_global_indices [$sel get index]
set selected_bondlists [$sel getbonds]
array unset global_to_local
set local_position 0
foreach global_index $selected_global_indices {{
    set global_to_local($global_index) $local_position
    incr local_position
}}
set bond_parts [list]
set local_position 0
foreach neighbor_globals $selected_bondlists {{
    set local_neighbors [list]
    foreach neighbor_global $neighbor_globals {{
        if {{[info exists global_to_local($neighbor_global)]}} {{
            set neighbor_local $global_to_local($neighbor_global)
            if {{$neighbor_local > $local_position}} {{
                lappend local_neighbors $neighbor_local
            }}
        }}
    }}
    lappend bond_parts [join $local_neighbors ","]
    incr local_position
}}
puts "__ADMDYN_ATOM_BONDS__ [join $bond_parts {{;}}]"

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
        raise RuntimeError(f"VMD failed while resolving Step 2 groups for index {file_index}. See {log_path}")

    group_indices = None
    masses = None
    bond_pairs = []
    for line in process.stdout.splitlines():
        if line.startswith("__ADMDYN_GROUP_POSITIONS__"):
            payload = line[len("__ADMDYN_GROUP_POSITIONS__"):].strip()
            group_indices = [
                [int(item) for item in group_payload.split(",") if item]
                for group_payload in payload.split(";")
                if group_payload
            ]
        elif line.startswith("__ADMDYN_ATOM_MASSES__"):
            payload = line[len("__ADMDYN_ATOM_MASSES__"):].strip()
            if payload:
                masses = np.array([float(item) for item in payload.split()], dtype=np.float64)
        elif line.startswith("__ADMDYN_ATOM_BONDS__"):
            payload = line[len("__ADMDYN_ATOM_BONDS__"):]
            # One entry per selected atom (in local order); each entry lists the
            # higher-numbered local positions it is bonded to. Entries are ";"
            # separated and may be empty.
            for local_position, entry in enumerate(payload.strip("\n").split(";")):
                entry = entry.strip()
                if not entry:
                    continue
                for neighbor in entry.split(","):
                    neighbor = neighbor.strip()
                    if neighbor:
                        bond_pairs.append((local_position, int(neighbor)))

    if not group_indices:
        raise ValueError(f"No {grouping_unit} groups were resolved for Step 2 group operations. See {log_path}")

    return {"group_indices": group_indices, "masses": masses, "bond_pairs": bond_pairs}


def _validate_group_indices_match(group_indices, usable_atoms, file_index, operation_label):
    grouped_positions = sorted(position for group in group_indices for position in group)
    if grouped_positions != list(range(usable_atoms)):
        raise ValueError(
            f"Step 2 {operation_label} metadata does not match coordinate width for index {file_index}: "
            f"VMD selection has {len(grouped_positions)} atoms, coordinate file has {usable_atoms} atoms. "
            "Make sure Step 1 target selection and Step 2 input files match."
        )


# ---------------------------------------------------------------------------
# Whole-molecule ("make molecules whole within the cell") support
# ---------------------------------------------------------------------------

def _parse_atoms_per_molecule(spec, total_atoms):
    """Parse an atoms-per-molecule spec into an explicit list of molecule sizes.

    Accepted forms (comma separated, whitespace ignored):
      * a single integer ``"3"``            -> uniform tiling into blocks of 3
      * a repeat form ``"<atoms>x<count>"`` -> ``atoms`` repeated ``count`` times
      * a mixed list ``"3,3,10,4x50"``      -> taken in order

    The parsed sizes must sum to ``total_atoms`` (a lone integer must divide it).
    """
    text = str(spec or "").strip()
    if not text:
        raise ValueError(
            "Whole-molecule mode without a PSF requires an 'atoms per molecule' value."
        )
    sizes = []
    for token in text.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if "x" in token:
            left, right = token.split("x", 1)
            atoms, count = int(left.strip()), int(right.strip())
            if atoms <= 0 or count <= 0:
                raise ValueError(f"Invalid atoms-per-molecule token: {token!r}")
            sizes.extend([atoms] * count)
        else:
            atoms = int(token)
            if atoms <= 0:
                raise ValueError(f"Invalid atoms-per-molecule token: {token!r}")
            sizes.append(atoms)
    if not sizes:
        raise ValueError("atoms-per-molecule spec parsed to nothing.")
    if len(sizes) == 1:
        block = sizes[0]
        if total_atoms % block != 0:
            raise ValueError(
                f"atoms-per-molecule={block} does not divide the {total_atoms} atoms "
                "in the coordinate file."
            )
        return [block] * (total_atoms // block)
    if sum(sizes) != total_atoms:
        raise ValueError(
            f"atoms-per-molecule spec sums to {sum(sizes)} atom(s) but the coordinate "
            f"file has {total_atoms} atom(s)."
        )
    return sizes


def _blocks_to_group_indices(block_sizes):
    """Turn a list of molecule sizes into contiguous column-position groups."""
    groups = []
    start = 0
    for size in block_sizes:
        groups.append(list(range(start, start + size)))
        start += size
    return groups


def _build_bfs_levels(group_indices, bond_pairs, n_atoms):
    """Build a breadth-first spanning forest over intra-molecule bonds.

    Returns ``(levels, fallback_groups)`` where ``levels`` is a list of
    ``(parent_columns, child_columns)`` int arrays ordered by BFS depth (every
    parent is finalized before its children) and ``fallback_groups`` holds any
    molecule whose bond graph does not connect all of its atoms; those are made
    whole by the topology-free centroid method instead.
    """
    from collections import deque, defaultdict

    group_of = np.full(n_atoms, -1, dtype=np.intp)
    for gi, cols in enumerate(group_indices):
        for col in cols:
            group_of[col] = gi

    adjacency = defaultdict(list)
    for a, b in bond_pairs:
        if 0 <= a < n_atoms and 0 <= b < n_atoms and group_of[a] == group_of[b] >= 0:
            adjacency[a].append(b)
            adjacency[b].append(a)

    level_edges = defaultdict(list)
    fallback_groups = []
    for gi, cols in enumerate(group_indices):
        if len(cols) <= 1:
            continue
        root = cols[0]
        seen = {root}
        queue = deque([(root, 0)])
        local_edges = []
        while queue:
            node, depth = queue.popleft()
            for neighbor in adjacency.get(node, ()):  # neighbours are same-group by construction
                if neighbor not in seen:
                    seen.add(neighbor)
                    local_edges.append((depth + 1, node, neighbor))
                    queue.append((neighbor, depth + 1))
        if len(seen) == len(cols):
            for depth, parent, child in local_edges:
                level_edges[depth].append((parent, child))
        else:
            fallback_groups.append(list(cols))

    if not level_edges:
        return None, fallback_groups

    levels = []
    for depth in range(1, max(level_edges) + 1):
        pairs = level_edges.get(depth)
        if not pairs:
            continue
        parents = np.fromiter((p for p, _ in pairs), dtype=np.intp, count=len(pairs))
        children = np.fromiter((c for _, c in pairs), dtype=np.intp, count=len(pairs))
        levels.append((parents, children))
    return levels, fallback_groups


def _resolve_group_metadata(file_index, baseDir, grouping_source, psf_pattern,
                            target_selection, vmd_path, grouping_unit,
                            atoms_per_molecule, usable_atoms, common_term=""):
    """Resolve per-molecule column groups (+ masses + bond spanning forest).

    ``grouping_source`` is ``"psf"`` (VMD resolves residues/chains/segments,
    masses and bonds) or ``"blocks"`` (fixed atoms-per-molecule, no topology).
    """
    source = str(grouping_source or "psf").strip().lower()
    if source in {"blocks", "block", "fixed", "no_psf", "atoms_per_molecule"}:
        block_sizes = _parse_atoms_per_molecule(atoms_per_molecule, usable_atoms)
        group_indices = _blocks_to_group_indices(block_sizes)
        return {
            "group_indices": group_indices,
            "levels": None,
            "fallback_groups": group_indices,
            "masses": None,
            "source": "blocks",
        }

    metadata = _resolve_group_metadata_via_vmd(
        file_index, baseDir, psf_pattern, target_selection, vmd_path, grouping_unit, common_term
    )
    group_indices = metadata["group_indices"]
    levels, fallback_groups = _build_bfs_levels(
        group_indices, metadata.get("bond_pairs") or [], usable_atoms
    )
    if fallback_groups:
        warnings.warn(
            f"{len(fallback_groups)} molecule(s) have an incomplete bond graph in the PSF "
            "selection; those are made whole by the topology-free centroid method."
        )
    return {
        "group_indices": group_indices,
        "levels": levels,
        "fallback_groups": fallback_groups,
        "masses": metadata.get("masses"),
        "source": "psf",
    }


def _box_broadcast(box, n_frames):
    """Return box as a float64 array broadcastable to (n_frames, n_atoms, 3)."""
    box = np.asarray(box, dtype=np.float64)
    if box.ndim == 1:
        return box.reshape(1, 1, 3)
    if box.ndim == 2:
        return box.reshape(box.shape[0], 1, 3)
    raise ValueError(f"Unexpected box shape {box.shape}")


def _make_groups_whole_bonded(coords, box, levels):
    """Make molecules whole by walking bonds, one BFS level at a time (in place)."""
    box_b = _box_broadcast(box, coords.shape[0])
    for parents, children in levels:
        anchor = coords[:, parents, :]
        delta = coords[:, children, :] - anchor
        delta -= box_b * np.round(delta / box_b)
        coords[:, children, :] = anchor + delta
    return coords


def _make_groups_whole_centroid(coords, box, group_col_lists):
    """Make molecules whole without topology (in place).

    Each atom is imaged to the running centroid of the already-placed atoms of
    the same molecule. Exact when consecutive selection atoms stay within half a
    box of the molecule centroid (true for chain-ordered PSFs and any molecule
    smaller than L/2 -- water, ions, small solvents).
    """
    from collections import defaultdict

    box_b = _box_broadcast(box, coords.shape[0])[:, 0, :]  # (nframes_or_1, 3)
    by_size = defaultdict(list)
    for cols in group_col_lists:
        if len(cols) > 1:
            by_size[len(cols)].append(cols)
    for size, cols_list in by_size.items():
        cols = np.asarray(cols_list, dtype=np.intp)          # (n_groups, size)
        sub = coords[:, cols, :]                              # (nframes, n_groups, size, 3)
        box_g = box_b[:, None, :]                             # (nframes_or_1, 1, 3)
        running_sum = sub[:, :, 0, :].astype(np.float64).copy()
        for j in range(1, size):
            centroid = running_sum / j
            delta = sub[:, :, j, :] - centroid
            delta -= box_g * np.round(delta / box_g)
            sub[:, :, j, :] = centroid + delta
            running_sum += sub[:, :, j, :]
        coords[:, cols, :] = sub
    return coords


def _make_groups_whole(coords, box, metadata):
    """Dispatch to the bonded and/or centroid whole-molecule builders (in place)."""
    if metadata.get("levels"):
        _make_groups_whole_bonded(coords, box, metadata["levels"])
    if metadata.get("fallback_groups"):
        _make_groups_whole_centroid(coords, box, metadata["fallback_groups"])
    return coords


def _wrap_whole_groups(coords, box, group_indices, masses=None, center_mode="cell"):
    """Translate each (already whole) molecule so its centre lands in the cell.

    ``center_mode`` ``"cell"`` places every molecule centre in ``[0, L)``;
    ``"extracted selection com"`` first shifts each frame so the whole selection
    centre of mass sits at the box centre (useful for keeping a solute framed).
    Centres are mass-weighted when masses are available, else geometric.
    """
    from collections import defaultdict

    box = np.asarray(box, dtype=np.float64)
    per_frame_box = box.ndim == 2
    n_frames = coords.shape[0]
    frame_box = box if per_frame_box else np.broadcast_to(box, (n_frames, 3))

    if str(center_mode or "cell").strip().lower() in {
        "extracted selection com", "selection com", "selection_com",
    }:
        if masses is not None:
            weights = masses / masses.sum()
            selection_centre = np.einsum("fnc,n->fc", coords.astype(np.float64), weights)
        else:
            selection_centre = coords.mean(axis=1)
        coords += (0.5 * frame_box - selection_centre)[:, None, :]

    box_g = _box_broadcast(box, n_frames)[:, 0, :][:, None, :]  # (nframes_or_1, 1, 3)
    by_size = defaultdict(list)
    for cols in group_indices:
        by_size[len(cols)].append(cols)
    for size, cols_list in by_size.items():
        cols = np.asarray(cols_list, dtype=np.intp)           # (n_groups, size)
        sub = coords[:, cols, :]                              # (nframes, n_groups, size, 3)
        if masses is not None and size > 1:
            group_weights = masses[cols]                      # (n_groups, size)
            group_weights = group_weights / group_weights.sum(axis=1, keepdims=True)
            centre = np.einsum("fgsc,gs->fgc", sub.astype(np.float64), group_weights)
        else:
            centre = sub.mean(axis=2)                         # (nframes, n_groups, 3)
        shift = box_g * np.floor(centre / box_g)              # (nframes, n_groups, 3)
        coords[:, cols, :] = sub - shift[:, :, None, :]
    return coords


def _group_extent_violations(coords, box, group_indices, frac=0.5):
    """Count (frame, molecule) pairs whose atom spread exceeds ``frac`` * box."""
    from collections import defaultdict

    box = np.asarray(box, dtype=np.float64)
    per_frame_box = box.ndim == 2
    box_min = box.min(axis=1) if per_frame_box else np.full(coords.shape[0], float(box.min()))
    threshold = box_min[:, None] * frac

    violations = 0
    worst = 0.0
    by_size = defaultdict(list)
    for cols in group_indices:
        if len(cols) > 1:
            by_size[len(cols)].append(cols)
    for size, cols_list in by_size.items():
        cols = np.asarray(cols_list, dtype=np.intp)
        sub = coords[:, cols, :]
        extent = sub.max(axis=2) - sub.min(axis=2)            # (nframes, n_groups, 3)
        max_extent = extent.max(axis=2)                       # (nframes, n_groups)
        mask = max_extent > threshold
        if mask.any():
            violations += int(mask.sum())
            worst = max(worst, float(max_extent[mask].max()))
    return violations, worst


def _flag_large_steps(disp, box, frac=0.5):
    """Return (frame_indices, atom_count) for MIC steps larger than ``frac`` * box.

    A corrected inter-frame displacement above half the box means the raw jump
    exceeded the minimum-image assumption: a dropped/duplicated/out-of-order
    frame, a concatenation seam, a wrong box, or a save interval too coarse for
    the fastest atoms. The unwrap silently mis-rounds such steps, so surface them.
    """
    box = np.asarray(box, dtype=np.float64)
    if box.ndim == 2:
        threshold = box[1:].min(axis=1).reshape(-1, 1, 1) * frac
    else:
        threshold = float(box.min()) * frac
    over = np.abs(disp) > threshold
    if not over.any():
        return None
    frame_hits = np.nonzero(np.any(over, axis=(1, 2)))[0] + 1  # step k joins frames k-1, k
    atom_count = int(np.count_nonzero(np.any(over, axis=2)))
    return frame_hits, atom_count


def _unwrap_single_file(file_index, baseDir, input_pattern, output_pattern, xsc_pattern,
                       box_size, chunk_size, common_term, input_io_spec=None, output_io_spec=None,
                       psf_pattern=None, target_selection=None, vmd=None, grouping_unit="residue",
                       repair_first_frame=False, wrap_groups_into_cell=False, wrap_center="cell",
                       ensemble="nvt", mode="continuous", grouping_source="psf",
                       atoms_per_molecule=None):
    """Process a single coordinate file.

    ``mode="continuous"`` makes per-atom trajectories continuous across periodic
    images (for MSD), optionally repairing molecules split in the first frame.
    ``mode="whole_in_cell"`` instead makes every molecule whole in every frame
    and wraps each whole molecule back into the primary cell (for structure,
    density, dipoles); every frame is treated independently.
    """

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

        box_trajectory = None
        if ensemble == "npt":
            xsc_file_rel = expand_path_pattern(xsc_pattern, common_term, file_index)
            xsc_full_path = os.path.join(baseDir, xsc_file_rel)
            if not os.path.exists(xsc_full_path):
                raise FileNotFoundError(
                    f"XSC file not found for constant-pressure (NPT) unwrapping of index {file_index}: {xsc_full_path}"
                )
            box_trajectory = _read_xsc_box_trajectory(xsc_full_path)
            if box_trajectory.shape[0] != n_frames:
                raise ValueError(
                    f"NPT ensemble: XSC file {xsc_full_path} has {box_trajectory.shape[0]} box-dimension "
                    f"row(s) but the coordinate file for index {file_index} has {n_frames} frame(s). "
                    "For Constant Pressure, the XSC file must contain one box row per coordinate frame."
                )
            frame_box = box_trajectory
        else:
            frame_box = box_size

        make_whole_mode = str(mode or "continuous").strip().lower() == "whole_in_cell"

        group_metadata = None
        if make_whole_mode or repair_first_frame:
            group_metadata = _resolve_group_metadata(
                file_index, baseDir, grouping_source, psf_pattern, target_selection,
                vmd, grouping_unit, atoms_per_molecule, usable_atoms, common_term,
            )
            _validate_group_indices_match(
                group_metadata["group_indices"],
                usable_atoms,
                file_index,
                "whole-molecule wrapping" if make_whole_mode else "first-frame repair",
            )

        if make_whole_mode:
            # Every frame is independent: chunk over frames to bound memory and
            # rewrite each chunk in place (peak RAM ~= one copy of the file).
            group_indices = group_metadata["group_indices"]
            masses = group_metadata.get("masses")
            step = int(chunk_size) if chunk_size else 2000
            for start in range(0, n_frames, step):
                stop = min(start + step, n_frames)
                block = coords[start:stop]
                block_box = box_trajectory[start:stop] if box_trajectory is not None else box_size
                _make_groups_whole(block, block_box, group_metadata)
                _wrap_whole_groups(block, block_box, group_indices, masses, wrap_center)

            if not np.all(np.isfinite(coords)):
                raise ValueError(f"Non-finite coordinates after whole-molecule wrapping for index {file_index}.")
            violations, worst = _group_extent_violations(
                coords, box_trajectory if box_trajectory is not None else box_size, group_indices
            )
            if violations:
                print(
                    f"⚠ {violations} molecule-frame(s) still span > 0.5·box (worst {worst:.1f} Å): "
                    "a molecule may genuinely be larger than half the box, or the grouping/atom "
                    "order is wrong for this selection."
                )
            _save_coordinate_array(output_file, coords.reshape(n_frames, -1), io_spec=output_io_spec)
            print(
                f"✓ Made {len(group_indices)} molecule(s) whole in {n_frames} frame(s) "
                f"and wrapped into the cell: {input_file_rel} -> {output_file_rel}"
            )
            return

        if repair_first_frame:
            frame0_box = box_trajectory[0] if box_trajectory is not None else box_size
            first_frame = coords[0:1].copy()
            _make_groups_whole(first_frame, frame0_box, group_metadata)
            # Centre each repaired molecule in the primary cell. Every atom shift
            # here is an exact integer number of box vectors, so adding it to all
            # frames leaves every inter-frame displacement (and the unwrap) intact.
            _wrap_whole_groups(
                first_frame, frame0_box, group_metadata["group_indices"],
                group_metadata.get("masses"), "cell",
            )
            atom_shifts = first_frame[0] - coords[0]
            shifted_atoms = int(np.count_nonzero(np.any(np.abs(atom_shifts) > 1e-6, axis=1)))
            coords += atom_shifts.reshape(1, usable_atoms, 3)
            coords[0] = first_frame[0]
            print(
                f"✓ Fixed first frame using {len(group_metadata['group_indices'])} molecule(s); "
                f"applied shifts to {shifted_atoms} atom(s) across all frames"
            )

        # Vectorized unwrapping algorithm - much more efficient than loops
        if n_frames == 1:
            # Single frame case - first-frame group repair may still have been applied.
            unwrapped = coords
        else:
            # Displacements between consecutive frames: shape (T-1, N, 3)
            disp = np.diff(coords, axis=0)

            # Apply minimum image convention in one vectorized step:
            # disp_MI = disp - box * round(disp / box)
            # For NPT, each displacement step uses the box of the frame it arrives at
            # (box_trajectory[1:]); for NVT the box is constant across all steps.
            step_box = box_trajectory[1:].reshape(-1, 1, 3) if box_trajectory is not None else box_size
            disp -= step_box * np.round(disp / step_box)

            # Diagnostic: a corrected step above half the box means the raw jump
            # broke the minimum-image assumption and this frame was mis-unwrapped.
            large_steps = _flag_large_steps(disp, frame_box)
            if large_steps is not None:
                frame_hits, atom_count = large_steps
                preview = ", ".join(str(int(f)) for f in frame_hits[:10])
                more = "" if len(frame_hits) <= 10 else f" (+{len(frame_hits) - 10} more)"
                print(
                    f"⚠ index {file_index}: {atom_count} atom-step(s) across {len(frame_hits)} frame(s) "
                    f"exceed 0.5·box after minimum-image correction [frames {preview}{more}]. "
                    "Likely a dropped/duplicated/out-of-order frame, a concatenation seam, a wrong "
                    "box, or a save interval too coarse for the fastest atoms; the unwrap of those "
                    "frames is unreliable."
                )

            # Now integrate displacements to get unwrapped positions
            unwrapped = np.empty_like(coords)
            unwrapped[0] = coords[0]                               # reference frame
            unwrapped[1:] = coords[0] + np.cumsum(disp, axis=0)    # cumulative sum

        if group_metadata is not None:
            violations, worst = _group_extent_violations(
                unwrapped, frame_box, group_metadata["group_indices"]
            )
            if violations:
                print(
                    f"⚠ index {file_index}: {violations} molecule-frame(s) span > 0.5·box after "
                    f"unwrapping (worst {worst:.1f} Å) -- some molecules were split by a mis-rounded "
                    "step. Inspect the flagged frames or re-run with a finer save interval / correct box."
                )

        # Flatten back to (n_frames, n_cols)
        unwrapped_flat = unwrapped.reshape(n_frames, -1)
        _save_coordinate_array(output_file, unwrapped_flat, io_spec=output_io_spec)

        print(f"✓ Completed {input_file_rel} -> {output_file_rel}")
        
    except Exception as e:
        print(f"✗ Failed {input_file_rel}: {e}")
        raise
