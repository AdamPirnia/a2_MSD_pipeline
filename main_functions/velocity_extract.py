import os
import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from difflib import get_close_matches
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


def _normalize_velocity_output_io_spec(output_io_spec):
    spec = dict(output_io_spec or {})
    mode = str(spec.get("mode", "text")).strip().lower()
    precision = str(spec.get("precision", "single")).strip().lower()
    decimals = spec.get("decimals")
    if mode not in {"text", "binary"}:
        mode = "text"
    if precision not in {"single", "double", "custom"}:
        precision = "single"
    if precision == "custom":
        try:
            decimals = int(decimals)
        except Exception:
            decimals = 6
    else:
        decimals = None
    return {"mode": mode, "precision": precision, "decimals": decimals}


def _velocity_temp_text_output_path(output_file):
    return f"{output_file}.vmdtmp"

def extract_velocities(baseDir, psf_pattern, veldcd_pattern, output_pattern=None, num_dcd=None, target_selection=None, grouping_unit="residue", vmd=None, max_workers=None, dcd_indices=None, common_term="", output_io_spec=None, stride=1):
    """
    Extracts center-of-mass velocities from a series of VELDCD trajectories using VMD in parallel,
    by generating per-segment Tcl scripts, running them in batch, and saving
    the results as text files.

    For each trajectory chunk, this function:
      1. Creates output directories based on output pattern.
      2. Writes a VMD Tcl script that:
         - Loads the PSF and VELDCD for that chunk using the expanded patterns.
         - Iterates over every frame.
         - Calculates center-of-mass velocities for each molecule/residue.
         - Writes results using the output pattern.
      3. Invokes VMD in text mode (`-dispdev text`) in parallel processes.
      4. Captures stdout/stderr to log files in the same directory as output files.

    Parameters
    ----------
    baseDir : str
        Root directory for the simulation.
    psf_pattern : str
        Path pattern for PSF files. Can contain * (common term) and {i} (file index).
        Example: "trajectories_*/run_{i}/system.psf"
    veldcd_pattern : str
        Path pattern for VELDCD files. Can contain * (common term) and {i} (file index).
        Example: "trajectories_*/run_{i}/traj.veldcd"
    output_pattern : str
        Path pattern for output files. Can contain * (common term) and {i} (file index).
        Example: "path/to/velCOM_{i}.dat"
    num_dcd : int
        Number of trajectory chunks to process. Generates scripts for
        indices `0` through `num_dcd-1`.
    target_selection : str
        VMD atom selection used to choose the atoms included in COM velocity extraction.
    grouping_unit : str
        Grouping unit used to build COM velocity particles. One of residue, chain, segname.
    vmd : str
        The path to the VMD executable.
    max_workers : int, optional
        Maximum number of parallel workers. Defaults to min(num_dcd, CPU count).
    dcd_indices : list, optional
        List of DCD indices to process (e.g., [0, 1, 4, 5] to process only DCDs 0, 1, 4, 5).
        If None, processes all DCDs from 0 to num_dcd-1. Default is None.
    common_term : str, optional
        Value to replace * placeholders in patterns. Default is "".
    
    Side Effects
    ------------
    - Creates directories:
        - `{baseDir}/{output_pattern}`
        - `writenCodes`
        - Log files in output directory
    - Writes Tcl scripts to `writenCodes/vel_{i}.tcl`.
    - Runs VMD on each script in parallel, logging to the same directory as output files.
    - Produces velocity files `velCOM_{i}.dat` in `{baseDir}/{output_pattern}`.

    Returns
    -------
    dict
        Summary of processing results with success/failure counts and timing.

    Example
    -------
    >>> results = extract_velocities(
    ...     baseDir="/home/user/sim",
    ...     psf_pattern="trajectories_*/run_{i}/system.psf",
    ...     veldcd_pattern="trajectories_*/run_{i}/traj.veldcd",
    ...     output_pattern="path/to/velCOM_{i}.dat",
    ...     num_dcd=6,
    ...     target_selection="water",
    ...     grouping_unit="residue",
    ...     vmd="/usr/local/bin/vmd",
    ...     common_term="260"
    ... )
    """
    
    start_time = time.time()
    
    # Validate path patterns
    is_valid, error_msg = validate_path_pattern(psf_pattern)
    if not is_valid:
        raise ValueError(f"Invalid PSF pattern: {error_msg}")
        
    is_valid, error_msg = validate_path_pattern(veldcd_pattern)
    if not is_valid:
        raise ValueError(f"Invalid VELDCD pattern: {error_msg}")
        
    is_valid, error_msg = validate_path_pattern(output_pattern)
    if not is_valid:
        raise ValueError(f"Invalid output pattern: {error_msg}")
    
    print(f"{'='*50}")
    print(f"VELOCITY EXTRACTION")
    print(f"{'='*50}")
    print(f"Base directory: {baseDir}")
    print(f"PSF pattern: {psf_pattern}")
    print(f"VELDCD pattern: {veldcd_pattern}")
    print(f"Output pattern: {output_pattern}")
    print(f"Common term: {common_term}")
    print(f"Number of DCDs: {num_dcd}")
    print(f"Target selection: {target_selection}")
    print(f"Grouping unit: {grouping_unit}")
    print(f"VMD executable: {vmd}")
    print(f"Stride: {stride}")

    if grouping_unit not in {"residue", "chain", "segname"}:
        raise ValueError("grouping_unit must be one of: residue, chain, segname")
    if not str(target_selection or "").strip():
        raise ValueError("target_selection is required for velocity extraction")

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
    # Create logs directory in the same location as output files
    if dcd_list:
        logs_dir = output_dir
        os.makedirs(logs_dir, exist_ok=True)
    
    # Validate that input files exist for first DCD (sanity check)
    if dcd_list:
        test_psf = os.path.join(baseDir, expand_path_pattern(psf_pattern, common_term, dcd_list[0]))
        test_veldcd = os.path.join(baseDir, expand_path_pattern(veldcd_pattern, common_term, dcd_list[0]))
        
        if not os.path.exists(test_psf):
            raise FileNotFoundError(_format_missing_file_message("PSF", test_psf))
        if not os.path.exists(test_veldcd):
            raise FileNotFoundError(_format_missing_file_message("VELDCD", test_veldcd))
        
        print(f"✓ Input files validated for index {dcd_list[0]}")
    
    # Validate that all required input files exist
    missing_files = []
    for i in dcd_list:  # Check for existence of all files
        test_psf = os.path.join(baseDir, expand_path_pattern(psf_pattern, common_term, i))
        test_veldcd = os.path.join(baseDir, expand_path_pattern(veldcd_pattern, common_term, i))
        
        if not os.path.exists(test_psf):
            missing_files.append(f"PSF file missing for index {i}: {test_psf}")
        if not os.path.exists(test_veldcd):
            missing_files.append(f"VELDCD file missing for index {i}: {test_veldcd}")
    
    if missing_files:
        print("WARNING: Some input files are missing:")
        for missing in missing_files:
            print(f"  - {missing}")
    
    # Results tracking
    results = {
        'successful': [],
        'failed': [],
        'total_time': 0,
        'num_processed': len(dcd_list),
        'missing_files': missing_files
    }
    
    # Generate all TCL scripts first with pattern validation
    for i in dcd_list:
        success = _write_velocity_tcl_script(
            i,
            baseDir,
            psf_pattern,
            veldcd_pattern,
            output_pattern,
            target_selection,
            grouping_unit,
            common_term,
            stride,
            output_io_spec=output_io_spec,
        )
        if not success:
            print(f"ERROR: Failed to generate TCL script for chunk {i} due to pattern validation failure.")
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
            executor.submit(_run_velocity_vmd_script, i, vmd, common_term, output_dir, baseDir, output_pattern, output_io_spec): i 
            for i in dcd_list
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                success, stdout, stderr = future.result()
                
                # Additional verification: check if output file was actually created
                if success:
                    # Verify output file exists and has content
                    expected_output = os.path.join(baseDir, expand_path_pattern(output_pattern, common_term, index))
                    if os.path.exists(expected_output) and os.path.getsize(expected_output) > 0:
                        results['successful'].append(index)
                        print(f"✓ Completed velocity extraction for chunk {index}")
                        print(f"  Output file: {expected_output} ({os.path.getsize(expected_output)} bytes)")
                    else:
                        success = False  # Override VMD success if no output file
                        if os.path.exists(expected_output):
                            error_msg = f"Output file exists but is empty: {expected_output}"
                        else:
                            error_msg = f"Output file was not created: {expected_output}"
                        results['failed'].append(index)
                        print(f"✗ Failed velocity extraction for chunk {index}")
                        print(f"  Error: {error_msg}")
                        # Check VMD log for more details
                        log_filename = f"vel_log_{common_term}_{index}.log" if common_term else f"vel_log_{index}.log"
                        log_path = os.path.join(output_dir, log_filename)
                        if os.path.exists(log_path):
                            print(f"  Check VMD log: {log_path}")
                
                elif not success:
                    # Only add to failed if not already added above
                    if index not in results['failed']:
                        results['failed'].append(index)
                        print(f"✗ Failed velocity extraction for chunk {index}")
                    if stderr:
                        print(f"  Error: {stderr}")
                        
            except Exception as exc:
                results['failed'].append(index)
                print(f"✗ Exception in velocity extraction chunk {index}: {exc}")
    
    results['total_time'] = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"VELOCITY EXTRACTION SUMMARY")
    print(f"{'='*50}")
    print(f"Total chunks: {len(dcd_list)}")
    print(f"Successful: {len(results['successful'])}")
    print(f"Failed: {len(results['failed'])}")
    if results['failed']:
        print(f"Failed indices: {results['failed']}")
    print(f"Total time: {results['total_time']:.2f} seconds")
    print(f"Parallel efficiency: {max_workers} workers")
    print(f"{'='*50}\n")
    
    return results


def _write_velocity_tcl_script(i, baseDir, psf_pattern, veldcd_pattern, output_pattern, target_selection, grouping_unit, common_term="", stride=1, output_io_spec=None):
    """Write optimized TCL script for velocity extraction from a single trajectory chunk."""
    
    # Expand patterns to get actual file paths
    psf_path = expand_path_pattern(psf_pattern, common_term, i)
    veldcd_path = expand_path_pattern(veldcd_pattern, common_term, i)
    output_path = expand_path_pattern(output_pattern, common_term, i)
    
    # Validate patterns to ensure they match user input expectations
    def validate_pattern_consistency(pattern_name, pattern, expanded_path, common_term, index):
        """Validate that the pattern is clean and expansion matches expectation"""
        if not pattern or not common_term:
            return True  # Skip validation if no pattern or common term
        
        # First, check if the user's input pattern itself contains corrupted terms
        # Look for patterns that suggest mixing of old common terms with placeholders
        import re
        suspicious_pattern = re.search(r'output_NVE_(\d+)_[^*]*\*', pattern)
        if suspicious_pattern:
            old_term = suspicious_pattern.group(1)
            print(f"ERROR: {pattern_name} pattern contains corrupted data!")
            print(f"  User input pattern: {pattern}")
            print(f"  Detected old common term: {old_term}")
            print(f"  The pattern appears to mix hardcoded terms with placeholders.")
            print(f"  Expected clean pattern: Replace 'output_NVE_{old_term}_..._*' with 'output_NVE_*'")
            print(f"  Please fix your GUI input field to use only * and {{i}} placeholders.")
            return False
        
        # Check for other corrupted patterns
        if re.search(r'_newDipole_', pattern):
            print(f"ERROR: {pattern_name} pattern contains '_newDipole_' which suggests corruption!")
            print(f"  User input pattern: {pattern}")
            print(f"  This appears to be from an old dipole calculation pattern.")
            print(f"  Please clean your pattern to remove any hardcoded terms.")
            return False
        
        # Second, validate that the expansion matches what we expect from clean input
        expected_pattern = pattern.replace('*', common_term)
        # Replace {i} with the actual index
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
            print(f"  This indicates an unexpected issue with pattern expansion.")
            return False
        
        return True
    
    # Validate PSF and VELDCD patterns
    psf_valid = validate_pattern_consistency("PSF", psf_pattern, psf_path, common_term, i)
    veldcd_valid = validate_pattern_consistency("VELDCD", veldcd_pattern, veldcd_path, common_term, i)
    
    if not psf_valid or not veldcd_valid:
        print(f"ERROR: Pattern validation failed for chunk {i}")
        print(f"Please fix the corrupted patterns in your GUI input fields before proceeding.")
        return False
    
    # Create absolute paths
    psf_full_path = os.path.join(baseDir, psf_path)
    veldcd_full_path = os.path.join(baseDir, veldcd_path)
    output_full_path = os.path.join(baseDir, output_path)
    normalized_output = _normalize_velocity_output_io_spec(output_io_spec)
    vmd_output_path = output_full_path
    if not (normalized_output["mode"] == "text" and normalized_output["precision"] == "single"):
        vmd_output_path = _velocity_temp_text_output_path(output_full_path)
    
    print(f"DEBUG Velocity TCL Script {i}:")
    print(f"  Output pattern: {output_pattern}")
    print(f"  Expanded output: {output_path}")
    print(f"  Full output path: {output_full_path}")
    if vmd_output_path != output_full_path:
        print(f"  Temporary VMD text output: {vmd_output_path}")
    print(f"  Output directory: {os.path.dirname(output_full_path)}")
    
    # Ensure output directory exists before creating TCL script
    output_dir = os.path.dirname(output_full_path)
    if output_dir and output_dir != baseDir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create more specific TCL filename with common term
    tcl_filename = f"vel_{common_term}_{i}.tcl" if common_term else f"vel_{i}.tcl"
    
    with open(f"writenCodes/{tcl_filename}", "w") as f:
        f.write(f"""# Velocity extraction script for chunk {i}
# Generated by velocity_extract.py

puts "Starting velocity extraction for chunk {i}"
puts "Timestamp: [clock format [clock seconds]]"

# Set paths
set baseDir "{_tcl_quote(baseDir)}"

# Open output file
set outfile [open "{_tcl_quote(vmd_output_path)}" w]

# Load trajectory
set PSF "{_tcl_quote(psf_full_path)}"
set VELDCD "{_tcl_quote(veldcd_full_path)}"

puts "Loading PSF: $PSF"
puts "Loading VELDCD: $VELDCD"

# Debug output - check file existence
puts "About to load: PSF=$PSF, VELDCD=$VELDCD"
if {{![file exists $PSF]}} {{
    puts "ERROR: PSF file does not exist: $PSF"
    exit 1
}}
if {{![file exists $VELDCD]}} {{
    puts "ERROR: VELDCD file does not exist: $VELDCD"
    exit 1
}}

set traj [mol load psf $PSF dcd $VELDCD]
puts "Velocity trajectory loaded with ID: $traj"

# Validate that trajectory was loaded successfully
if {{$traj == -1}} {{
    puts "ERROR: Failed to load trajectory"
    close $outfile
    exit 1
}}

# Get number of frames and COM groups
set nf [molinfo $traj get numframes]
set stride {stride}
puts "Total frames: $nf"
puts "Stride: $stride"

set target_selection "{_tcl_quote(target_selection)}"
set grouping_unit "{_tcl_quote(grouping_unit)}"
set base_sel [atomselect top $target_selection]
set selected_atoms [$base_sel num]
puts "Target selection: $target_selection"
puts "Grouping unit: $grouping_unit"
puts "Selected atoms: $selected_atoms"

if {{$selected_atoms == 0}} {{
    puts "ERROR: No atoms selected with target selection $target_selection"
    $base_sel delete
    close $outfile
    mol delete $traj
    exit 1
}}

set atom_indices [$base_sel get index]
if {{$grouping_unit eq "residue"}} {{
    set atom_group_segids [$base_sel get segid]
    set atom_group_resids [$base_sel get resid]
    set atom_groups [list]
    foreach group_segid $atom_group_segids group_resid $atom_group_resids {{
        lappend atom_groups [list $group_segid $group_resid]
    }}
}} else {{
    set atom_group_values [$base_sel get $grouping_unit]
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
    set group_sel [atomselect top "index [join $group_indices {{ }}]"]
    lappend group_sels $group_sel
}}
set num_groups [llength $group_sels]
puts "Resolved $num_groups COM group(s)"

if {{$num_groups <= 0}} {{
    puts "ERROR: No COM groups were resolved from the selected atoms"
    $base_sel delete
    close $outfile
    mol delete $traj
    exit 1
}}

$base_sel delete

# Validate frame count
if {{$nf <= 0}} {{
    puts "ERROR: No frames found in trajectory"
    close $outfile
    mol delete $traj
    exit 1
}}

# Extract center-of-mass velocities for all frames
puts "Extracting center-of-mass velocities..."
set frame_count 0
for {{set frame 0}} {{$frame < $nf}} {{incr frame $stride}} {{
    animate goto $frame
    set frameData ""
    
    foreach sel $group_sels {{
        $sel frame $frame
        if {{[$sel num] == 0}} {{
            puts "ERROR: Group selection became empty at frame $frame"
            close $outfile
            mol delete $traj
            exit 1
        }}
        set velcom [measure center $sel weight mass]
        append frameData [format " %.6f %.6f %.6f " [lindex $velcom 0] [lindex $velcom 1] [lindex $velcom 2]]
    }}
    
    puts $outfile $frameData
    incr frame_count
    
    if {{$frame_count % 100 == 0}} {{
        puts "Processed $frame_count frames..."
    }}
}}

close $outfile
foreach group_sel $group_sels {{
    $group_sel delete
}}
mol delete $traj

puts "Velocity extraction complete: $frame_count frames processed"
puts "Output written to: {_tcl_quote(vmd_output_path)}"

# Verify output file was created and has content
if {{[file exists "{_tcl_quote(vmd_output_path)}"]}} {{
    set filesize [file size "{_tcl_quote(vmd_output_path)}"]
    puts "✓ Output file verified: {_tcl_quote(vmd_output_path)} ($filesize bytes)"
}} else {{
    puts "✗ ERROR: Output file was not created: {_tcl_quote(vmd_output_path)}"
    exit 1
}}

puts "Timestamp: [clock format [clock seconds]]"

exit 0
""")
    
    return True


def _run_velocity_vmd_script(index, vmd_path, common_term="", output_dir="logs", baseDir="", output_pattern=None, output_io_spec=None):
    """Run a single VMD velocity extraction script and return results."""
    
    # Use the same naming convention as in _write_velocity_tcl_script
    tcl_filename = f"vel_{common_term}_{index}.tcl" if common_term else f"vel_{index}.tcl"
    script_path = f"writenCodes/{tcl_filename}"
    # Create log file in the same directory as output files
    log_filename = f"vel_log_{common_term}_{index}.log" if common_term else f"vel_log_{index}.log"
    log_path = os.path.join(output_dir, log_filename)
    
    # Build VMD command - ensure proper quoting for paths with spaces
    command = [vmd_path, "-dispdev", "text", "-e", script_path]
    
    print(f"Command: {' '.join(command)}")
    print(f"Script path: {script_path}")
    print(f"Log path: {log_path}")
    
    try:
        # Run VMD process
        process = subprocess.run(
            command, 
            shell=False,
            capture_output=True, 
            text=True,
            timeout=None  # No timeout for large trajectories
        )
        
        # Save log file
        with open(log_path, 'w') as log_file:
            log_file.write(f"Command: {' '.join(command)}\n")
            log_file.write(f"Return code: {process.returncode}\n")
            log_file.write(f"STDOUT:\n{process.stdout}\n")
            log_file.write(f"STDERR:\n{process.stderr}\n")
        
        success = process.returncode == 0
        if success and output_pattern and output_io_spec:
            output_file = os.path.join(baseDir, expand_path_pattern(output_pattern, common_term, index))
            normalized_output = _normalize_velocity_output_io_spec(output_io_spec)
            if not (normalized_output["mode"] == "text" and normalized_output["precision"] == "single"):
                input_file = _velocity_temp_text_output_path(output_file)
                data = load_numeric_array(
                    input_file,
                    {"mode": "text", "precision": "single"},
                    default_mode="text",
                    default_precision="single",
                )
                save_numeric_array(
                    output_file,
                    data,
                    output_io_spec,
                    default_mode="text",
                    default_precision="single",
                )
                if os.path.exists(input_file):
                    os.remove(input_file)
        return success, process.stdout, process.stderr
        
    except subprocess.TimeoutExpired:
        error_msg = f"VMD process for velocity chunk {index} timed out after 1 hour"
        with open(log_path, 'w') as log_file:
            log_file.write(f"ERROR: {error_msg}\n")
        return False, "", error_msg
        
    except Exception as e:
        error_msg = f"Exception running VMD for velocity chunk {index}: {str(e)}"
        with open(log_path, 'w') as log_file:
            log_file.write(f"ERROR: {error_msg}\n")
        return False, "", error_msg
