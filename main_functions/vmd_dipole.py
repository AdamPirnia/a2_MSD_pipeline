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

def vmd_dipole_collective(baseDir, INdir, OUTdir, psf, dcd, num_dcd, target, vmd, max_workers=None, dcd_indices=None, neutral=False, output_pattern=None, output_io_spec=None):
    """
    Calculates collective dipole moments from a series of DCD trajectories using VMD in parallel.
    
    This function computes the collective dipole moment of selected atoms using VMD's 
    measure dipole command, which is useful for analyzing overall molecular orientations
    and dipole fluctuations in MD simulations.

    For each trajectory chunk (0→1 ns, 1→2 ns, …), this function:
      1. Creates output directories based on output pattern or OUTdir.
      2. Writes a VMD Tcl script that:
         - Loads the PSF and DCD for that chunk.
         - Iterates over every frame.
         - Selects atoms according to the target selection.
         - Calculates collective dipole moment using VMD's measure dipole.
         - Writes results using output pattern or to `OUTdir/dipole_{i}.dat`.
      3. Invokes VMD in text mode (`-dispdev text`) in parallel processes.
      4. Captures stdout/stderr to `logs/log_{i}.log`.

    Parameters
    ----------
    baseDir : str
        Root directory containing the `INdir` subfolders named
        "0to1ns", "1to2ns", … up to `num_dcd-1`to`num_dcd`ns.
    INdir : str
        Name of the subfolder under `baseDir` where all trajectory chunks live.
    OUTdir : str
        Name of the output subdirectory under `baseDir` in which results will be 
        created as the `.dat` files. OUTdir will be created if it does not exist.
        (Used when output_pattern is None for backward compatibility)
    psf : str
        Base filename (+path between OUTdir and the file without extension) of the 
        topology file in each chunk (e.g. `"system"` if your files are `system.psf`).
    dcd : str
        Base filename (+path between OUTdir and the file without extension) of the 
        trajectory file in each chunk (e.g. `"traj"` if your files are `traj.dcd`).
    output_pattern : str, optional
        Path pattern for output files. Can contain * (common term) and {i} (file index).
        Example: "path/to/dipole_{i}.dat". If None, uses OUTdir for backward compatibility.
    num_dcd : int
        Number of trajectory chunks to process.  Generates scripts for
        indices `0` through `num_dcd-1`.
    target : str
        VMD atom selection string (e.g., "water", "resname WAT", "protein", 
        "segname PROT and backbone", "index 0 to 999").
    vmd : str
        The path to the VMD executable.
    max_workers : int, optional
        Maximum number of parallel workers. Defaults to min(num_dcd, CPU count).
    dcd_indices : list, optional
        List of DCD indices to process (e.g., [0, 1, 4, 5] to process only DCDs 0, 1, 4, 5).
        If None, processes all DCDs from 0 to num_dcd-1. Default is None.
    neutral : bool, optional
        If True, use a geometry-centered dipole evaluation because no COM correction is
        needed for a neutral selection. If False, use mass-centered dipoles.
    
    Side Effects
    ------------
    - Creates directories:
        - `{baseDir}/{OUTdir}`
        - `writenCodes`
        - `logs`
    - Writes Tcl scripts to `writenCodes/dipole_calculator_{i}.tcl`.
    - Runs VMD on each script in parallel, logging to `logs/log_{i}.log`.
    - Produces dipole files `dipole_{i}.dat` in `{baseDir}/{OUTdir}`.

    Returns
    -------
    dict
        Summary of processing results with success/failure counts and timing.
        
    Example
    -------
    >>> results = vmd_dipole_collective(
    ...     baseDir="/home/user/sim",
    ...     INdir="trajectories", 
    ...     OUTdir="dipoles",
    ...     psf="system",
    ...     dcd="traj",
    ...     num_dcd=6,
    ...     target="resname WAT",
    ...     vmd="/usr/local/bin/vmd",
    ...     max_workers=4
    ... )
    # Creates:
    #   Output files according to output_pattern or /home/user/sim/dipoles/dipole_0.dat, …, dipole_5.dat
    #   writenCodes/dipole_calculator_0.tcl, …, dipole_calculator_5.tcl
    #   logs/log_0.log, …, log_5.log
    """
    
    start_time = time.time()

    if not str(vmd or "").strip():
        raise ValueError("VMD executable path is required for collective dipole calculation")
    if not str(psf or "").strip():
        raise ValueError("PSF pattern is required for collective dipole calculation")
    if not str(dcd or "").strip():
        raise ValueError("DCD pattern is required for collective dipole calculation")
    if not str(target or "").strip():
        raise ValueError("Target selection is required for collective dipole calculation")
    if not str(output_pattern or OUTdir or "").strip():
        raise ValueError("Output pattern is required for collective dipole calculation")
    
    # Create output directories - handle both pattern and legacy formats
    if output_pattern:
        # New pattern-based approach
        try:
            from .path_utils import expand_path_pattern
        except ImportError:
            from path_utils import expand_path_pattern
        # Create directory from first output file pattern
        first_output = expand_path_pattern(output_pattern, "", 0)
        output_dir = os.path.dirname(os.path.join(baseDir, first_output))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        # Legacy approach for backward compatibility
        os.makedirs(f'{baseDir}/{OUTdir}', exist_ok=True)
    
    os.makedirs('writenCodes', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Determine which DCDs to process
    if dcd_indices is None:
        dcd_list = list(range(num_dcd))
    else:
        dcd_list = dcd_indices
    
    # Set up parallel processing
    if max_workers is None:
        max_workers = min(len(dcd_list), mp.cpu_count())
    
    print(f"Processing DCDs: {dcd_list}")
    print(f"Using {max_workers} parallel workers for {len(dcd_list)} trajectory chunks...")
    print(f"Neutral dipole handling: {'enabled' if neutral else 'disabled'}")
    
    # Generate all TCL scripts first with pattern validation
    for i in dcd_list:
        success = _write_dipole_tcl_script(i, baseDir, INdir, OUTdir, psf, dcd, target, neutral, output_pattern)
        if not success:
            print(f"ERROR: Failed to generate TCL script for dipole chunk {i} due to pattern validation failure.")
            print("Please fix the corrupted patterns in your GUI input fields and try again.")
            return {'success': 0, 'failed': dcd_list, 'total_time': 0}
    
    # Process VMD scripts in parallel
    results = {'success': 0, 'failed': [], 'total_time': 0}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_index = {
            executor.submit(_run_vmd_dipole_script, i, vmd, baseDir, output_pattern, output_io_spec): i 
            for i in dcd_list
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                success, stdout, stderr = future.result()
                if success:
                    results['success'] += 1
                    print(f"✓ Completed dipole calculation for chunk {index}")
                else:
                    results['failed'].append(index)
                    print(f"✗ Failed dipole calculation for chunk {index}")
                    if stderr:
                        print(f"  Error: {stderr}")
                        
            except Exception as exc:
                results['failed'].append(index)
                print(f"✗ Exception in dipole calculation for chunk {index}: {exc}")
    
    results['total_time'] = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"COLLECTIVE DIPOLE CALCULATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total chunks: {num_dcd}")
    print(f"Successful: {results['success']}")
    print(f"Failed: {len(results['failed'])}")
    if results['failed']:
        print(f"Failed indices: {results['failed']}")
    print(f"Total time: {results['total_time']:.2f} seconds")
    print(f"Parallel efficiency: {max_workers} workers")
    print(f"{'='*50}\n")
    
    return results


def _write_dipole_tcl_script(i, baseDir, INdir, OUTdir, psf, dcd, target, neutral=False, output_pattern=None):
    """Write optimized TCL script for collective dipole moment calculation."""
    
    # Validate patterns to ensure they match user input expectations
    def validate_dipole_pattern_consistency(pattern_name, pattern, expanded_path, index):
        """Validate that the expanded pattern matches what should be expected from user input"""
        if not pattern:
            return True  # Skip validation if no pattern
        
        # Create the expected expanded pattern by manually expanding the user's input
        # Note: dipole patterns typically don't use common_term (passed as "")
        expected_pattern = pattern
        # Replace {i} with the actual index
        import re
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
            print(f"  Index: {index}")
            print(f"  Expected expanded: {expected_pattern}")
            print(f"  Actual expanded:   {expanded_path}")
            print(f"  This indicates the pattern may contain corrupted data or unexpected terms.")
            print(f"  Please verify your GUI input fields contain clean patterns with only {{i}} placeholders.")
            return False
        
        return True
    
    try:
        from .path_utils import expand_path_pattern
    except ImportError:
        from path_utils import expand_path_pattern

    traj_path = expand_path_pattern(INdir, "", i)
    psf_path = expand_path_pattern(psf, "", i)
    dcd_path = expand_path_pattern(dcd, "", i)

    # Determine output file path
    if output_pattern:
        output_path = expand_path_pattern(output_pattern, "", i)
        
        # Validate output pattern
        if not validate_dipole_pattern_consistency("Output", output_pattern, output_path, i):
            print(f"ERROR: Pattern validation failed for dipole chunk {i}")
            print("Please fix the corrupted patterns in your GUI input fields before proceeding.")
            return False
        
        output_full_path = os.path.join(baseDir, output_path)
    else:
        # Legacy format for backward compatibility
        output_full_path = f"{baseDir}/{OUTdir}/dipole_{i}.dat"

    traj_full_path = os.path.join(baseDir, traj_path) if traj_path else baseDir
    psf_full_path = os.path.join(baseDir, psf_path)
    dcd_full_path = os.path.join(baseDir, dcd_path)
    dipole_center_option = "-geocenter" if neutral else "-masscenter"
    
    if not os.path.exists(psf_full_path):
        raise FileNotFoundError(_format_missing_file_message("PSF", psf_full_path))
    if not os.path.exists(dcd_full_path):
        raise FileNotFoundError(_format_missing_file_message("DCD", dcd_full_path))

    with open(f"writenCodes/dipole_calculator_{i}.tcl", "w") as f:
        f.write(f"""# Optimized collective dipole moment calculation script for chunk {i}
# Generated by vmd_dipole.py

puts "Starting collective dipole moment calculation for chunk {i}"
puts "Timestamp: [clock format [clock seconds]]"

# Open output file
set outfile [open "{_tcl_quote(output_full_path)}" w]

# Load trajectory
set trajDir "{_tcl_quote(traj_full_path)}"
set PSF "{_tcl_quote(psf_full_path)}"
set DCD "{_tcl_quote(dcd_full_path)}"

puts "Loading PSF: $PSF"
puts "Loading DCD: $DCD"

set molid [mol load psf $PSF dcd $DCD]
puts "Molecule loaded with ID: $molid"

# Get trajectory info
set nframes [molinfo $molid get numframes]
puts "Number of frames: $nframes"

# Create selection (done once for efficiency)
set sel [atomselect $molid "{_tcl_quote(target)}"]
set natoms [$sel num]
puts "Selected $natoms atoms with selection: {target}"

# Verify selection is valid
if {{$natoms == 0}} {{
    puts "ERROR: No atoms selected with selection '{target}'"
    exit 1
}}

# Write header to output file
puts $outfile "# Collective dipole moment calculation"
puts $outfile "# Selection: {target} ($natoms atoms)"
puts $outfile "# Neutral handling: {'geometric center' if neutral else 'center of mass'}"
puts $outfile "# Format: frame dipole_x dipole_y dipole_z magnitude_debye"

# Process each frame efficiently
for {{set frame 0}} {{$frame < $nframes}} {{incr frame}} {{
    # Update selection to current frame
    $sel frame $frame
    
    # Calculate dipole moment vector and magnitude
    set mu_vector [measure dipole $sel {dipole_center_option}]
    set mu_magnitude [veclength [measure dipole $sel {dipole_center_option} -debye]]
    
    # Extract individual components
    set mu_x [lindex $mu_vector 0]
    set mu_y [lindex $mu_vector 1] 
    set mu_z [lindex $mu_vector 2]
    
    # Write frame data with fixed precision
    puts $outfile [format "%d %.8f %.8f %.8f %.8f" $frame $mu_x $mu_y $mu_z $mu_magnitude]
    
    # Progress indicator for large trajectories
    if {{$frame % 100 == 0}} {{
        puts "Processed frame $frame/$nframes"
    }}
}}

# Cleanup
$sel delete
mol delete $molid
close $outfile

puts "Collective dipole moment calculation completed for chunk {i}"
puts "Output saved to: {output_full_path}"

# Force garbage collection and exit
gc
exit
""")
    
    return True


def _run_vmd_dipole_script(index, vmd_path, baseDir="", output_pattern=None, output_io_spec=None):
    """Run a single VMD dipole script and return results."""
    
    script_path = f"writenCodes/dipole_calculator_{index}.tcl"
    log_path = f"logs/log_{index}.log"

    if not str(vmd_path or "").strip():
        return False, "", "VMD executable path is empty"
    
    # Build VMD command
    command = [vmd_path, "-dispdev", "text", "-nt", "1", "-e", script_path]
    
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
            try:
                from .path_utils import expand_path_pattern
            except ImportError:
                from path_utils import expand_path_pattern
            output_file = os.path.join(baseDir, expand_path_pattern(output_pattern, "", index))
            mode = str(output_io_spec.get("mode", "text")).lower()
            precision = str(output_io_spec.get("precision", "custom")).lower()
            decimals = output_io_spec.get("decimals", 8)
            if not (mode == "text" and precision == "custom" and int(decimals) == 8):
                data = load_numeric_array(
                    output_file,
                    {"mode": "text", "precision": "custom", "decimals": 8},
                    default_mode="text",
                    default_precision="custom",
                    default_decimals=8,
                )
                save_numeric_array(
                    output_file,
                    data,
                    output_io_spec,
                    default_mode="text",
                    default_precision="custom",
                    default_decimals=8,
                )
        return success, process.stdout, process.stderr
        
    except subprocess.TimeoutExpired:
        error_msg = f"VMD process for chunk {index} timed out after 1 hour"
        with open(log_path, 'w') as log_file:
            log_file.write(f"ERROR: {error_msg}\n")
        return False, "", error_msg
        
    except Exception as e:
        error_msg = f"Exception running VMD for chunk {index}: {str(e)}"
        with open(log_path, 'w') as log_file:
            log_file.write(f"ERROR: {error_msg}\n")
        return False, "", error_msg



    
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
