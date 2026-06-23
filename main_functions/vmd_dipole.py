import os
import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
from difflib import get_close_matches
try:
    from .numeric_io import load_numeric_array, save_numeric_array
except ImportError:
    from numeric_io import load_numeric_array, save_numeric_array

def vmd_dipole_collective(baseDir, psf, dcd, num_dcd, target, vmd, stride=1, dipole_unit="e·Å",
                          max_workers=None, dcd_indices=None, neutral=False, output_pattern=None,
                          output_io_spec=None, wrap_settings=None):
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
    stride : int, optional
        Process every Nth frame. Default is 1.
    dipole_unit : str, optional
        Output unit for vector components and magnitudes. Either `e·Å` or `Debye`.
    max_workers : int, optional
        Maximum number of parallel workers. Defaults to min(num_dcd, CPU count).
    dcd_indices : list, optional
        List of DCD indices to process (e.g., [0, 1, 4, 5] to process only DCDs 0, 1, 4, 5).
        If None, processes all DCDs from 0 to num_dcd-1. Default is None.
    neutral : bool, optional
        If True, use a geometry-centered dipole evaluation because no COM correction is
        needed for a neutral selection. If False, use mass-centered dipoles.
    wrap_settings : dict, optional
        pbctools wrapping settings. If enabled, each frame is wrapped before measuring
        the collective dipole.
    
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
    if not str(output_pattern or "").strip():
        raise ValueError("Output pattern is required for collective dipole calculation")
    if int(stride) < 1:
        raise ValueError("Stride must be at least 1 for collective dipole calculation")
    if dipole_unit not in {"Debye", "e·Å"}:
        raise ValueError("Dipole unit must be either 'Debye' or 'e·Å'")
    
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
    print(f"Stride: {stride}")
    print(f"Dipole unit: {dipole_unit}")
    print(f"Wrap enabled: {bool((wrap_settings or {}).get('enabled'))}")
    
    # Generate all TCL scripts first with pattern validation
    for i in dcd_list:
        success = _write_dipole_tcl_script(i, baseDir, psf, dcd, target, stride, dipole_unit, neutral, output_pattern, wrap_settings)
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


def _write_dipole_tcl_script(i, baseDir, psf, dcd, target, stride=1, dipole_unit="e·Å", neutral=False,
                             output_pattern=None, wrap_settings=None):
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
    psf_full_path = os.path.join(baseDir, psf_path)
    dcd_full_path = os.path.join(baseDir, dcd_path)
    dipole_center_option = "-geocenter" if neutral else "-masscenter"
    dipole_unit_option = " -debye" if dipole_unit == "Debye" else ""
    header_unit = "Debye" if dipole_unit == "Debye" else "e·Å"
    wrap_block, wrap_frame_block = _build_wrap_tcl_blocks(wrap_settings, target)
    
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
set PSF "{_tcl_quote(psf_full_path)}"
set DCD "{_tcl_quote(dcd_full_path)}"

puts "Loading PSF: $PSF"
puts "Loading DCD: $DCD"

set molid [mol load psf $PSF dcd $DCD]
puts "Molecule loaded with ID: $molid"
{wrap_block}

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
puts $outfile "# Format: frame dipole_x dipole_y dipole_z magnitude_{header_unit}"

# Process each frame efficiently
for {{set frame 0}} {{$frame < $nframes}} {{incr frame {int(stride)}}} {{
    animate goto $frame
{wrap_frame_block}
    # Update selection to current frame
    $sel frame $frame
    
    # Calculate dipole moment vector and magnitude
    set mu_vector [measure dipole $sel {dipole_center_option}{dipole_unit_option}]
    set mu_magnitude [veclength $mu_vector]
    
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


def _build_wrap_tcl_blocks(wrap_settings, target_selection):
    wrap_settings = wrap_settings or {}
    if not bool(wrap_settings.get("enabled")):
        return "", ""

    atomselection = str(wrap_settings.get("atomselection") or "").strip() or target_selection or "all"
    wrap_shape = wrap_settings.get("shape") or "parallelepiped"
    wrap_compound = wrap_settings.get("compound")
    wrap_center = wrap_settings.get("center") or "unitcell"
    wrap_option_flags = wrap_settings.get("options", {})

    setup_lines = [
        "",
        "# Load pbctools and configure per-frame wrapping before dipole calculation",
        "if {[catch {package require pbctools} pbctools_err]} {",
        '    puts "ERROR: Failed to load pbctools package: $pbctools_err"',
        "    exit 1",
        "}",
        "set wrap_args [list -molid $molid -now]",
    ]
    if wrap_shape in {"parallelepiped", "orthorhombic"}:
        setup_lines.append(f"lappend wrap_args -{wrap_shape}")
    if wrap_option_flags.get("sel"):
        setup_lines.append(f'lappend wrap_args -sel "{_tcl_quote(atomselection)}"')
    if wrap_compound in {"res", "segid", "chain", "fragment"}:
        setup_lines.append(f"lappend wrap_args -compound {wrap_compound}")
    if wrap_option_flags.get("compoundref"):
        setup_lines.append(f'lappend wrap_args -compoundref "{_tcl_quote(atomselection)}"')
    if wrap_center in {"origin", "unitcell", "com", "bb"}:
        setup_lines.append(f"lappend wrap_args -center {wrap_center}")
    if wrap_option_flags.get("centersel"):
        setup_lines.append(f'lappend wrap_args -centersel "{_tcl_quote(atomselection)}"')
    if wrap_option_flags.get("verbose"):
        setup_lines.append("lappend wrap_args -verbose")
    if wrap_option_flags.get("draw"):
        setup_lines.append("lappend wrap_args -draw")
    setup_lines.append('puts "Configured per-frame wrapping with: pbc wrap [join $wrap_args { }]"')

    frame_block = """    if {[catch {eval pbc wrap $wrap_args} wrap_err]} {
        puts "ERROR: pbc wrap failed at frame $frame: $wrap_err"
        exit 1
    }
"""
    return "\n".join(setup_lines), frame_block.rstrip()


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



    
def vmd_dipole_individual_from_dcd(
    baseDir, psf, dcd, num_dcd, target, vmd, grouping_unit,
    dipole_unit="e·Å", stride=1, max_workers=None, dcd_indices=None,
    vectors_pattern=None, magnitudes_pattern=None,
    vectors_output_io_spec=None, magnitudes_output_io_spec=None,
    common_term="",
):
    """
    Calculate per-group individual dipole moments directly from DCD files using VMD.

    The TCL script re-evaluates the target selection every frame, so groups that
    enter or leave the selection at different frames are all captured.  Each dipole
    line carries a stable string label for the group (``SEGID:RESID``, chain name,
    segname, or ``ALL``).

    Python collects the union of all group labels seen across every frame, assigns
    stable integer indices, then builds NaN-padded rectangular arrays of shape
    (n_frames, n_groups) so no computed dipole moment is ever discarded.
    Groups absent from a particular frame are stored as NaN, distinguishable from
    a true-zero dipole.
    """
    import multiprocessing as mp
    try:
        from .path_utils import expand_path_pattern
        from .numeric_io import save_numeric_array
    except ImportError:
        from path_utils import expand_path_pattern
        from numeric_io import save_numeric_array
    import numpy as np

    if not str(vmd or "").strip():
        raise ValueError("VMD executable path is required for custom dipole calculation")
    if not str(psf or "").strip():
        raise ValueError("PSF pattern is required for custom dipole calculation")
    if not str(dcd or "").strip():
        raise ValueError("DCD pattern is required for custom dipole calculation")
    if not str(target or "").strip():
        raise ValueError("Target selection is required for custom dipole calculation")
    if not str(vectors_pattern or "").strip():
        raise ValueError("Vectors output pattern is required for custom dipole calculation")
    if grouping_unit not in {"residue", "chain", "segname", "all"}:
        raise ValueError(f"grouping_unit must be residue, chain, segname, or all; got {grouping_unit!r}")
    if dipole_unit not in {"Debye", "e·Å"}:
        raise ValueError("dipole_unit must be 'Debye' or 'e·Å'")
    if int(stride) < 1:
        raise ValueError("stride must be >= 1")

    start_time = time.time()
    if dcd_indices is None:
        dcd_list = list(range(num_dcd))
    else:
        dcd_list = list(dcd_indices)

    if max_workers is None:
        max_workers = min(len(dcd_list), mp.cpu_count())

    os.makedirs("writenCodes", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    dipole_center_option = "-geocenter" if dipole_unit == "Debye" else "-masscenter"
    debye_flag = " -debye" if dipole_unit == "Debye" else ""

    def _write_tcl(i):
        psf_path = expand_path_pattern(psf, common_term, i)
        dcd_path = expand_path_pattern(dcd, common_term, i)
        psf_full = os.path.join(baseDir, psf_path)
        dcd_full = os.path.join(baseDir, dcd_path)
        if not os.path.exists(psf_full):
            raise FileNotFoundError(_format_missing_file_message("PSF", psf_full))
        if not os.path.exists(dcd_full):
            raise FileNotFoundError(_format_missing_file_message("DCD", dcd_full))

        tcl_path = f"writenCodes/custom_dipole_{i}.tcl"
        stride_val = int(stride)
        with open(tcl_path, "w") as f:
            # KEY OPTIMIZATIONS:
            # 1. Load DCD with step=stride so VMD keeps only the frames we need
            #    in memory (10x less RAM when stride=10, avoids paging with
            #    33 GB DCDs and multiple parallel workers).
            # 2. Do the expensive "within X of ..." atomselect ONCE per frame
            #    via the master selection; sub-group selections use cheap
            #    index-based queries that touch no spatial data structure.
            # 3. Use a TCL array (hash map) for group deduplication: O(1) per
            #    atom instead of O(N) lsearch, which was O(N^2) overall.
            f.write(f"""# Custom per-group dipole — chunk {i}
# DCD loaded with step={stride_val}: VMD keeps only strided frames in memory.
# The expensive "within X of ..." selection runs ONCE per frame; groups use
# cheap index-based sub-selections that avoid re-running the spatial query.
mol new "{_tcl_quote(psf_full)}" type psf waitfor all
mol addfile "{_tcl_quote(dcd_full)}" type dcd first 0 last -1 step {stride_val} waitfor all
set molid [molinfo top]
set nframes [molinfo $molid get numframes]
set target_sel "{_tcl_quote(target)}"
set grouping "{grouping_unit}"

for {{set frame 0}} {{$frame < $nframes}} {{incr frame}} {{
    animate goto $frame

    # Single expensive spatial query for this frame
    set master [atomselect $molid $target_sel]
    $master frame $frame

    if {{[$master num] == 0}} {{
        $master delete
        continue
    }}

    if {{$grouping eq "residue"}} {{
        # Retrieve index+segid+resid in one batch — no repeated spatial query
        set triples [$master get {{index segid resid}}]
        $master delete
        array unset gi
        set order {{}}
        foreach t $triples {{
            set key "[lindex $t 1]:[lindex $t 2]"
            if {{![info exists gi($key)]}} {{ lappend order $key }}
            lappend gi($key) [lindex $t 0]
        }}
        foreach key $order {{
            # "index ..." selection: O(k), no spatial work at all
            set sel [atomselect $molid "index $gi($key)"]
            $sel frame $frame
            set mu [measure dipole $sel {dipole_center_option}{debye_flag}]
            puts "DIPOLE $frame $key [lindex $mu 0] [lindex $mu 1] [lindex $mu 2] [veclength $mu]"
            $sel delete
        }}
        array unset gi

    }} elseif {{$grouping eq "chain"}} {{
        set pairs [$master get {{index chain}}]
        $master delete
        array unset gi
        set order {{}}
        foreach p $pairs {{
            set key "chain:[lindex $p 1]"
            if {{![info exists gi($key)]}} {{ lappend order $key }}
            lappend gi($key) [lindex $p 0]
        }}
        foreach key $order {{
            set sel [atomselect $molid "index $gi($key)"]
            $sel frame $frame
            set mu [measure dipole $sel {dipole_center_option}{debye_flag}]
            puts "DIPOLE $frame $key [lindex $mu 0] [lindex $mu 1] [lindex $mu 2] [veclength $mu]"
            $sel delete
        }}
        array unset gi

    }} elseif {{$grouping eq "segname"}} {{
        set pairs [$master get {{index segname}}]
        $master delete
        array unset gi
        set order {{}}
        foreach p $pairs {{
            set key "segname:[lindex $p 1]"
            if {{![info exists gi($key)]}} {{ lappend order $key }}
            lappend gi($key) [lindex $p 0]
        }}
        foreach key $order {{
            set sel [atomselect $molid "index $gi($key)"]
            $sel frame $frame
            set mu [measure dipole $sel {dipole_center_option}{debye_flag}]
            puts "DIPOLE $frame $key [lindex $mu 0] [lindex $mu 1] [lindex $mu 2] [veclength $mu]"
            $sel delete
        }}
        array unset gi

    }} else {{
        # "all" grouping: master selection is the single group
        set mu [measure dipole $master {dipole_center_option}{debye_flag}]
        puts "DIPOLE $frame ALL [lindex $mu 0] [lindex $mu 1] [lindex $mu 2] [veclength $mu]"
        $master delete
    }}
}}

mol delete $molid
exit
""")
        return tcl_path

    def _run_one(i):
        tcl_path = _write_tcl(i)
        log_path = f"logs/custom_dipole_{i}.log"
        cmd = [vmd, "-dispdev", "text", "-nt", "1", "-e", tcl_path]
        proc = subprocess.run(cmd, shell=False, capture_output=True, text=True, timeout=None)
        with open(log_path, "w") as lf:
            lf.write(f"Command: {' '.join(cmd)}\nReturn: {proc.returncode}\n"
                     f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\n")
        return proc.returncode == 0, proc.stdout, proc.stderr, i

    results = {"success": 0, "failed": [], "total_time": 0}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_one, i): i for i in dcd_list}
        for future in as_completed(futures):
            i = futures[future]
            try:
                ok, stdout, stderr, idx = future.result()
            except Exception as exc:
                results["failed"].append(i)
                print(f"✗ Exception for chunk {i}: {exc}")
                continue

            if not ok:
                results["failed"].append(i)
                print(f"✗ Failed custom dipole for chunk {i}")
                if stderr:
                    print(f"  {stderr[:400]}")
                continue

            # ----------------------------------------------------------------
            # Parse VMD stdout.
            # Format: DIPOLE <frame_int> <label_str> <dx> <dy> <dz> <mag>
            # Labels may vary per frame; we build the union across all frames
            # and use NaN for groups absent from a particular frame.
            # ----------------------------------------------------------------
            per_frame: dict[int, list[tuple[str, float, float, float, float]]] = {}
            for line in stdout.splitlines():
                line = line.strip()
                if not line.startswith("DIPOLE"):
                    continue
                parts = line.split()
                if len(parts) != 7:
                    continue
                try:
                    frame_num = int(parts[1])
                    label = parts[2]
                    dx, dy, dz, mag = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                except (ValueError, IndexError):
                    continue
                per_frame.setdefault(frame_num, []).append((label, dx, dy, dz, mag))

            if not per_frame:
                results["failed"].append(i)
                print(f"✗ No parseable DIPOLE lines for chunk {i}")
                continue

            # Build stable ordered group label list (union, first-seen order)
            all_labels: list[str] = []
            seen_labels: set[str] = set()
            for frame_num in sorted(per_frame):
                for label, *_ in per_frame[frame_num]:
                    if label not in seen_labels:
                        seen_labels.add(label)
                        all_labels.append(label)
            label_to_idx = {lbl: idx for idx, lbl in enumerate(all_labels)}
            n_groups = len(all_labels)

            sorted_frames = sorted(per_frame)
            n_frames = len(sorted_frames)
            frame_map = {f: fi for fi, f in enumerate(sorted_frames)}

            # NaN fill: groups absent from a frame are NaN, not zero, so
            # downstream code can distinguish missing from a true zero dipole.
            vec_arr = np.full((n_frames, n_groups, 3), np.nan, dtype=np.float64)
            mag_arr = np.full((n_frames, n_groups), np.nan, dtype=np.float64)

            for frame_num, entries in per_frame.items():
                fi = frame_map[frame_num]
                for label, dx, dy, dz, mag in entries:
                    gi = label_to_idx[label]
                    vec_arr[fi, gi] = [dx, dy, dz]
                    mag_arr[fi, gi] = mag

            vec_flat = vec_arr.reshape(n_frames, n_groups * 3)

            # Save group label mapping alongside the data file so downstream
            # code can map column indices back to group identifiers.
            vec_file = os.path.join(baseDir, expand_path_pattern(vectors_pattern, common_term, i))
            vec_dir = os.path.dirname(vec_file)
            os.makedirs(vec_dir or ".", exist_ok=True)
            save_numeric_array(vec_file, vec_flat, vectors_output_io_spec,
                               default_mode="text", default_precision="double")
            _save_group_labels(vec_file, all_labels)

            if magnitudes_pattern:
                mag_file = os.path.join(baseDir, expand_path_pattern(magnitudes_pattern, common_term, i))
                os.makedirs(os.path.dirname(mag_file) or ".", exist_ok=True)
                save_numeric_array(mag_file, mag_arr, magnitudes_output_io_spec,
                                   default_mode="text", default_precision="double")

            n_frames_with_varying = sum(
                1 for entries in per_frame.values() if len(entries) != n_groups
            )
            if n_frames_with_varying:
                print(f"  Note: {n_frames_with_varying}/{n_frames} frames had fewer than "
                      f"{n_groups} groups; missing entries stored as NaN.")

            results["success"] += 1
            print(f"✓ Completed custom dipole for chunk {i}: "
                  f"{n_frames} frames, {n_groups} unique groups")

    results["total_time"] = time.time() - start_time
    print(f"\nCustom dipole summary — success: {results['success']}, "
          f"failed: {len(results['failed'])}, time: {results['total_time']:.2f}s")
    return results


def _save_group_labels(data_file: str, labels: list) -> None:
    """Write a companion <data_file>.groups.txt listing each column's group label."""
    labels_path = data_file + ".groups.txt"
    try:
        with open(labels_path, "w") as f:
            f.write("# column_index  group_label\n")
            for idx, lbl in enumerate(labels):
                f.write(f"{idx}\t{lbl}\n")
    except OSError as exc:
        print(f"  Warning: could not write group labels file {labels_path}: {exc}")


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
