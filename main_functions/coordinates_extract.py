import os
import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from difflib import get_close_matches
import numpy as np
try:
    from .numeric_io import (
        DEFAULT_DOUBLE_DECIMALS,
        DEFAULT_SINGLE_DECIMALS,
        load_numeric_array,
        save_numeric_array,
    )
except ImportError:
    from numeric_io import (
        DEFAULT_DOUBLE_DECIMALS,
        DEFAULT_SINGLE_DECIMALS,
        load_numeric_array,
        save_numeric_array,
    )
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


def _normalize_output_io_spec(output_io_spec):
    spec = dict(output_io_spec or {})
    mode = str(spec.get("mode") or "binary").strip().lower()
    precision = str(spec.get("precision") or "single").strip().lower()
    decimals = spec.get("decimals")

    if mode not in {"binary", "text"}:
        mode = "binary"
    if precision not in {"single", "double", "custom"}:
        precision = "single"
    if precision == "custom":
        try:
            decimals = max(0, int(decimals))
        except Exception:
            decimals = DEFAULT_SINGLE_DECIMALS
    else:
        decimals = None

    return {"mode": mode, "precision": precision, "decimals": decimals}


def _text_format_string(io_spec):
    if io_spec["precision"] == "double":
        decimals = DEFAULT_DOUBLE_DECIMALS
    elif io_spec["precision"] == "custom":
        decimals = int(io_spec["decimals"])
    else:
        decimals = DEFAULT_SINGLE_DECIMALS
    return f"%.{decimals}f"


def _raw_binary_output_path(output_file):
    return f"{output_file}.rawf32"


def _raw_binary_shape_path(output_file):
    return f"{output_file}.shape"


def _finalize_binary_coordinate_output(output_file, output_io_spec):
    """Convert VMD raw float32 output plus shape metadata into the requested final file."""
    raw_output_file = _raw_binary_output_path(output_file)
    shape_file = _raw_binary_shape_path(output_file)

    if not os.path.exists(raw_output_file):
        # Backward compatibility for older runs that may have written the final file directly.
        if os.path.exists(output_file):
            load_numeric_array(
                output_file,
                {"mode": "binary", "precision": "single"},
                default_mode="binary",
                default_precision="single",
            )
            return
        raise FileNotFoundError(f"Raw binary coordinate output not found: {raw_output_file}")

    if not os.path.exists(shape_file):
        raise FileNotFoundError(f"Binary coordinate shape metadata not found: {shape_file}")

    with open(shape_file, "r", encoding="utf-8") as handle:
        shape_tokens = handle.read().strip().split()
    if len(shape_tokens) != 2:
        raise ValueError(f"Invalid binary coordinate shape metadata in {shape_file}")

    try:
        n_frames, n_columns = (int(shape_tokens[0]), int(shape_tokens[1]))
    except ValueError as exc:
        raise ValueError(f"Non-integer shape metadata in {shape_file}") from exc

    data = np.fromfile(raw_output_file, dtype=np.float32)
    expected_size = n_frames * n_columns
    if data.size != expected_size:
        raise ValueError(
            f"Raw binary coordinate size mismatch for {raw_output_file}: "
            f"expected {expected_size} float32 values, found {data.size}"
        )

    data = data.reshape(n_frames, n_columns)
    save_numeric_array(
        output_file,
        data,
        output_io_spec,
        default_mode="binary",
        default_precision="single",
    )

    os.remove(raw_output_file)
    os.remove(shape_file)

def raw_coords(baseDir, psf_pattern, dcd_pattern, output_pattern=None, num_dcd=None, target_selection=None, vmd=None, max_workers=None, dcd_indices=None, common_term="", stride=1, wrap_settings=None, output_io_spec=None):
    """
    Extracts raw XYZ coordinates from a series of DCD trajectories using VMD in parallel,
    by generating per-segment Tcl scripts, running them in batch, and saving
    the results directly as NumPy `.npy` files.

    For each trajectory chunk, this function:
      1. Creates output directories based on output pattern.
      2. Writes a VMD Tcl script that:
         - Loads the PSF and DCD for that chunk using the expanded patterns.
         - Iterates over every frame.
         - Selects atoms using the target_selection criteria.
         - Extracts their x,y,z coordinates efficiently.
         - Writes results using the output pattern.
      3. Invokes VMD in text mode (`-dispdev text`) in parallel processes.
      4. Captures stdout/stderr to `logs/log_{i}.log`.

    Parameters
    ----------
    baseDir : str
        Root directory for the simulation.
    psf_pattern : str
        Path pattern for PSF files. Can contain * (common term) and {i} (file index).
        Example: "trajectories_*/run_{i}/system.psf"
    dcd_pattern : str
        Path pattern for DCD files. Can contain * (common term) and {i} (file index).
        Example: "trajectories_*/run_{i}/traj.dcd"
    output_pattern : str
        Path pattern for output files. Can contain * (common term) and {i} (file index).
        Example: "path/to/xyz_{i}.npy"
    num_dcd : int
        Number of trajectory chunks to process.  Generates scripts for
        indices `0` through `num_dcd-1`.
    target_selection : str
        VMD atom selection string (e.g., "resname WAT and residue 0 to 999", "water", "protein").
    vmd : str
        The path to the VMD executable.
    max_workers : int, optional
        Maximum number of parallel workers. Defaults to min(num_dcd, CPU count).
    dcd_indices : list, optional
        List of DCD indices to process (e.g., [0, 1, 4, 5] to process only DCDs 0, 1, 4, 5).
        If None, processes all DCDs from 0 to num_dcd-1. Default is None.
    common_term : str, optional
        Value to replace * placeholders in patterns. Default is "".
    stride : int, optional
        Frame stride for extraction. Use 1 to save every frame, 2 to save every
        other frame, etc. Default is 1.
    wrap_settings : dict, optional
        Optional pbctools wrapping configuration. When enabled, a `pbc wrap`
        command is inserted before coordinates are extracted.
    
    Side Effects
    ------------
    - Creates directories:
        - `{baseDir}/{output_pattern}`
        - `writenCodes`
        - `logs`
    - Writes Tcl scripts to `writenCodes/coords_{i}.tcl`.
    - Runs VMD on each script in parallel, logging to `logs/log_{i}.log`.
    - Produces coordinate files `xyz_{i}.npy` in `{baseDir}/{output_pattern}`.

    Returns
    -------
    dict
        Summary of processing results with success/failure counts and timing.

    Example
    -------
    >>> results = raw_coords(
    ...     baseDir="/home/user/sim",
    ...     psf_pattern="trajectories_*/run_{i}/system.psf",
    ...     dcd_pattern="trajectories_*/run_{i}/traj.dcd",
    ...     output_pattern="path/to/xyz_{i}.npy",
    ...     num_dcd=6,
    ...     target_selection="resname WAT and residue 0 to 999",
    ...     vmd="/usr/local/bin/vmd",
    ...     common_term="240"
    ... )
    # Creates:
    #   /home/user/sim/path/to/xyz_0.npy, …, path/to/xyz_5.npy
    #   writenCodes/coords_0.tcl, …, coords_5.tcl
    #   logs/log_0.log, …, log_5.log
    """
    
    start_time = time.time()
    
    # Validate path patterns
    is_valid, error_msg = validate_path_pattern(psf_pattern)
    if not is_valid:
        raise ValueError(f"Invalid PSF pattern: {error_msg}")
        
    is_valid, error_msg = validate_path_pattern(dcd_pattern)
    if not is_valid:
        raise ValueError(f"Invalid DCD pattern: {error_msg}")
        
    is_valid, error_msg = validate_path_pattern(output_pattern)
    if not is_valid:
        raise ValueError(f"Invalid output pattern: {error_msg}")
    
    print(f"{'='*50}")
    print(f"COORDINATE EXTRACTION")
    print(f"{'='*50}")
    print(f"Base directory: {baseDir}")
    print(f"PSF pattern: {psf_pattern}")
    print(f"DCD pattern: {dcd_pattern}")
    print(f"Output pattern: {output_pattern}")
    print(f"Common term: {common_term}")
    print(f"Number of DCDs: {num_dcd}")
    print(f"Target selection: {target_selection}")
    print(f"VMD executable: {vmd}")
    print(f"Stride: {stride}")
    print(f"Wrap enabled: {bool((wrap_settings or {}).get('enabled'))}")

    # Validate stride
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
    os.makedirs("logs", exist_ok=True)
    
    # Note: With flexible target_selection, we can't easily estimate output size
    # The actual number of selected atoms will be determined by VMD at runtime
    
    # Validate that input files exist for first DCD (sanity check)
    if dcd_list:
        test_psf = os.path.join(baseDir, expand_path_pattern(psf_pattern, common_term, dcd_list[0]))
        test_dcd = os.path.join(baseDir, expand_path_pattern(dcd_pattern, common_term, dcd_list[0]))
        
        if not os.path.exists(test_psf):
            raise FileNotFoundError(_format_missing_file_message("PSF", test_psf))
        if not os.path.exists(test_dcd):
            raise FileNotFoundError(_format_missing_file_message("DCD", test_dcd))
        
        print(f"✓ Input files validated for index {dcd_list[0]}")
    
    # Results tracking
    results = {
        'successful': [],
        'failed': [],
        'total_time': 0,
        'num_processed': len(dcd_list)
    }
    
    # Generate all TCL scripts first with pattern validation
    for i in dcd_list:
        success = _write_tcl_script(
            i,
            baseDir,
            psf_pattern,
            dcd_pattern,
            output_pattern,
            target_selection,
            common_term,
            stride,
            wrap_settings,
            output_io_spec,
        )
        if not success:
            print(f"ERROR: Failed to generate TCL script for coordinates chunk {i} due to pattern validation failure.")
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
            executor.submit(
                _run_vmd_script, i, vmd, baseDir, output_pattern, common_term, output_io_spec
            ): i 
            for i in dcd_list
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                success, stdout, stderr = future.result()
                if success:
                    results['successful'].append(index)
                    print(f"✓ Completed trajectory chunk {index}")
                else:
                    results['failed'].append(index)
                    print(f"✗ Failed trajectory chunk {index}")
                    if stderr:
                        print(f"  Error: {stderr}")
                        
            except Exception as exc:
                results['failed'].append(index)
                print(f"✗ Exception in trajectory chunk {index}: {exc}")
    
    results['total_time'] = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"COORDINATE EXTRACTION SUMMARY")
    print(f"{'='*50}")
    print(f"Total chunks: {num_dcd}")
    print(f"Successful: {len(results['successful'])}")
    print(f"Failed: {len(results['failed'])}")
    if results['failed']:
        print(f"Failed indices: {results['failed']}")
    print(f"Total time: {results['total_time']:.2f} seconds")
    print(f"Parallel efficiency: {max_workers} workers")
    print(f"{'='*50}\n")
    
    # Note: Output shape validation removed since expected count depends on VMD selection
    
    return results



def _write_tcl_script(
    i,
    baseDir,
    psf_pattern,
    dcd_pattern,
    output_pattern,
    target_selection,
    common_term="",
    stride=1,
    wrap_settings=None,
    output_io_spec=None,
):
    """Write optimized TCL script for a single trajectory chunk."""
    
    # Expand patterns to get actual file paths
    psf_path = expand_path_pattern(psf_pattern, common_term, i)
    dcd_path = expand_path_pattern(dcd_pattern, common_term, i)
    output_path = expand_path_pattern(output_pattern, common_term, i)
    
    # Validate patterns to ensure they match user input expectations
    def validate_coords_pattern_consistency(pattern_name, pattern, expanded_path, common_term, index):
        """Validate that the expanded pattern matches what should be expected from user input"""
        if not pattern or not common_term:
            return True  # Skip validation if no pattern or common term
        
        # Create the expected expanded pattern by manually expanding the user's input
        expected_pattern = pattern.replace('*', common_term)
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
            print(f"  Common term: {common_term}")
            print(f"  Index: {index}")
            print(f"  Expected expanded: {expected_pattern}")
            print(f"  Actual expanded:   {expanded_path}")
            print(f"  This indicates the pattern may contain corrupted data or unexpected terms.")
            print(f"  Please verify your GUI input fields contain clean patterns with only * and {{i}} placeholders.")
            return False
        
        return True
    
    # Validate PSF, DCD, and output patterns
    psf_valid = validate_coords_pattern_consistency("PSF", psf_pattern, psf_path, common_term, i)
    dcd_valid = validate_coords_pattern_consistency("DCD", dcd_pattern, dcd_path, common_term, i)
    output_valid = validate_coords_pattern_consistency("Output", output_pattern, output_path, common_term, i)
    
    if not psf_valid or not dcd_valid or not output_valid:
        print(f"ERROR: Pattern validation failed for coordinates chunk {i}")
        print(f"Please fix the corrupted patterns in your GUI input fields before proceeding.")
        return False
    
    # Create absolute paths
    psf_full_path = os.path.join(baseDir, psf_path)
    dcd_full_path = os.path.join(baseDir, dcd_path)
    output_full_path = os.path.join(baseDir, output_path)
    
    print(f"DEBUG TCL Script {i}:")
    print(f"  Output pattern: {output_pattern}")
    print(f"  Expanded output: {output_path}")
    print(f"  Full output path: {output_full_path}")
    print(f"  Output directory: {os.path.dirname(output_full_path)}")
    
    # Ensure output directory exists before creating TCL script
    output_dir = os.path.dirname(output_full_path)
    if output_dir and output_dir != baseDir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create more specific TCL filename with common term
    tcl_filename = f"coords_{common_term}_{i}.tcl" if common_term else f"coords_{i}.tcl"

    wrap_settings = wrap_settings or {}
    normalized_output = _normalize_output_io_spec(output_io_spec)
    wrap_enabled = bool(wrap_settings.get("enabled"))
    atomselection = wrap_settings.get("atomselection", "").strip() or target_selection or "all"
    wrap_option_flags = wrap_settings.get("options", {})

    wrap_lines = []
    if wrap_enabled:
        wrap_shape = wrap_settings.get("shape") or "parallelepiped"
        wrap_compound = wrap_settings.get("compound")
        wrap_center = wrap_settings.get("center") or "unitcell"

        wrap_lines.extend([
            "",
            "# Load pbctools and wrap coordinates before extraction",
            "if {[catch {package require pbctools} pbctools_err]} {",
            '    puts "ERROR: Failed to load pbctools package: $pbctools_err"',
            "    exit 1",
            "}",
            "set wrap_args [list -molid $molid -now]",
        ])

        if wrap_shape in {"parallelepiped", "orthorhombic"}:
            wrap_lines.append(f'lappend wrap_args -{wrap_shape}')
        if wrap_option_flags.get("sel"):
            wrap_lines.append(f'lappend wrap_args -sel "{atomselection}"')
        if wrap_compound in {"res", "segid", "chain", "fragment"}:
            wrap_lines.append(f'lappend wrap_args -compound {wrap_compound}')
        if wrap_option_flags.get("compoundref"):
            wrap_lines.append(f'lappend wrap_args -compoundref "{atomselection}"')
        if wrap_center in {"origin", "unitcell", "com", "bb"}:
            wrap_lines.append(f'lappend wrap_args -center {wrap_center}')
        if wrap_option_flags.get("centersel"):
            wrap_lines.append(f'lappend wrap_args -centersel "{atomselection}"')
        if wrap_option_flags.get("verbose"):
            wrap_lines.append("lappend wrap_args -verbose")
        if wrap_option_flags.get("draw"):
            wrap_lines.append("lappend wrap_args -draw")

        wrap_lines.extend([
            'puts "Configured per-frame wrapping with: pbc wrap [join $wrap_args { }]"',
        ])

    wrap_block = "\n".join(wrap_lines)
    text_output = normalized_output["mode"] == "text"
    text_format = _text_format_string(normalized_output)

    if text_output:
        output_header_block = ""
        output_open_block = f'set outfile [open "{_tcl_quote(output_full_path)}" w]'
        output_frame_block = f"""
    $sel frame $frame
    set coords [$sel get {{x y z}}]
    set line_parts [list]
    foreach coord $coords {{
        foreach value $coord {{
            lappend line_parts [format "{text_format}" $value]
        }}
    }}
    puts $outfile [join $line_parts " "]
"""
        output_summary_label = "Text output written to"
    else:
        raw_binary_path = _raw_binary_output_path(output_full_path)
        binary_shape_path = _raw_binary_shape_path(output_full_path)
        output_open_block = "\n".join([
            f'set outfile [open "{_tcl_quote(raw_binary_path)}" w]',
            "fconfigure $outfile -translation binary -encoding binary",
        ])
        output_header_block = f"""
# Persist raw float32 coordinates and let Python finalize the NumPy container.
set output_frames [expr {{($num_frames + $stride - 1) / $stride}}]
set num_columns [expr {{$num_atoms * 3}}]
set shapefile [open "{_tcl_quote(binary_shape_path)}" w]
puts $shapefile "$output_frames $num_columns"
close $shapefile
"""
        output_frame_block = """
    $sel frame $frame
    set coords [$sel get {x y z}]
    
    # Stream binary float32 output in chunks to avoid building enormous Tcl objects
    set binary_chunk ""
    set chunk_atoms 0
    foreach coord $coords {
        append binary_chunk [binary format r3 $coord]
        incr chunk_atoms
        if {$chunk_atoms >= 4096} {
            puts -nonewline $outfile $binary_chunk
            set binary_chunk ""
            set chunk_atoms 0
        }
    }
    if {$chunk_atoms > 0} {
        puts -nonewline $outfile $binary_chunk
    }
"""
        output_summary_label = "Raw binary coordinate output written to"
    
    with open(f"writenCodes/{tcl_filename}", "w") as f:
        f.write(f"""# Optimized coordinate extraction script for chunk {i}
# Generated by coordinates_extract.py

puts "Starting coordinate extraction for chunk {i}"
puts "Timestamp: [clock format [clock seconds]]"

# Set paths
set baseDir "{_tcl_quote(baseDir)}"

# Open output file
{output_open_block}

# Load trajectory
set PSF "{_tcl_quote(psf_full_path)}"
set DCD "{_tcl_quote(dcd_full_path)}"

puts "Loading PSF: $PSF"
puts "Loading DCD: $DCD"

# Debug output - check file existence
puts "About to load: PSF=$PSF, DCD=$DCD"
if {{![file exists $PSF]}} {{
    puts "ERROR: PSF file does not exist: $PSF"
    exit 1
}}
if {{![file exists $DCD]}} {{
    puts "ERROR: DCD file does not exist: $DCD"
    exit 1
}}

set molid [mol load psf $PSF dcd $DCD]
puts "Molecule loaded with ID: $molid"

# Get number of frames
set num_frames [molinfo $molid get numframes]
puts "Total frames: $num_frames"
set stride {stride}
{wrap_block}

# Create selection
set sel [atomselect $molid "{_tcl_quote(target_selection)}"]
set num_atoms [$sel num]
puts "Selected $num_atoms atoms"

if {{$num_atoms == 0}} {{
    puts "ERROR: No atoms selected with criteria '{target_selection}'"
    exit 1
}}

{output_header_block}

# Extract coordinates for all frames
puts "Extracting coordinates..."
set frame_count 0
for {{set frame 0}} {{$frame < $num_frames}} {{incr frame $stride}} {{
    animate goto $frame
""")
        if wrap_enabled:
            f.write(f"""
    if {{[catch {{eval pbc wrap $wrap_args}} wrap_err]}} {{
        puts "ERROR: pbc wrap failed at frame $frame: $wrap_err"
        exit 1
    }}
""")
        f.write(f"""
{output_frame_block}
    
    incr frame_count
    if {{$frame_count % 100 == 0}} {{
        puts "Processed $frame_count frames..."
    }}
}}

$sel delete
close $outfile
mol delete $molid

puts "Extraction complete: $frame_count frames processed"
puts "{output_summary_label}: {output_full_path}"
puts "Timestamp: [clock format [clock seconds]]"

exit 0
""")
    
    return True


def _run_vmd_script(index, vmd_path, baseDir, output_pattern, common_term="", output_io_spec=None):
    """Run a single VMD script and return results."""
    
    # Use the same naming convention as in _write_tcl_script
    tcl_filename = f"coords_{common_term}_{index}.tcl" if common_term else f"coords_{index}.tcl"
    script_path = f"writenCodes/{tcl_filename}"
    log_path = f"logs/log_{index}.log"
    
    # Build VMD command - ensure proper quoting for paths with spaces
    command = [vmd_path, "-dispdev", "text", "-nt", "1", "-e", script_path]
    
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
        if success:
            output_file = os.path.join(baseDir, expand_path_pattern(output_pattern, common_term, index))
            normalized_output = _normalize_output_io_spec(output_io_spec)
            if normalized_output["mode"] == "binary":
                _finalize_binary_coordinate_output(output_file, normalized_output)
        return success, process.stdout, process.stderr
        
    except subprocess.TimeoutExpired:
        error_msg = f"VMD process for chunk {index} timed out (timeout removed for large trajectories)"
        with open(log_path, 'w') as log_file:
            log_file.write(f"ERROR: {error_msg}\n")
        return False, "", error_msg
        
    except Exception as e:
        error_msg = f"Exception running VMD for chunk {index}: {str(e)}"
        with open(log_path, 'w') as log_file:
            log_file.write(f"ERROR: {error_msg}\n")
        return False, "", error_msg



    
