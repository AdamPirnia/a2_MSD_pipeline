# ADMDynAnlz User Manual

This manual explains what each module does, what each section is for, what to enter in each input field, and how the main options affect the generated workflow.

The Qt interface is now the maintained ADMDynAnlz GUI.

## General Concepts

### Base Directory

`Base Directory` is the root folder that contains the files and subfolders used in your workflow. Most file patterns are interpreted relative to this location.

### File patterns

Many fields expect a file pattern rather than a single file name.

When a field asks for a `Pattern`, it expects a full path or a full `path + filename`, depending on the section. For numeric data files generated or reused by ADMDynAnlz, the safest rule is to provide the full `path + filename + extension`.

Paths containing spaces, parentheses, and similar non-regular characters are supported. It is still best to enter the exact full path carefully so that missing-file validation can catch real typos.

- `*` is replaced by the value of `Common Term 1`
- `**` is replaced by the value of `Common Term 2`
- `{i}` is replaced by the zero-based DCD, trajectory, or coordinate-file index

Example with one common term:

```text
anlz/NVT_*/wrapped/xyz_{i}.dat
```

If `Common Term = sqp`, the pattern becomes:

```text
anlz/NVT_sqp/wrapped/xyz_0.dat
anlz/NVT_sqp/wrapped/xyz_1.dat
...
```

Example with two common terms:

```text
anlz/*/**/wrapped/xyz_{i}.dat
```

If `Common Term 1 = NVT` and `Common Term 2 = 300`, the pattern becomes:

```text
anlz/NVT/300/wrapped/xyz_0.dat
anlz/NVT/300/wrapped/xyz_1.dat
...
```

### Common Terms

`Common Terms` provides two optional replacement fields:

- left field: replaces `*`
- right field: replaces `**`

Each field can contain either:

- a single value, such as `NVT`
- a comma-separated list, such as `NVT, NVE`

Example:

- `Common Term 1`: `NVT, NVE`
- `Common Term 2`: `300, 320, 340`

Then a pattern like:

```text
anlz/*/**/coords_{i}.npy
```

expands across all combinations:

- `anlz/NVT/300/coords_0.npy`
- `anlz/NVT/320/coords_0.npy`
- `anlz/NVT/340/coords_0.npy`
- `anlz/NVE/300/coords_0.npy`
- `anlz/NVE/320/coords_0.npy`
- `anlz/NVE/340/coords_0.npy`

The generated workflow handles the nested looping. The low-level analysis functions still receive fully expanded concrete paths.

### VMD Path on macOS, Linux, and Windows

For any field that asks for a `VMD Path` or `VMD Executable Path`, enter the actual VMD executable path that can be run directly by the software.

Do not rely on:

- shell aliases that exist only in your interactive terminal
- desktop shortcuts
- wrapper entries that work only when expanded by the shell

Platform guidance:

- macOS: use the real executable from inside the VMD application bundle, not `startup.command.csh`
- macOS: to inspect the bundle from Terminal, run `ls -l "/Applications/VMD ... .app/Contents/vmd"` and use the actual binary you find there, for example `vmd_MACOSXARM64`
- macOS: if trajectory extraction produces a missing, empty, or invalid `.vmdtmp.bin` coordinate temp file, do not use `startup.command.csh`; use the actual VMD executable inside the app bundle
- Linux: use the real VMD executable path, or a real executable/symlink that is on `PATH`
- Linux: if `vmd` works in your terminal, run `which -a vmd` to find the executable that should be entered into the software
- Windows: use the actual `vmd.exe` path rather than a shortcut or non-executable launcher entry
- Windows: find the installed `vmd.exe`, typically under `C:\Program Files\VMD\...`, and enter that full path

Generic examples:

- macOS: `/Applications/VMD ... .app/Contents/vmd/vmd_MACOSXARM64`
- Linux: `/usr/local/bin/vmd` or another real executable path returned by `which -a vmd`
- Windows: `C:\Program Files\VMD\...\vmd.exe`

If `vmd` works in your terminal but the generated workflow says it cannot find VMD, that usually means your terminal is using an alias or wrapper that the generated workflow cannot use directly.

### Precision

Each relevant section has a `Precision` button near the `Skip` checkbox. The current setting is always shown next to that button. This applies only to ADMDynAnlz numeric data files, not to external structure or trajectory files such as `PSF`, `DCD`, `VELDCD`, or `XSC`.

Available choices:

- `binary single`
- `binary double`
- `binary custom`
- `text single`
- `text double`
- `text custom`

Meaning:

- `binary`: store or read the file as binary NumPy-style numeric data
- `text`: store or read the file as plain text numeric data
- `single`: use single-precision numeric values
- `double`: use double-precision numeric values
- `custom`: use a user-specified number of decimal places

When `custom` is selected, a small input field appears next to the dropdown. Enter the number of decimal places you want.

Practical notes:

- for output fields, this setting controls how the generated numeric file is written
- for input fields, this setting tells the generated workflow how to read and convert the file before calculation
- when a workflow already has a memory-saving option, such as `Use Memory Map`, that option still affects the in-memory processing strategy

### Number of DCDs

`Number of DCDs` is the total number of trajectory segments that belong to the workflow. The software uses this count to generate the list of files and validate optional DCD selections.

### DCD Selection

Any `DCD Selection` field is optional and lets you restrict processing to only some trajectory indices.

Supported examples:

- `5`
- `0,1,4,7`
- `3-10`
- `1,3-5,9`
- `range(10,20)`
- `[0, 2, 4, 6]`

Indices are zero-based. If the field is left empty, the module processes all DCD indices from `0` to `Number of DCDs - 1`.

### Skip checkboxes

If a section has a `Skip` checkbox, checking it tells the software not to generate that part of the workflow. The section becomes inactive visually and no script for that block is produced.

### Smart Optimization

`Smart Optimization` is optional in all modules. It does not change the scientific method or modify any existing data. It only recommends practical run settings such as:

- chunk size
- memory request
- worker count

The input prompts are different in each module so the recommendation matches that module’s workflow.

For Module 1 Step 3, `COM groups` is optional. If you know the number of COM output groups resolved by the Step 3 PSF selection and grouping unit, enter it for a more accurate memory estimate. If left blank, the optimizer assumes the conservative worst case of up to one output group per selected atom.

### SLURM Submission Parameters

This section exists in all modules. It controls whether a SLURM submit script is generated and what resource requests that script contains.

Fields:

- `Nodes`: number of requested compute nodes
- `Partition`: SLURM partition or queue
- `QOS`: quality-of-service name if your cluster uses one
- `CPUs`: CPU cores per job
- `Tasks`: number of tasks or processes
- `Memory (GB)`: requested memory in GB
- `Walltime`: maximum runtime, for example `24:00:00`
- `Output prefix`: prefix used for SLURM log files
- `Email`: notification address for SLURM mail messages
- `Module`: optional environment module to load before execution

Option:

- `Skip SLURM submission`: generate only the main analysis script and do not generate a SLURM submit script

### Output Files

This section exists in all modules.

Fields:

- `Output folder name`: folder that will be created for this module’s generated files
- `Main script file`: main Python analysis script that will be generated
- `Submit script file`: SLURM submit script file name

Buttons:

- `Generate Scripts`: generates the files for the current module
- `Multi-Run Anlz.`: generates a helper shell script that runs all generated `.sh` files in the output folder

`Multi-Run Anlz.` is useful only when SLURM submit scripts were generated.

## Module 1: Coordinates Extraction

This module prepares coordinate-based trajectory data for later analysis. It is organized into three steps.

## Common Parameters

- `Base Directory`: root folder containing the raw trajectories, PSF files, and output folders
- `Number of DCDs`: total number of input trajectory segments
- `Number of Particles`: number of molecules or analysis units used later in COM generation
- `Common Terms`: two optional shared replacement fields for `*` and `**` in path patterns
- `Max Workers`: maximum CPU workers available for parallel processing

## Step 1: Extraction

This step uses VMD to read trajectory files and write out coordinate data for the selected atoms.

Fields:

- `PSF Pattern`: pattern pointing to PSF structure files
- `DCD Pattern`: pattern pointing to input trajectory DCD files
- `Output Pattern`: full output `path + filename + extension` pattern for extracted coordinate files
- `Target Selections`: VMD atom selection string, such as `water`, `protein`, or a more specific selection
- `?` help button: opens the VMD atom-selection reference in your default browser
- `VMD path`: full path to the VMD executable
- `DCD Selection`: optional subset of DCD indices to extract
- `Stride`: required frame sampling stride; use `1` for every frame, `10` for every tenth frame
- `Frame Chunks: VMD`: required number of saved frames loaded into VMD per Step 1 DCD batch
- `Frame Chunks: Python`: required number of frame rows converted from VMD temporary binary output at once

Options:

- `Use Parallel VMD`: runs multiple extraction jobs in parallel when possible
- `Wrap Coordinates`: applies VMD `pbctools` wrapping during extraction
- `Settings`: opens the wrapping popup where you can configure shape, centering, compound mode, atom selection, and optional wrap flags

When Step 1 `COM` output is enabled and `Grouping Unit = residue`, residues are grouped by VMD `segid + resid` in first-occurrence atom order. This avoids VMD-internal `residue` numbering behavior that can treat each atom as a separate residue for some PSF/PDB inputs.

### What `Wrap Coordinates` does

When `Wrap Coordinates` is enabled, the extracted coordinates are wrapped back into the primary periodic box during Step 1. This is useful when you want a compact wrapped trajectory for later processing or visualization.

This option does not perform continuous unwrapping. That is the purpose of Step 2.

Practical effect:

- enabled: Step 1 output is wrapped into the unit cell
- disabled: Step 1 output remains as directly extracted coordinates without that wrapping step

Use it when your workflow benefits from first producing wrapped coordinates and then reconstructing continuity later. If you already have a suitable wrapped coordinate representation, Step 2 can then convert it into continuous coordinates.

Output unit:

- extracted coordinate files use the same length unit as the input trajectory coordinates

## Step 2: Continuous coordinates

This step removes periodic-boundary discontinuities and reconstructs continuous particle trajectories.

Fields:

- `Input Pattern`: full input `path + filename + extension` pattern for wrapped coordinate files
- `Output Pattern`: full output `path + filename + extension` pattern for continuous-coordinate files
- `XSC file`: file containing simulation box dimensions
- `Num atoms`: total number of atoms in each extracted coordinate file
- `Interval`: optional frame interval expression to restrict the processed frame range
- `Stride`: frame stride used during unwrapping
- `DCD Selection`: optional subset of DCD indices to process
- `Chunk Size`: memory-related processing chunk size, or `auto`

Option:

- `Use Parallel`: enable parallel processing for the unwrapping step
- `Fix 1st Frame`: optionally repair molecules that are already split across periodic boundaries in the first input frame before continuous coordinates are reconstructed

When `Fix 1st Frame` is checked, Step 2 enables extra metadata fields in the Options section:

- `PSF Pattern`: PSF file pattern used to resolve atom grouping for the first-frame repair
- `Target Selection`: VMD atom selection matching the atom order in the Step 2 coordinate input files
- `VMD Path`: VMD executable used to resolve the PSF and atom selection
- `Grouping Unit`: grouping used to define each molecule or unit to repair; currently `residue`, `chain`, or `segname`; `residue` groups by VMD `segid + resid`

### What `Fix 1st Frame` does

Use this option when the Step 2 input trajectory starts from a later part of a longer trajectory and the first saved frame already contains molecules split across periodic boundaries.

When enabled, Step 2:

- resolves the selected atoms and groups using the Step 2 `PSF Pattern`, `Target Selection`, `VMD Path`, and `Grouping Unit`
- repairs the first coordinate frame group by group so each selected group is made whole using minimum-image offsets
- computes the exact per-atom coordinate shifts between the original first frame and the repaired first frame
- applies those same per-atom shifts to every frame in the Step 2 input coordinate file
- then runs the normal continuous-coordinate unwrapping algorithm

The `Target Selection` used for `Fix 1st Frame` must match the atom order in the Step 2 input coordinate files. If the resolved atom count does not match the coordinate file width, the workflow stops with an error.

If `Fix 1st Frame` is unchecked, Step 2 behaves as before and uses the first frame exactly as it appears in the input coordinate file.

What this section does:

- reads wrapped coordinates
- uses the box information from the XSC file
- reconstructs continuous trajectories across periodic boundaries

Output unit:

- continuous-coordinate files use the same length unit as the input trajectory coordinates

Use `auto` for `Chunk Size` when you want the software to choose a safe value, or run `Smart Optimization` first and use the recommended value.

## Step 3: COM calculation

This step computes center-of-mass trajectories from the continuous coordinates.
Step 3 now has its own metadata inputs and does not inherit PSF / selection / VMD settings from Step 1.

Fields:

- `Input Pattern`: full input `path + filename + extension` pattern for continuous-coordinate files from Step 2
- `Output Pattern`: full output `path + filename + extension` pattern for generated COM files
- `PSF Pattern`: PSF file pattern used to resolve atom masses and topology grouping for the COM calculation
- `Target Selection`: atom selection defining which atoms participate in the COM calculation
- `VMD Path`: VMD executable used to interpret the selection and PSF topology metadata
- `COM Mode`: choose `individual` or `collective`
- `Grouping Unit`: used only in `individual` mode; currently `residue`, `chain`, or `segname`; `residue` groups by VMD `segid + resid`
- `DCD Selection`: optional subset of DCD indices to process

Options:

- `Use Parallel`: enable parallel COM calculation
- `Use Memory Map`: use memory mapping to reduce RAM pressure on large files

What this section does:

- resolves the selected atoms from the PSF and target selection
- obtains atom masses from the PSF
- computes center-of-mass positions frame by frame

If `COM Mode = individual`:

- the software resolves the selected atoms and groups them using `Grouping Unit`
- each resolved group gets its own COM per frame
- groups do not need to have the same number of atoms
- selections such as `same residue as ...` are interpreted by VMD before the COM calculation, so full residues/segments/chains can be pulled into the final selection if the selection syntax requests that behavior

Examples:

- if `Grouping Unit = residue`, one COM is computed for each `segid + resid` residue inside the final selection
- if `Grouping Unit = chain`, one COM is computed for each chain inside the final selection
- if the target selection is `same residue as resname TIP3 and index 10 to 90`, the final atom list follows VMD selection semantics before COMs are calculated

If `COM Mode = collective`:

- `Grouping Unit` is not used
- all selected atoms are treated as one single group
- one COM is computed per frame for the whole selection

Output unit:

- COM files use the same length unit as the input trajectory coordinates

## Module 2: Velocities and Dipoles

This module contains two workflows in separate tabs.

- `Velocity Extraction`
- `Dipole Calculations`

You can enable either one or both, depending on the scripts you want to generate.

## Common Parameters

- `Base Directory`: root folder for inputs and outputs
- `Number of DCDs`: total number of trajectory segments
- `Common Terms`: two optional shared replacement fields for `*` and `**` in patterns
- `Max Workers`: maximum CPU workers to use

## Velocity Extraction tab

This workflow generates scripts for extracting center-of-mass velocities from velocity trajectories using VMD.

Fields:

- `PSF Pattern`: pattern for the PSF files
- `VELDCD Pattern`: pattern for velocity-trajectory files
- `Output Pattern`: full output `path + filename + extension` pattern for generated velocity files
- `Target Selection`: VMD atom-selection string for the atoms whose COM velocities should be extracted
- `Grouping Unit`: grouping used to define each COM velocity, currently `residue`, `chain`, or `segname`; `residue` groups by VMD `segid + resid`
- `Stride`: frame sampling stride for velocity extraction
- `VMD Executable Path`: full path to the VMD executable
- `DCD Selection`: optional subset of DCD indices to process

Notes:

- this workflow does not use a module-level `Number of Particles` field
- VMD determines which atoms are included from `Target Selection`
- `Grouping Unit` tells the workflow how those selected atoms are grouped into COM velocities
- for correct results, the PSF and velocity trajectory must describe the same system ordering

What this section does:

- loads PSF and velocity trajectory files
- extracts velocity information for each selected trajectory
- writes generated velocity-analysis workflow files

Output unit:

- velocity files use the same velocity unit as the input velocity trajectory

## Dipole Calculations tab

This workflow can generate either individual dipole scripts or collective dipole scripts.

Field:

- `Calculation Method`: choose `individual` or `collective`
- `PSF Pattern`: full input `path + filename + extension` pattern for the PSF used by both dipole methods
- `Target Selection`: VMD atom selection defining the atoms included in dipole calculations
- `VMD Path`: full path to the VMD executable
- `Stride`: frame sampling stride used by both individual and collective dipole calculations
- `Dipole Unit`: output dipole unit, either `Debye` or `e·Å`
- `DCD Selection`: optional subset of DCD indices to process

### If `Calculation Method = individual`

Use these fields:

- `Coordinates Pattern`: full input `path + filename + extension` pattern for coordinate files
- `All neutral particles`: check only if every net charge per selected group is zero
- `COM Patterns`: full input `path + filename + extension` pattern for center-of-mass files used when `All neutral particles` is not checked
- `Grouping Unit`: grouping used to define each individual dipole: `residue`, `chain`, `segname`, or `all`; `residue` groups by VMD `segid + resid`
- `Dipole Vectors Pattern`: full output `path + filename + extension` pattern for dipole-vector files
- `Magnitudes Pattern`: optional full output `path + filename + extension` pattern for dipole-magnitude files
- checkbox next to `Magnitudes Pattern`: enables or disables dipole-magnitude calculation and saving

What it does:

- resolves atom masses, charges, and group membership from the PSF and target selection
- loads atomic coordinate data for the selected atoms
- if `All neutral particles` is checked, computes dipoles directly from atomic coordinates without subtracting centers of mass
- if `All neutral particles` is not checked, loads the COM file and subtracts one COM per resolved group before computing dipoles
- computes one dipole vector per resolved group for each frame
- always computes dipole vectors for each frame
- computes and saves dipole magnitudes only if the `Magnitudes Pattern` checkbox is enabled

Output units:

- dipole vector files are written in the selected `Dipole Unit`
- dipole magnitude files are written in the selected `Dipole Unit` when enabled

How individual molecules are distinguished:

- this workflow groups atoms using `Grouping Unit`
- if `Grouping Unit = residue`, one dipole is computed for each `segid + resid` residue inside the final selection
- if `Grouping Unit = chain`, one dipole is computed for each chain inside the final selection
- if `Grouping Unit = all`, one dipole is computed from all selected/input coordinates
- groups do not need to have the same number of atoms
- selections such as `same residue as ...` are interpreted by VMD before dipoles are computed

Important input note:

- `Coordinates Pattern` must point to per-atom coordinates, for example `xyz` or `unwr_xyz`
- do not use COM-coordinate files such as `com_xyz` as the dipole coordinate input
- the coordinate file must match the atom selection resolved from the PSF
- when `All neutral particles` is not checked, the `COM Patterns` file must contain one COM vector per resolved group for each frame

### If `Calculation Method = collective`

Use these fields:

- `DCD Pattern`: full input `path + filename + extension` pattern for collective dipole analysis
- `Output Pattern`: full output `path + filename + extension` pattern for the collective dipole result
- `Options`: enables or disables parallel VMD execution and optional pbctools coordinate wrapping before collective dipoles are measured

What it does:

- loads the selected atoms in VMD
- computes the collective dipole of the selected group
- writes output for the full selected set rather than per-molecule dipoles

Notes:

- shared fields stay active regardless of `Calculation Method`
- method-specific fields are grouped at the top of the dipole section
- `Skip` disables the dipole block entirely

Output units:

- the collective output file contains frame index, dipole vector components, and dipole magnitude
- both vector components and magnitude are written in the selected `Dipole Unit`

## Module 3: MSD and NGP / Anisotropic NGP

This module generates scripts for mean-squared displacement and non-Gaussian analysis from COM trajectory data.

## Common Parameters

- `Base Directory`: root folder for the COM files and generated outputs
- `Number of DCDs`: total number of input trajectory segments
- `Number of Particles`: number of particles represented in the COM files
- `Common Terms`: two optional shared replacement fields for `*` and `**` in file patterns
- `Max Workers`: maximum CPU workers available

## α₂(t) / α_{ani}(t) Calculation

This block generates one analysis workflow at a time depending on the selected calculation type.

Fields:

- `Calculation Type`: choose `alpha2_msd` or `alpha_xz`
- `Input File Pattern`: full input `path + filename + extension` pattern for COM files
- `Alpha2 Output File Pattern`: shown when `Calculation Type = alpha2_msd`; full output `path + filename + extension` for `α₂(t)` results
- `MSD Output File Pattern`: shown when `Calculation Type = alpha2_msd`; full output `path + filename + extension` for MSD results
- `Anisotropic NGP Output File Pattern`: shown when `Calculation Type = alpha_xz`; full output `path + filename + extension` used as the base for anisotropic NGP result files
- `Min frames`: optional minimum number of frames expected in each input file
- `DCD Selection`: optional subset of DCD indices to process

Option:

- `Validate Data`: enables input-data validation checks during the generated calculation

### If `Calculation Type = alpha2_msd`

The generated workflow computes:

- MSD
- NGP, written as `α₂(t)`

Use this when you want translational dynamics and non-Gaussian behavior from COM trajectories.

The two output fields are written independently. For example, you can provide:

```text
Alpha2 Output File Pattern = anlz/NVT_*/analysis/ngp_results_a2.dat
MSD Output File Pattern = anlz/NVT_*/analysis/ngp_results_MSD.dat
```

and the generated workflow will save each result to the matching path.

Output units:

- MSD output is in squared length, using the square of the coordinate unit
- `α₂(t)` output is dimensionless

### If `Calculation Type = alpha_xz`

The generated workflow computes anisotropy-related quantities, including the anisotropy parameter `α_{ani}(t)`.

Use this when you want directional anisotropy information rather than the standard `α₂(t)` and MSD workflow.

The `Anisotropic NGP Output File Pattern` is used as the base name, and the anisotropy workflow generates related result files derived from that base name. For example:

```text
Anisotropic NGP Output File Pattern = anlz/NVT_*/analysis/ani_results.dat
```

generates result files such as `ani_results_axz.dat`, `ani_results_axy.dat`, `ani_results_ayz.dat`, and `ani_results_anisotropy.dat`.

Output units:

- `α_xz(t)`, `α_xy(t)`, `α_yz(t)`, and the anisotropy parameter are dimensionless

### What `Validate Data` does

When enabled, the generated calculation includes validation checks on the input data before or during processing. This is helpful when you want extra protection against malformed or inconsistent trajectory-derived files.

When disabled, the workflow runs without those additional checks and may be faster on already trusted inputs.

## Module 4: Correlation Functions

This module generates scripts for time-correlation analysis from numeric array files that already exist on disk.

## Common Parameters

- `Base Directory`: root folder for correlation inputs and outputs
- `Num. Corr. Functions`: total number of indexed correlation functions to calculate
- `Number of Particles`: retained for consistency with other modules
- `Particles Selection`: optional zero-based subset of particles to use from multi-particle inputs; supports forms such as `0`, `1,3,5`, `0-10`, `[0, 2, 4]`, and `range(0, 10)`
- `Common Terms`: two optional shared replacement fields for `*` and `**` in file patterns
- `Max Workers`: maximum independent correlation functions to calculate at the same time

## Correlation Function Parameters

Fields:

- `Array 1 Pattern`: full input `path + filename + extension` pattern for the first numeric array
- `Array 2 Pattern`: full input `path + filename + extension` pattern for the second numeric array
- `Corr. Func. Output`: full output `path + filename + extension` pattern for the normalized correlation results
- `Variance Output`: full output `path + filename + extension` pattern for the saved variance values
- `Shift (delta)`: frame or sample stride used in the correlation sum
- `Max Length (num. frames)`: largest lag index to evaluate
- `Step (between frames)`: physical step associated with one frame increment
- `Coefficient`: scaling factor used only for the saved variance output; the standard choice is `3`
- `DCD Selection`: optional subset of indices to process
- `Data Type`: choose `scalar` or `vector`
- `Particle Count`: choose `single` or `multiple`
- `Correlation Function Mode`: choose `Subtracting Mean` or `Not Subtracting Mean`

## Precision

The correlation module provides precision settings for:

- `Array 1 input`
- `Array 2 input`
- `Output files`

These settings control whether the generated script reads and writes text or binary numeric files and how numeric precision is handled.

## How the type and mode choices work

### Scalar vs vector

- `scalar`: use this when each frame contains scalar values
- `vector`: use this when each frame contains 3-component vectors

### Single vs multiple

- `single`: one scalar time series or one vector time series per file
- `multiple`: many scalar variables or many vectors contained in the same file, averaged together in the final correlation

### Correlation Function Mode

- `Subtracting Mean`: subtracts the mean from each input before computing the correlation
- `Not Subtracting Mean`: computes the correlation directly from the raw input without subtracting the mean

### Correlation normalization

- the saved correlation function is normalized as `C(t)/C(0)`
- the saved variance output stores `C(0) * Coefficient^2`
- the `Coefficient` is not applied to the saved correlation-function curve itself

Output units:

- the first output column uses the unit implied by `Step (between frames)`
- `Step (between frames)` determines the unit of the horizontal axis, meaning the independent variable shown along the correlation-function x-axis, such as time
- the second output column uses the product of the units of the two input arrays

## Current limitation

The current module supports:

- scalar-scalar correlations
- vector-vector correlations

It does not currently support mixed scalar/vector pairs in one calculation.

## Choosing output folder and script names

A good practice is to use a distinct `Output folder name` for each module or each run configuration, for example:

- `coords_output`
- `vel_dipole_output`
- `ngp_output`
- `corr_output`

This keeps generated scripts and submit files separate and makes `Multi-Run Anlz.` easier to use.

## Recommended workflow order

For a standard coordinate-to-analysis workflow:

1. Use `Coordinates extraction` to generate extracted, continuous, and COM data.
2. Use `Velocities and Dipoles` if you need velocity or dipole workflows.
3. Use `MSD and NGP / Anisotropic NGP` on the COM outputs.
4. Use `Correlation Functions` when you already have numeric scalar or vector arrays and want correlation-analysis scripts for them.
5. Use `Diffusion constant` when you already have saved VACF files, MSD files, or both, and want diffusion estimates from either route.
6. Use `Static Structure Factor` when you want isotropic `S(k)` from saved coordinate arrays.
7. Use `Radial Distribution Function` when you want `g(r)` from saved coordinate or COM arrays.

## Module 5: Diffusion Constant

This module generates scripts for computing diffusion constants from saved VACF data, saved MSD data, or both.

The module has two independent subsections:

- `D VACF`
- `D MSD`

Each subsection has its own enable checkbox. At least one must be enabled.

## Common Parameters

- `Base Directory`: root folder used to resolve relative VACF paths, MSD paths, and the analysis output file path
- `Number of VACFs`: number of VACF files to use when the resolved VACF input points to multiple files
- `Common Terms`: two optional shared replacement fields for `*` and `**`

Important behavior:

- `VACF Path`, `MSD Path`, and `Analysis Output File` support `*` and `**`
- when `VACF Path` resolves to multiple files, the files are sorted and the first `Number of VACFs` files are used
- when `Analysis Output File` is relative, it is interpreted relative to `Base Directory`

## Diffusion Constant Parameters

### D VACF

Fields:

- `VACF Path`: single file, directory, or glob pattern for the saved VACF data
- `Time Axis Exist`: tells the software whether the VACF file already contains a time column
- `Saved Frame dt`: used only when `Time Axis Exist` is unchecked
- `Time Axis Unit`: currently recorded as input metadata for the user; the actual parsing decision is controlled by `Time Axis Exist`
- `Normalized VACF`: tells the software whether the VACF has already been divided by its `t = 0` value
- `Velocity Units`: `namd_internal` or `angstrom_per_ps`
- `Variance`: required when `Normalized VACF` is checked; used for the VACF-based diffusion estimate in `A^2/ps^2`

How VACF input is interpreted:

- if `Time Axis Exist` is checked:
  - column 1 is used as time
  - column 2 is used as VACF
- if `Time Axis Exist` is unchecked:
  - the software builds the time axis as `t = (i - 1) * dt`
  - the VACF is read from the numeric series in the file

How normalization and units are handled:

- if `Normalized VACF` is unchecked:
  - the software reads the VACF as an unnormalized physical VACF
  - if `Velocity Units = namd_internal`, velocities are converted using the built-in factor before the variance-based diffusion estimate is formed
  - each VACF is normalized internally by its own `VACF[0]` before integrating `tau`
  - `VACF var, ang^2/ps^2` is reported in the output
- if `Normalized VACF` is checked:
  - the software assumes the input is already normalized
  - the velocity-unit menu is disabled and no unit conversion is applied
  - the `Variance` field becomes active and must be provided by the user
  - the diffusion estimate uses the user-provided variance
  - `VACF var, ang^2/ps^2` in the output corresponds to the user-provided variance
  - the equipartition variance is still reported separately for comparison

### D MSD

Fields:

- `MSD Path`: single file path or pattern for the MSD data
- `Time Axis Exist`: tells the software whether the MSD file already contains a time column
- `Saved Frame dt`: used only when `Time Axis Exist` is unchecked
- `Time Axis Unit`: recorded as user metadata

How MSD input is interpreted:

- if `Time Axis Exist` is checked:
  - column 1 is used as time
  - column 2 is used as MSD
- if `Time Axis Exist` is unchecked:
  - the numeric MSD table is flattened into one MSD series
  - the time axis is built as `t = (i - 1) * dt`
  - this matches the current Mathematica-style workflow used in the project

How `D MSD` is calculated:

- the software fits `MSD(t) = a t` through the origin
- then reports `D = a / 6`

### Shared lower fields

- `Temperature (K)`: used in the equipartition comparison and optional infinite-size correction
- `Molar Mass (g/mol)`: used in the equipartition comparison
- `inf. size correction`: enables the optional infinite-size correction
- `Parameters`: opens the infinite-size correction popup
- `Analysis Output File`: output text file written by the generated workflow

Temperature handling:

- the temperature field accepts one value or multiple values separated by commas, spaces, or semicolons
- when multiple temperatures are given, the generated script handles all of them in one Python script
- if the temperature list exactly matches `Common Term 1` or `Common Term 2`, the workflow does not nest those loops separately
- instead, each temperature is paired with the corresponding common-term entry
- in multi-temperature runs, the output files receive a temperature suffix such as `_T300K`

### Infinite size correction parameters

Clicking `Parameters` opens a window titled `Infinite size correction parameters`.

That window contains:

- a warning that the correction is not recommended for non-cubic or strongly non-cubic cells
- `Averaged Cubic Side Length (Å)`
- one viscosity field per temperature, for example `η(300K)`, `η(320K)`, and so on

How the correction is used:

- the software first computes the raw diffusion constants
- then it applies the infinite-size correction separately for each temperature
- if both `D VACF` and `D MSD` are enabled, the correction is applied to both

## Output contents

The output file is plain text. It does not contain a trailing numeric data table.

Its high-level structure is:

- title line
- `Inputs:` section
- `Results:` section
- optional `Results per VACF file:` and `Average results:` blocks when multiple VACF files are used
- optional `D with Infinite size correction` section

### Inputs section

The `Inputs:` section can include:

- `source`
- `time axis exists`
- `time axis unit`
- `VACF timestep, ps`
- `velocity units`
- `velocity scale to angstrom/ps`
- `VACF already normalized`
- `MSD source`
- `MSD timestep, ps`
- `MSD time axis exists`
- `MSD time axis unit`
- `D VACF enabled`
- `D MSD enabled`
- `T, K`
- `molar mass, g/mol`
- `viscosity, cP`
- `averaged cubic side length, ang`

Some lines appear only when they are relevant to the enabled subsection or correction settings.

### Results section

Possible result lines include:

- `D VACF, ang^2/ps`
- `D VACF, cm^2/s`
- `tau, ps`
- `VACF var, ang^2/ps^2`
- `equipartition var, ang^2/ps^2`
- `variance difference, %`
- `D MSD, ang^2/ps`
- `D MSD, cm^2/s`

Important details:

- `VACF var, ang^2/ps^2` is written only when the VACF input was not marked as already normalized
- `equipartition var, ang^2/ps^2` is reported for the VACF route because it is part of the comparison against the measured or inferred variance
- for a single VACF file, the results section directly reports the final single-file values
- for multiple VACF files, the file reports:
  - `Results per VACF file:`
  - one line per input VACF
  - `Average results:`

### Multiple VACF files

When multiple VACF files are used:

- each VACF file is processed separately
- `tau` and `D VACF` are computed separately for each file
- the final reported `D VACF` and `tau` are the averages over those per-file values

### Infinite size correction output

If the correction is enabled, the output file ends with:

- a blank line
- `D with Infinite size correction`
- `D VACF = ...` when the VACF route is enabled
- `D MSD = ...` when the MSD route is enabled

## Units summary

- `VACF timestep, ps`: picoseconds
- `MSD timestep, ps`: picoseconds
- `T, K`: Kelvin
- `molar mass, g/mol`: grams per mole
- `Averaged Cubic Side Length (Å)`: angstrom
- `viscosity, cP`: centipoise
- `D VACF, ang^2/ps` and `D MSD, ang^2/ps`: angstrom squared per picosecond
- `D VACF, cm^2/s` and `D MSD, cm^2/s`: centimeters squared per second
- `tau, ps`: picoseconds
- `VACF var, ang^2/ps^2` and `equipartition var, ang^2/ps^2`: angstrom squared per picosecond squared

## Module 6: Static Structure Factor

This module generates scripts for density structure factors and charge-dipole structure factors from saved numeric trajectory data.

## Common Parameters

- `Base Directory`: root folder used to resolve relative input and output paths
- `Common Terms`: optional `*` and `**` replacements, exactly like the other modules
- `Number of Trajectories`: optional limit on how many discovered trajectory files or file sets are used; leave blank to use all discovered inputs
- `Max Workers`: maximum worker processes used when the density structure-factor calculations average over multiple trajectory files
- `Resume`: when `Use existing per-trajectory outputs` is checked, trajectories whose per-trajectory report files already exist and match the current `k` grid are reused; use this only when rerunning the same inputs/settings after an interrupted job; missing trajectories are computed, and the final total includes both reused and newly computed reports

## Density Structure Factor Tab

This tab contains three density-based calculations:

- `Isotropic`
- `Directional`
- `Along k components`

All three use saved coordinate arrays, not raw DCD files.

### Shared Density-Tab Fields

- `Coordinate Path`: path pattern for coordinate arrays; supports `*`, `**`, and `{i}`
- `Stride`: frame stride applied before the structure-factor calculation; `1` means use every saved frame
- `Output Path`: final output file written by that calculation
- `L_x`, `L_y`, `L_z`: orthorhombic box lengths used to build reciprocal-space vectors and to apply minimum-image distances
- `Shell Width`: positive tolerance used when nearby `|k|` values are grouped together
- `CO-1`, `CO-2`, `CO-3`: optional real-space cutoffs applied to the first, second, and third `k` tiers
- `CS-1`, `CS-2`, `CS-3`: optional cell-list sizes paired with `CO-1`, `CO-2`, and `CO-3`; each cell-size value is used only for the matching cutoff tier
- `k Max 1`, `k Max 2`, `k Max 3`: upper bounds for the first, second, and third reciprocal-space tiers
- `Resolution: res. 1`, `res. 2`, `res. 3`: minimum `|k|` spacing kept in the first, second, and third reciprocal-space tiers
- `Trajectory desired length`: optional field; a single integer means use the first `N` frames from each file; `range(start, stop[, step])` means use that frame window from each file
- `Trajectory Selection`: optional zero-based subset of trajectory indices to process from `Number of Trajectories`; supports the same syntax as `DCD Selection`
- `Chunk Sizes`: opens the dialog for density-structure-factor chunk sizes

### What The Three `k` Tiers Mean

The density calculations build one reciprocal-space sampling list in three contiguous ranges:

- tier 1: `0 < |k| <= k Max 1`
- tier 2: `k Max 1 < |k| <= k Max 2`
- tier 3: `k Max 2 < |k| <= k Max 3`

Rules:

- `k Max 1` may be `0`; `k Max 2` and `k Max 3` must be positive
- the values must satisfy `k Max 1 <= k Max 2 <= k Max 3`
- each tier has its own `k` resolution and optional cutoff
- each tier can also have its own optional cell size for the cutoff-based neighbor search
- if `k Max 2`, `k Max 3`, `res. 2`, or `res. 3` are left empty, the generated script reuses the previous tier value

Practical meaning:

- smaller resolution, such as `0.05`: keep more closely spaced reciprocal-space magnitudes or vectors in that tier
- larger resolution, such as `0.20`: keep a sparser `k` sampling in that tier
- smaller `k Max`: lower reciprocal-space coverage
- larger `k Max`: more reciprocal-space coverage and more work

### What `k_x`, `k_y`, and `k_z` Mean

The directional and component-based calculations use explicit reciprocal-space vectors

```text
k = (k_x, k_y, k_z)
```

with

```text
k_x = 2*pi*n_x/L_x
k_y = 2*pi*n_y/L_y
k_z = 2*pi*n_z/L_z
```

where the generated density-tab vectors use non-negative integers `n_x`, `n_y`, and `n_z`, and the all-zero vector is excluded.

So:

- `k_x`, `k_y`, `k_z` are the Cartesian components of the sampled reciprocal-space vector
- `|k| = sqrt(k_x^2 + k_y^2 + k_z^2)` is its magnitude
- `L_x`, `L_y`, `L_z` set the spacing of allowed reciprocal-space values along each box direction

### Isotropic

This calculation averages the density structure factor over unique `|k|` values.

How the `k` list is built:

- candidate reciprocal-space magnitudes are generated from the box lengths and the three `k Max` tiers
- nearby magnitudes are merged using `Shell Width`
- the final isotropic table is written as one row per unique `|k|`

Main output format:

- column 1: `|k|`
- column 2: `S(k)`

Notes:

- this mode produces an isotropic average, so the output does not preserve the original `k_x`, `k_y`, `k_z` direction of each reciprocal vector
- `CO-1`, `CO-2`, and `CO-3` apply to pair distances in real space for the matching `k` tier

### Directional

This calculation keeps each sampled reciprocal vector explicitly instead of collapsing everything to `|k|`.

Main output format:

- column 1: `k_x`
- column 2: `k_y`
- column 3: `k_z`
- column 4: `|k|`
- column 5: `S(k)`

Use this when you want anisotropic information, for example to distinguish different reciprocal-space directions that have the same `|k|`.

In the current implementation, the directional output keeps every sampled `k` vector as its own row, so `Shell Width` does not merge rows in this mode.

### Along `k` Components

This calculation restricts the reciprocal-space vectors to selected axis combinations and writes one output file per selection.

Field:

- `Components`: comma-separated selections such as `x`, `y`, `z`, `x+y`, `x+z`, `y+z`, or `x+y+z`

How selections are interpreted:

- `x`: only the `x` direction is active, so vectors are of the form `(k_x, 0, 0)` with `k_x > 0`
- `y`: vectors are `(0, k_y, 0)` with `k_y > 0`
- `z`: vectors are `(0, 0, k_z)` with `k_z > 0`
- `x+y`: vectors are `(k_x, k_y, 0)` with `k_x` and `k_y` sampled and `k_z = 0`
- `x+z`: vectors are `(k_x, 0, k_z)`
- `y+z`: vectors are `(0, k_y, k_z)`
- `x+y+z`: vectors are `(k_x, k_y, k_z)`

Important meaning:

- a component selection names which reciprocal-space axes are allowed to vary
- it does not mean arithmetic addition of already-computed one-dimensional curves
- for mixed selections such as `x+y`, the calculation still uses the full vector magnitude internally

Output naming:

- if you request one component only, the exact `Output Path` is used
- if you request multiple components, the software appends the component label before the extension, for example `output_kx.dat`, `output_ky.dat`, or `output_kxy.dat`

Output format for one-axis selections:

- column 1: `|k_x|`, `|k_y|`, or `|k_z|` for the selected axis
- column 2: `S(k)`

Output format for multi-axis selections such as `x+y`, `x+z`, `y+z`, and `x+y+z`:

- column 1: shell-averaged `|k|`
- column 2: shell-averaged `S(k)`

In other words:

- one-axis selections stay one-dimensional in the chosen Cartesian component
- multi-axis selections are grouped by total magnitude using `Shell Width`

### Density-Tab Auxiliary Files

The generated scripts may also write:

- per-trajectory reports next to the main output, with names like `output_1.dat`, `output_2.dat`, and so on
- `status.log` in the output directory

Each per-trajectory report contains the same column layout as the corresponding main output, but for one trajectory file only.

## Charge-Dipole Structure Factor Tab

This tab computes charge-dipole structure factors from:

- one charge-values file
- one set of charge-coordinate files
- one set of dipole-position files
- one set of dipole-vector files

The charge-coordinate, dipole-position, and dipole-vector patterns must resolve to the same number of files.

The charge-values file is normally a constant-charge table. If the file has more than one column, the calculation uses the last column as the charge value. For two-column files, the first column may be a charge-site index for human bookkeeping, but it is not used to reorder coordinates; charge rows are expected to match the charge-coordinate site order. If optional `Mask Charge Index/Indices` is set, those charge values are set to `0.0` before nonzero-charge filtering.

Charge sites with charge exactly `0.0` are ignored before the charge-dipole calculation starts. This means they do not contribute to `Sqp(k)`, and they also do not define cutoff neighborhoods. With a cutoff enabled, only dipoles within the cutoff distance of nonzero charge sites are included.

### Charge-Dipole Fields

- `Calculation Mode`: either `Directional` or `Isotropic`
- `Small k approx`: when checked, uses the first-order small-`k` approximation for the selected charge-dipole mode
- `Charge Values Path`: file containing the charge values; if multiple columns are present, the last column is used
- `Charge Coordinates Path`: coordinate pattern for charge positions
- `Charge Coordinates Stride`: frame stride for the charge-coordinate input
- `Dipole Positions Path`: coordinate pattern for dipole positions
- `Dipole Positions Stride`: frame stride for dipole-position input
- `Dipole Vectors Path`: vector pattern for dipole directions; the GUI keeps its stride matched to the dipole-position stride
- `Isotropic Output Path`: isotropic charge-dipole output file
- `Directional Output Path`: directional charge-dipole output file; required only in directional mode
- `CS-1`, `CS-2`, `CS-3`: optional cell-list sizes paired with `CO-1`, `CO-2`, and `CO-3`
- `Mask Charge Index/Indices`: optional zero-based charge index or comma-separated charge indices whose charge values are set to `0.0` before calculation; for example, `0, 67` masks Python entries `charges[0]` and `charges[67]`; the charge-coordinate array is not changed
- `Trajectory desired length`: optional field with the same syntax as the density tab
- `Trajectory Selection`: optional zero-based subset of trajectory indices to process from `Number of Trajectories`; supports the same syntax as `DCD Selection`

### Charge-Dipole `k` Parameters

The charge-dipole tab uses the same three `k`-resolution tiers in both directional and isotropic modes.

- `Cell 1`, `Cell 2`, and `Cell 3` define the orthorhombic cell dimensions used for the first, second, and third `k` tiers.
- `Cell 1` must contain three positive lengths. If `Cell 2` or `Cell 3` is left empty, it reuses the previous tier's cell dimensions.
- `Resolution: res. 1`, `res. 2`, `res. 3` are the minimum spacing kept between neighboring `|k|` values within the first, second, and third tiers.
- Directional mode keeps all reciprocal-vector directions whose `|k|` value survives that tier's resolution filter.
- The same cell dimensions, `k` tiers, cutoffs, and cell-list settings are used by both full charge-dipole formulas and the `Small k approx` formulas.

For all charge-dipole calculations, the distance vector is defined as:

$$
{\Huge
\mathbf r_{q,p} = \mathbf r_p - \mathbf r_q
}
$$

where `q` is the charge site and `p` is the dipole position. This vector points from the charge site to the dipole position.

### Charge-Dipole Directional Mode

This mode writes both a directional table and an isotropic shell average derived from it.

With `Small k approx` unchecked, the directional charge-dipole structure factor is:

$$
{\Huge
S_{qp}(\mathbf k) =
i\sum_q\sum_p
z_q\left(\hat{\mathbf e}_p\cdot\hat{\mathbf k}\right)
\exp\!\left(i\mathbf k\cdot\mathbf r_{q,p}\right)
}
$$

With `Small k approx` checked, the first-order small-`k` directional approximation is:

$$
{\Huge
S_{qp}(\mathbf k) \simeq
-k\sum_q\sum_p
z_q\left(\hat{\mathbf e}_p\cdot\hat{\mathbf k}\right)
\left(\hat{\mathbf k}\cdot\mathbf r_{q,p}\right)
}
$$

The small-`k` directional approximation is real-valued. The output still keeps the same real/imaginary column layout as the full directional calculation; the imaginary column is zero in small-`k` mode.

Directional output format:

- column 1: `k_x`
- column 2: `k_y`
- column 3: `k_z`
- column 4: `|k|`
- column 5: real part of the charge-dipole structure factor
- column 6: imaginary part of the charge-dipole structure factor

The directional values are raw accumulated values; they are not divided by the number of frames, trajectories, or dipoles.

Isotropic output format produced from the directional data:

- column 1: shell-averaged `|k|`
- column 2: real part
- column 3: imaginary part
- column 4: number of directional `k` vectors contributing to that shell

`Shell Width` is required in this mode because it controls that shell average.

### Charge-Dipole Isotropic Mode

This mode writes one isotropic table directly.

With `Small k approx` unchecked, the isotropic charge-dipole structure factor is:

$$
{\Huge
S_{qp}(k) =
-\sum_q\sum_p
z_q\left(\hat{\mathbf e}_p\cdot\hat{\mathbf r}_{q,p}\right)
j_1\!\left(k|\mathbf r_{q,p}|\right)
}
$$

With `Small k approx` checked, the first-order small-`k` isotropic approximation is:

$$
{\Huge
S_{qp}(k) \simeq
-\frac{k}{3}
\sum_q\sum_p
z_q\left(\hat{\mathbf e}_p\cdot\mathbf r_{q,p}\right)
}
$$

Output format:

- column 1: `|k|`
- column 2: raw accumulated charge-dipole structure factor, `Sqp(k)`; it is not divided by the number of frames, trajectories, or dipoles
- column 3: average number of dipoles included by `CO-1`
- column 4: average number of dipoles included by `CO-2`
- column 5: average number of dipoles included by `CO-3`

The output begins with `#` header lines, so NumPy text loaders skip them by default. The header records the charge/dipole input paths, strides, box lengths, `k` tiers, cutoffs, cell sizes, frame/trajectory counts, and column labels.

In this mode, `Shell Width` is not used by the calculation.

### Charge-Dipole Cutoffs And Extra Output

`CO-1`, `CO-2`, and `CO-3` are optional real-space cutoffs matched to the three `k` tiers.

`CS-1`, `CS-2`, and `CS-3` are the matching cell sizes used with those cutoff tiers. If a cutoff is provided and the matching cell size is left empty, the generated workflow falls back to using the cutoff value as the cell size.

Cutoffs are applied to distances between dipole positions and charge-coordinate sites whose charge value is nonzero. Zero-charge coordinate sites are masked out before the neighbor search, so they cannot add nearby dipoles to the cutoff-limited `Sqp(k)`.

When a cutoff is used:

- the script tracks how many dipoles are inside the cutoff for each processed frame
- it writes one `dipole_count_<trajectory>.dat` file per processed trajectory in a `dipole_counts` directory next to the charge-dipole output

### Charge-Dipole Auxiliary Files

The generated scripts may also write:

- per-file-set reports such as `output_1.dat`, `output_2.dat`, and so on
- `status.log`
- `dipole_counts/dipole_count_<trajectory>.dat`

## Static Structure Factor Output File Structures

This section summarizes the output tables for all static-structure-factor calculations. Unless a file explicitly begins with `#` header lines, the output is a plain numeric table with one data row per sampled `k` value, `k` vector, or shell.

### Density Isotropic Output

Rows:

- one row per retained isotropic `|k|` value
- rows are ordered by increasing `|k|`

Columns:

- column 1: `|k|`, the magnitude of the reciprocal-space vector
- column 2: `S(k)`, the isotropic density structure factor averaged over processed frames and trajectory files

Per-trajectory reports use the same two-column layout and are combined into the final output using frame-weighted averaging.

With resume enabled, an existing per-trajectory density report is reused only if its `|k|` column matches the current settings. The code still reads the matching coordinate file metadata so the final frame-weighted average uses the correct frame count.

### Density Directional Output

Rows:

- one row per sampled reciprocal-space vector
- vectors with the same `|k|` remain separate if their directions differ

Columns:

- column 1: `k_x`
- column 2: `k_y`
- column 3: `k_z`
- column 4: `|k|`
- column 5: `S(k)` for that explicit vector

Per-trajectory reports use the same five-column layout and are combined into the final output using frame-weighted averaging.

With resume enabled, an existing per-trajectory directional report is reused only if its `k_x`, `k_y`, `k_z`, and `|k|` columns match the current settings. The final output is rebuilt from all reused and newly computed reports.

### Density Along-Components Output

For one-axis selections such as `x`, `y`, or `z`:

- each row is one allowed value along the selected axis
- column 1 is `|k_x|`, `|k_y|`, or `|k_z|`
- column 2 is `S(k)` for that one-dimensional selection

For multi-axis selections such as `x+y`, `x+z`, `y+z`, or `x+y+z`:

- each row is one shell-averaged total `|k|`
- column 1 is shell-averaged `|k|`
- column 2 is shell-averaged `S(k)`

When multiple component selections are requested, each selection gets its own output file by appending a component label such as `_kx`, `_ky`, `_kxy`, or `_kxyz` to the requested output path.

Resume is checked separately for each component output file, so one component can reuse existing per-trajectory reports while another component computes missing reports.

### Charge-Dipole Directional Output

Directional mode writes two main output files.

The directional output file has one row per sampled reciprocal-space vector:

- column 1: `k_x`
- column 2: `k_y`
- column 3: `k_z`
- column 4: `|k|`
- column 5: real part of `Sqp(k)`
- column 6: imaginary part of `Sqp(k)`

The isotropic output file produced from directional mode has one row per shell:

- column 1: shell-averaged `|k|`
- column 2: shell-averaged real part
- column 3: shell-averaged imaginary part
- column 4: number of directional `k` vectors contributing to that shell

In directional small-`k` mode, the imaginary column is kept for format consistency and is written as zero.

With resume enabled, the directional per-trajectory report is the file that controls reuse. Its `k_x`, `k_y`, `k_z`, and `|k|` columns must match the current settings, and the matching `dipole_counts/dipole_count_<trajectory>.dat` file must exist for the same frame selection.

### Charge-Dipole Isotropic Output

Direct isotropic charge-dipole mode writes one main output file. It begins with `#` header lines that record input paths, strides, box lengths, `k` tiers, cutoff settings, masked zero-based charge index/indices, frame counts, and trajectory counts. Numeric loaders such as NumPy skip those header lines by default.

Rows:

- one row per generated isotropic `|k|` value
- rows are ordered by increasing `|k|`

Columns:

- column 1: `|k|`
- column 2: raw accumulated `Sqp(k)`
- column 3: average number of dipoles included by `CO-1`
- column 4: average number of dipoles included by `CO-2`
- column 5: average number of dipoles included by `CO-3`

The `Sqp(k)` value is raw accumulated output: it is not divided by the number of frames, trajectory files, or dipoles.

With resume enabled, direct isotropic charge-dipole per-trajectory reports are reused only if their `|k|` column matches the current settings. The matching dipole-count file is also required so the final `CO-1`, `CO-2`, and `CO-3` average-count columns can include reused trajectories.

### Charge-Dipole Dipole-Count Outputs

When charge-dipole calculations run, the workflow writes one `dipole_counts/dipole_count_<trajectory>.dat` file per processed trajectory. These files begin with `#` header lines describing the trajectory index and cutoff values.

Rows:

- one row per processed frame in that trajectory
- `Frame` is the processed-frame counter in the charge-dipole workflow, starting from `1`

Columns:

- column 1: `Frame`
- column 2: number of unique dipoles included by `CO-1`
- column 3: number of unique dipoles included by `CO-2`
- column 4: number of unique dipoles included by `CO-3`

If a cutoff tier is not set, its count column may remain zero or unused for that tier.

### Status Logs And Per-Trajectory Reports

For long structure-factor jobs, `status.log` records periodic progress lines with elapsed time, processed frame count, coordinate or charge count, number of `k` values or vectors, and, for charge-dipole runs, the current dipole count.

Per-trajectory or per-file-set reports are saved next to the main output using names derived from the output path, such as `output_1.dat` and `output_2.dat`. These reports use the same column layout as the corresponding main output, but contain only one trajectory or one charge/dipole file set.

When resume is enabled and an existing per-trajectory report has the wrong number of columns, a different `k` grid, or a missing/mismatched charge-dipole dipole-count file, the run stops with a clear resume error instead of silently mixing incompatible data.

## Extra Module: Radial Distribution Function

The `Radial Distribution Function` module generates scripts for computing `g(r)` from saved coordinate arrays.

## RDF Common Parameters

- `Base Directory`: root folder for RDF inputs and relative outputs
- `Number of Coordinate Files`: number of indexed coordinate files; indices are zero-based, from `0` to `N - 1`
- `Common Terms`: two optional replacement fields for `*` and `**`
- `Max Workers`: maximum coordinate files to process at the same time
- `Smart Optimization`: estimates RDF pair workload and memory from frames, particle count, selections, available memory, and CPU workers; it applies recommended `Max Workers`, `Chunk Size 1`, `Chunk Size 2`, and SLURM CPU/memory/walltime fields

## RDF Fields

- `Coordinate Path`: coordinate-file pattern. Text inputs may be flattened as `x1 y1 z1 x2 y2 z2 ...`; binary inputs may also use shape `(frames, particles, 3)`.
- `stride`: frame stride applied before RDF calculation
- `Output Path`: main RDF output file
- `Cell Dimensions`: one cubic length or three dimensions `Lx Ly Lz`; used for orthorhombic minimum-image distances
- `r Max`: optional maximum RDF distance. If empty, the calculation uses half of the smallest cell dimension.
- `Bin Width`: RDF radial bin width
- `Density-Normalize`: includes Selection 2 density in the RDF denominator. If unchecked, column 2 is a shell-volume-normalized radial number-density profile around Selection 1 instead of a bulk-density-normalized RDF.
- `Chunkify`: enables the chunk-size fields. If unchecked, both chunk sizes are treated as `1`.
- `Chunk Size 1`: number of Selection 1 particles processed at once during pair-distance calculation
- `Chunk Size 2`: number of Selection 2 particles processed at once during pair-distance calculation
- `Selection 1`: optional zero-based particle-index subset for RDF centers. Empty means all particles.
- `Selection 2`: optional zero-based particle-index subset for RDF partners. Empty means all particles.
- `Exclude Self-Pairs`: excludes pairs where Selection 1 and Selection 2 refer to the same zero-based particle index
- `Trajectory desired length`: optional field; a single integer uses the first `N` frames from each file; `range(start, stop[, step])` uses zero-based frame indices
- `Trajectory Selection`: optional zero-based subset of coordinate-file indices

`Selection 1`, `Selection 2`, and `Trajectory Selection` support comma lists, inclusive dash ranges, Python lists, and `range(...)`, such as `0,3,5`, `0-10`, `[0, 2, 4]`, and `range(0, 10)`.

## RDF Output

The RDF output has four columns:

- column 1: radial bin center `r`
- column 2: radial distribution function `g(r)` when `Density-Normalize` is checked; radial number-density profile when unchecked
- column 3: running coordination number
- column 4: `hist`, the raw pair-count histogram before RDF normalization

## Final notes

- The GUI generates scripts and workflow files. The heavy calculations are carried out later when those generated scripts are executed.
- Use `Smart Optimization` when you want help choosing chunk sizes, memory requests, and worker counts.
- Use `Skip` when part of the pipeline has already been completed and you only want to generate the remaining steps.
