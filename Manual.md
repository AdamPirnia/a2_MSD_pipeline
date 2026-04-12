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
- `{i}` is replaced by the DCD index

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

### VMD Path on macOS and Windows

For any field that asks for a `VMD Path` or `VMD Executable Path`, enter the actual VMD executable path or an actual runnable launcher command.

Do not rely on:

- shell aliases that exist only in your interactive terminal
- desktop shortcuts
- wrapper entries that work only when expanded by the shell

Platform guidance:

- macOS: use the real path from inside the VMD application bundle rather than an alias such as `vmd`
- Windows: use the actual `vmd.exe` path rather than a shortcut or non-executable launcher entry

Generic examples:

- macOS: `/Applications/VMD ... .app/Contents/MacOS/...`
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

Any `DCD Selection (optional)` field lets you restrict processing to only some trajectory indices.

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
- `DCD Selection (optional)`: optional subset of DCD indices to extract
- `Stride (optional)`: frame sampling stride; use `1` for every frame, `10` for every tenth frame

Options:

- `Use Parallel VMD`: runs multiple extraction jobs in parallel when possible
- `Wrap Coordinates`: applies VMD `pbctools` wrapping during extraction
- `Settings`: opens the wrapping popup where you can configure shape, centering, compound mode, atom selection, and optional wrap flags

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
- `Interval (optional)`: optional frame interval expression to restrict the processed frame range
- `Stride (optional)`: frame stride used during unwrapping
- `DCD Selection (optional)`: optional subset of DCD indices to process
- `Chunk Size`: memory-related processing chunk size, or `auto`

Option:

- `Use Parallel`: enable parallel processing for the unwrapping step

What this section does:

- reads wrapped coordinates
- uses the box information from the XSC file
- reconstructs continuous trajectories across periodic boundaries

Output unit:

- continuous-coordinate files use the same length unit as the input trajectory coordinates

Use `auto` for `Chunk Size` when you want the software to choose a safe value, or run `Smart Optimization` first and use the recommended value.

## Step 3: COM calculation

This step computes center-of-mass trajectories from the continuous coordinates.

Fields:

- `Input Pattern`: full input `path + filename + extension` pattern for continuous-coordinate files from Step 2
- `Output Pattern`: full output `path + filename + extension` pattern for generated COM files
- `Atoms per particle`: number of atoms in each molecule or particle
- `Mass list`: comma-separated atomic masses in the same order used in each molecule
- `DCD Selection (optional)`: optional subset of DCD indices to process

Options:

- `Use Parallel`: enable parallel COM calculation
- `Use Memory Map`: use memory mapping to reduce RAM pressure on large files

What this section does:

- groups atoms into molecules or particles
- applies the supplied masses
- computes center-of-mass positions frame by frame

How molecule or particle boundaries are determined:

- this step assumes the atoms in each frame are ordered molecule-by-molecule in a consistent repeating pattern
- `Atoms per particle` tells the software how many consecutive atoms belong to one molecule or particle
- the first `Atoms per particle` atoms are treated as particle 1, the next block as particle 2, and so on
- the `Mass list` must match that per-particle atom order

Example:

- if water is stored as `O H H O H H ...`, then `Atoms per particle = 3`
- the software groups atoms as `(O,H,H)`, `(O,H,H)`, `(O,H,H)`, ...

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
- `Number of Particles`: number of molecules or particles in the system
- `Common Terms`: two optional shared replacement fields for `*` and `**` in patterns
- `Max Workers`: maximum CPU workers to use

## Velocity Extraction tab

This workflow generates scripts for extracting center-of-mass velocities from velocity trajectories using VMD.

Fields:

- `PSF Pattern`: pattern for the PSF files
- `VELDCD Pattern`: pattern for velocity-trajectory files
- `Output Pattern`: full output `path + filename + extension` pattern for generated velocity files
- `Stride`: frame sampling stride for velocity extraction
- `VMD Executable Path`: full path to the VMD executable
- `DCD Selection (optional)`: optional subset of DCD indices to process

Notes:

- this workflow uses the module-level `Number of Particles` value from `Common Parameters`
- there is no separate velocity-only molecule-count input

How molecule or particle boundaries are determined:

- the velocity workflow assumes the velocity trajectory follows the same particle ordering as the structure
- `Number of Particles` tells the workflow how many molecule or residue COM velocities to extract
- the VMD script computes one center-of-mass velocity per particle/residue in that ordering
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

### If `Calculation Method = individual`

Use these fields:

- `Coordinates Pattern`: full input `path + filename + extension` pattern for coordinate files
- `COM Pattern`: full input `path + filename + extension` pattern for COM files
- `Dipole Vectors Pattern`: full output `path + filename + extension` pattern for dipole-vector files
- `Dipole Magnitudes Pattern`: full output `path + filename + extension` pattern for dipole-magnitude files
- `Atomic charges`: comma-separated atomic charges for one molecule
- `Atoms per molecule`: number of atoms in each molecule
- `Stride`: frame sampling stride for individual dipole calculation
- `DCD Selection (optional)`: optional subset of DCD indices to process
- `Neutral molecule(s)`: indicates that the molecules are neutral and the calculation should treat them accordingly

What it does:

- loads coordinate data for each molecule
- uses the supplied charges
- computes dipole vectors and magnitudes for each frame

Output units:

- dipole vector files are written in Debye
- dipole magnitude files are written in Debye

How individual molecules are distinguished:

- this workflow assumes the atomic coordinates are ordered molecule-by-molecule in a consistent repeating pattern
- `Atoms per molecule` tells the software how many consecutive atoms belong to one molecule
- `Atomic charges` are applied to each molecule using that same atom order
- the first `Atoms per molecule` atoms are treated as molecule 1, the next block as molecule 2, and so on

Example:

- if each water molecule is stored as `O H H`, use `Atoms per molecule = 3`
- the software interprets the coordinates as `(O,H,H)`, `(O,H,H)`, `(O,H,H)`, ...

Important input note:

- `Coordinates Pattern` must point to per-atom coordinates, for example `xyz` or `unwr_xyz`
- do not use COM-coordinate files such as `com_xyz` as the dipole coordinate input
- if the atom ordering in the coordinate file does not follow the expected repeating per-molecule pattern, the dipole grouping will be wrong

### If `Calculation Method = collective`

Use these fields:

- `Trajectory Pattern`: full input `path + filename + extension` pattern used for collective dipole analysis when applicable
- `PSF Pattern`: full input `path + filename + extension` pattern for collective dipole analysis
- `DCD Pattern`: full input `path + filename + extension` pattern for collective dipole analysis
- `Output Pattern`: full output `path + filename + extension` pattern for the collective dipole result
- `Target Selection`: VMD atom selection defining the group whose collective dipole will be computed
- `VMD Path`: full path to the VMD executable
- `DCD Selection (optional)`: optional subset of DCD indices to process

What it does:

- loads the selected atoms in VMD
- computes the collective dipole of the selected group
- writes output for the full selected set rather than per-molecule dipoles

Notes:

- both individual and collective fields are visible in the tab, but the selected `Calculation Method` determines which workflow the generated script uses
- `Skip` disables the dipole block entirely

Output units:

- the collective output file contains frame index, dipole vector components, and dipole magnitude
- the dipole magnitude is in Debye
- the vector components use the dipole unit returned by VMD for the collective vector output

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
- `Output File Pattern`: full output `path + filename + extension` used as the base name for generated analysis results
- `Min frames`: minimum number of frames expected in each input file
- `DCD Selection (optional)`: optional subset of DCD indices to process

Option:

- `Validate Data`: enables input-data validation checks during the generated calculation

### If `Calculation Type = alpha2_msd`

The generated workflow computes:

- MSD
- NGP, written as `α₂(t)`

Use this when you want translational dynamics and non-Gaussian behavior from COM trajectories.

The `Output File Pattern` is used as the base name of the generated result files. For example, if you provide:

```text
anlz/NVT_*/analysis/ngp_results.dat
```

the generated files will use that base path and produce module-specific result names such as:

- `ngp_results_MSD.dat`
- `ngp_results_a2.dat`

Output units:

- MSD output is in squared length, using the square of the coordinate unit
- `α₂(t)` output is dimensionless

### If `Calculation Type = alpha_xz`

The generated workflow computes anisotropy-related quantities, including the anisotropy parameter `α_{ani}(t)`.

Use this when you want directional anisotropy information rather than the standard `α₂(t)` and MSD workflow.

The `Output File Pattern` is again used as the base name, and the anisotropy workflow generates related result files derived from that base name.

Output units:

- `α_xz(t)`, `α_xy(t)`, `α_yz(t)`, and the anisotropy parameter are dimensionless

### What `Validate Data` does

When enabled, the generated calculation includes validation checks on the input data before or during processing. This is helpful when you want extra protection against malformed or inconsistent trajectory-derived files.

When disabled, the workflow runs without those additional checks and may be faster on already trusted inputs.

## Module 4: Correlation Functions

This module generates scripts for time-correlation analysis from numeric array files that already exist on disk.

## Common Parameters

- `Base Directory`: root folder for correlation inputs and outputs
- `Number of DCDs`: total number of indexed files to process
- `Number of Particles`: retained for consistency with other modules
- `Common Terms`: two optional shared replacement fields for `*` and `**` in file patterns
- `Max Workers`: retained for workflow consistency; current correlation generation runs serially

## Correlation Function Parameters

Fields:

- `Array 1 Pattern`: full input `path + filename + extension` pattern for the first numeric array
- `Array 2 Pattern`: full input `path + filename + extension` pattern for the second numeric array
- `Output Pattern`: full output `path + filename + extension` pattern for the generated correlation results
- `Lag Step (delta)`: frame or sample stride used in the correlation sum
- `Maximum Lag`: largest lag index to evaluate
- `Time per Lag (t1)`: physical time corresponding to one lag increment
- `Vector Coefficient`: scaling factor used for vector-vector correlations; the standard choice is `3`
- `DCD Selection (optional)`: optional subset of indices to process
- `Array 1 Type`: choose `scalar` or `vector`
- `Array 2 Type`: choose `scalar` or `vector`
- `Variable Count`: choose `single` or `multiple`
- `Correlation Mode`: choose `acf` or `fluctuation`

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

### Correlation Mode

- `acf`: subtracts the mean from each input before computing the correlation
- `fluctuation`: computes the correlation directly from the raw input without subtracting the mean

### Correlation normalization

The correlation module does not divide the output by the `t = 0` correlation value.

- scalar and vector results are written as raw correlations
- for vector-vector correlations, the `Vector Coefficient` still applies its usual scaling

Output units:

- the first output column uses the same time unit as `Time per Lag (t1)`
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
- `VACF Already Normalized`: tells the software whether the VACF has already been divided by its `t = 0` value
- `Velocity Units`: `namd_internal` or `angstrom_per_ps`

How VACF input is interpreted:

- if `Time Axis Exist` is checked:
  - column 1 is used as time
  - column 2 is used as VACF
- if `Time Axis Exist` is unchecked:
  - the software builds the time axis as `t = (i - 1) * dt`
  - the VACF is read from the numeric series in the file

How normalization and units are handled:

- if `VACF Already Normalized` is unchecked:
  - the software reads the VACF as an unnormalized physical VACF
  - if `Velocity Units = namd_internal`, velocities are converted using the built-in factor before the variance-based diffusion estimate is formed
  - each VACF is normalized internally by its own `VACF[0]` before integrating `tau`
  - `VACF var, ang^2/ps^2` is reported in the output
- if `VACF Already Normalized` is checked:
  - the software assumes the input is already normalized
  - the velocity-unit menu is disabled and no unit conversion is applied
  - the diffusion estimate uses the equipartition variance instead of a measured `VACF[0]`
  - `VACF var, ang^2/ps^2` is not reported in the results

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

This module generates scripts for isotropic static-structure-factor analysis from saved coordinate arrays.

## Common Parameters

- `Base Directory`: root folder used to resolve relative coordinate paths
- `Common Terms`: optional `*` and `**` replacements, exactly like the other modules

## Structure Factor

### Isotropic Structure Factor

- `Coordinate Path`: path pattern for the coordinate dataset
- `Output Path`: full output path and filename for the generated `S(k)` table
- `k Max`: maximum reciprocal-space magnitude included in the generated `k` list
- `L_x`, `L_y`, `L_z`: orthorhombic box lengths used for reciprocal-vector generation and minimum-image distances
- `tolerance`: tolerance used to merge nearby `|k|` magnitudes into unique isotropic sampling values
- `Trajectory desired length (optional)`: a single integer uses the first `N` frames from each coordinate file; `range(start, stop[, step])` uses that frame window from each file

Notes:

- `Coordinate Path` and `Output Path` support `*` and `**`; `Coordinate Path` also supports `{i}`
- the current module writes one `S(k)` table to the specified output path
- isothermal compressibility is intentionally not included in this version of the module

## Final notes

- The GUI generates scripts and workflow files. The heavy calculations are carried out later when those generated scripts are executed.
- Use `Smart Optimization` when you want help choosing chunk sizes, memory requests, and worker counts.
- Use `Skip` when part of the pipeline has already been completed and you only want to generate the remaining steps.
