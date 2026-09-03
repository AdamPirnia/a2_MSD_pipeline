#!/usr/bin/env python3
"""
Reference-point ("single") dipole calculation.

Given a trajectory of one moving reference point, a cutoff, and a user-named
probe atom, this module restricts a dipole analysis to only those molecules
(residues, or another grouping unit) whose probe atom is currently within the
cutoff of the reference point, and ties every result back to that point.

Inputs
------
* A pre-extracted coordinate file (Module 1 / Step 2 output) plus its
  ``<coords>.atoms.npz`` sidecar, which maps every coordinate column to a
  0-based PSF atom index and resid.
* A PSF file, read directly here (no VMD) for per-atom charge / mass / name /
  resname / segid / resid.
* A reference-point file shaped ``(n_frames, 3)``.

Outputs (``calc_type`` in {"individual", "custom"})
---------------------------------------------------
Per frame the qualifying groups vary, so results are left-packed into padded
rectangular arrays with a ``.counts.npy`` int32 sidecar giving the true number
of qualifying groups per frame, and a ``.groups`` sidecar pair
(``.groups.npy`` + ``.groups.json``) mapping every packed slot to its group.

* ``vec_<base>``           dipole vectors,   ``(n_frames, max_hits * 3)``
* ``mag_<base>``           dipole magnitudes,``(n_frames, max_hits)``
* ``distancevec_<base>``   center - ref,     ``(n_frames, max_hits * 3)``
* ``distancemag_<base>``   |center - ref|,   ``(n_frames, max_hits)``

Outputs (``calc_type == "collective"``)
---------------------------------------
Over every atom ``i`` of every qualifying group at frame ``f``:

* scalar  ``S_f = sum_i q_i * ||r_i - p_f||``      -> ``<base>``, ``(n_frames, 1)``
* vector  ``M_f = sum_i q_i * (r_i - p_f)``        -> ``vec_<base>``, ``(n_frames, 3)``

``S_f`` is reference-origin dependent and is *not* a rotationally invariant
dipole; it is computed verbatim per the analysis request.

Conventions
-----------
* Vectors use ``position - reference``.
* Distances are plain Euclidean (no minimum-image / PBC); wrap upstream if
  needed.
* Dipole sign follows ``sum q_i (r_i - c)`` (negative to positive pole), the
  same convention as :mod:`dipole_function`.
* ``Debye`` divides by :data:`dipole_function.DEBYE_CONVERSION`; ``e·Å`` passes
  through.
"""
import json
import os
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

try:
    from .numeric_io import (
        counts_sidecar_path,
        load_numeric_array,
        save_numeric_array,
    )
    from .path_utils import expand_path_pattern, validate_path_pattern
    from .dipole_function import (
        DEBYE_CONVERSION,
        _compute_group_com_vectorized,
        _compute_group_dipoles,
    )
except ImportError:  # pragma: no cover - exercised only outside the package
    from numeric_io import (
        counts_sidecar_path,
        load_numeric_array,
        save_numeric_array,
    )
    from path_utils import expand_path_pattern, validate_path_pattern
    from dipole_function import (
        DEBYE_CONVERSION,
        _compute_group_com_vectorized,
        _compute_group_dipoles,
    )


_ATOMS_SIDECAR_SUFFIX = ".atoms.npz"


def _atoms_sidecar_path(coord_path):
    return f"{coord_path}{_ATOMS_SIDECAR_SUFFIX}"


def _prepend_prefix_to_basename(pattern, prefix):
    if not pattern:
        return pattern
    directory = os.path.dirname(pattern)
    basename = os.path.basename(pattern)
    return os.path.join(directory, prefix + basename) if directory else prefix + basename


# ---------------------------------------------------------------------------
# PSF parsing
# ---------------------------------------------------------------------------

def _coerce_resid(token):
    """Return an integer resid, tolerating insertion codes such as ``10A``."""
    try:
        return int(token)
    except (TypeError, ValueError):
        digits = re.sub(r"[^0-9-]", "", str(token))
        try:
            return int(digits)
        except ValueError:
            return 0


def parse_psf_atoms(psf_path):
    """Read the ``!NATOM`` section of a PSF file.

    Returns a dict of NumPy arrays indexed by 0-based atom index:
    ``segid`` (str), ``resid`` (int32), ``resname`` (str), ``name`` (str),
    ``type`` (str), ``charge`` (float64), ``mass`` (float64).

    Handles the standard and EXT column layouts by splitting on whitespace,
    which is correct for CHARMM/NAMD PSFs whose segid / resname / atom-name
    fields contain no spaces.
    """
    if not psf_path or not os.path.exists(psf_path):
        raise FileNotFoundError(f"PSF file not found: {psf_path}")

    with open(psf_path, "r", errors="replace") as handle:
        lines = handle.readlines()

    natom = None
    start = None
    for i, line in enumerate(lines):
        if "!NATOM" in line:
            try:
                natom = int(line.split()[0])
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Could not parse !NATOM count in {psf_path}") from exc
            start = i + 1
            break
    if natom is None:
        raise ValueError(f"No !NATOM section found in PSF: {psf_path}")

    segid = np.empty(natom, dtype=object)
    resid = np.zeros(natom, dtype=np.int32)
    resname = np.empty(natom, dtype=object)
    name = np.empty(natom, dtype=object)
    atype = np.empty(natom, dtype=object)
    charge = np.zeros(natom, dtype=np.float64)
    mass = np.zeros(natom, dtype=np.float64)

    filled = 0
    for line in lines[start:]:
        if filled >= natom:
            break
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) < 8:
            raise ValueError(
                f"Malformed PSF atom record in {psf_path} (need >= 8 columns): {stripped!r}"
            )
        try:
            atom_id = int(fields[0])
        except ValueError as exc:
            raise ValueError(
                f"Malformed PSF atom record in {psf_path}: {stripped!r}"
            ) from exc
        slot = atom_id - 1
        if slot < 0 or slot >= natom:
            raise ValueError(
                f"PSF atom id {atom_id} outside 1..{natom} in {psf_path}"
            )
        segid[slot] = fields[1]
        resid[slot] = _coerce_resid(fields[2])
        resname[slot] = fields[3]
        name[slot] = fields[4]
        atype[slot] = fields[5]
        charge[slot] = float(fields[6])
        mass[slot] = float(fields[7])
        filled += 1

    if filled != natom:
        raise ValueError(
            f"PSF {psf_path} declared {natom} atoms but only {filled} records were read"
        )

    return {
        "natom": natom,
        "segid": segid,
        "resid": resid,
        "resname": resname,
        "name": name,
        "type": atype,
        "charge": charge,
        "mass": mass,
    }


# ---------------------------------------------------------------------------
# Metadata: coordinate columns -> groups, charges, masses, probe slots
# ---------------------------------------------------------------------------

def _load_atom_label_sidecar(coord_path):
    sidecar = _atoms_sidecar_path(coord_path)
    if not os.path.exists(sidecar):
        raise FileNotFoundError(
            f"Atom-label sidecar not found: {sidecar}. Reference-point dipole mode "
            "requires a Module 1 / Step 2 coordinate file (it always writes "
            "'<coords>.npy.atoms.npz')."
        )
    with np.load(sidecar, allow_pickle=False) as data:
        index = np.asarray(data["index"])
        resid = np.asarray(data["resid"])
    if index.ndim != 1 or resid.ndim != 1:
        raise ValueError(
            f"Atom-label sidecar {sidecar} is 2-D (per-frame selection). "
            "Reference-point dipole mode needs a static / reference-frame "
            "coordinate file with constant membership."
        )
    return index.astype(np.int64), resid.astype(np.int64)


def _resolve_group_keys(psf_atoms, psf_index, grouping_unit):
    """Return (group_indices_all, group_labels) for the coordinate columns.

    ``group_labels`` is a list of dicts (``segid``/``resid``) in ascending
    order, so packed output slots are deterministic.
    """
    seg = psf_atoms["segid"][psf_index]
    res = psf_atoms["resid"][psf_index]

    if grouping_unit == "all":
        keys = [("", 0)] * len(psf_index)
    elif grouping_unit == "residue":
        keys = list(zip((str(s) for s in seg), (int(r) for r in res)))
    elif grouping_unit in ("segname", "chain"):
        # PSF has no chain field; segid is the closest equivalent.
        keys = [(str(s), -1) for s in seg]
    else:
        raise ValueError(f"Unsupported grouping unit: {grouping_unit}")

    ordered = sorted(set(keys))
    lookup = {key: i for i, key in enumerate(ordered)}
    group_indices_all = np.fromiter((lookup[k] for k in keys), dtype=np.int32, count=len(keys))
    group_labels = [{"segid": seg_val, "resid": res_val} for (seg_val, res_val) in ordered]
    return group_indices_all, group_labels


def _build_reference_metadata(
    coord_path,
    n_atoms,
    psf_atoms,
    grouping_unit,
    probe_resname,
    probe_atom_name,
):
    psf_index, _sidecar_resid = _load_atom_label_sidecar(coord_path)
    if psf_index.shape[0] != n_atoms:
        raise ValueError(
            f"Coordinate file has {n_atoms} atom columns but its atom-label sidecar "
            f"lists {psf_index.shape[0]} atoms."
        )
    if psf_index.min() < 0 or psf_index.max() >= psf_atoms["natom"]:
        raise ValueError(
            "Atom-label sidecar references PSF atom indices outside the PSF "
            f"(0..{psf_atoms['natom'] - 1}). Wrong PSF for this coordinate file?"
        )

    atom_charges = psf_atoms["charge"][psf_index].astype(np.float64)
    atom_masses = psf_atoms["mass"][psf_index].astype(np.float64)
    atom_names = psf_atoms["name"][psf_index]
    atom_resnames = psf_atoms["resname"][psf_index]

    group_indices_all, group_labels = _resolve_group_keys(psf_atoms, psf_index, grouping_unit)
    group_count = len(group_labels)

    com_atom_mask = atom_masses != 0.0
    if not np.any(com_atom_mask):
        raise ValueError("Selected atoms carry no nonzero mass; check the PSF.")
    com_group_indices = group_indices_all[com_atom_mask]
    com_masses = atom_masses[com_atom_mask]
    group_masses = np.zeros(group_count, dtype=np.float64)
    np.add.at(group_masses, com_group_indices, com_masses)
    if np.any(group_masses <= 0.0):
        raise ValueError("At least one group has non-positive total mass.")

    probe_slots = np.where(
        (atom_resnames == probe_resname) & (atom_names == probe_atom_name)
    )[0].astype(np.int64)
    if probe_slots.size == 0:
        raise ValueError(
            f"No probe atom '{probe_atom_name}' in resname '{probe_resname}' is present "
            "among the extracted coordinates. Check the Probe fields and the "
            "Module 1 target selection."
        )
    probe_group = group_indices_all[probe_slots]

    groups_with_probe = np.unique(probe_group)
    if groups_with_probe.size < group_count:
        print(
            f"  Note: {group_count - groups_with_probe.size} of {group_count} group(s) "
            "have no probe atom and can never qualify."
        )
    if probe_slots.size > groups_with_probe.size:
        print(
            f"  Note: {probe_slots.size} probe atoms across {groups_with_probe.size} group(s) "
            "(a group qualifies if any of its probe atoms is within the cutoff)."
        )

    return {
        "atom_count": int(n_atoms),
        "group_indices_all": group_indices_all,
        "group_count": group_count,
        "group_labels": group_labels,
        "atom_charges": atom_charges,
        "atom_masses": atom_masses,
        "com_atom_mask": com_atom_mask,
        "com_group_indices": com_group_indices,
        "com_masses": com_masses,
        "group_masses": group_masses,
        "probe_slots": probe_slots,
        "probe_group": probe_group,
    }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _compute_group_geometric_centers(atom_coords, metadata):
    n_frames = atom_coords.shape[0]
    n_groups = metadata["group_count"]
    centers = np.zeros((n_frames, n_groups, 3), dtype=np.float64)
    gi = metadata["group_indices_all"]
    for group_index in range(n_groups):
        atom_mask = gi == group_index
        centers[:, group_index, :] = atom_coords[:, atom_mask, :].mean(axis=1)
    return centers


def _group_qualification(atom_coords, ref_points, metadata, cutoff):
    """Return a ``(n_frames, n_groups)`` bool array: group's probe within cutoff."""
    probe_slots = metadata["probe_slots"]
    probe_group = metadata["probe_group"]
    probe_coords = atom_coords[:, probe_slots, :]                       # (F, P, 3)
    probe_dist = np.linalg.norm(probe_coords - ref_points[:, None, :], axis=2)  # (F, P)
    within = probe_dist <= cutoff                                       # (F, P)

    n_frames = atom_coords.shape[0]
    group_qualifies = np.zeros((n_frames, metadata["group_count"]), dtype=bool)
    for probe_local, group_index in enumerate(probe_group):
        group_qualifies[:, group_index] |= within[:, probe_local]
    return group_qualifies


def _left_pack(values, mask, counts, max_hits):
    """Left-pack ``values`` rows by ``mask`` (qualifying first), zeroing padding.

    ``values`` shape ``(F, n_groups, ...)``; result ``(F, max_hits, ...)``.
    """
    n_frames = values.shape[0]
    order = np.argsort(~mask, axis=1, kind="stable")[:, :max_hits]       # (F, max_hits)
    rows = np.arange(n_frames)[:, None]
    packed = values[rows, order]
    valid = np.arange(max_hits)[None, :] < counts[:, None]              # (F, max_hits)
    if packed.ndim == 3:
        packed = packed * valid[:, :, None]
    else:
        packed = packed * valid
    return packed, order, valid


# ---------------------------------------------------------------------------
# Per-file worker
# ---------------------------------------------------------------------------

def _load_reference_points(baseDir, ref_point_pattern, common_term, file_idx):
    rel = expand_path_pattern(ref_point_pattern, common_term, file_idx)
    path = os.path.join(baseDir, rel)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference-point file not found: {path}")
    try:
        data = load_numeric_array(
            path, None, default_mode="binary", default_precision="double"
        )
    except Exception:
        data = load_numeric_array(
            path, {"mode": "text", "precision": "double"},
            default_mode="text", default_precision="double",
        )
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 3:
        raise ValueError(
            f"Reference-point file {path} must have exactly 3 columns (x y z); "
            f"got shape {data.shape}."
        )
    return data


def _process_single_refpoint_file(file_idx, ctx):
    try:
        baseDir = ctx["baseDir"]
        common_term = ctx["common_term"]
        stride = ctx["stride"]
        calc_type = ctx["calc_type"]
        dipole_unit = ctx["dipole_unit"]
        all_neutral = ctx["all_neutral"]
        cutoff = ctx["cutoff"]

        coord_rel = expand_path_pattern(ctx["coords_pattern"], common_term, file_idx)
        coord_file = os.path.join(baseDir, coord_rel)
        psf_rel = expand_path_pattern(ctx["psf_pattern"], common_term, file_idx)
        psf_file = os.path.join(baseDir, psf_rel)
        print(f"Processing file {file_idx}:")
        print(f"  Coord file: {coord_file}")
        print(f"  PSF file:   {psf_file}")

        if not os.path.exists(coord_file):
            return {"success": False, "error": f"Coordinate file not found: {coord_file}", "file_idx": file_idx}
        if os.path.exists(counts_sidecar_path(coord_file)):
            return {
                "success": False,
                "file_idx": file_idx,
                "error": (
                    f"Coordinate file {coord_file} has a per-frame count sidecar "
                    "(per_frame selection). Reference-point dipole mode needs a "
                    "static / reference-frame coordinate file."
                ),
            }

        coord_data = load_numeric_array(
            coord_file, ctx["coords_input_io_spec"],
            default_mode="binary", default_precision="single",
        )
        coord_data = np.asarray(coord_data, dtype=np.float64)
        if coord_data.ndim != 2 or coord_data.shape[1] % 3 != 0:
            raise ValueError(
                f"Coordinate file {coord_file} must be 2-D with a column count divisible by 3; "
                f"got shape {coord_data.shape}."
            )
        n_atoms = coord_data.shape[1] // 3

        psf_atoms = parse_psf_atoms(psf_file)
        metadata = _build_reference_metadata(
            coord_file, n_atoms, psf_atoms, ctx["grouping_unit"],
            ctx["probe_resname"], ctx["probe_atom_name"],
        )

        ref_points = _load_reference_points(baseDir, ctx["ref_point_pattern"], common_term, file_idx)

        if ref_points.shape[0] != coord_data.shape[0]:
            raise ValueError(
                f"Reference-point file has {ref_points.shape[0]} frames but the coordinate file "
                f"has {coord_data.shape[0]}. They must match (stride is applied to both)."
            )

        if stride > 1:
            coord_data = coord_data[::stride]
            ref_points = ref_points[::stride]

        n_frames = coord_data.shape[0]
        atom_coords = coord_data.reshape(n_frames, n_atoms, 3)
        print(f"  {n_frames} frame(s), {n_atoms} atom column(s), {metadata['group_count']} group(s)")

        group_qualifies = _group_qualification(atom_coords, ref_points, metadata, cutoff)
        counts = group_qualifies.sum(axis=1).astype(np.int32)
        print(f"  Qualifying groups per frame: min={int(counts.min())}, "
              f"max={int(counts.max())}, mean={float(counts.mean()):.2f}")

        if calc_type == "collective":
            outputs = _write_collective(
                ctx, file_idx, atom_coords, ref_points, metadata, group_qualifies, counts, dipole_unit
            )
        else:
            outputs = _write_individual(
                ctx, file_idx, atom_coords, ref_points, metadata, group_qualifies, counts,
                dipole_unit, all_neutral,
            )

        print(f"  SUCCESS: file {file_idx}, {n_frames} frames")
        return {
            "success": True,
            "file_idx": file_idx,
            "frames_processed": n_frames,
            "outputs": outputs,
            "qualifying_groups_mean": float(counts.mean()),
        }
    except Exception as exc:  # noqa: BLE001 - report, do not crash the pool
        return {
            "success": False,
            "file_idx": file_idx,
            "error": f"File {file_idx} failed: {exc}\nTraceback: {traceback.format_exc()}",
        }


def _convert_unit(array, dipole_unit):
    if dipole_unit == "Debye":
        return array / DEBYE_CONVERSION
    if dipole_unit != "e·Å":
        raise ValueError(f"Unsupported dipole unit: {dipole_unit}")
    return array


def _write_individual(ctx, file_idx, atom_coords, ref_points, metadata, group_qualifies,
                      counts, dipole_unit, all_neutral):
    n_frames = atom_coords.shape[0]
    max_hits = int(max(1, counts.max()))

    if all_neutral:
        centers = _compute_group_geometric_centers(atom_coords, metadata)
        dipole_center_arg = None
    else:
        centers = _compute_group_com_vectorized(atom_coords, metadata)
        dipole_center_arg = centers

    dipole_vectors, dipole_mags = _compute_group_dipoles(
        atom_coords, metadata, centers_of_mass=dipole_center_arg,
        dipole_unit=dipole_unit, calculate_magnitudes=True,
    )                                                          # (F, G, 3), (F, G)
    distance_vectors = centers - ref_points[:, None, :]         # (F, G, 3)
    distance_mags = np.linalg.norm(distance_vectors, axis=2)    # (F, G)

    packed_vec, order, valid = _left_pack(dipole_vectors, group_qualifies, counts, max_hits)
    packed_mag = dipole_mags[np.arange(n_frames)[:, None], order] * valid
    packed_dvec, _order2, _valid2 = _left_pack(distance_vectors, group_qualifies, counts, max_hits)
    packed_dmag = distance_mags[np.arange(n_frames)[:, None], order] * valid
    packed_gid = np.where(valid, order.astype(np.int32), np.int32(-1))

    base = ctx["base_output_pattern"]
    rel_base = expand_path_pattern(base, ctx["common_term"], file_idx)
    vec_spec = ctx["vectors_output_io_spec"]
    scal_spec = ctx["scalars_output_io_spec"]

    vec_path = os.path.join(ctx["baseDir"], _prepend_prefix_to_basename(rel_base, "vec_"))
    mag_path = os.path.join(ctx["baseDir"], _prepend_prefix_to_basename(rel_base, "mag_"))
    dvec_path = os.path.join(ctx["baseDir"], _prepend_prefix_to_basename(rel_base, "distancevec_"))
    dmag_path = os.path.join(ctx["baseDir"], _prepend_prefix_to_basename(rel_base, "distancemag_"))

    save_numeric_array(vec_path, packed_vec.reshape(n_frames, -1), vec_spec,
                       default_mode="text", default_precision="double")
    save_numeric_array(mag_path, packed_mag, scal_spec,
                       default_mode="text", default_precision="double")
    save_numeric_array(dvec_path, packed_dvec.reshape(n_frames, -1), vec_spec,
                       default_mode="text", default_precision="double")
    save_numeric_array(dmag_path, packed_dmag, scal_spec,
                       default_mode="text", default_precision="double")

    np.save(f"{vec_path}.counts.npy", counts.astype(np.int32))
    np.save(f"{vec_path}.groups.npy", packed_gid)
    with open(f"{vec_path}.groups.json", "w") as handle:
        json.dump(
            {
                "grouping_unit": ctx["grouping_unit"],
                "note": (
                    "groups.npy holds the global group id of each packed slot "
                    "(-1 = padding); look it up in 'groups' below."
                ),
                "groups": [
                    {"gid": i, "segid": lbl["segid"], "resid": lbl["resid"]}
                    for i, lbl in enumerate(metadata["group_labels"])
                ],
            },
            handle,
            indent=1,
        )

    for path in (vec_path, mag_path, dvec_path, dmag_path):
        load_numeric_array(path, vec_spec, default_mode="text", default_precision="double")
    print(f"  wrote {vec_path}")
    print(f"        {mag_path}")
    print(f"        {dvec_path}")
    print(f"        {dmag_path}")
    print(f"        {vec_path}.counts.npy / .groups.npy / .groups.json  (max_hits={max_hits})")
    return {"vectors": vec_path, "magnitudes": mag_path,
            "distance_vectors": dvec_path, "distance_magnitudes": dmag_path}


def _write_collective(ctx, file_idx, atom_coords, ref_points, metadata, group_qualifies,
                      counts, dipole_unit):
    n_frames = atom_coords.shape[0]
    atom_qual = group_qualifies[:, metadata["group_indices_all"]]        # (F, n_atoms) bool
    rel = atom_coords - ref_points[:, None, :]                           # (F, n_atoms, 3)
    weight = metadata["atom_charges"][None, :] * atom_qual               # (F, n_atoms)

    m_vec = np.einsum("fa,fad->fd", weight, rel)                         # (F, 3)
    atom_dist = np.linalg.norm(rel, axis=2)                             # (F, n_atoms)
    s_scalar = np.einsum("fa,fa->f", weight, atom_dist)                 # (F,)

    m_vec = _convert_unit(m_vec, dipole_unit)
    s_scalar = _convert_unit(s_scalar, dipole_unit)

    base = ctx["base_output_pattern"]
    rel_base = expand_path_pattern(base, ctx["common_term"], file_idx)
    vec_spec = ctx["vectors_output_io_spec"]
    scal_spec = ctx["scalars_output_io_spec"]

    scalar_path = os.path.join(ctx["baseDir"], rel_base)
    vec_path = os.path.join(ctx["baseDir"], _prepend_prefix_to_basename(rel_base, "vec_"))

    save_numeric_array(scalar_path, s_scalar.reshape(n_frames, 1), scal_spec,
                       default_mode="text", default_precision="double")
    save_numeric_array(vec_path, m_vec, vec_spec,
                       default_mode="text", default_precision="double")
    np.save(f"{scalar_path}.counts.npy", counts.astype(np.int32))

    load_numeric_array(scalar_path, scal_spec, default_mode="text", default_precision="double")
    load_numeric_array(vec_path, vec_spec, default_mode="text", default_precision="double")
    print(f"  wrote {scalar_path}  (S_f = sum q_i |r_i - p_f|)")
    print(f"        {vec_path}  (M_f = sum q_i (r_i - p_f))")
    print(f"        {scalar_path}.counts.npy")
    return {"scalar": scalar_path, "vector": vec_path}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def reference_point_dipole(
    baseDir,
    coords_pattern,
    psf_pattern,
    ref_point_pattern,
    cutoff,
    probe_resname,
    probe_atom_name,
    grouping_unit,
    dipole_unit,
    all_neutral,
    calc_type,
    output_pattern,
    num_dcds,
    stride=1,
    max_workers=1,
    dcd_indices=None,
    common_term="",
    coords_input_io_spec=None,
    vectors_output_io_spec=None,
    scalars_output_io_spec=None,
    progress_callback=None,
):
    """Run the reference-point ("single") dipole analysis over a set of files.

    See the module docstring for the output contract. ``calc_type`` is one of
    ``"individual"``, ``"custom"`` (identical behaviour), or ``"collective"``.
    """
    start_time = time.time()

    for label, pattern in (
        ("coords", coords_pattern),
        ("PSF", psf_pattern),
        ("reference point", ref_point_pattern),
        ("output", output_pattern),
    ):
        ok, msg = validate_path_pattern(pattern or "")
        if not ok:
            raise ValueError(f"Invalid {label} pattern: {msg}")

    try:
        cutoff = float(cutoff)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cutoff distance must be a number: {cutoff!r}") from exc
    if cutoff <= 0:
        raise ValueError("Cutoff distance must be > 0.")

    if grouping_unit not in {"residue", "chain", "segname", "all"}:
        raise ValueError("grouping_unit must be one of: residue, chain, segname, all")
    if dipole_unit not in {"Debye", "e·Å"}:
        raise ValueError("dipole_unit must be one of: Debye, e·Å")
    if calc_type not in {"individual", "custom", "collective"}:
        raise ValueError("calc_type must be one of: individual, custom, collective")
    if not str(probe_resname or "").strip():
        raise ValueError("probe_resname is required")
    if not str(probe_atom_name or "").strip():
        raise ValueError("probe_atom_name is required")

    stride = max(1, int(stride or 1))

    if dcd_indices is not None:
        index_list = list(dcd_indices)
    else:
        index_list = list(range(int(num_dcds)))
    if not index_list:
        raise ValueError("No trajectory indices selected for reference-point dipole calculation.")

    ctx = {
        "baseDir": baseDir,
        "coords_pattern": coords_pattern,
        "psf_pattern": psf_pattern,
        "ref_point_pattern": ref_point_pattern,
        "cutoff": cutoff,
        "probe_resname": str(probe_resname).strip(),
        "probe_atom_name": str(probe_atom_name).strip(),
        "grouping_unit": grouping_unit,
        "dipole_unit": dipole_unit,
        "all_neutral": bool(all_neutral),
        "calc_type": calc_type,
        "base_output_pattern": output_pattern,
        "stride": stride,
        "common_term": common_term,
        "coords_input_io_spec": coords_input_io_spec,
        "vectors_output_io_spec": vectors_output_io_spec,
        "scalars_output_io_spec": scalars_output_io_spec,
    }

    print("=" * 60)
    print("REFERENCE-POINT (SINGLE) DIPOLE CALCULATION")
    print("=" * 60)
    print(f"Base directory : {baseDir}")
    print(f"Coords pattern : {coords_pattern}")
    print(f"PSF pattern    : {psf_pattern}")
    print(f"Ref. point     : {ref_point_pattern}")
    print(f"Cutoff         : {cutoff}")
    print(f"Probe          : name '{ctx['probe_atom_name']}' in resname '{ctx['probe_resname']}'")
    print(f"Grouping unit  : {grouping_unit}")
    print(f"Dipole unit    : {dipole_unit}")
    print(f"Calc type      : {calc_type}")
    print(f"Indices        : {index_list}")
    print(f"Workers        : {max_workers}")

    results = []
    successful, failed = [], []

    def _handle(result):
        results.append(result)
        if result["success"]:
            successful.append(result["file_idx"])
            print(f"✓ File {result['file_idx']}: {result['frames_processed']} frames, "
                  f"avg qualifying groups {result['qualifying_groups_mean']:.2f}")
        else:
            failed.append(result)
            print(f"✗ File {result['file_idx']}: {result['error']}")
        if progress_callback:
            progress_callback(len(results), len(index_list))

    if int(max_workers) > 1 and len(index_list) > 1:
        try:
            with ProcessPoolExecutor(max_workers=int(max_workers)) as executor:
                futures = {
                    executor.submit(_process_single_refpoint_file, i, ctx): i
                    for i in index_list
                }
                for future in as_completed(futures):
                    _handle(future.result())
        except (TypeError, AttributeError) as exc:
            if "pickle" in str(exc).lower() or "local object" in str(exc):
                print(f"⚠️  Parallel execution failed ({exc}); falling back to single-threaded.")
                for i in index_list:
                    _handle(_process_single_refpoint_file(i, ctx))
            else:
                raise
    else:
        for i in index_list:
            _handle(_process_single_refpoint_file(i, ctx))

    total_time = time.time() - start_time
    summary = {
        "success": len(successful),
        "successful": sorted(successful),
        "failed": failed,
        "total": len(index_list),
        "total_time": total_time,
        "calc_type": calc_type,
    }
    print("\nReference-point dipole calculation complete:")
    print(f"  Successful files: {len(successful)}/{len(index_list)}")
    print(f"  Total time: {total_time:.2f}s")
    return summary


# Alias for symmetry with dipole_function.dipole_calculation
reference_point_dipole_calculation = reference_point_dipole
