# α₂(t) Pipeline from MD

This repository bundles four scripts plus a single driver (`run_pipeline.py`) to take you seamlessly from raw trajectories to mean-square displacements (MSD) and the non-Gaussian parameter α₂(t).

## Requirements

- **Python** 3.7+
  - `numpy`
  - `pandas`
- **VMD** (for `coordinates_extract.py` step)

## Directory Layout

```
α2-MD-pipeline/
├── .gitignore
├── README.md
├── requirements.txt
├── coordinates_extract.py
├── unwrap_coords.py
├── COM_calc.py
├── alpha2_MSD.py
└── run_pipeline.py
```

## Files & Pipeline Steps

1. **coordinates_extract.py**  
   Extract raw XYZ coordinates from DCD via VMD Tcl.

2. **unwrap_coords.py**  
   Remove periodic-boundary wrapping from those coordinates.

3. **COM_calc.py**  
   Compute per-frame center-of-mass (COM) for each multi-atom molecule.

4. **alpha2_MSD.py**  
   Calculate ensemble-averaged MSD ⟨|Δr(t)|²⟩ and non-Gaussian parameter α₂(t).

5. **run_pipeline.py**  
   One-shot driver that calls steps 1–4 in order (with optional skipping of step 3).

---

## Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline with a single command

```bash
python3 run_pipeline.py \
  --baseDir   /path/to/sim \
  --INdir     trajectories \
  --OUTdir    results \
  --num_dcd   6 \
  --num_mols  1000 \
  --numFrames 1000 \
  --psf       system.psf \
  --dcd       traj.dcd \
  --vmd       /usr/local/bin/vmd \
  --xsc       restart_equil.xsc \
  --num_atoms 3000 \
  --masses    16.00 1.008 1.008
```

- `--num_atoms == --num_mols` (single-particle mode) **skips** the COM calculation (step 3) and feeds the unwrapped coordinates directly into `alpha2_MSD.py`.
- Otherwise, step 3 runs normally and its output is used by step 4.

## Outputs

- **MSD** files in:  `results/MSDs/MSD_<tag>.dat`
- **α₂(t)** files in: `results/alpha2s/a2_<tag>.dat`

Replace `<tag>` with your chosen `OUTdir` name.