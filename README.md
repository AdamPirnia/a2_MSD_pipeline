# a2\_MSD Pipeline App

A self‑contained desktop application for generating **Mean‑Square‑Displacement (MSD)** and **non‑Gaussian parameter (α₂)** analyses from NAMD DCD trajectories—plus a ready‑to‑submit SLURM job script.

| Artifact                                 | Description                                                       | Platform                           |
| ---------------------------------------- | ----------------------------------------------------------------- | ---------------------------------- |
| `` (or `a2_MSD_pipeline.exe` on Windows) | Stand‑alone executable containing the full GUI + analysis engine. | Linux • macOS • Windows (if built) |

---

## Quick Start

```bash
# 1 – clone the public repo
$ git clone https://github.com/AdamPirnia/a2_MSD_pipeline.git
$ cd a2_MSD_pipeline/dist

# 2 – make executable (Linux/macOS)
$ chmod +x a2_MSD_pipeline

# 3 – run
$ ./a2_MSD_pipeline        # Linux / macOS
> a2_MSD_pipeline.exe      # Windows
```

The GUI opens—fill in the required (\*) fields, choose output paths for a driver script and SLURM submission file, then click **Generate Files**.

---

## What the App Does

| Step | Task                                            | Output             |
| ---- | ----------------------------------------------- | ------------------ |
|  1   | Extract raw coordinates (`coordinates_extract`) | User‑chosen OUTdir |
|  2   | Unwrap PBC (`unwrap_coords`)                    | User‑chosen OUTdir |
|  3   | Center‑of‑Mass calc (`COM_calc`)                | User‑chosen OUTdir |
|  4   | MSD & α₂ (`alpha2_MSD`)                         | User‑chosen OUTdir |

The app stitches those steps into a single driver (`main_<run>.py`) and a matching SLURM script (`submit_<run>.sh`).

---

## Typical Workflow

1. **Common**   – Base directory & number of DCDs.
2. **Step 1**    – Particles range (`0 to 120`), Resname (`SOL`), etc.
3. **SLURM**    – Partition, wall‑time, CPUs, email.
4. **Generate** – Two files appear where you specified.  Submit with `sbatch submit_<run>.sh`.

The GUI remembers your last inputs in `~/.pipeline_gui_config.json`.

---

## Requirements on the Cluster

Nothing to install—Python 3 and the standard library are bundled into the executable.

---

## FAQ

|  Q                              |  A                                                                                     |
| ------------------------------- | -------------------------------------------------------------------------------------- |
| *Where is the source?*          | This public repo distributes only the compiled app.                                    |
| *Does it phone home?*           | No. All computation is local; the only external call is `sbatch`.                      |
| *Can I rebuild for another OS?* | Clone the private source repo and run the PyInstaller build workflow on that platform. |

---

© 2025 Adam Pirnia — All rights reserved.

