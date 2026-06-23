# FitSNAP ACE Tutorials

This directory contains the classroom-focused tutorial notebooks and matching `.py` exports for ACE-first workflows in FitSNAP.

## Run order

1. `tutorial.ipynb`
   - Start here.
   - Introduces reusable FitSNAP workflow in Python: setup, data loading, descriptor construction, linear fitting, and output of `.pace` files.
   - Uses Ta ACE as the baseline system.

2. `tutorial_ace_solvers_and_optimization.ipynb`
   - Builds on the Ta setup.
   - Compares solvers (SVD, RIDGE, ARD), adds small hyperparameter scans, and includes a tiny GA group-weight example.
   - Adds discussion of overfitting signs (test-vs-train gaps, large coefficient size).

3. `tutorial_ace_inp_and_lammps_md.ipynb`
   - Adds a compact multi-element InP ACE training section.
   - Shows how to run a short Ta LAMMPS MD workflow and save stability-check logs.
   - Includes a separate optional local 10,000-step check helper.

## Included files

- `tutorials/` contains the notebook files intended for students.
- `scripts/` contains matching `.py` exports and helper generator `make_ace_tutorials.py`.
- `environment_notes.md` records the stack used for validation in this project.

## Notes before running

- Keep this folder clean: training logs, `.yace`, `.mod`, `.acecoeff`, `.npy`, `log.lammps`, and run-specific output are written outside the committed tutorial files (by default into a writable `FITSNAP_TUTORIAL_WORK` directory you control).
- For reproducibility, record:
  - `FITSNAP_DIR`
  - `LAMMPS_DIR`
  - output folder path
  - FitSNAP and LAMMPS revisions
- Both local WSL/Linux and Google Colab are supported by the same notebooks.
  - Colab setup/installation is guarded by `ON_COLAB = "google.colab" in sys.modules`.
  - Local runs should set `FITSNAP_DIR` and `LAMMPS_DIR` for your machine.

## Recommended local workflow

From inside `tutorials/`:

1. Activate your environment (for this project: `fs_lammps` in WSL).
2. Open and run
   - `tutorial.ipynb`
   - `tutorial_ace_solvers_and_optimization.ipynb`
   - `tutorial_ace_inp_and_lammps_md.ipynb`
3. Use the notebook callouts before each section to keep edits focused.

## Version tracking for stability checks

For 10,000-step MD checks, keep versioned log filenames in your working directory (e.g. timestamped
`<system>_10000__fit<gitSHA>__lammps-<gitSHA>__nvt.log`) and store a short manifest entry that records:
- potential path
- fit revision
- LAMMPS revision
- run length

This makes it easy to compare baseline vs GA and other variants later.