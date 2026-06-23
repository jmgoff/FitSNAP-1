# -*- coding: utf-8 -*-
"""Exported from tutorial.ipynb."""

# %% [markdown]
# # FitSNAP ACE tutorial: Ta linear potential
#
# This notebook is a compact ACE-first replacement for the older SNAP/PyTorch-centered tutorial. It keeps the library-mode flow visible: define settings, scrape configurations, calculate descriptors, fit a linear model, inspect errors, and write LAMMPS/PACE output files.
#
# The settings below are a reduced-runtime Ta ACE example derived from the current FitSNAP `examples/Ta_PACE/Ta.in` pattern.
#
# You will follow three steps in order: set inputs, process descriptors, then fit and export a potential.
# Numerical values are for learning workflow, not production benchmarking.

# %%
import os
import sys
import shutil
import subprocess
from pathlib import Path

ON_COLAB = "google.colab" in sys.modules

if ON_COLAB:
    # Explicit Colab bootstrap: install dependencies, build LAMMPS with ACE/PACE,
    # and install FitSNAP in editable mode for the tutorial workflow.
    subprocess.run("apt-get update", shell=True, check=True)
    subprocess.run(
        "apt-get install -y cmake build-essential git ccache openmpi-bin "
        "libopenmpi-dev python3-dev",
        shell=True,
        check=True,
    )
    subprocess.run(
        f"{sys.executable} -m pip install --upgrade pip wheel setuptools",
        shell=True,
        check=True,
    )
    subprocess.run(
        f"{sys.executable} -m pip install numpy scipy scikit-learn virtualenv "
        "psutil pandas tabulate mpi4py Cython sympy pyyaml ase matplotlib",
        shell=True,
        check=True,
    )
    os.chdir("/content")
    if not Path("FitSNAP").exists():
        subprocess.run("git clone https://github.com/FitSNAP/FitSNAP.git", shell=True, check=True)
    if not Path("lammps").exists():
        subprocess.run("git clone https://github.com/lammps/lammps.git", shell=True, check=True)
    # Check out the stable tag used by this tutorial stack.
    os.chdir("/content/lammps")
    subprocess.run("git checkout stable_22Jul2025_update4", shell=True, check=True)
    build = Path("build-fitsnap-ace")
    build.mkdir(exist_ok=True)
    os.chdir(build)
    # Shared library + ML packages keep the Python interface and ACE path available.
    cmake_cmd = [
        "cmake", "../cmake",
        "-DLAMMPS_EXCEPTIONS=yes",
        "-DBUILD_SHARED_LIBS=yes",
        "-DMLIAP_ENABLE_PYTHON=yes",
        "-DMLIAP_ENABLE_ACE=yes",
        "-DPKG_PYTHON=yes",
        "-DPKG_ML-SNAP=yes",
        "-DPKG_ML-IAP=yes",
        "-DPKG_ML-PACE=yes",
        "-DPKG_SPIN=yes",
        f"-DPYTHON_EXECUTABLE:FILEPATH={sys.executable}",
    ]
    subprocess.run(cmake_cmd, check=True)
    subprocess.run(["cmake", "--build", ".", "-j2"], check=True)
    subprocess.run(["cmake", "--build", ".", "--target", "install-python"], check=True)
    os.chdir("/content/FitSNAP")
    subprocess.run(f"{sys.executable} -m pip install -e . --no-deps", shell=True, check=True)

# Default to the paths used by the WSL environment we prepared, but allow override
# through environment variables for portability.
FITSNAP_DIR = Path(os.environ.get("FITSNAP_DIR", "/content/FitSNAP" if ON_COLAB else "~/software/FitSNAP")).expanduser()
LAMMPS_DIR = Path(os.environ.get("LAMMPS_DIR", "/content/lammps" if ON_COLAB else "~/software/lammps")).expanduser()
WORK_ROOT = Path(os.environ.get("FITSNAP_TUTORIAL_WORK", Path.cwd() / "fitsnap_tutorial_work")).expanduser()
WORK_ROOT.mkdir(parents=True, exist_ok=True)

if not FITSNAP_DIR.exists():
    raise FileNotFoundError(f"FitSNAP checkout not found: {FITSNAP_DIR}")
if str(FITSNAP_DIR) not in sys.path:
    sys.path.insert(0, str(FITSNAP_DIR))

# Add a short marker to help students orient where execution-specific logic diverges.
print("Execution context -> ON_COLAB:", ON_COLAB)
print("Start here for local tutorial workflow; ON_COLAB blocks are only needed in Colab runtimes.")
print(f"ON_COLAB = {ON_COLAB}")
print(f"FITSNAP_DIR = {FITSNAP_DIR}")
print(f"LAMMPS_DIR = {LAMMPS_DIR}")
print(f"WORK_ROOT = {WORK_ROOT}")

# %% [markdown]
# ### Next edit: This is the first setup cell for local + Colab and should run before ...
# - This is the first setup cell for local + Colab and should run before all downstream imports.
# - In class rooms, keep package install choices stable unless you intentionally switch stacks.

# %%
import importlib.metadata as metadata
import platform

def run_text(cmd, cwd=None):
    # Small helper to keep version checks robust in notebooks and terminals.
    try:
        result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
        text = (result.stdout or result.stderr).strip()
        return text.splitlines()[0] if text else f"exit status {result.returncode}"
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"

def git_version(path):
    # Track exact upstream revision for future reproducibility.
    if not Path(path, ".git").exists():
        return "not a git checkout"
    desc = run_text(["git", "describe", "--tags", "--always", "--dirty"], cwd=path)
    sha = run_text(["git", "rev-parse", "--short", "HEAD"], cwd=path)
    return f"{desc} ({sha})"

print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())
print("Executable:", sys.executable)
print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX", "not set"))
print("FitSNAP git:", git_version(FITSNAP_DIR))
print("LAMMPS git:", git_version(LAMMPS_DIR))

for package in ["fitsnap3", "lammps", "numpy", "scipy", "scikit-learn", "pandas", "mpi4py", "ase", "sympy", "pyyaml", "matplotlib"]:
    try:
        print(f"{package}: {metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"{package}: not installed as a package")

try:
    import lammps
    lmp = lammps.lammps()
    print("LAMMPS Python version integer:", lmp.version())
    lmp.close()
except Exception as exc:
    print("LAMMPS Python check failed:", repr(exc))

# %% [markdown]
# ### Next edit: This version check cell is a good checkpoint for reproducibility.
# - This version check cell is a good checkpoint for reproducibility.
# - If versions differ from your class image, note that explicitly in your notebook header.

# %%
from copy import deepcopy

import numpy as np
import pandas as pd
from mpi4py import MPI
from fitsnap3lib.fitsnap import FitSnap

# Keep one communicator object and print rank info so students can run both 1-proc
# and multi-proc examples without changing the code.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(f"MPI rank {rank} of {comm.Get_size()}")

# %% [markdown]
# ### Next edit: Your first run, only edit fitsnap_dir and lammps_dir in the envir...
# - For your first run, only edit `FITSNAP_DIR` and `LAMMPS_DIR` in the environment section
#   if your repository checkout is not in `~/software`.
# - Instructors: add package checks by extending `VERSION_CELL` if your class requires a fixed
#   dependency set.

# %% [markdown]
# Students should start here for local execution (WSL/Linux or an installed Python stack).
# The Colab setup code path is handled by the `ON_COLAB` flag in `COMMON_SETUP`.

# %% [markdown]
# Local-first workflow:
# 1) environment checks  
# 2) input/model definitions  
# 3) fitting workflow
#
# If you are on Google Colab, dependency install and build steps happen automatically in the setup cell before the shared workflow starts.

# %% [markdown]
# ## Reproducible ACE starting values with `tools/default_ACE_settings.py`
#
# `default_ACE_settings.py` is a helper script that estimates physically reasonable
# `rcutfac` and `lambda` starting values from element radii.
#
# For a classroom workflow:
# 1) start from a small input (single element: `Ta`),
# 2) then run for multi-element systems (`In` and `P`) when needed.
# Small edits are often necessary, so we copy the script to your work folder and
# edit the `elems` line there, instead of touching your installed package file.

# %% [markdown]
# ### Classroom pattern for this helper script
#
# We keep a local copy in the working directory so students can make small, visible
# edits. We then run it and print the suggested starting hyperparameters in notebook
# output.

# %%
default_tool_src = FITSNAP_DIR / "tools" / "default_ACE_settings.py"
default_tool_local = WORK_ROOT / "default_ACE_settings.py"
helper_script = WORK_ROOT / "default_ACE_settings_run.py"

if not default_tool_src.exists():
    raise FileNotFoundError(f"default_ACE_settings.py not found: {default_tool_src}")

if not default_tool_local.exists():
    shutil.copy2(default_tool_src, default_tool_local)
print(f"default_ACE_settings.py copied/loaded at: {default_tool_local}")
print("Preview (first 40 lines):")
for line_num, line in enumerate(default_tool_local.read_text().splitlines()[:40], start=1):
    print(f"{line_num:3d}: {line}")


def run_default_ace_settings(elements, nshell=2.2, label=""):
    # Copy the tool into a temporary file, replace the element list,
    # then print its suggested hyperparameter lines. This keeps the original
    # repository script untouched and makes edits explicit for classroom notes.
    label = label or str(elements)
    text = default_tool_local.read_text()
    marker = "elems = ['N','W']"
    replacement = f"elems = {elements}"
    if marker not in text:
        raise RuntimeError(f"Expected marker '{marker}' in copied default_ACE_settings.py")
    text = text.replace(marker, replacement, 1)
    text = text.replace("nshell=2.2", f"nshell={nshell}", 1)
    helper_script.write_text(text)
    result = subprocess.run(
        [sys.executable, str(helper_script)],
        text=True,
        capture_output=True,
        check=False,
    )
    print(f"\nRunning default_ACE_settings for {label}")
    print("return code:", result.returncode)
    output = (result.stdout or result.stderr or "").strip()
    print(output if output else "No output produced.")


run_default_ace_settings("['Ta']", label="Ta")

# %% [markdown]
# ### Next edit: Ta, keep nshell = 2.2 as a first pass and inspect how much rcutfac
# - For Ta, keep `nshell = 2.2` as a first pass and inspect how much `rcutfac`
#   and `lambda` change from the printed suggestion.
# - If your element set is different, swap in `run_default_ace_settings("[...]", nshell=...)`
#   with explicit element labels and rerun this same cell.
# - To compare multiple projects, save the local helper as `default_ACE_settings_run.py`
#   and change only `elems` and `nshell` line-by-line.

# %% [markdown]
# ## What the model learns, and what is being predicted

# %% [markdown]
# FitSNAP solves a linear regression on descriptors:
#
# $\hat{y} = \Phi w$
#
# `y` are target labels from your training data (energy/force/virial components).  
# `\Phi` are descriptor values from atomic environments.  
# `w` are coefficients learned from the data.
#
# Once fitted, `w` is written into PACE outputs and used by LAMMPS for predictions.

# %% [markdown]
# ## Source example
#
# The current upstream Ta ACE example is kept as a reference. The tutorial uses a smaller descriptor schedule so it can run on a laptop or Colab runtime.

# %%
source_input = FITSNAP_DIR / "examples" / "Ta_PACE" / "Ta.in"
print(source_input)
print("\n".join(source_input.read_text().splitlines()[:35]))

# %% [markdown]
# ### Next edit: Compare this upstream Ta.in with your local input by changing the nex...
# - Compare this upstream `Ta.in` with your local input by changing the next cell's settings
#   block, not the upstream file.
# - If you want to reuse a class project dataset, edit only the local dataset path and group
#   weights first before changing descriptor hyperparameters.

# %% [markdown]
# ## Editable ACE settings
#
# The most common fields to edit are `type`, `ranks`, `lmax`, `nmax`, `nmaxbase`, cutoff parameters, solver choice, group weights, data path, and output path. Keep these visible when adapting the notebook to a new system.

# %%
# Base Ta ACE input. Changing this one block (ACE + GROUPS) is usually enough to start
# experimentation for a different system.
ta_data = FITSNAP_DIR / "examples" / "Ta_Linear_JCP2014" / "JSON"
settings = {
    "ACE": {
        "numTypes": 1,
        "ranks": "1 2 3",
        "lmax": "0 2 2",
        "nmax": "8 3 1",
        "nmaxbase": 8,
        "rcutfac": 4.604694451,
        "lambda": 3.059235105,
        "type": "Ta",
        "lmin": "0 0 1",
        "bzeroflag": 0,
        "b_basis": "minsub",
    },
    "CALCULATOR": {
        "calculator": "LAMMPSPACE",
        "energy": 1,
        "force": 1,
        "stress": 0,
    },
    "ESHIFT": {"Ta": 0.0},
    **{'SOLVER': {'solver': 'SVD', 'compute_testerrs': 1, 'detailed_errors': 1}},
    "SCRAPER": {"scraper": "JSON"},
    "PATH": {"dataPath": str(ta_data)},
    "OUTFILE": {
        "output_style": "PACE",
        "metrics": str(WORK_ROOT / "Ta_ace_basic_metrics.md"),
        "potential": str(WORK_ROOT / "Ta_ace_basic_pot"),
    },
    "REFERENCE": {
        "units": "metal",
        "atom_style": "atomic",
        "pair_style": "hybrid/overlay zero 5.0 zbl 4.0 4.8",
        "pair_coeff1": "* * zero",
        "pair_coeff2": "* * zbl 73 73",
    },
    "GROUPS": {'group_sections': 'name training_size testing_size eweight fweight vweight', 'group_types': 'str float float float float float', 'smartweights': 0, 'random_sampling': 0, 'Displaced_A15': '0.20 0.05 1.0 10.0 1.0e-8', 'Displaced_BCC': '0.20 0.05 1.0 10.0 1.0e-8', 'Displaced_FCC': '0.20 0.05 1.0 10.0 1.0e-8', 'Elastic_BCC': '0.20 0.05 1.0 10.0 1.0e-8', 'Elastic_FCC': '0.20 0.05 1.0 10.0 1.0e-8', 'GSF_110': '0.20 0.05 1.0 10.0 1.0e-8', 'GSF_112': '0.20 0.05 1.0 10.0 1.0e-8', 'Liquid': '0.20 0.05 1.0 10.0 1.0e-8', 'Surface': '0.20 0.05 1.0 10.0 1.0e-8', 'Volume_A15': '0.20 0.05 1.0 10.0 1.0e-8', 'Volume_BCC': '0.20 0.05 1.0 10.0 1.0e-8', 'Volume_FCC': '0.20 0.05 1.0 10.0 1.0e-8'},
    "EXTRAS": {
        "dump_descriptors": 0,
        "dump_truth": 0,
        "dump_weights": 0,
    },
    "MEMORY": {"override": 0},
}
print("Data path:", settings["PATH"]["dataPath"])
print("Potential output prefix:", settings["OUTFILE"]["potential"])

# %% [markdown]
# ### Next edit: Change settings["ACE"] first (e.g., ranks, lmax, nmax, rcutfac, lambd...
# - Change `settings["ACE"]` first (e.g., `ranks`, `lmax`, `nmax`, `rcutfac`, `lambda`) to
#   explore descriptor richness.
# - Change `settings["GROUPS"]` to prioritize data you trust more (for example, lower weight for
#   liquid cells if your use case is solids only).
# - Keep one block of edits at a time so results stay interpretable.

# %% [markdown]
# ## Run the fit

# %%
# Build a FitSnap object once settings are ready.
fs = FitSnap(settings, comm=comm, arglist=["--overwrite"])

# Read JSON trajectories into memory.
fs.scrape_configs()
print("Number of configurations scraped on this rank:", len(fs.data))

# Convert the raw structures into descriptor arrays.
fs.process_configs()

# Solve for ACE coefficients and print a concise error summary.
fs.perform_fit()

print("Error table:")
display(fs.solver.errors)
print("Number of fitted coefficients:", len(np.ravel(fs.solver.fit)))

# %% [markdown]
# ### Next edit: If perform_fit fails, check that dataPath exists and that type names ...
# - If `perform_fit` fails, check that `dataPath` exists and that `type` names in `settings["ACE"]["type"]`
#   match atomic symbols in the dataset.
# - Record `fs.solver.errors` now; later sections use the same error keys for comparison.

# %% [markdown]
# ## Write PACE output files

# %%
# Export everything required for downstream LAMMPS usage.
fs.output.write_lammps(fs.solver.fit)
fs.output.write_errors(fs.solver.errors)

for suffix in [".acecoeff", ".yace", ".mod"]:
    path = Path(settings["OUTFILE"]["potential"] + suffix)
    print(path, "exists:", path.exists(), "size:", path.stat().st_size if path.exists() else "missing")

# %% [markdown]
# ### Next edit: The .yace, .mod, and .acecoeff files feed LAMMPS and post-processing ...
# - The `.yace`, `.mod`, and `.acecoeff` files feed LAMMPS and post-processing tools.
# - Before moving on, note file sizes and names; you can reuse these exact paths in the MD notebook.

# %% [markdown]
# ## Extract ACE descriptors from ASE frames
#
# This section only inspects descriptor matrix shapes for a few structures. It checks that the calculator can process ASE `Atoms` objects; it does not validate a fitted model.

# %%
from ase.io import read
from fitsnap3lib.scrapers.ase_funcs import ase_scraper

# Read two structures from a small XYZ file and convert them into ASE Atoms objects.
xyz_path = FITSNAP_DIR / "examples" / "Ta_XYZ" / "XYZ" / "Displaced_FCC.xyz"
frames = read(str(xyz_path), ":2")
data = ase_scraper(frames)

descriptor_settings = deepcopy(settings)
# Keep fit disabled so we only check the descriptor contract and matrix sizes.
descriptor_settings["CALCULATOR"].update({"force": 0, "stress": 0, "per_atom_energy": 1})
descriptor_settings["ACE"]["bikflag"] = 1
descriptor_settings["ACE"]["bzeroflag"] = 1
descriptor_fs = FitSnap(descriptor_settings, comm=comm, arglist=["--overwrite", "--nofit"])

for i, configuration in enumerate(data):
    # Each call returns descriptor blocks (A), target vectors (b), and weights (w).
    a, b, w = descriptor_fs.calculator.process_single(configuration)
    print(f"configuration {i}: A shape={a.shape}, b shape={np.shape(b)}, w shape={np.shape(w)}")

# %% [markdown]
# ### Next edit: Add more structures by changing read(..., ":2") to a larger slice and...
# - Add more structures by changing `read(..., ":2")` to a larger slice and compare matrix size scaling.
# - Keep `--nofit` when checking `process_single`; it is a fast way to verify scraper compatibility
#   before a full refit.

# %% [markdown]
# ## Historical SNAP comparison
#
# The old tutorial centered on SNAP bispectrum settings. This optional cell shows the old style of calculator configuration without running it by default.

# %%
RUN_SNAP_COMPARISON = False

if RUN_SNAP_COMPARISON:
    snap_settings = {
        "BISPECTRUM": {
            "numTypes": 1,
            "twojmax": 6,
            "rcutfac": 4.67637,
            "rfac0": 0.99363,
            "rmin0": 0.0,
            "wj": 1.0,
            "radelem": 0.5,
            "type": "Ta",
            "wselfallflag": 0,
            "chemflag": 0,
            "bzeroflag": 0,
        },
        "CALCULATOR": {"calculator": "LAMMPSSNAP", "energy": 1, "force": 1, "stress": 0},
        "ESHIFT": {"Ta": 0.0},
        "SOLVER": {"solver": "SVD", "compute_testerrs": 1, "detailed_errors": 1},
        "SCRAPER": {"scraper": "JSON"},
        "PATH": {"dataPath": str(FITSNAP_DIR / "examples" / "Ta_Linear_JCP2014" / "JSON")},
        "OUTFILE": {"metrics": str(WORK_ROOT / "Ta_snap_metrics.md"), "potential": str(WORK_ROOT / "Ta_snap_pot")},
        "REFERENCE": {"units": "metal", "atom_style": "atomic", "pair_style": "zero 5.0", "pair_coeff": "* *"},
        "GROUPS": settings["GROUPS"],
    }
    snap_fs = FitSnap(snap_settings, comm=comm, arglist=["--overwrite"])
    snap_fs.scrape_configs()
    snap_fs.process_configs()
    snap_fs.perform_fit()
    display(snap_fs.solver.errors)
else:
    print("SNAP comparison skipped. Set RUN_SNAP_COMPARISON = True to run it.")

# %% [markdown]
# ### Next edit: Leave this section off during routine class runs to save time.
# - Leave this section off during routine class runs to save time.
# - To compare frameworks more directly, keep the same `GROUPS` and training split, then compare
#   only `SOLVER`/`CALCULATOR` sections.

# %% [markdown]
# ### Useful utility scripts to explore next (non-NN)
#
# Once this notebook runs, try these short utility scripts from `examples/library`:
#
# - `basic_examples/example1.py`: small linear fit workflow with compact diagnostics.
# - `loop_over_fits/example1.py`: sweep descriptor hyperparameters as an extension of this notebook.
# - `loop_over_fits/example2.py`: compute descriptors once, refit many times.
# - `transpose_trick/example.py` and `transpose_trick/README.md`: memory-aware descriptor assembly.
