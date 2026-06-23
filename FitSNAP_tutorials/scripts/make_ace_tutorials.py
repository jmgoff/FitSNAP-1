import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CELL_COUNTER = 0


def next_id():
    global CELL_COUNTER
    CELL_COUNTER += 1
    return f"cell-{CELL_COUNTER:04d}"


def md(source):
    return {"cell_type": "markdown", "id": next_id(), "metadata": {}, "source": source.strip() + "\n"}


def code(source):
    return {
        "cell_type": "code",
        "id": next_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def _title_from_callout(markdown_text):
    """Derive a short, stable, descriptive subsection title from the note text."""
    first = ""
    for line in markdown_text.strip().splitlines():
        clean = line.strip()
        if clean:
            first = clean
            break
    if first.startswith("- "):
        first = first[2:]
    first = first.replace("**", "").replace("`", "")
    if len(first) > 72:
        first = first[:69] + "..."
    if first.lower().startswith("for "):
        first = first[4:].capitalize()
    if not first:
        first = "Next step"
    return f"Next edit: {first}"


def what_next(markdown_text):
    """Create a short undergrad-friendly callout after a tutorial code block."""
    return md(f"### {_title_from_callout(markdown_text)}\n{markdown_text.strip()}\n")


def write_notebook(name, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    ipynb = ROOT / name
    ipynb.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    write_py_export(ipynb.with_suffix(".py"), cells)


def write_py_export(path, cells):
    lines = [
        "# -*- coding: utf-8 -*-",
        f'"""Exported from {path.with_suffix(".ipynb").name}."""',
        "",
    ]
    for cell in cells:
        if cell["cell_type"] == "markdown":
            lines.append("# %% [markdown]")
            lines.extend("# " + line if line else "#" for line in cell["source"].splitlines())
        else:
            lines.append("# %%")
            lines.extend(cell["source"].splitlines())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


COMMON_SETUP = code(
    r"""
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
"""
)


VERSION_CELL = code(
    r"""
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
"""
)


IMPORTS = code(
    r"""
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
"""
)


def default_ace_tool_cells():
    return [
        md(
            """
## Reproducible ACE starting values with `tools/default_ACE_settings.py`

`default_ACE_settings.py` is a helper script that estimates physically reasonable
`rcutfac` and `lambda` starting values from element radii.

For a classroom workflow:
1) start from a small input (single element: `Ta`),
2) then run for multi-element systems (`In` and `P`) when needed.
Small edits are often necessary, so we copy the script to your work folder and
edit the `elems` line there, instead of touching your installed package file.
"""
        ),
        md(
            """
### Classroom pattern for this helper script

We keep a local copy in the working directory so students can make small, visible
edits. We then run it and print the suggested starting hyperparameters in notebook
output.
"""
        ),
        code(
            r"""
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
    """
        ),
        what_next(
            """
- For Ta, keep `nshell = 2.2` as a first pass and inspect how much `rcutfac`
  and `lambda` change from the printed suggestion.
- If your element set is different, swap in `run_default_ace_settings("[...]", nshell=...)`
  with explicit element labels and rerun this same cell.
- To compare multiple projects, save the local helper as `default_ACE_settings_run.py`
  and change only `elems` and `nshell` line-by-line.
    """
        ),
    ]


TA_GROUPS = {
    "group_sections": "name training_size testing_size eweight fweight vweight",
    "group_types": "str float float float float float",
    "smartweights": 0,
    "random_sampling": 0,
    "Displaced_A15": "0.20 0.05 1.0 10.0 1.0e-8",
    "Displaced_BCC": "0.20 0.05 1.0 10.0 1.0e-8",
    "Displaced_FCC": "0.20 0.05 1.0 10.0 1.0e-8",
    "Elastic_BCC": "0.20 0.05 1.0 10.0 1.0e-8",
    "Elastic_FCC": "0.20 0.05 1.0 10.0 1.0e-8",
    "GSF_110": "0.20 0.05 1.0 10.0 1.0e-8",
    "GSF_112": "0.20 0.05 1.0 10.0 1.0e-8",
    "Liquid": "0.20 0.05 1.0 10.0 1.0e-8",
    "Surface": "0.20 0.05 1.0 10.0 1.0e-8",
    "Volume_A15": "0.20 0.05 1.0 10.0 1.0e-8",
    "Volume_BCC": "0.20 0.05 1.0 10.0 1.0e-8",
    "Volume_FCC": "0.20 0.05 1.0 10.0 1.0e-8",
}


def settings_cell(var_name, outfile_prefix, solver="SVD", extra_solver=None):
    solver_sections = {
        "SOLVER": {
            "solver": solver,
            "compute_testerrs": 1,
            "detailed_errors": 1,
        }
    }
    if extra_solver:
        solver_sections.update(extra_solver)
    return code(
        f"""
# Base Ta ACE input. Changing this one block (ACE + GROUPS) is usually enough to start
# experimentation for a different system.
ta_data = FITSNAP_DIR / "examples" / "Ta_Linear_JCP2014" / "JSON"
{var_name} = {{
    "ACE": {{
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
    }},
    "CALCULATOR": {{
        "calculator": "LAMMPSPACE",
        "energy": 1,
        "force": 1,
        "stress": 0,
    }},
    "ESHIFT": {{"Ta": 0.0}},
    **{solver_sections!r},
    "SCRAPER": {{"scraper": "JSON"}},
    "PATH": {{"dataPath": str(ta_data)}},
    "OUTFILE": {{
        "output_style": "PACE",
        "metrics": str(WORK_ROOT / "{outfile_prefix}_metrics.md"),
        "potential": str(WORK_ROOT / "{outfile_prefix}_pot"),
    }},
    "REFERENCE": {{
        "units": "metal",
        "atom_style": "atomic",
        "pair_style": "hybrid/overlay zero 5.0 zbl 4.0 4.8",
        "pair_coeff1": "* * zero",
        "pair_coeff2": "* * zbl 73 73",
    }},
    "GROUPS": {TA_GROUPS!r},
    "EXTRAS": {{
        "dump_descriptors": 0,
        "dump_truth": 0,
        "dump_weights": 0,
    }},
    "MEMORY": {{"override": 0}},
}}
print("Data path:", {var_name}["PATH"]["dataPath"])
print("Potential output prefix:", {var_name}["OUTFILE"]["potential"])
"""
    )


def beginner_cells():
    return [
        md(
            """
# FitSNAP ACE tutorial: Ta linear potential

This notebook is a compact ACE-first replacement for the older SNAP/PyTorch-centered tutorial. It keeps the library-mode flow visible: define settings, scrape configurations, calculate descriptors, fit a linear model, inspect errors, and write LAMMPS/PACE output files.

The settings below are a reduced-runtime Ta ACE example derived from the current FitSNAP `examples/Ta_PACE/Ta.in` pattern.

You will follow three steps in order: set inputs, process descriptors, then fit and export a potential.
Numerical values are for learning workflow, not production benchmarking.
        """
        ),
        COMMON_SETUP,
        what_next(
            """
- This is the first setup cell for local + Colab and should run before all downstream imports.
- In class rooms, keep package install choices stable unless you intentionally switch stacks.
"""
        ),
        VERSION_CELL,
        what_next(
            """
- This version check cell is a good checkpoint for reproducibility.
- If versions differ from your class image, note that explicitly in your notebook header.
"""
        ),
        IMPORTS,
        what_next(
            """
- For your first run, only edit `FITSNAP_DIR` and `LAMMPS_DIR` in the environment section
  if your repository checkout is not in `~/software`.
- Instructors: add package checks by extending `VERSION_CELL` if your class requires a fixed
  dependency set.
"""
        ),
        md(
            """
Students should start here for local execution (WSL/Linux or an installed Python stack).
The Colab setup code path is handled by the `ON_COLAB` flag in `COMMON_SETUP`.
"""
        ),
        md(
            """
Local-first workflow:
1) environment checks  
2) input/model definitions  
3) fitting workflow

If you are on Google Colab, dependency install and build steps happen automatically in the setup cell before the shared workflow starts.
"""
        ),
        *default_ace_tool_cells(),
        md("## What the model learns, and what is being predicted"),
        md(
            r"""
FitSNAP solves a linear regression on descriptors:

$\hat{y} = \Phi w$

`y` are target labels from your training data (energy/force/virial components).  
`\Phi` are descriptor values from atomic environments.  
`w` are coefficients learned from the data.

Once fitted, `w` is written into PACE outputs and used by LAMMPS for predictions.
"""
        ),
        md(
            """
## Source example

The current upstream Ta ACE example is kept as a reference. The tutorial uses a smaller descriptor schedule so it can run on a laptop or Colab runtime.
"""
        ),
        code(
            r"""
source_input = FITSNAP_DIR / "examples" / "Ta_PACE" / "Ta.in"
print(source_input)
print("\n".join(source_input.read_text().splitlines()[:35]))
"""
        ),
        what_next(
            """
- Compare this upstream `Ta.in` with your local input by changing the next cell's settings
  block, not the upstream file.
- If you want to reuse a class project dataset, edit only the local dataset path and group
  weights first before changing descriptor hyperparameters.
"""
        ),
        md(
            """
## Editable ACE settings

The most common fields to edit are `type`, `ranks`, `lmax`, `nmax`, `nmaxbase`, cutoff parameters, solver choice, group weights, data path, and output path. Keep these visible when adapting the notebook to a new system.
"""
        ),
        settings_cell("settings", "Ta_ace_basic", "SVD"),
        what_next(
            """
- Change `settings["ACE"]` first (e.g., `ranks`, `lmax`, `nmax`, `rcutfac`, `lambda`) to
  explore descriptor richness.
- Change `settings["GROUPS"]` to prioritize data you trust more (for example, lower weight for
  liquid cells if your use case is solids only).
- Keep one block of edits at a time so results stay interpretable.
"""
        ),
        md("## Run the fit"),
        code(
            r"""
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
"""
        ),
        what_next(
            """
- If `perform_fit` fails, check that `dataPath` exists and that `type` names in `settings["ACE"]["type"]`
  match atomic symbols in the dataset.
- Record `fs.solver.errors` now; later sections use the same error keys for comparison.
"""
        ),
        md("## Write PACE output files"),
        code(
            r"""
# Export everything required for downstream LAMMPS usage.
fs.output.write_lammps(fs.solver.fit)
fs.output.write_errors(fs.solver.errors)

for suffix in [".acecoeff", ".yace", ".mod"]:
    path = Path(settings["OUTFILE"]["potential"] + suffix)
    print(path, "exists:", path.exists(), "size:", path.stat().st_size if path.exists() else "missing")
"""
        ),
        what_next(
            """
- The `.yace`, `.mod`, and `.acecoeff` files feed LAMMPS and post-processing tools.
- Before moving on, note file sizes and names; you can reuse these exact paths in the MD notebook.
"""
        ),
        md(
            """
## Extract ACE descriptors from ASE frames

This section only inspects descriptor matrix shapes for a few structures. It checks that the calculator can process ASE `Atoms` objects; it does not validate a fitted model.
"""
        ),
        code(
            r"""
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
"""
        ),
        what_next(
            """
- Add more structures by changing `read(..., ":2")` to a larger slice and compare matrix size scaling.
- Keep `--nofit` when checking `process_single`; it is a fast way to verify scraper compatibility
  before a full refit.
"""
        ),
        md(
            """
## Historical SNAP comparison

The old tutorial centered on SNAP bispectrum settings. This optional cell shows the old style of calculator configuration without running it by default.
"""
        ),
        code(
            r"""
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
"""
        ),
        what_next(
            """
- Leave this section off during routine class runs to save time.
- To compare frameworks more directly, keep the same `GROUPS` and training split, then compare
  only `SOLVER`/`CALCULATOR` sections.
"""
        ),
        md(
            """
### Useful utility scripts to explore next (non-NN)

Once this notebook runs, try these short utility scripts from `examples/library`:

- `basic_examples/example1.py`: small linear fit workflow with compact diagnostics.
- `loop_over_fits/example1.py`: sweep descriptor hyperparameters as an extension of this notebook.
- `loop_over_fits/example2.py`: compute descriptors once, refit many times.
- `transpose_trick/example.py` and `transpose_trick/README.md`: memory-aware descriptor assembly.
"""
        ),
    ]


def solver_cells():
    return [
        md(
            """
# ACE solvers and small hyperparameter optimization

This notebook compares linear ACE fits with SVD, RIDGE, and ARD; scans a small RIDGE regularization list; and includes a reduced genetic-algorithm example. The GA demonstration optimizes group weights after descriptors are computed. It does not optimize descriptor-changing parameters such as `ranks`, `lmax`, `nmax`, or cutoffs.
"""
        ),
        md(
            """
Start here for student execution (local):
1) run the setup and version checks,
2) inspect the upstream examples for what was adapted,
3) run each subsection in order and read the short explanation before the code.

Google Colab execution:
1) keep the same cells; setup/build work is auto-ran when `ON_COLAB` is true,
2) keep all `~/` paths as written because this notebook is designed to be run from the same
    working layout in Colab and local WSL.
"""
        ),
        COMMON_SETUP,
        what_next(
            """
- This bootstrap cell is the first thing to run for local or Colab execution.
- In a university lab, keep this unchanged and only edit paths in `FITSNAP_DIR` if needed.
"""
        ),
        VERSION_CELL,
        what_next(
            """
- Confirm that the `pip` and import checks here match what your teaching environment expects.
- If a dependency is missing here, install it once in this environment before proceeding to the data inputs.
"""
        ),
        IMPORTS,
        what_next(
            """
- This is the base `FitSnap` + MPI setup used by all notebooks. Keep this as-is for class
  and focus edits on `settings` dictionaries below.
"""
        ),
        *default_ace_tool_cells(),
        what_next(
            """
- If this environment section fails, return to local/Colab setup notes and resolve the mismatch
  before moving into solver comparisons.
"""
        ),
        what_next(
            """
- Use `default_ACE_settings.py` once for each chemistry class and paste suggested
  values into the local `base_settings`.
- If you already have stable settings for Ta, skip to the solver comparison and keep the
  rest of the notebook focused on bias/variance tuning.
"""
        ),
        md("## What this notebook is fitting"),
        md(
r"""
In each subsection below, FitSNAP fits a linear model:

`y_hat = Phi w`

`y` are target labels from the data (energies, forces, and stress components), and `Phi`
contains ACE descriptor values built from each atomic neighborhood.

SVD solves a standard least-squares problem.
RIDGE also penalizes large coefficients:

`min_w ||Phi w - y||_2^2 + alpha * ||w||_2^2`

Larger `alpha` usually gives smoother models (smaller coefficients), while very small
`alpha` can increase variance and can overfit training structures.
"""
        ),
        md("## Upstream solver examples"),
        md(
            """
### What is being shown here

The next cell prints the original upstream inputs that inspired this notebook. This keeps your workflow traceable: you can see exactly what was changed to make the example classroom-sized.
"""
        ),
        code(
            r"""
# These are the upstream files we are adapting into smaller, explicit notebook runs.
for rel in ["examples/Ta_PACE/Ta.in", "examples/Ta_PACE_RIDGE/Ta.in", "examples/Ta_PACE_ARD/Ta.in", "examples/library/genetic_algorithm/ACE_Ta.in"]:
    path = FITSNAP_DIR / rel
    print("\n---", rel, "---")
    print("\n".join(path.read_text().splitlines()[:28]))
"""
        ),
        what_next(
            """
- Keep these files open as references; edit only the notebook settings dicts for experiments.
- For a first pass, keep data paths and group names identical and vary one solver input at a time.
"""
        ),
        md(
            """
### Why compare solvers

All three methods are linear models:
- SVD gives a direct pseudo-inverse estimate.
- RIDGE adds a penalty on large coefficients to reduce variance.
- ARD can shrink less useful coefficients, but its behavior depends on scikit-learn compatibility.

If we only improve training error while validation error does not improve, we are likely overfitting.
"""
        ),
        md(
            """
### Common pitfall map (what to adjust for what effect)

* `ranks`, `lmax`, `nmax`, `nmaxbase`: change basis richness (descriptor dimension).
* `rcutfac`, `lambda`, `rcinner` etc.: change the radial/angular descriptor scaling.
* group weights in `[GROUPS]`: change which configurations matter most.
* solver hyperparameters such as `alpha` in RIDGE: regularize the coefficient fit.

Keep one change at a time while learning, and compare one metric at a time to understand what changed.
"""
        ),
        settings_cell("base_settings", "Ta_ace_solver_base", "SVD"),
        what_next(
            """
- Edit one ACE knob here first (`ranks`, `lmax`, `nmax`, `nmaxbase`) before trying different solvers.
- Keep group weights fixed while you compare SVD/RIDGE/ARD behavior, then optimize weights after you understand solver effects.
"""
        ),
        md("## ARD compatibility note"),
        md(
            """
### What this section is doing and why

Sometimes the local `scikit-learn` version has changed argument names from older FitSNAP expectations. This small patch keeps the ARD section runnable for this notebook rather than failing at import time.
"""
        ),
        code(
            r"""
import inspect
from sklearn.linear_model import ARDRegression as SklearnARDRegression
import fitsnap3lib.solvers.ard as fitsnap_ard_module

# Older FitSNAP ARD calls can break with newer scikit-learn; this keeps the
# educational notebook runnable while still making the compatibility path explicit.
if "n_iter" not in inspect.signature(SklearnARDRegression).parameters:
    class ARDRegressionCompat(SklearnARDRegression):
        def __init__(
            self,
            *,
            n_iter=None,
            max_iter=300,
            tol=1.0e-3,
            alpha_1=1.0e-6,
            alpha_2=1.0e-6,
            lambda_1=1.0e-6,
            lambda_2=1.0e-6,
            compute_score=False,
            threshold_lambda=10000.0,
            fit_intercept=True,
            copy_X=True,
            verbose=False,
        ):
            self.n_iter = n_iter
            if n_iter is not None:
                max_iter = n_iter
            super().__init__(
                max_iter=max_iter,
                tol=tol,
                alpha_1=alpha_1,
                alpha_2=alpha_2,
                lambda_1=lambda_1,
                lambda_2=lambda_2,
                compute_score=compute_score,
                threshold_lambda=threshold_lambda,
                fit_intercept=fit_intercept,
                copy_X=copy_X,
                verbose=verbose,
            )

    fitsnap_ard_module.ARDRegression = ARDRegressionCompat
    print("Patched FitSNAP ARDRegression for scikit-learn max_iter API.")
else:
    print("No ARDRegression compatibility patch needed.")
"""
        ),
        what_next(
            """
- This is a local compatibility workaround; confirm the imported signature once each environment changes.
- If you see ARD import/fit failures, set `RUN_GA_DEMO` and continue with SVD/RIDGE as a controlled baseline.
"""
        ),
        md("## Compare SVD, RIDGE, and ARD"),
        md(
            """
### What is being done

Each run below uses the same Ta training data and group definitions, and changes only the solver settings. This isolates solver behavior from data choice.
"""
        ),
        code(
            r"""
def summarize_solver_errors(label, errors, fit_vector):
    # Return a compact numeric summary for a single fit.
    error_df = pd.DataFrame(errors) if not isinstance(errors, pd.DataFrame) else errors.copy()
    print(f"\n{label}: type(errors)={type(errors)}")
    print("columns:", list(error_df.columns))
    numeric = error_df.select_dtypes(include=[np.number])

    metric_cols = [c for c in error_df.columns if isinstance(c, str) and ("rmse" in c.lower() or "mae" in c.lower())]
    row = {"label": label}
    for col in metric_cols:
        values = pd.to_numeric(error_df[col], errors="coerce")
        row[f"{col}_mean"] = float(np.nanmean(values.to_numpy()))
        row[f"{col}_min"] = float(np.nanmin(values.to_numpy()))

    if not numeric.empty:
        flat = numeric.stack()
        if len(flat) > 0:
            row["mean_numeric_error_metric"] = float(np.nanmean(flat.to_numpy(dtype=float)))
    if "force_mae" not in row:
        # A very mild fallback so students still get a scalar summary.
        row["force_mae_mean"] = np.nan

    fit_arr = np.asarray(fit_vector, dtype=float).ravel()
    row["n_coeff"] = int(fit_arr.size)
    row["max_abs_coeff"] = float(np.max(np.abs(fit_arr))) if fit_arr.size > 0 else 0.0
    row["mean_abs_coeff"] = float(np.mean(np.abs(fit_arr))) if fit_arr.size > 0 else 0.0
    return row, error_df

def print_train_test_quality(label, error_df):
    # Print a few high-level scalar diagnostics if those entries are present.
    error_df = pd.DataFrame(error_df) if not isinstance(error_df, pd.DataFrame) else error_df.copy()
    print(f"\nTrain/test summary for {label}:")
    if isinstance(error_df.columns, pd.MultiIndex):
        lookup = {
            "train_force_mae": ("*ALL", "Unweighted", "Training", "Force", "mae"),
            "test_force_mae": ("*ALL", "Unweighted", "Testing", "Force", "mae"),
            "train_force_rmse": ("*ALL", "Unweighted", "Training", "Force", "rmse"),
            "test_force_rmse": ("*ALL", "Unweighted", "Testing", "Force", "rmse"),
            "train_energy_mae": ("*ALL", "Unweighted", "Training", "Energy", "mae"),
            "test_energy_mae": ("*ALL", "Unweighted", "Testing", "Energy", "mae"),
        }
        for name, key in lookup.items():
            try:
                if key in error_df.columns:
                    value = error_df.loc[:, key].iloc[0]
                else:
                    value = error_df.xs(key).iloc[0]
                print(f"{name}: {value}")
            except Exception:
                print(f"{name}: unavailable")
    else:
        for col in error_df.columns:
            lower = str(col).lower()
            for target in ("train", "test"):
                if target in lower and ("force" in lower or "energy" in lower):
                    if "mae" in lower or "rmse" in lower:
                        first = pd.to_numeric(error_df[col], errors="coerce").iloc[0]
                        print(f"{col}: {first}")

solver_runs = []
summary_rows = []

for label, solver_name, extra_section in [
        ("SVD", "SVD", {}),
        ("RIDGE alpha=1e-5", "RIDGE", {"RIDGE": {"local_solver": 1, "alpha": 1.0e-5}}),
        ("ARD", "ARD", {"ARD": {"directmethod": 0, "scap": 1.0e-4, "scai": 1.0e-4, "logcut": 0.3}}),
]:
    # Rebuild settings per trial so each solver sees the same training split.
    trial = deepcopy(base_settings)
    trial["SOLVER"]["solver"] = solver_name
    trial["OUTFILE"]["metrics"] = str(WORK_ROOT / f"{solver_name.lower()}_metrics.md")
    trial["OUTFILE"]["potential"] = str(WORK_ROOT / f"{solver_name.lower()}_pot")
    trial.update(extra_section)

    print(f"\n=== {label} ===")
    snap = FitSnap(trial, comm=comm, arglist=["--overwrite"])
    snap.scrape_configs()
    snap.process_configs()
    try:
        snap.perform_fit()
    except TypeError as exc:
        if solver_name == "ARD" and "n_iter" in str(exc):
            print("ARD did not run even after the notebook compatibility patch.")
            print("Recorded error:", exc)
            solver_runs.append((label, None, str(exc)))
            continue
        raise
    display(snap.solver.errors)
    print_train_test_quality(label, snap.solver.errors)
    row, _ = summarize_solver_errors(label, snap.solver.errors, snap.solver.fit)
    summary_rows.append(row)
    solver_runs.append((label, snap.solver.errors.copy(), None))

pd.DataFrame(summary_rows)
"""
        ),
        what_next(
            """
- Compare only one thing at once: run one solver first, inspect the summary, then switch to
  the next solver.
- Keep track of which metric moves most: force MAE often dominates structure relaxation quality.
"""
        ),
        md(
            """
### How to read the summary

The summary table below is a lightweight comparison:
- smaller RMSE/MAE columns are usually better,
- `max_abs_coeff` can suggest potential instability (large values are often a hint to increase regularization or rescale inputs),
- a change in `n_coeff` means the model size changed.
"""
        ),
        md("## Small RIDGE alpha scan"),
        md(
            """
### Why scan `alpha`

`alpha` controls ridge strength. In this subsection we vary only `alpha` and keep the basis fixed so students can see the fit bias/variance trade-off.
"""
        ),
        code(
            r"""
alpha_results = []
ridge_rows = []
np.random.seed(0)

for alpha in [1.0e-6, 1.0e-5, 1.0e-4]:
    # Keep the solver fixed and only change the regularization strength.
    trial = deepcopy(base_settings)
    trial["SOLVER"]["solver"] = "RIDGE"
    trial["RIDGE"] = {"local_solver": 1, "alpha": alpha}
    trial["OUTFILE"]["metrics"] = str(WORK_ROOT / f"ridge_alpha_{alpha:.0e}_metrics.md")
    trial["OUTFILE"]["potential"] = str(WORK_ROOT / f"ridge_alpha_{alpha:.0e}_pot")

    snap = FitSnap(trial, comm=comm, arglist=["--overwrite"])
    snap.scrape_configs()
    snap.process_configs()
    snap.perform_fit()
    errors = snap.solver.errors
    print(f"alpha={alpha}")
    display(errors)
    alpha_results.append((alpha, errors.copy()))
    print_train_test_quality(f"RIDGE alpha={alpha}", errors)
    row, _ = summarize_solver_errors(f"ridge alpha={alpha}", errors, snap.solver.fit)
    row["alpha"] = alpha
    ridge_rows.append(row)

pd.DataFrame(ridge_rows)
"""
        ),
        what_next(
            """
- Make your own 3-point scan on `alpha` by editing only this list.
- A stable `alpha` trend often looks like: training error rises slowly, test error bottoms out
  near an intermediate value.
"""
        ),
        md(
            """
### Descriptor-size scan

This section changes basis-size knobs while holding solver style fixed. Bigger descriptor spaces can reduce training error but often increase coefficient magnitude and risk overfitting if the data set is not rich enough.
"""
        ),
        md("## Descriptor-size scan"),
        code(
            r"""
descriptor_results = []
descriptor_rows = []
np.random.seed(0)

for ranks, lmax, nmax, nbase in [
    ("1 2", "0 1", "6 2", 6),
    ("1 2 3", "0 2 2", "8 3 1", 8),
]:
    # Change one descriptor knob at a time to compare model size and error behavior.
    trial = deepcopy(base_settings)
    trial["ACE"].update({"ranks": ranks, "lmax": lmax, "nmax": nmax, "nmaxbase": nbase})
    trial["SOLVER"]["solver"] = "RIDGE"
    trial["RIDGE"] = {"local_solver": 1, "alpha": 1.0e-5}
    trial["OUTFILE"]["potential"] = str(WORK_ROOT / f"descriptor_scan_r{ranks.replace(' ', '')}_pot")
    trial["OUTFILE"]["metrics"] = str(WORK_ROOT / f"descriptor_scan_r{ranks.replace(' ', '')}_metrics.md")

    snap = FitSnap(trial, comm=comm, arglist=["--overwrite"])
    snap.scrape_configs()
    snap.process_configs()
    snap.perform_fit()
    ncoeff = len(np.ravel(snap.solver.fit))
    print(f"ranks={ranks}; coefficients={ncoeff}")
    display(snap.solver.errors)
    descriptor_results.append((ranks, ncoeff, snap.solver.errors.copy()))
    print_train_test_quality(f"descriptor ranks={ranks}", snap.solver.errors)
    row, _ = summarize_solver_errors(f"descriptor {ranks}", snap.solver.errors, snap.solver.fit)
    row["ranks"] = ranks
    row["lmax"] = lmax
    row["nmax"] = nmax
    row["nbase"] = nbase
    descriptor_rows.append(row)

pd.DataFrame(descriptor_rows)
"""
        ),
        what_next(
            """
- Edit one descriptor knob at a time so students can connect model complexity changes to
  `n_coeff` growth and error behavior.
- If coefficient sizes jump sharply, reduce complexity or increase regularization.
"""
        ),
        md(
            """
### Tiny GA example: what to vary

The GA is a short optimization of group weights only (not descriptor parameters). In practice, students often see better gains from adjusting group weights before changing basis-size hyperparameters.

FitSNAP's optimization objectives are often written as a weighted RMSE combination:

`Q = w_E * RMSE(E) + w_F * RMSE(F)`

In this tiny notebook run we keep this objective unchanged and only explore a very small
set of weight combinations so students can see the workflow end-to-end.
"""
        ),
        md("## Tiny genetic-algorithm group-weight example"),
        code(
            r"""
RUN_GA_DEMO = True

ga_dir = WORK_ROOT / "ga_demo"
ga_dir.mkdir(parents=True, exist_ok=True)

# Copy the upstream GA example files into a private scratch folder.
for filename in ["script_optimize.py", "libmod_optimize.py", "ACE_Ta.in"]:
    shutil.copy2(FITSNAP_DIR / "examples" / "library" / "genetic_algorithm" / filename, ga_dir / filename)

# Make the run tiny for notebook execution: 4 individuals, 1 generation.
script_path = ga_dir / "script_optimize_tiny.py"
script_text = (ga_dir / "script_optimize.py").read_text()
script_text = script_text.replace("population_size = 30", "population_size = 4")
script_text = script_text.replace("ngenerations = 20", "ngenerations = 1")
script_text = script_text.replace("conv_check = 0.5", "conv_check = 1.0")
script_text = script_text.replace("write_to_json = False", "write_to_json = True")
script_path.write_text(script_text)

input_path = ga_dir / "ACE_Ta.in"
input_text = input_path.read_text()
input_text = input_text.replace("dataPath = ../../Ta_Linear_JCP2014/JSON", f"dataPath = {FITSNAP_DIR / 'examples' / 'Ta_Linear_JCP2014' / 'JSON'}")
input_text = input_text.replace("1.0    0.0        1.0           10.0", "0.10   0.0        1.0           10.0")
input_path.write_text(input_text)

if RUN_GA_DEMO:
    # Run the GA in the copied example directory so FITSNAP paths stay simple.
    result = subprocess.run(
        [sys.executable, script_path.name, "--fitsnap_in", input_path.name, "--optimization_style", "genetic_algorithm"],
        cwd=ga_dir,
        text=True,
        capture_output=True,
        check=False,
        timeout=1800,
    )
    print(result.stdout[-4000:])
    if result.returncode != 0:
        print(result.stderr[-4000:])
        raise RuntimeError(f"GA demo failed with exit status {result.returncode}")

    # Highlight a few final status lines for a quick classroom read.
    for line in result.stdout.splitlines():
        low = line.lower()
        if "best" in low or "score" in low or "generation" in low:
            print(line)
else:
    print("GA demo skipped. Set RUN_GA_DEMO = True to run it.")
"""
        ),
        what_next(
            """
- This GA run is intentionally tiny; increase `population_size` and `ngenerations` in a separate
  script after students understand how to read `best` and `score`.
- Check whether group-weight changes improve test errors more than just train errors.
"""
        ),
        md(
            """
### Next steps for this notebook

If you want a stronger comparison, increase the number of training configurations and then run the exact same loops with longer RIDGE scans and larger descriptor schedules in a script (outside the notebook) using fixed random seeds.

For larger production-style studies, check:
* `examples/library/loop_over_fits/example1.py` for repeatedly changing descriptor hyperparameters,
* `examples/library/loop_over_fits/example2.py` for reusing descriptors in fit loops,
* `examples/library/basic_examples/example1.py` for compact fit/error diagnostics,
* `examples/library/transpose_trick/example.py` for descriptor matrix memory strategies when models get large,
* `examples/library/transpose_trick/README.md` for when to use `mpirun` and how this affects workflows.
"""
        ),
    ]


def advanced_cells():
    return [
        md(
            """
# Multi-element ACE and LAMMPS MD workflow example

This notebook walks through two connected steps: first a compact InP multi-element ACE training example, then a Ta-to-LAMMPS MD example using a fitted potential.
The MD example is pedagogical. In this notebook we keep runs short; in non-interactive scripts, you can extend them for longer production-style checks.
"""
        ),
        md(
            """
Student start:
1) read the local/Colab note and version checks,
2) run the optional `default_ACE_settings.py` snippet for your element set,
3) run the InP fit cell before the LAMMPS MD cell so the data path exists.
"""
        ),
        COMMON_SETUP,
        what_next(
            """
- Run this once and verify dependencies before launching the InP or MD sections.
- If path assumptions differ from your setup, set `FITSNAP_DIR` and `LAMMPS_DIR` once here.
"""
        ),
        VERSION_CELL,
        what_next(
            """
- Run this once at the notebook top to verify path and import assumptions on your machine or Colab.
- If `FITSNAP_DIR`/`LAMMPS_DIR` are different, set them once and keep them constant.
"""
        ),
        IMPORTS,
        what_next(
            """
- This section creates the communicator and common imports used by all solver loops.
- Keep the `comm` object unchanged unless you are intentionally running true multi-rank MPI examples.
"""
        ),
        *default_ace_tool_cells(),
        what_next(
            """
- For a local run, keep `FITSNAP_DIR` and `LAMMPS_DIR` fixed until both the InP and MD steps work.
- This section assumes the previous utilities are available; if `default_ACE_settings.py` fails,
  verify that `FITSNAP_DIR` is a complete checkout first.
"""
        ),
        md("## Suggested `default_ACE_settings.py` values for InP"),
        code(
            r"""
# InP is a two-element system, so we ask for a two-element starting suggestion.
run_default_ace_settings("['In', 'P']", label="In, P", nshell=2.2)
"""
        ),
        what_next(
            """
- Compare the suggested In/P values to the Ta values and use them only as starting points.
- For binary systems, element order in `type = In P` is important for later `pair_coeff` and weight consistency.
"""
        ),
        md("## InP multi-element ACE input"),
        md(
            """
### What is being read

`InP-example.in` has type declarations (`type = In P`), element-specific cutoff arrays,
and a training/testing split that is useful to reuse for your own binary systems.
"""
        ),
        code(
            r"""
inp_source = FITSNAP_DIR / "examples" / "InP_PACE" / "InP-example.in"
print("\n".join(inp_source.read_text().splitlines()[:80]))
"""
        ),
        what_next(
            """
- This file is the exact upstream layout. Keep file section order and only swap the few
  training/validation weights and paths in the next notebook cell for fast classroom experiments.
- Confirm your binary dataset path matches the environment checkout before fitting.
"""
        ),
        md("## Reduced InP fit"),
        md(
            """
### What is being changed

To keep this notebook quick, group weights are reduced from full-production defaults and file paths are made absolute. The ACE descriptor block is otherwise kept intact so students can see which edits are safe defaults vs. structural edits.
"""
        ),
        code(
            r"""
inp_work = WORK_ROOT / "inp_demo"
inp_work.mkdir(parents=True, exist_ok=True)
inp_input = inp_work / "InP-example-reduced.in"
text = inp_source.read_text()
# Keep the sample fast by using small group weights and explicit absolute paths.
text = text.replace("dataPath = ../InP_JPCA2020/JSON", f"dataPath = {FITSNAP_DIR / 'examples' / 'InP_JPCA2020' / 'JSON'}")
text = text.replace("metrics = InP_metrics.md", f"metrics = {inp_work / 'InP_metrics.md'}")
text = text.replace("potential = InP_pot", f"potential = {inp_work / 'InP_pot'}")
for group in ["aa", "aIn", "aP", "Bulk", "EOS", "iIn", "iP", "s_aa", "s_aIn", "s_aP", "Shear", "s_iIn", "s_iP", "Strain", "s_vIn", "s_vP", "s_vv", "vP"]:
    text = text.replace(f"{group}       =     0.95      0.05", f"{group}       =     0.10      0.02")
inp_input.write_text(text)

RUN_INP_FIT = True
if RUN_INP_FIT:
    original_cwd = Path.cwd()
    os.chdir(inp_work)
    try:
        # Use a local filename after chdir to avoid path translation edge cases.
        inp_snap = FitSnap(inp_input.name, comm=comm, arglist=["--overwrite"])
        inp_snap.scrape_configs()
        print("InP configurations on this rank:", len(inp_snap.data))
        inp_snap.process_configs()
        inp_snap.perform_fit()
        display(inp_snap.solver.errors)
    finally:
        os.chdir(original_cwd)
else:
    print("InP fit skipped. Set RUN_INP_FIT = True to run it.")
"""
        ),
        what_next(
            """
- If fitting is slow, reduce groups first or increase `dataPath` cutoff rather than touching
  descriptor math first.
- Save the printed `inp_snap.solver.errors` as your benchmark before editing the next section.
"""
        ),
        md("## Train a Ta ACE potential for LAMMPS workflow example"),
        md(
            """
### Why include Ta training here

This keeps the downstream MD section concrete: this is the same exported `.yace` pattern used in a production script, but with a reduced basis for quick completion.
"""
        ),
        settings_cell("md_settings", "Ta_md_demo", "RIDGE", {"RIDGE": {"local_solver": 1, "alpha": 1.0e-5}}),
        what_next(
            """
- This is the Ta workflow block the MD section depends on; keep it working before changing InP settings.
- You can switch `RIDGE` settings here, or set `local_solver` to a different solver for comparison runs.
"""
        ),
        code(
            r"""
# This uses the same fit pipeline as the starter notebook.
md_snap = FitSnap(md_settings, comm=comm, arglist=["--overwrite"])
md_snap.scrape_configs()
md_snap.process_configs()
md_snap.perform_fit()
md_snap.output.write_lammps(md_snap.solver.fit)
md_snap.output.write_errors(md_snap.solver.errors)

yace_path = Path(md_settings["OUTFILE"]["potential"] + ".yace")
print("Wrote:", yace_path, "exists:", yace_path.exists())
display(md_snap.solver.errors)
"""
        ),
        what_next(
            """
- Reuse the same settings template for any single-element ACE training notebook.
- Keep this cell runnable even if you skip InP fitting; it creates the `.yace` file needed by the MD step.
"""
        ),
        md("## Run a Ta ACE MD example in LAMMPS"),
        md(
            """
### Educational MD workflow

In a notebook we run 100 steps to keep the workflow quick in class.

For the longer 10,000-step check in the same potential, we run a separate local script at
the end of this section (not inside the classroom notebook cell) so students can keep the
teaching path short while still verifying longer-time stability on their own machine.
"""
        ),
        code(
            r"""
lmp_exe = LAMMPS_DIR / "build-fs-lammps" / "lmp"
if ON_COLAB:
    lmp_exe = LAMMPS_DIR / "build-fitsnap-ace" / "lmp"
if not lmp_exe.exists():
    raise FileNotFoundError(f"LAMMPS executable not found: {lmp_exe}")

md_run_dir = WORK_ROOT / "md_workflow"
md_run_dir.mkdir(parents=True, exist_ok=True)
nsteps = 100
# Keep this short in notebook form so students can complete the full workflow.

# Build a minimal LAMMPS input for a short MD trajectory.
in_run = md_run_dir / "in.run"
in_lines = [
    "variable nrep equal 2",
    "units metal",
    "boundary p p p",
    "lattice bcc 3.316",
    "region box block 0 2 0 2 0 2",
    "create_box 1 box",
    "create_atoms 1 box",
    "mass 1 180.88",
    "pair_style pace product",
    f"pair_coeff * * {yace_path} Ta",
    "compute eatom all pe/atom",
    "compute energy all reduce sum c_eatom",
    "thermo_style custom step temp epair c_energy etotal press",
    "thermo 5",
    "thermo_modify norm yes",
    "timestep 0.5e-3",
    "neighbor 1.0 bin",
    "neigh_modify once no every 1 delay 0 check yes",
    "velocity all create 300.0 4928459 loop geom",
    "fix 1 all nvt temp 300.0 300.0 0.1",
    f"run {nsteps}",
]
in_run.write_text("\\n".join(in_lines) + "\\n")

print("Running LAMMPS input:", in_run)
result = subprocess.run([str(lmp_exe), "-in", str(in_run)], cwd=md_run_dir, text=True, capture_output=True, check=False)
print(result.stdout[-3000:])
if result.returncode != 0:
    print(result.stderr[-3000:])
    raise RuntimeError(f"LAMMPS MD example failed with exit status {result.returncode}")
"""
        ),
        what_next(
            """
- Keep `nsteps = 100` for this notebook cell. For your separate 10,000-step stability check, run the optional
  local helper cell below with `run_long_md = True`.
- Long-check logs are written to `WORK_ROOT / "md_long_logs"` with versioned file names and summarized in
  `version_manifest.txt`, so you can match each trajectory to the exact FitSNAP/LAMMPS revisions.
- For classroom comparison, log `epair` and temperature versus step and compare to a reference classical potential.
- Next step: switch `pair_style`/`pair_coeff` to a binary example once `Ta` path flow is validated.
"""
        ),
        md(
            """
## Optional local 10,000-step verification

This helper cell is intentionally separate so we do not accidentally run long jobs in the
teaching notebook, but you can execute it locally to verify your trained potential over a
longer trajectory.
"""
        ),
        code(
            r"""
run_long_md = False
RUN_LONG_MD_STEPS = 10000

md_long_log_dir = WORK_ROOT / "md_long_logs"
md_long_log_dir.mkdir(parents=True, exist_ok=True)
md_long_manifest = md_long_log_dir / "version_manifest.txt"

def _short_git_sha(path_like):
    # Keep short run logs keyed by source revisions for easy future recovery.
    git_sha = run_text(["git", "rev-parse", "--short", "HEAD"], cwd=str(path_like))
    return git_sha if git_sha else "unknown"

if run_long_md:
    long_in = md_run_dir / "in.run.10000"
    long_lines = in_lines[:]
    long_lines[-1] = f"run {RUN_LONG_MD_STEPS}"
    long_in.write_text("\\n".join(long_lines) + "\\n")

    fitsnap_sha = _short_git_sha(FITSNAP_DIR)
    lammps_sha = _short_git_sha(LAMMPS_DIR)
    system = Path(yace_path).stem
    long_log = md_long_log_dir / f"{system}_10000__fit{fitsnap_sha}__lammps-{lammps_sha}__nvt.log"

    print("Running optional long MD check:", long_in)
    long_result = subprocess.run(
        [str(lmp_exe), "-in", str(long_in), "-log", str(long_log)],
        cwd=md_run_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    print(long_result.stdout[-3000:])
    if long_result.returncode != 0:
        print(long_result.stderr[-3000:])
        raise RuntimeError(f"Optional 10,000-step MD check failed with exit status {long_result.returncode}")
    print(f"Long-MD log saved to: {long_log}")
    with md_long_manifest.open("a", encoding="utf-8") as fp:
        fp.write(f"{Path(yace_path).name} | fit={fitsnap_sha} | lammps={lammps_sha} | steps={RUN_LONG_MD_STEPS} | log={long_log}\n")
else:
    print("Optional long MD check skipped. Set run_long_md = True to run it locally.")
"""
        ),
        what_next(
            """
- If the optional 10,000-step check becomes unstable, reduce `alpha` in RIDGE and keep
  the same thermostat settings when you restart.
- After this run, move from single-element to the multi-element InP workflow with `pair_coeff * * InP_pot.yace In P`.
"""
        ),
        md(
            """
### Useful utilities to explore next (no neural networks)

For further undergrad investigation, try these scripts from `examples/library`:

- `basic_examples/example1.py` (compact fit/error loop),
- `loop_over_fits/example2.py` (reuse descriptors for repeated fits),
- `transpose_trick/example.py` (memory-efficient descriptor handling),
- `genetic_algorithm/script_optimize.py` (group-weight optimization workflow),
- `genetic_algorithm/README.md` (important notes on reproducibility and GA objective).
"""
        ),
    ]



write_notebook("tutorial.ipynb", beginner_cells())
write_notebook("tutorial_ace_solvers_and_optimization.ipynb", solver_cells())
write_notebook("tutorial_ace_inp_and_lammps_md.ipynb", advanced_cells())

