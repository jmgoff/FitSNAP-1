# -*- coding: utf-8 -*-
"""Exported from tutorial_ace_solvers_and_optimization.ipynb."""

# %% [markdown]
# # ACE solvers and small hyperparameter optimization
#
# This notebook compares linear ACE fits with SVD, RIDGE, and ARD; scans a small RIDGE regularization list; and includes a reduced genetic-algorithm example. The GA demonstration optimizes group weights after descriptors are computed. It does not optimize descriptor-changing parameters such as `ranks`, `lmax`, `nmax`, or cutoffs.

# %% [markdown]
# Start here for student execution (local):
# 1) run the setup and version checks,
# 2) inspect the upstream examples for what was adapted,
# 3) run each subsection in order and read the short explanation before the code.
#
# Google Colab execution:
# 1) keep the same cells; setup/build work is auto-ran when `ON_COLAB` is true,
# 2) keep all `~/` paths as written because this notebook is designed to be run from the same
#     working layout in Colab and local WSL.

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
# ### Next edit: This bootstrap cell is the first thing to run for local or Colab exec...
# - This bootstrap cell is the first thing to run for local or Colab execution.
# - In a university lab, keep this unchanged and only edit paths in `FITSNAP_DIR` if needed.

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
# ### Next edit: Confirm that the pip and import checks here match what your teaching ...
# - Confirm that the `pip` and import checks here match what your teaching environment expects.
# - If a dependency is missing here, install it once in this environment before proceeding to the data inputs.

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
# ### Next edit: This is the base FitSnap + MPI setup used by all notebooks. Keep this...
# - This is the base `FitSnap` + MPI setup used by all notebooks. Keep this as-is for class
#   and focus edits on `settings` dictionaries below.

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
# ### Next edit: If this environment section fails, return to local/Colab setup notes ...
# - If this environment section fails, return to local/Colab setup notes and resolve the mismatch
#   before moving into solver comparisons.

# %% [markdown]
# ### Next edit: Use default_ACE_settings.py once for each chemistry class and paste s...
# - Use `default_ACE_settings.py` once for each chemistry class and paste suggested
#   values into the local `base_settings`.
# - If you already have stable settings for Ta, skip to the solver comparison and keep the
#   rest of the notebook focused on bias/variance tuning.

# %% [markdown]
# ## What this notebook is fitting

# %% [markdown]
# In each subsection below, FitSNAP fits a linear model:
#
# `y_hat = Phi w`
#
# `y` are target labels from the data (energies, forces, and stress components), and `Phi`
# contains ACE descriptor values built from each atomic neighborhood.
#
# SVD solves a standard least-squares problem.
# RIDGE also penalizes large coefficients:
#
# `min_w ||Phi w - y||_2^2 + alpha * ||w||_2^2`
#
# Larger `alpha` usually gives smoother models (smaller coefficients), while very small
# `alpha` can increase variance and can overfit training structures.

# %% [markdown]
# ## Upstream solver examples

# %% [markdown]
# ### What is being shown here
#
# The next cell prints the original upstream inputs that inspired this notebook. This keeps your workflow traceable: you can see exactly what was changed to make the example classroom-sized.

# %%
# These are the upstream files we are adapting into smaller, explicit notebook runs.
for rel in ["examples/Ta_PACE/Ta.in", "examples/Ta_PACE_RIDGE/Ta.in", "examples/Ta_PACE_ARD/Ta.in", "examples/library/genetic_algorithm/ACE_Ta.in"]:
    path = FITSNAP_DIR / rel
    print("\n---", rel, "---")
    print("\n".join(path.read_text().splitlines()[:28]))

# %% [markdown]
# ### Next edit: Keep these files open as references; edit only the notebook settings ...
# - Keep these files open as references; edit only the notebook settings dicts for experiments.
# - For a first pass, keep data paths and group names identical and vary one solver input at a time.

# %% [markdown]
# ### Why compare solvers
#
# All three methods are linear models:
# - SVD gives a direct pseudo-inverse estimate.
# - RIDGE adds a penalty on large coefficients to reduce variance.
# - ARD can shrink less useful coefficients, but its behavior depends on scikit-learn compatibility.
#
# If we only improve training error while validation error does not improve, we are likely overfitting.

# %% [markdown]
# ### Common pitfall map (what to adjust for what effect)
#
# * `ranks`, `lmax`, `nmax`, `nmaxbase`: change basis richness (descriptor dimension).
# * `rcutfac`, `lambda`, `rcinner` etc.: change the radial/angular descriptor scaling.
# * group weights in `[GROUPS]`: change which configurations matter most.
# * solver hyperparameters such as `alpha` in RIDGE: regularize the coefficient fit.
#
# Keep one change at a time while learning, and compare one metric at a time to understand what changed.

# %%
# Base Ta ACE input. Changing this one block (ACE + GROUPS) is usually enough to start
# experimentation for a different system.
ta_data = FITSNAP_DIR / "examples" / "Ta_Linear_JCP2014" / "JSON"
base_settings = {
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
        "metrics": str(WORK_ROOT / "Ta_ace_solver_base_metrics.md"),
        "potential": str(WORK_ROOT / "Ta_ace_solver_base_pot"),
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
print("Data path:", base_settings["PATH"]["dataPath"])
print("Potential output prefix:", base_settings["OUTFILE"]["potential"])

# %% [markdown]
# ### Next edit: Edit one ACE knob here first (ranks, lmax, nmax, nmaxbase) before try...
# - Edit one ACE knob here first (`ranks`, `lmax`, `nmax`, `nmaxbase`) before trying different solvers.
# - Keep group weights fixed while you compare SVD/RIDGE/ARD behavior, then optimize weights after you understand solver effects.

# %% [markdown]
# ## ARD compatibility note

# %% [markdown]
# ### What this section is doing and why
#
# Sometimes the local `scikit-learn` version has changed argument names from older FitSNAP expectations. This small patch keeps the ARD section runnable for this notebook rather than failing at import time.

# %%
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

# %% [markdown]
# ### Next edit: This is a local compatibility workaround; confirm the imported signat...
# - This is a local compatibility workaround; confirm the imported signature once each environment changes.
# - If you see ARD import/fit failures, set `RUN_GA_DEMO` and continue with SVD/RIDGE as a controlled baseline.

# %% [markdown]
# ## Compare SVD, RIDGE, and ARD

# %% [markdown]
# ### What is being done
#
# Each run below uses the same Ta training data and group definitions, and changes only the solver settings. This isolates solver behavior from data choice.

# %%
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

# %% [markdown]
# ### Next edit: Compare only one thing at once: run one solver first, inspect the sum...
# - Compare only one thing at once: run one solver first, inspect the summary, then switch to
#   the next solver.
# - Keep track of which metric moves most: force MAE often dominates structure relaxation quality.

# %% [markdown]
# ### How to read the summary
#
# The summary table below is a lightweight comparison:
# - smaller RMSE/MAE columns are usually better,
# - `max_abs_coeff` can suggest potential instability (large values are often a hint to increase regularization or rescale inputs),
# - a change in `n_coeff` means the model size changed.

# %% [markdown]
# ## Small RIDGE alpha scan

# %% [markdown]
# ### Why scan `alpha`
#
# `alpha` controls ridge strength. In this subsection we vary only `alpha` and keep the basis fixed so students can see the fit bias/variance trade-off.

# %%
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

# %% [markdown]
# ### Next edit: Make your own 3-point scan on alpha by editing only this list.
# - Make your own 3-point scan on `alpha` by editing only this list.
# - A stable `alpha` trend often looks like: training error rises slowly, test error bottoms out
#   near an intermediate value.

# %% [markdown]
# ### Descriptor-size scan
#
# This section changes basis-size knobs while holding solver style fixed. Bigger descriptor spaces can reduce training error but often increase coefficient magnitude and risk overfitting if the data set is not rich enough.

# %% [markdown]
# ## Descriptor-size scan

# %%
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

# %% [markdown]
# ### Next edit: Edit one descriptor knob at a time so students can connect model comp...
# - Edit one descriptor knob at a time so students can connect model complexity changes to
#   `n_coeff` growth and error behavior.
# - If coefficient sizes jump sharply, reduce complexity or increase regularization.

# %% [markdown]
# ### Tiny GA example: what to vary
#
# The GA is a short optimization of group weights only (not descriptor parameters). In practice, students often see better gains from adjusting group weights before changing basis-size hyperparameters.
#
# FitSNAP's optimization objectives are often written as a weighted RMSE combination:
#
# `Q = w_E * RMSE(E) + w_F * RMSE(F)`
#
# In this tiny notebook run we keep this objective unchanged and only explore a very small
# set of weight combinations so students can see the workflow end-to-end.

# %% [markdown]
# ## Tiny genetic-algorithm group-weight example

# %%
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

# %% [markdown]
# ### Next edit: This GA run is intentionally tiny; increase population_size and ngene...
# - This GA run is intentionally tiny; increase `population_size` and `ngenerations` in a separate
#   script after students understand how to read `best` and `score`.
# - Check whether group-weight changes improve test errors more than just train errors.

# %% [markdown]
# ### Next steps for this notebook
#
# If you want a stronger comparison, increase the number of training configurations and then run the exact same loops with longer RIDGE scans and larger descriptor schedules in a script (outside the notebook) using fixed random seeds.
#
# For larger production-style studies, check:
# * `examples/library/loop_over_fits/example1.py` for repeatedly changing descriptor hyperparameters,
# * `examples/library/loop_over_fits/example2.py` for reusing descriptors in fit loops,
# * `examples/library/basic_examples/example1.py` for compact fit/error diagnostics,
# * `examples/library/transpose_trick/example.py` for descriptor matrix memory strategies when models get large,
# * `examples/library/transpose_trick/README.md` for when to use `mpirun` and how this affects workflows.
