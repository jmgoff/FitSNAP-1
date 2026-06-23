# -*- coding: utf-8 -*-
"""Exported from tutorial_ace_inp_and_lammps_md.ipynb."""

# %% [markdown]
# # Multi-element ACE and LAMMPS MD workflow example
#
# This notebook walks through two connected steps: first a compact InP multi-element ACE training example, then a Ta-to-LAMMPS MD example using a fitted potential.
# The MD example is pedagogical. In this notebook we keep runs short; in non-interactive scripts, you can extend them for longer production-style checks.

# %% [markdown]
# Student start:
# 1) read the local/Colab note and version checks,
# 2) run the optional `default_ACE_settings.py` snippet for your element set,
# 3) run the InP fit cell before the LAMMPS MD cell so the data path exists.

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
# ### Next edit: Run this once and verify dependencies before launching the InP or MD ...
# - Run this once and verify dependencies before launching the InP or MD sections.
# - If path assumptions differ from your setup, set `FITSNAP_DIR` and `LAMMPS_DIR` once here.

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
# ### Next edit: Run this once at the notebook top to verify path and import assumptio...
# - Run this once at the notebook top to verify path and import assumptions on your machine or Colab.
# - If `FITSNAP_DIR`/`LAMMPS_DIR` are different, set them once and keep them constant.

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
# ### Next edit: This section creates the communicator and common imports used by all ...
# - This section creates the communicator and common imports used by all solver loops.
# - Keep the `comm` object unchanged unless you are intentionally running true multi-rank MPI examples.

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
# ### Next edit: A local run, keep fitsnap_dir and lammps_dir fixed until both the...
# - For a local run, keep `FITSNAP_DIR` and `LAMMPS_DIR` fixed until both the InP and MD steps work.
# - This section assumes the previous utilities are available; if `default_ACE_settings.py` fails,
#   verify that `FITSNAP_DIR` is a complete checkout first.

# %% [markdown]
# ## Suggested `default_ACE_settings.py` values for InP

# %%
# InP is a two-element system, so we ask for a two-element starting suggestion.
run_default_ace_settings("['In', 'P']", label="In, P", nshell=2.2)

# %% [markdown]
# ### Next edit: Compare the suggested In/P values to the Ta values and use them only ...
# - Compare the suggested In/P values to the Ta values and use them only as starting points.
# - For binary systems, element order in `type = In P` is important for later `pair_coeff` and weight consistency.

# %% [markdown]
# ## InP multi-element ACE input

# %% [markdown]
# ### What is being read
#
# `InP-example.in` has type declarations (`type = In P`), element-specific cutoff arrays,
# and a training/testing split that is useful to reuse for your own binary systems.

# %%
inp_source = FITSNAP_DIR / "examples" / "InP_PACE" / "InP-example.in"
print("\n".join(inp_source.read_text().splitlines()[:80]))

# %% [markdown]
# ### Next edit: This file is the exact upstream layout. Keep file section order and o...
# - This file is the exact upstream layout. Keep file section order and only swap the few
#   training/validation weights and paths in the next notebook cell for fast classroom experiments.
# - Confirm your binary dataset path matches the environment checkout before fitting.

# %% [markdown]
# ## Reduced InP fit

# %% [markdown]
# ### What is being changed
#
# To keep this notebook quick, group weights are reduced from full-production defaults and file paths are made absolute. The ACE descriptor block is otherwise kept intact so students can see which edits are safe defaults vs. structural edits.

# %%
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

# %% [markdown]
# ### Next edit: If fitting is slow, reduce groups first or increase dataPath cutoff r...
# - If fitting is slow, reduce groups first or increase `dataPath` cutoff rather than touching
#   descriptor math first.
# - Save the printed `inp_snap.solver.errors` as your benchmark before editing the next section.

# %% [markdown]
# ## Train a Ta ACE potential for LAMMPS workflow example

# %% [markdown]
# ### Why include Ta training here
#
# This keeps the downstream MD section concrete: this is the same exported `.yace` pattern used in a production script, but with a reduced basis for quick completion.

# %%
# Base Ta ACE input. Changing this one block (ACE + GROUPS) is usually enough to start
# experimentation for a different system.
ta_data = FITSNAP_DIR / "examples" / "Ta_Linear_JCP2014" / "JSON"
md_settings = {
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
    **{'SOLVER': {'solver': 'RIDGE', 'compute_testerrs': 1, 'detailed_errors': 1}, 'RIDGE': {'local_solver': 1, 'alpha': 1e-05}},
    "SCRAPER": {"scraper": "JSON"},
    "PATH": {"dataPath": str(ta_data)},
    "OUTFILE": {
        "output_style": "PACE",
        "metrics": str(WORK_ROOT / "Ta_md_demo_metrics.md"),
        "potential": str(WORK_ROOT / "Ta_md_demo_pot"),
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
print("Data path:", md_settings["PATH"]["dataPath"])
print("Potential output prefix:", md_settings["OUTFILE"]["potential"])

# %% [markdown]
# ### Next edit: This is the Ta workflow block the MD section depends on; keep it work...
# - This is the Ta workflow block the MD section depends on; keep it working before changing InP settings.
# - You can switch `RIDGE` settings here, or set `local_solver` to a different solver for comparison runs.

# %%
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

# %% [markdown]
# ### Next edit: Reuse the same settings template for any single-element ACE training ...
# - Reuse the same settings template for any single-element ACE training notebook.
# - Keep this cell runnable even if you skip InP fitting; it creates the `.yace` file needed by the MD step.

# %% [markdown]
# ## Run a Ta ACE MD example in LAMMPS

# %% [markdown]
# ### Educational MD workflow
#
# In a notebook we run 100 steps to keep the workflow quick in class.
#
# For the longer 10,000-step check in the same potential, we run a separate local script at
# the end of this section (not inside the classroom notebook cell) so students can keep the
# teaching path short while still verifying longer-time stability on their own machine.

# %%
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

# %% [markdown]
# ### Next edit: Keep nsteps = 100 for this notebook cell. For your separate 10,000-st...
# - Keep `nsteps = 100` for this notebook cell. For your separate 10,000-step stability check, run the optional
#   local helper cell below with `run_long_md = True`.
# - Long-check logs are written to `WORK_ROOT / "md_long_logs"` with versioned file names and summarized in
#   `version_manifest.txt`, so you can match each trajectory to the exact FitSNAP/LAMMPS revisions.
# - For classroom comparison, log `epair` and temperature versus step and compare to a reference classical potential.
# - Next step: switch `pair_style`/`pair_coeff` to a binary example once `Ta` path flow is validated.

# %% [markdown]
# ## Optional local 10,000-step verification
#
# This helper cell is intentionally separate so we do not accidentally run long jobs in the
# teaching notebook, but you can execute it locally to verify your trained potential over a
# longer trajectory.

# %%
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

# %% [markdown]
# ### Next edit: If the optional 10,000-step check becomes unstable, reduce alpha in R...
# - If the optional 10,000-step check becomes unstable, reduce `alpha` in RIDGE and keep
#   the same thermostat settings when you restart.
# - After this run, move from single-element to the multi-element InP workflow with `pair_coeff * * InP_pot.yace In P`.

# %% [markdown]
# ### Useful utilities to explore next (no neural networks)
#
# For further undergrad investigation, try these scripts from `examples/library`:
#
# - `basic_examples/example1.py` (compact fit/error loop),
# - `loop_over_fits/example2.py` (reuse descriptors for repeated fits),
# - `transpose_trick/example.py` (memory-efficient descriptor handling),
# - `genetic_algorithm/script_optimize.py` (group-weight optimization workflow),
# - `genetic_algorithm/README.md` (important notes on reproducibility and GA objective).
