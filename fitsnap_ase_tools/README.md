# FitSNAP ASE Tools

Importable utilities for converting FitSNAP JSON datasets into ASE `Atoms` objects and ASE-friendly storage formats.

This is meant for notebooks, Python scripts, and data-science/AI workflows where you want to inspect, filter, transform, and export structures programmatically.

## Supported input layout

The default discovery logic expects the common FitSNAP repository layout:

```text
JSON/
  sub_folder/
    structure_file.json
```

A JSON file is expected to contain:

```python
{
    "Dataset": {
        "Label": "...",
        "LatticeStyle": "angstrom",
        "EnergyStyle": "electronvolt",
        "StressStyle": "bar",
        "AtomTypeStyle": "chemicalsymbol",
        "PositionsStyle": "angstrom",
        "ForcesStyle": "electronvoltperangstrom",
        "Data": [
            {
                "NumAtoms": 2,
                "Lattice": [[...], [...], [...]],
                "Energy": ...,
                "Stress": [[...], [...], [...]],
                "AtomTypes": ["W", "Be"],
                "Positions": [[...], [...]],
                "Forces": [[...], [...]],
            }
        ],
    }
}
```

Whole-line `#` comments at the top of files are accepted.

# Installation/Setup
See dependencies below.

To install as a python module, run

```
pip install .
```

in this directory `fitsnap_ase_tools` folder containing the README.md

alternatively, you can add this folder to your pythonpath:

`export PYTHONPATH=$PYTHONPATH:/path/to/FitSNAP/fitsnap_ase_tools`

If on windows, this will work best under Windows Subsystem Linux (WSL).


## Dependencies

```bash
pip install ase numpy
```

Optional:

```bash
pip install pandas json5
```

`pandas` gives nicer summary tables. `json5` is a fallback for loose JSON files.

## Quick usage

```python
from fitsnap_ase_tools import (
    FitSnapLoadConfig,
    load_fitsnap_json_folder,
    load_one_fitsnap_structure,
    summarize_atoms,
    filter_atoms,
    write_dataset,
    write_grouped_extxyz,
)

# Load everything under JSON/*/*.json.
atoms = load_fitsnap_json_folder("/path/to/FitSNAP/repo")

# Inspect as a pandas DataFrame when pandas is installed.
df = summarize_atoms(atoms)
print(df.head())

# Work with one structure as an ASE Atoms object.
a = atoms[0]
print(a)
print(a.info["fitsnap_group"])
print(a.get_potential_energy())
print(a.get_forces())

# Write a logically concatenated extended XYZ file.
write_dataset(atoms, "converted/all.extxyz")

# Write compact ASE-native formats.
write_dataset(atoms, "converted/all.traj")
write_dataset(atoms, "converted/all.db")

# Write one extxyz file per FitSNAP group/subfolder.
write_grouped_extxyz(atoms, "converted/by_group")
```

## Load subsets

### By FitSNAP group/subfolder

```python
cfg = FitSnapLoadConfig(include_groups=["EOS_1"])
atoms = load_fitsnap_json_folder("/path/to/repo", cfg)
```

### By filename regex

```python
cfg = FitSnapLoadConfig(include_file_regex=r"EOS|vacancy")
atoms = load_fitsnap_json_folder("/path/to/repo", cfg)
```

### By metadata callback before ASE construction

The callback receives a metadata dictionary with fields such as `fitsnap_group`, `fitsnap_file`, `fitsnap_natoms`, `fitsnap_energy`, and `fitsnap_elements`.

```python
def small_low_energy(meta):
    return meta["fitsnap_natoms"] <= 20 and meta.get("fitsnap_energy", 0.0) < 0.0

cfg = FitSnapLoadConfig(selection=small_low_energy)
atoms = load_fitsnap_json_folder("/path/to/repo", cfg)
```

### By ordinary Python after loading

```python
small_eos = filter_atoms(
    atoms,
    groups=["EOS_1"],
    contains_elements=["W", "Be"],
    max_atoms=20,
)
```

### Single structure from one file

```python
a = load_one_fitsnap_structure("JSON/EOS_1/EOS_B2_1.json", structure_index=0)
```

## Non-symbol atom types

If a dataset uses numeric or custom atom types instead of chemical symbols, pass a mapping:

```python
cfg = FitSnapLoadConfig(species_map={1: "W", 2: "Be"})
atoms = load_fitsnap_json_folder("/path/to/repo", cfg)
```

## Stress handling

FitSNAP example files often use `StressStyle='bar'`. ASE calculator stress expects eV/Å³ in Voigt order. Also, stress sign conventions can differ between codes/datasets.

Therefore this loader keeps raw stress in metadata by default:

```python
atoms[0].info["fitsnap_stress_raw"]
atoms[0].info["fitsnap_stress_style"]
```

To attach converted stress to the ASE `SinglePointCalculator`, opt in explicitly:

```python
cfg = FitSnapLoadConfig(
    include_stress_in_calculator=True,
    stress_sign=1.0,   # change to -1.0 if your target convention requires it
)
atoms = load_fitsnap_json_folder("/path/to/repo", cfg)
print(atoms[0].get_stress())
```

## API overview

Main loading functions:

- `discover_fitsnap_json_files(root, config=None)`
- `iter_fitsnap_records(root, config=None)`
- `load_fitsnap_json_folder(root, config=None)`
- `load_fitsnap_json_file(path, config=None)`
- `load_one_fitsnap_structure(path, structure_index=0, config=None)`

Main export functions:

- `write_dataset(atoms, path, fmt=None)`
- `write_extxyz(atoms, path)`
- `write_traj(atoms, path)`
- `write_ase_db(atoms, path)`
- `write_grouped_extxyz(atoms, output_dir)`

Exploration utilities:

- `summarize_atoms(atoms)`
- `filter_atoms(atoms, ...)`
