"""Notebook/script-friendly examples for fitsnap_ase_tools.

This file is not a command-line entry point. Copy pieces of it into notebooks or
import the functions directly in your own code.
"""

from fitsnap_ase_tools import (
    FitSnapLoadConfig,
    filter_atoms,
    load_fitsnap_json_folder,
    load_one_fitsnap_structure,
    summarize_atoms,
    write_dataset,
    write_grouped_extxyz,
)


# Example 1: load a whole repository or its JSON folder.
cfg = FitSnapLoadConfig()
# atoms = load_fitsnap_json_folder("/path/to/fitsnap/repo", cfg)


# Example 2: load only EOS structures.
eos_cfg = FitSnapLoadConfig(include_groups=["EOS_1"])
# eos_atoms = load_fitsnap_json_folder("/path/to/fitsnap/repo", eos_cfg)


# Example 3: load a metadata-filtered subset before ASE objects are created.
def only_small_structures(meta):
    return meta["fitsnap_natoms"] <= 64


#small_cfg = FitSnapLoadConfig(selection=only_small_structures)
# small_atoms = load_fitsnap_json_folder("/path/to/fitsnap/repo", small_cfg)


# Example 4: post-load filtering.
# be_w_structures = filter_atoms(atoms, contains_elements=["Be", "W"], max_atoms=100)


# Example 5: one structure from one JSON file.
one_atoms = load_one_fitsnap_structure("JSON/EOS_1/EOS_B2_1.json")


# Example 6: summary and export.
# df = summarize_atoms(atoms)
# write_dataset(atoms, "converted/all.extxyz")
# write_dataset(atoms, "converted/all.traj")
# write_dataset(atoms, "converted/all.db")
# write_grouped_extxyz(atoms, "converted/by_group")
