"""Importable utilities for loading FitSNAP JSON datasets into ASE objects.

This package intentionally has no command-line interface. It is meant for
notebooks, Python scripts, and data-science workflows.
"""

from .core import (
    BAR_TO_EV_PER_A3,
    FitSnapLoadConfig,
    FitSnapRecord,
    discover_fitsnap_json_files,
    filter_atoms,
    iter_fitsnap_records,
    load_fitsnap_json_file,
    load_fitsnap_json_folder,
    load_one_fitsnap_structure,
    read_fitsnap_json,
    summarize_atoms,
    write_ase_db,
    write_dataset,
    write_extxyz,
    write_grouped_extxyz,
    write_traj,
)

__all__ = [
    "BAR_TO_EV_PER_A3",
    "FitSnapLoadConfig",
    "FitSnapRecord",
    "discover_fitsnap_json_files",
    "filter_atoms",
    "iter_fitsnap_records",
    "load_fitsnap_json_file",
    "load_fitsnap_json_folder",
    "load_one_fitsnap_structure",
    "read_fitsnap_json",
    "summarize_atoms",
    "write_ase_db",
    "write_dataset",
    "write_extxyz",
    "write_grouped_extxyz",
    "write_traj",
]
