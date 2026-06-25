"""Notebook/script-friendly examples for fitsnap_ase_tools.
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
cfg = FitSnapLoadConfig(
    include_stress_in_calculator=True,
    stress_sign=1.0,
)
atoms = load_fitsnap_json_folder("JSON", cfg)


# Print pandas dataframe about structures
#['structure_id', 'group', 'file', 'index', 'natoms', 'elements', 'formula_counts', 'energy_eV', 'energy_eV_per_atom']
df = summarize_atoms(atoms)
print('summary\n')
print(df)
print('end summary\n\n\n')

# count the number of structures
number_of_structures = len(atoms)
# select first structure (index 0)
first_structure = atoms[0] #NOTE this is the ASE atoms object you want to get SOAP fingerprints for

#TODO evaluate fingerprints for first structure (e.g. SOAP fingerprints)

print(df.iloc[0], first_structure) # optionally you can print the summarized information for that 'first_structure'


# Exmaples of writing out the structures in a non-json format (easier for later loading)

#write_dataset(atoms, "converted/all.extxyz")
#write_dataset(atoms, "converted/all.traj")
#write_dataset(atoms, "converted/all.db")
#write_grouped_extxyz(atoms, "converted/by_group")
