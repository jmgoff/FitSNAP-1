from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence, Union
import json
import math
import re
import warnings

import numpy as np

PathLike = Union[str, Path]
Metadata = Mapping[str, Any]
SelectionCallback = Callable[[Metadata], bool]

# 1 eV / Angstrom^3 = 160.2176634 GPa = 1.602176634e11 Pa.
# 1 bar = 1e5 Pa.
BAR_TO_EV_PER_A3 = 1.0e5 / 1.602176634e11


@dataclass(frozen=True)
class FitSnapLoadConfig:
    """Options for reading FitSNAP JSON folders into ASE atoms.

    The defaults are conservative and data-science friendly:

    - If ``root/JSON`` exists, search there and ignore duplicated JSONs beside it.
    - Attach energy/forces through ASE's ``SinglePointCalculator`` when possible.
    - Keep FitSNAP stress as raw metadata by default, because FitSNAP examples often
      use ``StressStyle='bar'`` and sign conventions can differ between datasets.
    - Use callbacks and filters instead of command-line flags.
    """

    # File discovery ---------------------------------------------------------
    prefer_json_subdir: bool = True
    recursive: bool = True
    glob_pattern: str = "*.json"
    include_groups: Optional[Sequence[str]] = None
    exclude_groups: Sequence[str] = field(default_factory=tuple)
    include_file_regex: Optional[str] = None
    exclude_file_regex: Optional[str] = None

    # Structure selection ----------------------------------------------------
    structure_indices: Optional[Sequence[int]] = None
    max_structures: Optional[int] = None
    max_structures_per_file: Optional[int] = None
    selection: Optional[SelectionCallback] = None

    # Chemistry/ASE conversion -----------------------------------------------
    species_map: Optional[Mapping[Any, str]] = None
    pbc: Union[bool, Sequence[bool]] = True
    attach_calculator: bool = True
    include_forces: bool = True

    # Stress handling --------------------------------------------------------
    # FitSNAP examples commonly store stress in bar. ASE calculators expect
    # stress in eV/A^3 in Voigt order. Because sign conventions vary, raw stress
    # is retained in atoms.info by default, and calculator stress is opt-in.
    include_stress_in_calculator: bool = False
    stress_sign: float = 1.0
    keep_raw_stress_info: bool = True

    # Metadata ---------------------------------------------------------------
    keep_dataset_styles: bool = True
    keep_raw_record: bool = False
    config_type_from_group: bool = True

    # Error policy: "raise", "warn", or "skip".
    on_error: str = "raise"

    def with_updates(self, **kwargs: Any) -> "FitSnapLoadConfig":
        """Return a modified copy, convenient for notebooks/config dictionaries."""
        return replace(self, **kwargs)


@dataclass
class FitSnapRecord:
    """One structure read from a FitSNAP JSON file."""

    atoms: Any
    path: Path
    group: str
    structure_index: int
    metadata: dict[str, Any]
    dataset_metadata: dict[str, Any]
    raw_data: Optional[dict[str, Any]] = None


def _require_ase() -> tuple[Any, Any, Any, Any]:
    """Import ASE lazily so metadata utilities still import without ASE."""
    try:
        from ase import Atoms
        from ase.calculators.singlepoint import SinglePointCalculator
        from ase.io import write as ase_write
        from ase.db import connect as ase_db_connect
    except ImportError as exc:
        raise ImportError(
            "These utilities need ASE for Atoms conversion and writing. Install with "
            "`pip install ase`. Optional: `pip install json5 pandas`."
        ) from exc
    return Atoms, SinglePointCalculator, ase_write, ase_db_connect


def _strip_comment_lines(text: str) -> str:
    """Remove FitSNAP-style header/comment lines before JSON parsing.

    FitSNAP JSON examples often start with lines such as ``# A test JSON file``.
    Standard JSON parsers do not allow these lines. This function removes only
    whole lines whose first non-whitespace character is ``#``.
    """
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def read_fitsnap_json(path: PathLike) -> dict[str, Any]:
    """Read one FitSNAP JSON file.

    The reader first strips whole-line ``#`` comments, then tries Python's
    standard ``json`` module. If that fails, it falls back to ``json5`` when
    installed, which is useful for loose JSON variants.
    """
    path = Path(path)
    text = _strip_comment_lines(path.read_text())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import json5  # type: ignore
        except ImportError as exc:
            raise ValueError(
                f"Could not parse {path} as strict JSON. Install json5 or clean the file."
            ) from exc
        return json5.loads(text)


def _search_root(root: Path, config: FitSnapLoadConfig) -> Path:
    if config.prefer_json_subdir and root.name != "JSON" and (root / "JSON").is_dir():
        return root / "JSON"
    return root


def _group_for_path(path: Path, search_root: Path) -> str:
    parent = path.relative_to(search_root).parent
    if str(parent) in ("", "."):
        return "root"
    return parent.as_posix()


def _matches_file_filters(path: Path, group: str, config: FitSnapLoadConfig) -> bool:
    if config.include_groups is not None and group not in set(config.include_groups):
        return False
    if config.exclude_groups and group in set(config.exclude_groups):
        return False
    text = path.as_posix()
    if config.include_file_regex and re.search(config.include_file_regex, text) is None:
        return False
    if config.exclude_file_regex and re.search(config.exclude_file_regex, text) is not None:
        return False
    return True


def discover_fitsnap_json_files(
    root: PathLike, config: Optional[FitSnapLoadConfig] = None
) -> list[Path]:
    """Find FitSNAP JSON files under ``root``.

    If ``root/JSON`` exists and ``config.prefer_json_subdir=True``, discovery
    uses ``root/JSON``. This avoids double-counting repositories that contain
    both copied example files and the canonical ``JSON/sub_folder/file.json``
    layout.
    """
    config = config or FitSnapLoadConfig()
    root = Path(root)
    search_root = _search_root(root, config)
    if not search_root.exists():
        raise FileNotFoundError(f"No such FitSNAP JSON root: {search_root}")

    if config.recursive:
        files = sorted(search_root.rglob(config.glob_pattern))
    else:
        files = sorted(search_root.glob(config.glob_pattern))

    selected: list[Path] = []
    for path in files:
        group = _group_for_path(path, search_root)
        if _matches_file_filters(path, group, config):
            selected.append(path)
    return selected


def _as_dataset(document: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
    if "Dataset" in document:
        return document["Dataset"]
    # Helpful for tests or hand-made documents that contain the dataset directly.
    if "Data" in document:
        return document
    raise KeyError(f"{path} does not contain a `Dataset` object with a `Data` list.")


def _dataset_metadata(dataset: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in dataset.items() if k != "Data"}


def _coerce_symbols(
    atom_types: Sequence[Any],
    atom_type_style: Optional[str],
    species_map: Optional[Mapping[Any, str]],
    path: Path,
) -> list[str]:
    if species_map is not None:
        try:
            return [species_map[x] for x in atom_types]
        except KeyError as exc:
            raise KeyError(
                f"Atom type {exc.args[0]!r} in {path} is missing from species_map."
            ) from exc

    # The common FitSNAP style in the example files is chemical symbols.
    if atom_type_style is None or str(atom_type_style).lower() in {
        "chemicalsymbol",
        "chemicalsymbols",
        "symbol",
        "symbols",
    }:
        if all(isinstance(x, str) and x for x in atom_types):
            return [str(x) for x in atom_types]

    # If the JSON has type ids like 1, 2, 3, we cannot infer chemistry safely.
    if any(isinstance(x, (int, np.integer, float, np.floating)) for x in atom_types):
        raise ValueError(
            f"{path} uses non-symbol atom types. Pass a species_map, e.g. "
            "FitSnapLoadConfig(species_map={1: 'W', 2: 'Be'})."
        )

    return [str(x) for x in atom_types]


def _jsonable_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def _stress_matrix_to_voigt6(matrix: np.ndarray) -> np.ndarray:
    """Convert a symmetric 3x3 stress matrix to ASE Voigt order.

    ASE Voigt order is xx, yy, zz, yz, xz, xy.
    """
    m = np.asarray(matrix, dtype=float)
    if m.shape != (3, 3):
        raise ValueError(f"Expected stress shape (3, 3), got {m.shape}.")
    return np.array([m[0, 0], m[1, 1], m[2, 2], m[1, 2], m[0, 2], m[0, 1]])


def _stress_to_ev_per_a3(
    stress: Any, stress_style: Optional[str], config: FitSnapLoadConfig
) -> np.ndarray:
    style = (stress_style or "bar").lower().replace(" ", "")
    matrix = np.asarray(stress, dtype=float)
    if style in {"bar", "bars"}:
        factor = BAR_TO_EV_PER_A3
    elif style in {"ev/angstrom^3", "ev/ang^3", "electronvoltperangstromcubed", "ev/a^3"}:
        factor = 1.0
    elif style in {"gpa", "gigapascal", "gigapascals"}:
        # 1 GPa = 1e9 Pa.
        factor = 1.0e9 / 1.602176634e11
    else:
        raise ValueError(
            f"Unsupported StressStyle={stress_style!r}. Keep raw stress metadata or add a conversion."
        )
    return config.stress_sign * factor * matrix


def _make_metadata(
    *,
    path: Path,
    group: str,
    structure_index: int,
    dataset: Mapping[str, Any],
    data: Mapping[str, Any],
    symbols: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    atom_types = list(symbols if symbols is not None else data.get("AtomTypes", []))
    counts = dict(Counter(str(x) for x in atom_types))
    natoms = int(data.get("NumAtoms", len(atom_types)))
    energy = _safe_float(data.get("Energy"))
    rel_id = f"{group}/{path.stem}:{structure_index}" if group != "root" else f"{path.stem}:{structure_index}"
    metadata: dict[str, Any] = {
        "structure_id": rel_id,
        "fitsnap_source_path": str(path),
        "fitsnap_file": path.name,
        "fitsnap_stem": path.stem,
        "fitsnap_group": group,
        "fitsnap_structure_index": int(structure_index),
        "fitsnap_label": dataset.get("Label"),
        "fitsnap_natoms": natoms,
        "fitsnap_elements": ",".join(sorted(counts)),
        "fitsnap_element_counts": counts,
        "fitsnap_num_elements": len(counts),
    }
    if energy is not None:
        metadata["fitsnap_energy"] = energy
        if natoms:
            metadata["fitsnap_energy_per_atom"] = energy / natoms
    return metadata


def fitsnap_data_to_atoms(
    data: Mapping[str, Any],
    dataset_metadata: Mapping[str, Any],
    *,
    path: PathLike = "<memory>",
    group: str = "root",
    structure_index: int = 0,
    config: Optional[FitSnapLoadConfig] = None,
) -> Any:
    """Convert one FitSNAP ``Data`` entry to an ASE ``Atoms`` object."""
    Atoms, SinglePointCalculator, _ase_write, _ase_db_connect = _require_ase()
    config = config or FitSnapLoadConfig()
    path = Path(path)

    required = ["AtomTypes", "Positions", "Lattice"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{path} Data[{structure_index}] is missing required keys: {missing}")

    symbols = _coerce_symbols(
        data["AtomTypes"], dataset_metadata.get("AtomTypeStyle"), config.species_map, path
    )
    positions = np.asarray(data["Positions"], dtype=float)
    cell = np.asarray(data["Lattice"], dtype=float)
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=config.pbc)

    info = _make_metadata(
        path=path,
        group=group,
        structure_index=structure_index,
        dataset=dataset_metadata,
        data=data,
        symbols=symbols,
    )
    if config.config_type_from_group:
        info["config_type"] = group
    if config.keep_dataset_styles:
        for key, value in dataset_metadata.items():
            if key.endswith("Style"):
                info[f"fitsnap_{key}"] = _jsonable_scalar(value)
    if config.keep_raw_stress_info and "Stress" in data:
        info["fitsnap_stress_raw"] = np.asarray(data["Stress"], dtype=float).tolist()
        info["fitsnap_stress_style"] = dataset_metadata.get("StressStyle")

    atoms.info.update(info)

    # Optional per-atom extras.
    if "Charges" in data:
        charges = np.asarray(data["Charges"], dtype=float)
        try:
            atoms.set_initial_charges(charges)
        except Exception:
            atoms.arrays["initial_charges"] = charges
        atoms.arrays["fitsnap_charges"] = charges
    if "CoulPots" in data:
        atoms.arrays["fitsnap_coul_pots"] = np.asarray(data["CoulPots"], dtype=float)
    if "MagneticMoments" in data:
        magmoms = np.asarray(data["MagneticMoments"], dtype=float)
        try:
            atoms.set_initial_magnetic_moments(magmoms)
        except Exception:
            atoms.arrays["initial_magmoms"] = magmoms
        atoms.arrays["fitsnap_magnetic_moments"] = magmoms

    if config.attach_calculator:
        results: dict[str, Any] = {}
        energy = _safe_float(data.get("Energy"))
        if energy is not None:
            results["energy"] = energy
        if config.include_forces and "Forces" in data:
            results["forces"] = np.asarray(data["Forces"], dtype=float)
        if config.include_stress_in_calculator and "Stress" in data:
            stress_matrix = _stress_to_ev_per_a3(
                data["Stress"], dataset_metadata.get("StressStyle"), config
            )
            results["stress"] = _stress_matrix_to_voigt6(stress_matrix)
            atoms.info["fitsnap_stress_ev_per_a3"] = stress_matrix.tolist()
            atoms.info["fitsnap_stress_sign_used_for_ase"] = config.stress_sign
        if results:
            atoms.calc = SinglePointCalculator(atoms, **results)
    else:
        # If users do not want a calculator, keep forces as an ordinary array.
        if config.include_forces and "Forces" in data:
            atoms.arrays["forces"] = np.asarray(data["Forces"], dtype=float)

    return atoms


def _handle_error(exc: Exception, path: Path, config: FitSnapLoadConfig) -> bool:
    """Return True if caller should continue after the error."""
    if config.on_error == "raise":
        raise exc
    message = f"Skipping {path}: {exc}"
    if config.on_error == "warn":
        warnings.warn(message)
        return True
    if config.on_error == "skip":
        return True
    raise ValueError("on_error must be 'raise', 'warn', or 'skip'.")


def _iter_records_from_files(
    files: Iterable[Path],
    *,
    search_root: Path,
    config: FitSnapLoadConfig,
) -> Iterator[FitSnapRecord]:
    emitted_total = 0
    index_filter = set(config.structure_indices) if config.structure_indices is not None else None

    for path in files:
        group = _group_for_path(path, search_root) if path.is_relative_to(search_root) else path.parent.name
        try:
            document = read_fitsnap_json(path)
            dataset = _as_dataset(document, path)
            ds_meta = _dataset_metadata(dataset)
            data_list = list(dataset.get("Data", []))
        except Exception as exc:
            if _handle_error(exc, path, config):
                continue

        emitted_this_file = 0
        for structure_index, data in enumerate(data_list):
            if index_filter is not None and structure_index not in index_filter:
                continue
            if (
                config.max_structures_per_file is not None
                and emitted_this_file >= config.max_structures_per_file
            ):
                break
            if config.max_structures is not None and emitted_total >= config.max_structures:
                return

            try:
                # Make enough metadata before constructing ASE so selection callbacks
                # can filter by group, file, atom count, energy, etc.
                pre_meta = _make_metadata(
                    path=path,
                    group=group,
                    structure_index=structure_index,
                    dataset=ds_meta,
                    data=data,
                )
                if config.selection is not None and not config.selection(pre_meta):
                    continue
                atoms = fitsnap_data_to_atoms(
                    data,
                    ds_meta,
                    path=path,
                    group=group,
                    structure_index=structure_index,
                    config=config,
                )
                metadata = dict(atoms.info)
                raw_data = dict(data) if config.keep_raw_record else None
                yield FitSnapRecord(
                    atoms=atoms,
                    path=path,
                    group=group,
                    structure_index=structure_index,
                    metadata=metadata,
                    dataset_metadata=ds_meta,
                    raw_data=raw_data,
                )
                emitted_total += 1
                emitted_this_file += 1
            except Exception as exc:
                if _handle_error(exc, path, config):
                    continue


def iter_fitsnap_records(
    root: PathLike, config: Optional[FitSnapLoadConfig] = None
) -> Iterator[FitSnapRecord]:
    """Yield ``FitSnapRecord`` objects from a FitSNAP JSON folder."""
    config = config or FitSnapLoadConfig()
    root = Path(root)
    search_root = _search_root(root, config)
    files = discover_fitsnap_json_files(root, config)
    yield from _iter_records_from_files(files, search_root=search_root, config=config)


def load_fitsnap_json_folder(
    root: PathLike, config: Optional[FitSnapLoadConfig] = None
) -> list[Any]:
    """Load all selected structures under a FitSNAP JSON folder as ASE Atoms."""
    return [record.atoms for record in iter_fitsnap_records(root, config)]


def load_fitsnap_json_file(
    path: PathLike, config: Optional[FitSnapLoadConfig] = None
) -> list[Any]:
    """Load all selected structures from one FitSNAP JSON file as ASE Atoms.

    For a path like ``JSON/EOS_1/EOS_B2_1.json``, the default group metadata
    is ``EOS_1``.
    """
    config = config or FitSnapLoadConfig(prefer_json_subdir=False)
    path = Path(path)
    search_root = path.parent.parent if path.parent != path.parent.parent else path.parent
    return [
        record.atoms
        for record in _iter_records_from_files([path], search_root=search_root, config=config)
    ]


def load_one_fitsnap_structure(
    path: PathLike,
    structure_index: int = 0,
    config: Optional[FitSnapLoadConfig] = None,
) -> Any:
    """Load one structure from one FitSNAP JSON file as an ASE Atoms object."""
    base = config or FitSnapLoadConfig(prefer_json_subdir=False)
    cfg = base.with_updates(structure_indices=[structure_index], max_structures=1)
    atoms_list = load_fitsnap_json_file(path, cfg)
    if not atoms_list:
        raise IndexError(f"No structure index {structure_index} found in {path}.")
    return atoms_list[0]


def _as_atoms_list(atoms_or_records: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    for item in atoms_or_records:
        if isinstance(item, FitSnapRecord):
            out.append(item.atoms)
        else:
            out.append(item)
    return out


def write_extxyz(
    atoms_or_records: Iterable[Any],
    output_path: PathLike,
    *,
    append: bool = False,
) -> Path:
    """Write a logically concatenated extended XYZ file."""
    _Atoms, _SPC, ase_write, _db_connect = _require_ase()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atoms_list = _as_atoms_list(atoms_or_records)
    ase_write(str(output_path), atoms_list, format="extxyz", append=append)
    return output_path


def write_traj(
    atoms_or_records: Iterable[Any],
    output_path: PathLike,
    *,
    append: bool = False,
) -> Path:
    """Write an ASE trajectory file.

    ``.traj`` is compact and ASE-native. It is convenient for Python/ASE
    workflows, but less universal than extended XYZ.
    """
    _Atoms, _SPC, ase_write, _db_connect = _require_ase()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atoms_list = _as_atoms_list(atoms_or_records)
    ase_write(str(output_path), atoms_list, format="traj", append=append)
    return output_path


def _db_safe_key_values(info: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "structure_id",
        "config_type",
        "fitsnap_file",
        "fitsnap_stem",
        "fitsnap_group",
        "fitsnap_structure_index",
        "fitsnap_natoms",
        "fitsnap_num_elements",
        "fitsnap_elements",
        "fitsnap_energy",
        "fitsnap_energy_per_atom",
    }
    safe: dict[str, Any] = {}
    for key in allowed:
        if key not in info:
            continue
        value = info[key]
        if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
            safe[key] = _jsonable_scalar(value)
    return safe


def write_ase_db(
    atoms_or_records: Iterable[Any],
    output_path: PathLike,
    *,
    append: bool = False,
) -> Path:
    """Write an ASE SQLite database.

    The database stores query-friendly scalar metadata as columns and a copy of
    the full ``atoms.info`` dictionary in the row ``data`` payload.
    """
    _Atoms, _SPC, _ase_write, ase_db_connect = _require_ase()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not append:
        output_path.unlink()

    atoms_list = _as_atoms_list(atoms_or_records)
    with ase_db_connect(str(output_path)) as db:
        for atoms in atoms_list:
            kv = _db_safe_key_values(atoms.info)
            # Keep nested metadata in data, not query columns.
            data = {"info": dict(atoms.info)}
            db.write(atoms, data=data, **kv)
    return output_path


def write_dataset(
    atoms_or_records: Iterable[Any],
    output_path: PathLike,
    *,
    fmt: Optional[str] = None,
    append: bool = False,
) -> Path:
    """Write atoms to an ASE-friendly format.

    Parameters
    ----------
    atoms_or_records
        Iterable of ASE ``Atoms`` or ``FitSnapRecord`` objects.
    output_path
        Output file path.
    fmt
        One of ``'extxyz'``, ``'xyz'``, ``'db'``, ``'ase.db'``, ``'traj'``.
        If omitted, the suffix decides.
    append
        Append to an existing file/database when supported.
    """
    output_path = Path(output_path)
    suffix = output_path.suffix.lower().lstrip(".")
    fmt_norm = (fmt or suffix).lower()
    if fmt_norm in {"extxyz", "xyz"}:
        return write_extxyz(atoms_or_records, output_path, append=append)
    if fmt_norm in {"db", "ase.db", "sqlite"}:
        return write_ase_db(atoms_or_records, output_path, append=append)
    if fmt_norm in {"traj", "trajectory"}:
        return write_traj(atoms_or_records, output_path, append=append)
    raise ValueError("fmt must be one of: extxyz, xyz, db, ase.db, sqlite, traj")


def _safe_filename(text: str) -> str:
    text = text.strip() or "ungrouped"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def write_grouped_extxyz(
    atoms_or_records: Iterable[Any],
    output_dir: PathLike,
    *,
    group_key: str = "fitsnap_group",
    append: bool = False,
) -> dict[str, Path]:
    """Write one concatenated extxyz per group, e.g. ``EOS_1.extxyz``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[Any]] = defaultdict(list)
    for atoms in _as_atoms_list(atoms_or_records):
        group = str(atoms.info.get(group_key, "ungrouped"))
        groups[group].append(atoms)

    written: dict[str, Path] = {}
    for group, atoms_list in sorted(groups.items()):
        path = output_dir / f"{_safe_filename(group)}.extxyz"
        written[group] = write_extxyz(atoms_list, path, append=append)
    return written


def summarize_atoms(atoms_or_records: Iterable[Any], *, as_dataframe: bool = True) -> Any:
    """Return a compact table of structures for data exploration.

    If pandas is installed and ``as_dataframe=True``, returns a DataFrame.
    Otherwise returns a list of dictionaries.
    """
    rows: list[dict[str, Any]] = []
    for atoms in _as_atoms_list(atoms_or_records):
        symbols = list(atoms.get_chemical_symbols())
        counts = Counter(symbols)
        info = atoms.info
        energy = info.get("fitsnap_energy")
        if energy is None and getattr(atoms, "calc", None) is not None:
            try:
                energy = float(atoms.get_potential_energy())
            except Exception:
                energy = None
        natoms = len(atoms)
        rows.append(
            {
                "structure_id": info.get("structure_id"),
                "group": info.get("fitsnap_group"),
                "file": info.get("fitsnap_file"),
                "index": info.get("fitsnap_structure_index"),
                "natoms": natoms,
                "elements": ",".join(sorted(counts)),
                "formula_counts": dict(counts),
                "energy_eV": energy,
                "energy_eV_per_atom": (float(energy) / natoms) if energy is not None and natoms else None,
            }
        )
    if as_dataframe:
        try:
            import pandas as pd  # type: ignore

            return pd.DataFrame(rows)
        except ImportError:
            pass
    return rows


def filter_atoms(
    atoms_or_records: Iterable[Any],
    *,
    groups: Optional[Sequence[str]] = None,
    contains_elements: Optional[Sequence[str]] = None,
    only_elements: Optional[Sequence[str]] = None,
    min_atoms: Optional[int] = None,
    max_atoms: Optional[int] = None,
    predicate: Optional[Callable[[Any], bool]] = None,
) -> list[Any]:
    """Filter already-loaded structures in ordinary Python.

    This is deliberately simple so it works naturally in notebooks and ML data
    preprocessing scripts.
    """
    group_set = set(groups) if groups is not None else None
    contains_set = set(contains_elements) if contains_elements is not None else None
    only_set = set(only_elements) if only_elements is not None else None

    selected: list[Any] = []
    for atoms in _as_atoms_list(atoms_or_records):
        atom_set = set(atoms.get_chemical_symbols())
        if group_set is not None and atoms.info.get("fitsnap_group") not in group_set:
            continue
        if contains_set is not None and not contains_set.issubset(atom_set):
            continue
        if only_set is not None and not atom_set.issubset(only_set):
            continue
        if min_atoms is not None and len(atoms) < min_atoms:
            continue
        if max_atoms is not None and len(atoms) > max_atoms:
            continue
        if predicate is not None and not predicate(atoms):
            continue
        selected.append(atoms)
    return selected
