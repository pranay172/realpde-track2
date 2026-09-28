"""Audit the official Track 2 release and build leakage-safe split manifests."""

from __future__ import annotations

import hashlib
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import h5py
import numpy as np


REAL_BAD_CASE = "train_real/7575_0.h5"
HELD_OUT_RE = (12675, 21600)
HELD_OUT_AOA = (15,)
# Complementary fold: lowest and highest audited nominal Re, and the first AoA
# strictly between 0 and the v1 holdout 15. Chosen from the release grid, not
# from real_regime_v1 validation scores.
COMPLEMENT_HELD_OUT_RE = (3750, 26700)
COMPLEMENT_HELD_OUT_AOA = (5,)
EXPECTED_ARCHIVES = {
    "train_real.tar.gz": {
        "bytes": 7_393_582_393,
        "sha256": "20f19e3910028d76573848064fc55d470a4fb4609bb56972dd7b387ca0d515ea",
        "members": 82,
        "unpacked_bytes": 7_410_338_948,
    },
    "train_sim.tar.gz": {
        "bytes": 8_826_593_178,
        "sha256": "09d0b94c3a23e0ee8dc954c4c903710ff565d1ed3e36f86c7a90816d32483657",
        "members": 100,
        "unpacked_bytes": 9_844_735_200,
    },
}
EXPECTED_KEYS = {
    "train_real": ("aoa", "re", "t", "u", "v", "x", "y"),
    "train_sim": ("aoa", "p", "re", "t", "u", "v", "x", "y"),
}
EXPECTED_DTYPES = {
    "train_real": {
        "aoa": "int32", "re": "int32", "t": "float32", "u": "float64",
        "v": "float64", "x": "float64", "y": "float64",
    },
    "train_sim": {
        "aoa": "int32", "p": "float32", "re": "int32", "t": "float64",
        "u": "float32", "v": "float32", "x": "float64", "y": "float64",
    },
}
CASE_RE = re.compile(r"(?P<re>\d+)_(?P<aoa>-?\d+)\.h5")


def parse_case_name(path: Path) -> tuple[int, int]:
    """Return nominal Reynolds number and AoA encoded by an HDF5 filename."""
    match = CASE_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected case filename: {path.name}")
    return int(match.group("re")), int(match.group("aoa"))


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_archive(path: Path, expected_prefix: str) -> dict[str, Any]:
    """Check archive integrity and reject unsafe or unexpected members."""
    member_count = 0
    unpacked_bytes = 0
    seen: set[str] = set()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            member_path = PurePosixPath(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe archive member: {member.name}")
            if not member.isfile():
                raise ValueError(f"Unexpected non-file archive member: {member.name}")
            if len(member_path.parts) != 2 or member_path.parts[0] != expected_prefix:
                raise ValueError(f"Unexpected archive path: {member.name}")
            if not member_path.name.endswith(".h5"):
                raise ValueError(f"Unexpected archive file type: {member.name}")
            if member.name in seen:
                raise ValueError(f"Duplicate archive member: {member.name}")
            seen.add(member.name)
            member_count += 1
            unpacked_bytes += member.size
    return {
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "members": member_count,
        "unpacked_bytes": unpacked_bytes,
        "safe_paths": True,
    }


def _iter_frame_chunks(dataset: h5py.Dataset, frames: int = 32) -> Iterable[np.ndarray]:
    for start in range(0, dataset.shape[0], frames):
        yield np.asarray(dataset[start : start + frames])


def _scan_values(dataset: h5py.Dataset) -> dict[str, float | int]:
    minimum = np.inf
    maximum = -np.inf
    nonfinite = 0
    for chunk in _iter_frame_chunks(dataset):
        finite = np.isfinite(chunk)
        nonfinite += int(chunk.size - np.count_nonzero(finite))
        if np.any(finite):
            minimum = min(minimum, float(np.min(chunk[finite])))
            maximum = max(maximum, float(np.max(chunk[finite])))
    return {"min": float(minimum), "max": float(maximum), "nonfinite": nonfinite}


def _dataset_sha256(dataset: h5py.Dataset) -> str:
    digest = hashlib.sha256()
    for chunk in _iter_frame_chunks(dataset):
        digest.update(np.ascontiguousarray(chunk).tobytes())
    return digest.hexdigest()


def inspect_case(
    path: Path, split: str, full_value_scan: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one trajectory and return compact metadata plus scan statistics."""
    nominal_re, nominal_aoa = parse_case_name(path)
    with h5py.File(path, "r") as handle:
        keys = tuple(sorted(handle.keys()))
        if keys != EXPECTED_KEYS[split]:
            raise ValueError(f"Unexpected keys in {path}: {keys}")
        observed_dtypes = {key: str(handle[key].dtype) for key in keys}
        if observed_dtypes != EXPECTED_DTYPES[split]:
            raise ValueError(f"Unexpected dtypes in {path}: {observed_dtypes}")

        stored_re = int(handle["re"][()])
        stored_aoa = int(handle["aoa"][()])
        if stored_aoa != nominal_aoa:
            raise ValueError(f"AoA metadata mismatch in {path}: {stored_aoa} != {nominal_aoa}")

        time = np.asarray(handle["t"])
        frames = int(time.shape[0])
        if time.ndim != 1 or frames < 21 or not np.all(np.isfinite(time)):
            raise ValueError(f"Invalid time vector in {path}")
        delta = np.diff(time.astype(np.float64))
        if not np.all(delta > 0):
            raise ValueError(f"Non-increasing time vector in {path}")

        for coordinate in ("x", "y"):
            values = np.asarray(handle[coordinate])
            if values.shape != (64, 128) or not np.all(np.isfinite(values)):
                raise ValueError(f"Invalid {coordinate} grid in {path}: {values.shape}")

        channels = ("u", "v") if split == "train_real" else ("u", "v", "p")
        for channel in channels:
            if handle[channel].shape != (frames, 64, 128):
                raise ValueError(f"Invalid {channel} shape in {path}: {handle[channel].shape}")

        schema = {
            key: {"dtype": str(handle[key].dtype), "shape": list(handle[key].shape)}
            for key in keys
        }
        record = {
            "path": f"{split}/{path.name}",
            "nominal_re": nominal_re,
            "stored_re": stored_re,
            "aoa": nominal_aoa,
            "frames": frames,
            "bytes": path.stat().st_size,
            "t_start": float(time[0]),
            "t_end": float(time[-1]),
            "dt_min": float(np.min(delta)),
            "dt_max": float(np.max(delta)),
        }
        stats: dict[str, Any] = {
            "schema": schema,
            "coordinate_hash": hashlib.sha256(
                np.ascontiguousarray(handle["x"][...]).tobytes()
                + np.ascontiguousarray(handle["y"][...]).tobytes()
            ).hexdigest(),
        }
        if full_value_scan:
            stats["channels"] = {channel: _scan_values(handle[channel]) for channel in channels}
            for channel, values in stats["channels"].items():
                if values["nonfinite"]:
                    raise ValueError(
                        f"Found {values['nonfinite']} non-finite {channel} values in {path}"
                    )
        return record, stats


def _schema_without_frames(schema: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
    signature = []
    for key, details in sorted(schema.items()):
        shape = list(details["shape"])
        if key in {"t", "u", "v", "p"} and shape:
            shape[0] = "frames"
        signature.append((key, details["dtype"], tuple(shape)))
    return tuple(signature)


def _summarize_split(
    records: list[dict[str, Any]], stats: list[dict[str, Any]], full_value_scan: bool
) -> dict[str, Any]:
    nominal_res = sorted({record["nominal_re"] for record in records})
    stored_res = sorted({record["stored_re"] for record in records})
    aoas = sorted({record["aoa"] for record in records})
    frame_counts = Counter(record["frames"] for record in records)
    cases_by_re: dict[int, list[int]] = defaultdict(list)
    cases_by_aoa: Counter[int] = Counter()
    for record in records:
        cases_by_re[record["nominal_re"]].append(record["aoa"])
        cases_by_aoa[record["aoa"]] += 1

    schema_counts = Counter(_schema_without_frames(item["schema"]) for item in stats)
    schemas = []
    for signature, count in sorted(schema_counts.items(), key=lambda item: repr(item[0])):
        schemas.append(
            {
                "count": count,
                "datasets": {
                    key: {"dtype": dtype, "shape": list(shape)}
                    for key, dtype, shape in signature
                },
            }
        )

    summary: dict[str, Any] = {
        "trajectories": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "nominal_re_values": nominal_res,
        "stored_re_values": stored_res,
        "aoa_values": aoas,
        "frame_counts": {str(key): value for key, value in sorted(frame_counts.items())},
        "cases_by_nominal_re": {
            str(key): sorted(value) for key, value in sorted(cases_by_re.items())
        },
        "case_counts_by_aoa": {
            str(key): value for key, value in sorted(cases_by_aoa.items())
        },
        "stored_re_mismatch_count": sum(
            record["nominal_re"] != record["stored_re"] for record in records
        ),
        "coordinate_grid_count": len({item["coordinate_hash"] for item in stats}),
        "schemas": schemas,
    }
    if full_value_scan:
        channel_names = sorted({key for item in stats for key in item["channels"]})
        summary["value_scan"] = {
            channel: {
                "min": min(item["channels"][channel]["min"] for item in stats),
                "max": max(item["channels"][channel]["max"] for item in stats),
                "nonfinite": sum(item["channels"][channel]["nonfinite"] for item in stats),
            }
            for channel in channel_names
        }
    return summary


def verify_known_duplicate(data_root: Path) -> dict[str, Any]:
    """Hash the announced duplicate's measured arrays independently."""
    result: dict[str, Any] = {"bad_case": REAL_BAD_CASE, "source_case": "train_real/6300_0.h5"}
    for dataset_name in ("u", "v"):
        hashes = []
        for relative in (REAL_BAD_CASE, result["source_case"]):
            with h5py.File(data_root / relative, "r") as handle:
                hashes.append(_dataset_sha256(handle[dataset_name]))
        result[f"{dataset_name}_sha256"] = hashes[0]
        result[f"{dataset_name}_identical"] = hashes[0] == hashes[1]
    result["measurement_duplicate_confirmed"] = bool(
        result["u_identical"] and result["v_identical"]
    )
    if not result["measurement_duplicate_confirmed"]:
        raise ValueError("Known bad case did not match 6300_0 measured arrays")
    return result


def audit_release(
    data_root: Path, *, full_value_scan: bool = False, verify_archives: bool = False
) -> dict[str, Any]:
    """Audit all extracted cases and optionally the compressed source archives."""
    data_root = data_root.resolve()
    audit: dict[str, Any] = {
        "schema_version": 1,
        "release": "AI4Science-WestlakeU/RealPDE-Competition-Data",
        "audit_policy": {
            "all_hdf5_structures": True,
            "all_channel_values_scanned": full_value_scan,
            "archives_verified": verify_archives,
        },
        "archives": {},
        "splits": {},
    }
    if verify_archives:
        for archive_name, expected in EXPECTED_ARCHIVES.items():
            observed = inspect_archive(data_root / archive_name, archive_name.split(".")[0])
            if observed != {**expected, "safe_paths": True}:
                raise ValueError(f"Archive mismatch for {archive_name}: {observed}")
            audit["archives"][archive_name] = observed
    else:
        audit["archives"] = EXPECTED_ARCHIVES

    all_records: dict[str, list[dict[str, Any]]] = {}
    for split, expected_count in (("train_real", 82), ("train_sim", 100)):
        paths = sorted((data_root / split).glob("*.h5"), key=parse_case_name)
        if len(paths) != expected_count:
            raise ValueError(f"Expected {expected_count} {split} cases, found {len(paths)}")
        records = []
        statistics = []
        for path in paths:
            record, case_stats = inspect_case(path, split, full_value_scan)
            records.append(record)
            statistics.append(case_stats)
        all_records[split] = records
        audit["splits"][split] = {
            "summary": _summarize_split(records, statistics, full_value_scan),
            "cases": records,
        }

    audit["known_duplicate"] = verify_known_duplicate(data_root)
    audit["valid_real_trajectories"] = len(all_records["train_real"]) - 1
    return audit


def build_split_manifest(
    audit: dict[str, Any],
    *,
    name: str = "real_regime_v1",
    held_out_re: tuple[int, ...] = HELD_OUT_RE,
    held_out_aoa: tuple[int, ...] = HELD_OUT_AOA,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Create deterministic whole-trajectory real and simulation partitions."""
    real_records = audit["splits"]["train_real"]["cases"]
    sim_records = audit["splits"]["train_sim"]["cases"]

    def membership(record: dict[str, Any]) -> tuple[bool, bool]:
        return record["nominal_re"] in held_out_re, record["aoa"] in held_out_aoa

    real_partitions = {"train": [], "val_re": [], "val_aoa": [], "val_joint": []}
    for record in real_records:
        if record["path"] == REAL_BAD_CASE:
            continue
        held_re, held_aoa = membership(record)
        partition = (
            "val_joint" if held_re and held_aoa else
            "val_re" if held_re else
            "val_aoa" if held_aoa else
            "train"
        )
        real_partitions[partition].append(record["path"])

    sim_standard = [record["path"] for record in sim_records]
    sim_strict_train = []
    sim_strict_heldout = []
    for record in sim_records:
        held_re, held_aoa = membership(record)
        target = sim_strict_heldout if held_re or held_aoa else sim_strict_train
        target.append(record["path"])

    all_real = [path for paths in real_partitions.values() for path in paths]
    if len(all_real) != len(set(all_real)) or len(all_real) != audit["valid_real_trajectories"]:
        raise ValueError("Real partitions are overlapping or incomplete")
    if REAL_BAD_CASE in all_real:
        raise ValueError("Known duplicate leaked into split")

    if notes is None:
        notes = [
            "Every HDF5 trajectory belongs to at most one real partition; "
            "windows must never cross partitions.",
            "Use standard_train to match the official sim-pretrain setting.",
            "Use strict_regime_train for a stronger unseen-parameter test that "
            "removes held-out Re/AoA from simulation too.",
            "Released sim_real_ft checkpoints saw the full real release and are "
            "not leakage-safe on these validation cases.",
        ]
    return {
        "schema_version": 1,
        "name": name,
        "case_key": "nominal Reynolds number and AoA parsed from filename",
        "seed": None,
        "held_out": {"nominal_re": list(held_out_re), "aoa": list(held_out_aoa)},
        "excluded": {
            REAL_BAD_CASE: "Organizer-announced duplicate of train_real/6300_0.h5 measured u/v"
        },
        "real": real_partitions,
        "simulation": {
            "standard_train": sim_standard,
            "strict_regime_train": sim_strict_train,
            "strict_regime_heldout": sim_strict_heldout,
        },
        "notes": notes,
    }


def build_complement_manifest(audit: dict[str, Any]) -> dict[str, Any]:
    """Hold out the complementary extreme-Re / AoA-5 slice unused by v1."""
    return build_split_manifest(
        audit,
        name="real_regime_complement_v1",
        held_out_re=COMPLEMENT_HELD_OUT_RE,
        held_out_aoa=COMPLEMENT_HELD_OUT_AOA,
        notes=[
            "Complementary fold of real_regime_v1: hold out the lowest and "
            "highest audited nominal Re (3750, 26700) and AoA 5.",
            "Holdouts were chosen from the release grid, not from v1 "
            "validation scores.",
            "A model trained on this train partition has seen v1 validation "
            "trajectories and is not leakage-safe on real_regime_v1.",
            "Every HDF5 trajectory belongs to at most one real partition.",
        ],
    )
