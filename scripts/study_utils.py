"""Small provenance guards for resumable experiment runners."""
import hashlib
import json
from pathlib import Path


def write_checked_config(path: Path, config: dict, report: Path) -> None:
    """Never overwrite controls for an already completed, differently configured run."""
    if report.exists():
        recorded = json.loads(report.read_text())
        if recorded.get("config") != config:
            raise RuntimeError(f"Cannot resume stale result {report}: configuration differs")
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError(f"Cannot overwrite different run controls {path}; use a fresh output")
    path.write_text(json.dumps(config, indent=2) + "\n")


def verify_checkpoint(path: Path, report: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    expected = json.loads(report.read_text())["checkpoint_sha256"]
    if actual != expected:
        raise RuntimeError(f"Checkpoint no longer matches training report: {path}")
    return actual
