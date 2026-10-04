"""Deterministic intake of selected Distributional MIPLIB splits."""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DMIPLIB_REPOSITORY = "weiminhu/D-MIPLIB"
MVC_CONFIGURATION = "MVC-easy"
MVC_SPLITS = {"train": 800, "validation": 100}


class DatasetFetchError(RuntimeError):
    """The remote dataset did not satisfy the expected immutable contract."""


def _decode_milp(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if not isinstance(value, str):
        raise DatasetFetchError(f"MILP field has unsupported type {type(value).__name__}")
    stripped = value.strip()
    if (stripped.startswith("b'") and stripped.endswith("'")) or (
        stripped.startswith('b"') and stripped.endswith('"')
    ):
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError) as exc:
            raise DatasetFetchError("Unable to decode byte-string MILP payload") from exc
        if isinstance(parsed, bytes):
            return parsed.decode("utf-8")
    return value


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _atomic_write_bytes(path: Path, content: bytes, *, force: bool) -> None:
    if path.exists():
        if path.read_bytes() == content:
            return
        if not force:
            raise DatasetFetchError(f"Refusing to overwrite different existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def fetch_dmiplib_mvc(
    output_dir: str | Path,
    *,
    revision: str | None = None,
    force: bool = False,
    dataset_loader: Callable[..., Iterable[dict[str, Any]]] | None = None,
    revision_resolver: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Fetch only the MVC-easy training and validation streams."""

    if dataset_loader is None:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise DatasetFetchError("Dataset download requires `uv sync --extra data`.") from exc
        dataset_loader = load_dataset
    if revision_resolver is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise DatasetFetchError("Dataset download requires `uv sync --extra data`.") from exc

        def resolve_revision(repository: str) -> str:
            return HfApi().dataset_info(repository).sha

        revision_resolver = resolve_revision

    resolved_revision = revision or revision_resolver(DMIPLIB_REPOSITORY)
    root = Path(output_dir).resolve()
    instance_dir = root / "instance"
    (root / "backbone").mkdir(parents=True, exist_ok=True)
    partitions: dict[str, list[str]] = {}
    file_records: list[dict[str, Any]] = []

    for split, expected_count in MVC_SPLITS.items():
        exported: list[str] = []
        rows = dataset_loader(
            DMIPLIB_REPOSITORY,
            MVC_CONFIGURATION,
            split=split,
            revision=resolved_revision,
            streaming=True,
        )
        for index, row in enumerate(rows):
            file_format = str(row.get("format", "lp")).lower()
            if file_format not in {"lp", "mps"}:
                raise DatasetFetchError(f"Unsupported D-MIPLIB format {file_format!r}")
            name = f"{'valid' if split == 'validation' else split}_{index:04d}.{file_format}"
            text = _decode_milp(row["MILP"])
            if not text.endswith("\n"):
                text += "\n"
            content = text.encode("utf-8")
            _atomic_write_bytes(instance_dir / name, content, force=force)
            exported.append(name)
            file_records.append(
                {
                    "split": split,
                    "row_index": index,
                    "source_id": row.get("Unnamed: 0", index),
                    "local_name": name,
                    "format": file_format,
                    "sha256": _sha256_bytes(content),
                }
            )
        if len(exported) != expected_count:
            raise DatasetFetchError(
                f"Expected {expected_count} rows for {split}, received {len(exported)}"
            )
        partitions[split] = exported

    partition_payload = {"schema_version": 1, "partitions": partitions}
    source_path = root / "source.json"
    acquired_at = datetime.now(UTC).isoformat()
    if source_path.exists():
        try:
            previous = json.loads(source_path.read_text(encoding="utf-8"))
            if previous.get("revision") == resolved_revision:
                acquired_at = previous.get("acquired_at", acquired_at)
        except json.JSONDecodeError:
            pass
    source_payload = {
        "schema_version": 1,
        "repository": DMIPLIB_REPOSITORY,
        "configuration": MVC_CONFIGURATION,
        "revision": resolved_revision,
        "acquired_at": acquired_at,
        "license": "CC BY 4.0",
        "status": "reconstructed from the published D-MIPLIB distribution",
        "files": sorted(file_records, key=lambda item: item["local_name"]),
    }
    _atomic_write_json(root / "partitions.json", partition_payload)
    _atomic_write_json(source_path, source_payload)
    return source_payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)
