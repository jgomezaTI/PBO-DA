"""Backbone label transformations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import IntEnum
from pathlib import Path


class BackboneLabel(IntEnum):
    """Encoding used by the reference pipeline."""

    B0 = 0
    B1 = 1
    NB = 2


class BackboneFormatError(ValueError):
    """The file does not follow the GuroBack output format used by BackPaS."""


def flip_backbone_label(label: BackboneLabel | int) -> BackboneLabel:
    """Swap B0/B1 and preserve NB."""

    normalized = BackboneLabel(label)
    if normalized == BackboneLabel.B0:
        return BackboneLabel.B1
    if normalized == BackboneLabel.B1:
        return BackboneLabel.B0
    return BackboneLabel.NB


def transform_backbone(
    labels: Mapping[str, BackboneLabel | int],
    flipped_variables: Iterable[str] | None = None,
) -> dict[str, BackboneLabel]:
    """Transform labels for a global or selective inversion."""

    selected = set(labels) if flipped_variables is None else set(flipped_variables)
    unknown = selected.difference(labels)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Variables without backbone labels: {names}")

    return {
        name: flip_backbone_label(label) if name in selected else BackboneLabel(label)
        for name, label in labels.items()
    }


def loads_backbone(
    content: str,
    variables: Iterable[str],
) -> dict[str, BackboneLabel]:
    """Read a GuroBack backbone and fill non-backbone variables with ``NB``.

    BackPaS uses ``b xN`` lines for B1, ``b -xN`` lines for B0, and a final ``b 0``
    line as the complete-extraction marker.
    """

    labels = {name: BackboneLabel.NB for name in variables}
    complete = False
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 2 or parts[0] != "b":
            raise BackboneFormatError(f"Invalid line {line_number}: {line!r}")
        token = parts[1]
        if token == "0":
            if line_number != len(lines):
                raise BackboneFormatError("The 'b 0' marker must be the last line")
            complete = True
            continue
        label = BackboneLabel.B0 if token.startswith("-") else BackboneLabel.B1
        variable = token[1:] if token.startswith("-") else token
        if variable not in labels:
            raise BackboneFormatError(f"Variable {variable!r} does not belong to the instance")
        labels[variable] = label

    if not complete:
        raise BackboneFormatError("Missing final 'b 0' marker")
    return labels


def read_backbone(path: str | Path, variables: Iterable[str]) -> dict[str, BackboneLabel]:
    """Read an ASCII ``.backbone`` file."""

    return loads_backbone(Path(path).read_text(encoding="ascii"), variables)


def dumps_backbone(labels: Mapping[str, BackboneLabel | int]) -> str:
    """Write a backbone in the format consumed by BackPaS."""

    lines = []
    for name in sorted(labels):
        label = BackboneLabel(labels[name])
        if label == BackboneLabel.B0:
            lines.append(f"b -{name}")
        elif label == BackboneLabel.B1:
            lines.append(f"b {name}")
    lines.append("b 0")
    return "\n".join(lines) + "\n"
