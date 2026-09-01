"""Transformaciones de etiquetas de backbone."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import IntEnum
from pathlib import Path


class BackboneLabel(IntEnum):
    """Codificación usada por el pipeline de referencia."""

    B0 = 0
    B1 = 1
    NB = 2


class BackboneFormatError(ValueError):
    """El archivo no sigue el formato de salida de GuroBack usado por BackPaS."""


def flip_backbone_label(label: BackboneLabel | int) -> BackboneLabel:
    """Intercambia B0/B1 y conserva NB."""

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
    """Transforma las etiquetas para una inversión global o selectiva."""

    selected = set(labels) if flipped_variables is None else set(flipped_variables)
    unknown = selected.difference(labels)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Variables sin etiqueta de backbone: {names}")

    return {
        name: flip_backbone_label(label) if name in selected else BackboneLabel(label)
        for name, label in labels.items()
    }


def loads_backbone(
    content: str,
    variables: Iterable[str],
) -> dict[str, BackboneLabel]:
    """Lee un backbone de GuroBack y completa las variables no-backbone con ``NB``.

    BackPaS utiliza líneas ``b xN`` para B1, ``b -xN`` para B0 y una línea final
    ``b 0`` como marcador de extracción completa.
    """

    labels = {name: BackboneLabel.NB for name in variables}
    complete = False
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 2 or parts[0] != "b":
            raise BackboneFormatError(f"Línea {line_number} inválida: {line!r}")
        token = parts[1]
        if token == "0":
            if line_number != len(lines):
                raise BackboneFormatError("El marcador 'b 0' debe ser la última línea")
            complete = True
            continue
        label = BackboneLabel.B0 if token.startswith("-") else BackboneLabel.B1
        variable = token[1:] if token.startswith("-") else token
        if variable not in labels:
            raise BackboneFormatError(f"Variable {variable!r} no pertenece a la instancia")
        labels[variable] = label

    if not complete:
        raise BackboneFormatError("Falta el marcador final 'b 0'")
    return labels


def read_backbone(path: str | Path, variables: Iterable[str]) -> dict[str, BackboneLabel]:
    """Lee un archivo ``.backbone`` en ASCII."""

    return loads_backbone(Path(path).read_text(encoding="ascii"), variables)


def dumps_backbone(labels: Mapping[str, BackboneLabel | int]) -> str:
    """Escribe un backbone en el formato consumido por BackPaS."""

    lines = []
    for name in sorted(labels):
        label = BackboneLabel(labels[name])
        if label == BackboneLabel.B0:
            lines.append(f"b -{name}")
        elif label == BackboneLabel.B1:
            lines.append(f"b {name}")
    lines.append("b 0")
    return "\n".join(lines) + "\n"
