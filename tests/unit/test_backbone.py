import pytest

from backbone_pbo.backbone import (
    BackboneFormatError,
    BackboneLabel,
    dumps_backbone,
    flip_backbone_label,
    loads_backbone,
    transform_backbone,
)


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        (BackboneLabel.B0, BackboneLabel.B1),
        (BackboneLabel.B1, BackboneLabel.B0),
        (BackboneLabel.NB, BackboneLabel.NB),
    ],
)
def test_flip_backbone_label(original, expected):
    assert flip_backbone_label(original) == expected


def test_transform_backbone_selectively():
    labels = {"x1": 0, "x2": 1, "x3": 2}

    transformed = transform_backbone(labels, {"x1", "x3"})

    assert transformed == {
        "x1": BackboneLabel.B1,
        "x2": BackboneLabel.B1,
        "x3": BackboneLabel.NB,
    }


def test_transform_backbone_rejects_unknown_variables():
    with pytest.raises(ValueError, match="x2"):
        transform_backbone({"x1": 0}, {"x2"})


def test_read_backpas_backbone_format():
    content = "b -x1\nb x2\nb 0\n"

    labels = loads_backbone(content, ["x1", "x2", "x3"])

    assert labels == {
        "x1": BackboneLabel.B0,
        "x2": BackboneLabel.B1,
        "x3": BackboneLabel.NB,
    }


def test_backbone_round_trip():
    labels = {"x1": BackboneLabel.B0, "x2": BackboneLabel.B1, "x3": BackboneLabel.NB}

    assert loads_backbone(dumps_backbone(labels), labels) == labels


@pytest.mark.parametrize("content", ["b x1\n", "b x1\nb 0\nb x2\n", "x x1\nb 0\n"])
def test_backbone_reader_rejects_incomplete_or_invalid_files(content):
    with pytest.raises(BackboneFormatError):
        loads_backbone(content, ["x1", "x2"])
