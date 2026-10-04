from backbone_pbo.backbone import BackboneLabel
from backbone_pbo.dataset.loader import create_dataset_splits, load_raw_dataset
from backbone_pbo.graph.converter import instance_to_bipartite_graph
from backbone_pbo.io.opb import loads_opb
from backbone_pbo.model import Constraint, LinearExpression, PBOInstance
from backbone_pbo.solver.exact import extract_backbone, solve_pbo
from backbone_pbo.training.gnn import PBOBackboneGNN


def _make_in_memory_fixture(
    num_vars: int,
    num_constraints: int,
) -> tuple[PBOInstance, dict[str, BackboneLabel]]:
    variables = [f"x{i}" for i in range(1, num_vars + 1)]
    instance = PBOInstance(
        objective=LinearExpression({variable: 1 for variable in variables}),
        constraints=tuple(
            Constraint(
                expression=LinearExpression({variables[index % num_vars]: 1}),
                operator=">=",
                rhs=0,
            )
            for index in range(num_constraints)
        ),
        comments=("in-memory unit-test fixture",),
    )
    labels = {variable: BackboneLabel.B0 for variable in variables}
    return instance, labels


def test_exact_solver_and_backbone():
    inst_text = """
    * #variable= 3 #constraint= 2
    min: +1 x1 +2 x2 +3 x3 ;
    +1 x1 +1 x2 >= 1 ;
    +1 x3 >= 1 ;
    """
    instance = loads_opb(inst_text, validate_header=False)
    opt_val, _assign = solve_pbo(instance)
    assert opt_val == 4.0  # x1=1, x2=0, x3=1 (1*1 + 2*0 + 3*1 = 4)
    labels = extract_backbone(instance)
    assert labels["x1"] == BackboneLabel.B1
    assert labels["x2"] == BackboneLabel.B0
    assert labels["x3"] == BackboneLabel.B1


def test_graph_converter_and_gnn_forward():
    instance, labels = _make_in_memory_fixture(num_vars=8, num_constraints=10)
    graph = instance_to_bipartite_graph(instance, labels)

    assert graph.x.shape[0] == 8 + 10
    assert graph.x.shape[1] == 4
    assert graph.edge_index.shape[0] == 2
    assert graph.var_mask.sum().item() == 8

    model = PBOBackboneGNN(in_channels=4, hidden_channels=32, num_layers=2)
    logits = model(graph.x, graph.edge_index)
    assert logits.shape == (8 + 10, 3)


def test_dataset_splits_and_augmentation():
    items = [
        (*_make_in_memory_fixture(num_vars=6, num_constraints=8), f"inst_{i}.opb")
        for i in range(10)
    ]
    splits_orig = create_dataset_splits(
        items, train_ratio=0.7, val_ratio=0.2, seed=42, augment_polarity=False
    )
    splits_aug = create_dataset_splits(
        items, train_ratio=0.7, val_ratio=0.2, seed=42, augment_polarity=True
    )

    assert len(splits_orig.train_graphs) == 7
    assert len(splits_aug.train_graphs) == 14  # doubled due to polarity augmentation
    assert len(splits_orig.val_graphs) == 2
    assert len(splits_aug.val_graphs) == 2  # validation unchanged


def test_loader_uses_backpas_singular_directory_names(repo_tmp_path):
    dataset_dir = repo_tmp_path / "MIS"
    instance_dir = dataset_dir / "instance"
    backbone_dir = dataset_dir / "backbone"
    instance_dir.mkdir(parents=True)
    backbone_dir.mkdir()

    (instance_dir / "train_tiny.opb").write_text(
        "* #variable= 2 #constraint= 1\nmin: +1 x1 +2 x2 ;\n+1 x1 +1 x2 >= 1 ;\n",
        encoding="ascii",
    )
    (backbone_dir / "train_tiny.opb.backbone").write_text(
        "b x1\nb -x2\nb 0\n",
        encoding="ascii",
    )

    items = load_raw_dataset(dataset_dir)

    assert len(items) == 1
    _, labels, filename = items[0]
    assert filename == "train_tiny.opb"
    assert labels == {"x1": BackboneLabel.B1, "x2": BackboneLabel.B0}
