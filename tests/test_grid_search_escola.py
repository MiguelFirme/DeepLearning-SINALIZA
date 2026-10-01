"""A busca usa validação real separada e nunca seleciona pelo teste."""
import json

from scripts.grid_search_escola import prepare_person_splits, ranking
from scripts.reselect_grid_accuracy import peak_validation_row


def test_person_grid_splits_keep_derivatives_with_their_signer(tmp_path):
    spec = json.loads(open("configs/grid_escola_23.json", encoding="utf-8").read())
    report = prepare_person_splits(spec, tmp_path)
    manifest = json.loads((tmp_path / "tune" / "manifest.json").read_text(encoding="utf-8"))
    tune = json.loads((tmp_path / "tune" / "splits.json").read_text(encoding="utf-8"))
    refit = json.loads((tmp_path / "refit" / "splits.json").read_text(encoding="utf-8"))
    assert len(report["classes"]) == 23
    assert len(tune["val"]) == len(tune["test"]) == 23
    assert {manifest[i]["signer"] for i in tune["train"]} == {"Articulador1"}
    assert {manifest[i]["signer"] for i in tune["val"]} == {"Articulador2"}
    assert {manifest[i]["signer"] for i in tune["test"]} == {"Articulador3"}
    assert all(not manifest[i].get("is_synthetic", False) for i in tune["val"] + tune["test"])
    assert {manifest[i]["signer"] for i in refit["train"]} == {"Articulador1", "Articulador2"}
    assert refit["val"] == []
    assert all(not manifest[i].get("is_synthetic", False) for i in refit["test"])


def test_grid_ranking_uses_validation_metrics():
    base = {"val_f1_macro": 0.1, "val_loss": 2.0, "parameter_count": 100, "name": "a"}
    better_accuracy = {**base, "val_top1": 0.2}
    better_loss = {**base, "val_top1": 0.1, "val_loss": 1.0}
    assert ranking(better_accuracy) < ranking(better_loss)


def test_peak_validation_selection_uses_accuracy_before_loss():
    history = [
        {"epoch": 1, "val_top1": 0.0, "val_f1_macro": 0.0,
         "val_top3": 0.1, "val_loss": 1.0, "train_top1": 0.1},
        {"epoch": 2, "val_top1": 0.2, "val_f1_macro": 0.1,
         "val_top3": 0.3, "val_loss": 2.0, "train_top1": 0.5},
    ]
    selected = peak_validation_row(history, {"best_epoch": 1})
    assert selected["best_epoch"] == 2
    assert selected["checkpoint_epoch_by_val_loss"] == 1
