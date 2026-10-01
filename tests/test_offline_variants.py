"""Variações offline mantêm o sinal e não contaminam o teste holdout."""
import csv
import json

import numpy as np

from ml.data.offline_variants import MAX_XY_DELTA, generate_category_variants
from ml.data.split import DataSplitter


def test_generate_seven_variants_and_keep_test_real(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    annotations = tmp_path / "annotations.csv"
    rows = []
    manifest = []
    originals = {}
    for signer_index in range(1, 4):
        signer = f"Articulador{signer_index}"
        name = f"Biblioteca_{signer}.npz"
        landmarks = np.zeros((6, 346), dtype=np.float32)
        landmarks[:, 0:63:3] = 0.3 + signer_index * 0.05
        landmarks[:, 1:63:3] = 0.4
        landmarks[:, 2:63:3] = 0.1
        landmarks[:, 126:226:4] = 0.5
        landmarks[:, 127:226:4] = 0.6
        landmarks[:, 129:226:4] = 0.9  # visibilidade da pose
        mask = np.zeros((6, 4), dtype=bool)
        mask[:, 0] = True
        mask[:, 2] = True
        mask[2, 0] = False
        landmarks[2, :63] = 0.0
        np.savez(source / name, landmarks=landmarks, mask=mask, metadata={"fps": 30})
        originals[name] = (landmarks, mask)
        manifest.append({"path": name, "label": "Biblioteca", "signer": signer})
        rows.append({"video_id": name, "class": "Biblioteca", "user_id": signer})
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with annotations.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "class", "user_id"])
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "output"
    report = generate_category_variants(source, output, annotations, ["escola"], seed=42)
    generated = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert (report["num_original"], report["num_synthetic"], report["num_total"]) == (3, 7, 10)
    assert len(generated) == 10
    for sample in generated:
        path = output / sample["path"]
        assert path.is_file()
        if not sample.get("is_synthetic", False):
            assert path.read_bytes() == (source / sample["path"]).read_bytes()
            continue
        origin, origin_mask = originals[sample["source_path"]]
        with np.load(path, allow_pickle=True) as data:
            variant = data["landmarks"]
            variant_mask = data["mask"]
        assert variant.shape == origin.shape
        assert np.array_equal(variant_mask, origin_mask)
        assert np.array_equal(variant[:, 2:63:3], origin[:, 2:63:3])
        assert np.array_equal(variant[:, 128:226:4], origin[:, 128:226:4])
        assert np.array_equal(variant[:, 129:226:4], origin[:, 129:226:4])
        assert np.array_equal(variant[2, :63], origin[2, :63])
        assert 0 < np.max(np.abs(variant - origin)) <= MAX_XY_DELTA + 1e-6

    splits = DataSplitter().split_by_signer(generated, test_signer="Articulador3",
                                            train_signers=["Articulador1", "Articulador2"])
    assert len(splits["test"]) == 1
    assert not generated[splits["test"][0]].get("is_synthetic", False)
    assert all(generated[index]["signer"] != "Articulador3" for index in splits["train"])
