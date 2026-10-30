import csv
import json
import numpy as np
import pytest

from ml.data.quality import hand_quality
from scripts.prepare_rejection_dataset import prepare


def test_hand_coverage_flags_missing_active_hand():
    n = 30
    landmarks = np.zeros((n, 346), dtype=np.float32)
    mask = np.ones((n, 4), dtype=bool)
    pose = landmarks[:, 126:226].reshape(n, 25, 4)
    pose[:, 5, :2] = [0.4, 0.5]
    pose[:, 6, :2] = [0.6, 0.5]
    pose[:, 9, :2] = np.column_stack((np.linspace(0.4, 0.8, n), np.full(n, 0.6)))
    pose[:, 10, :2] = [0.6, 0.6]
    mask[:, 0] = False
    quality = hand_quality(landmarks, mask)
    assert quality["active_hand"] == "left"
    assert quality["active_hand_coverage"] == 0
    assert quality["active_end"] <= n


def test_negative_pipeline_requires_reviewed_recordings(tmp_path):
    root = tmp_path / "landmarks"
    positive = root / "mixed_25"
    negative = root / "negatives"
    processed = tmp_path / "processed"
    positive.mkdir(parents=True)
    negative.mkdir()
    processed.mkdir()
    samples = [{"path": f"p{i}.npz", "label": "A", "signer": "minds:Sinalizador01"}
               for i in range(670)]
    (positive / "manifest.json").write_text(__import__("json").dumps(samples))
    (processed / "manifest.json").write_text(__import__("json").dumps(samples))
    (processed / "splits.json").write_text(__import__("json").dumps({"train": list(range(670)), "val": [], "test": []}))
    (processed / "label_map.json").write_text(__import__("json").dumps({f"C{i}": i for i in range(25)}))
    review = tmp_path / "review.csv"
    review.write_text("path,person_id,session_id,subtype,split,reviewed\n")
    with pytest.raises(ValueError, match="Nenhum negativo revisado"):
        prepare(positive, processed, tmp_path / "raw", negative, review,
                root / "mixed25_rejection", tmp_path / "processed_rejection")
    assert not (root / "mixed25_rejection").exists()


def test_negative_pipeline_builds_grouped_26_class_dataset(tmp_path):
    root = tmp_path / "landmarks"
    positive, negative = root / "mixed_25", root / "negatives"
    processed, raw = tmp_path / "processed", tmp_path / "raw"
    for directory in (positive, negative, processed, raw):
        directory.mkdir(parents=True)
    splits = {"train": list(range(500)), "val": list(range(500, 585)),
              "test": list(range(585, 670))}
    samples = [{"path": f"p{i}.npz", "label": f"C{i % 25}",
                "signer": f"minds:{'train' if i < 500 else 'val' if i < 585 else 'test'}"}
               for i in range(670)]
    for target in (positive, processed):
        (target / "manifest.json").write_text(json.dumps(samples))
    (processed / "splits.json").write_text(json.dumps(splits))
    (processed / "label_map.json").write_text(json.dumps({f"C{i}": i for i in range(25)}))
    review = tmp_path / "review.csv"
    with review.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "person_id", "session_id",
                                                  "subtype", "split", "reviewed"])
        writer.writeheader()
        for split, count in (("train", 20), ("val", 5), ("test", 5)):
            for i in range(count):
                name = f"{split}/n{i}.mp4"
                (raw / split).mkdir(exist_ok=True)
                (negative / split).mkdir(exist_ok=True)
                (raw / name).write_bytes(b"video-fixture")
                np.savez_compressed(negative / f"{split}/n{i}.npz",
                                    landmarks=np.zeros((24, 346), dtype=np.float32),
                                    mask=np.zeros((24, 4), dtype=bool))
                writer.writerow({"path": name, "person_id": f"negative:{split}",
                                 "session_id": "session1", "subtype": "repouso",
                                 "split": split, "reviewed": "yes"})
    output = root / "mixed25_rejection"
    derived = tmp_path / "processed_rejection"
    report = prepare(positive, processed, raw, negative, review, output, derived)
    assert report["num_classes"] == 26
    assert report["split_sizes"] == {"train": 520, "val": 90, "test": 90}
    assert json.loads((derived / "label_map.json").read_text())["__nenhum_sinal__"] == 25
