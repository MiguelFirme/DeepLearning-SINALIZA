import json

import numpy as np

from scripts.mine_idle_edges import mine


def test_boundary_mining_keeps_test_signer_out(tmp_path):
    data_dir = tmp_path / "landmarks"
    processed_dir = tmp_path / "processed"
    data_dir.mkdir()
    processed_dir.mkdir()
    manifest = []
    for index, split in enumerate(("train", "val", "test")):
        n = 120
        landmarks = np.zeros((n, 346), dtype=np.float32)
        mask = np.ones((n, 4), dtype=bool)
        pose = landmarks[:, 126:226].reshape(n, 25, 4)
        pose[:, 5, :2] = [0.4, 0.5]
        pose[:, 6, :2] = [0.6, 0.5]
        pose[:, 9, :2] = [0.4, 0.85]
        pose[:, 10, :2] = [0.6, 0.85]
        pose[45:75, 9, 0] = 0.4 + np.sin(np.linspace(0, 6 * np.pi, 30)) * 0.1
        np.savez_compressed(data_dir / f"sample{index}.npz", landmarks=landmarks, mask=mask)
        manifest.append({"path": f"sample{index}.npz", "video_id": f"minds:sample{index}.mp4",
                         "origin": "minds", "signer": f"minds:{split}", "label": "Acontecer"})
    (data_dir / "manifest.json").write_text(json.dumps(manifest))
    (processed_dir / "splits.json").write_text(
        json.dumps({"train": [0], "val": [1], "test": [2]}))
    candidates = mine(data_dir, processed_dir, tmp_path / "raw", length=32, threshold=0.2)
    assert {row["split"] for row in candidates} == {"train", "val"}
    assert len(candidates) == 4
    assert all(row["end_frame_exclusive"] - row["start_frame"] == 32 for row in candidates)
