from pathlib import Path

import pytest

from zero_watermarking.research_contract import assert_compatible_summaries, fingerprint, manifest_fingerprint


def test_fingerprint_is_stable():
    assert fingerprint({"b": 2, "a": 1}) == fingerprint({"a": 1, "b": 2})


def test_manifest_fingerprint_reads_file(tmp_path: Path):
    p = tmp_path / "manifest.csv"
    p.write_text("image_id,path\n1,a.png\n", encoding="utf-8")
    first = manifest_fingerprint(p)
    p.write_text("image_id,path\n1,b.png\n", encoding="utf-8")
    assert first != manifest_fingerprint(p)


def test_incompatible_protocol_is_rejected():
    left = {"manifest_id": "abc", "image_size": 128, "bits": 128, "attack_grid_id": "grid", "split": "test"}
    right = {"manifest_id": "def", "image_size": 128, "bits": 128, "attack_grid_id": "grid", "split": "test"}
    with pytest.raises(ValueError):
        assert_compatible_summaries(left, right)
