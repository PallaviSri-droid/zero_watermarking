import torch

from zero_watermarking.final import CAP_ZW_FINAL, FinalCAPZWConfig, build_final_config


def test_final_candidate_is_locked_between_v9_and_v11_regimes():
    cfg = FinalCAPZWConfig()
    assert CAP_ZW_FINAL == "CAP-ZW-final-candidate"
    assert 1.20 < cfg.lambda_binary_collision < 1.45
    assert cfg.binary_collision_target == 0.135
    assert cfg.robustness_guard_quantile == 0.85
    assert cfg.robustness_guard_lambda_init < 1.0
    assert cfg.topk_negatives == 12
    assert cfg.memory_size == 4096


def test_final_config_accepts_explicit_research_overrides():
    cfg = build_final_config(nbits=64, epochs=1, device="cpu")
    assert cfg.nbits == 64
    assert cfg.epochs == 1
    assert cfg.device == "cpu"


def test_final_config_is_serializable():
    cfg = FinalCAPZWConfig()
    assert torch.isfinite(torch.tensor(cfg.lambda_binary_collision))
    assert torch.isfinite(torch.tensor(cfg.robustness_guard_target))
