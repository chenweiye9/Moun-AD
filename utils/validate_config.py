# utils/validate_config.py
def validate_optim_config(cfg):
    if cfg.optimizer.name == "Muon":
        assert cfg.optimizer.orthogonal_update.enabled,
               "The Muon optimizer must have orthographic projection enabled"
        assert cfg.training.mixed_precision == "bf16",
               "Muon requires BF16 mixed-precision support"