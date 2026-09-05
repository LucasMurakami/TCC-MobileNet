from argparse import Namespace

from main import resolve_run_config
from train_timm_models import compute_adaptive_batch_strategy


def _args(**overrides):
    values = {
        'epochs': None,
        'patience': None,
        'batch_size': None,
        'lr_stage1': None,
        'lr_stage2': None,
        'seed': None,
        'img_size': None,
        'mel_threshold': None,
        'bcc_threshold': None,
        'malignant_threshold': None,
        'balanced_sampling': False,
        'logit_adjust': None,
        'mixup_alpha': None,
        'use_tta': False,
        'color_constancy': False,
        'stage1_epochs': None,
        'no_cudnn': False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_config_precedence():
    scenario = {'epochs': 30, 'batch_size': 16, 'mel_threshold': 'youden', 'mixup_alpha': 0.2, 'balanced_sampling': True, 'use_tta': True}
    model = {'epochs': 20, 'batch_size': 24}
    config = resolve_run_config(_args(epochs=10), scenario, model)
    assert config.epochs == 10
    assert config.batch_size == 24
    assert config.mel_threshold == 'youden'
    assert config.mixup_alpha == 0.2
    assert config.balanced_sampling
    assert config.use_tta


def test_rtx_5070_uses_12gb_tier():
    assert compute_adaptive_batch_strategy(11.9, 'v5') == {
        'micro_batch': 8, 'grad_accum_steps': 4, 'tier_gb': 12
    }
    assert compute_adaptive_batch_strategy(11.9, 'v1') == {
        'micro_batch': 32, 'grad_accum_steps': 1, 'tier_gb': 12
    }

