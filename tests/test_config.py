from argparse import Namespace

import json
from pathlib import Path

from main import resolve_run_config, resolve_seeds
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
        'eval_precision': None,
        'selection_min_delta': None,
        'no_cudnn': False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_config_precedence():
    scenario = {'epochs': 30, 'batch_size': 16, 'mel_threshold': 'sens90', 'mixup_alpha': 0.2, 'balanced_sampling': True, 'use_tta': True, 'eval_precision': 'amp', 'selection_min_delta': 0.001}
    model = {'epochs': 20, 'batch_size': 24}
    config = resolve_run_config(_args(epochs=10), scenario, model)
    assert config.epochs == 10
    assert config.batch_size == 24
    assert config.mel_threshold == 'sens90'
    assert config.eval_precision == 'amp'
    assert config.selection_min_delta == 0.001
    assert config.mixup_alpha == 0.2
    assert config.balanced_sampling
    assert config.use_tta
    assert config.loss == 'focal'
    assert config.temperature_scaling
    assert config.split_seed == 42


def test_loss_and_temperature_and_split_seed_resolution():
    scenario = {'loss': 'ce', 'temperature_scaling': False, 'split_seed': 7}
    config = resolve_run_config(_args(seed=43), scenario, {})
    assert config.loss == 'ce'
    assert not config.temperature_scaling
    assert config.split_seed == 7
    assert config.seed == 43
    cli = resolve_run_config(_args(loss='focal', no_temperature_scaling=True), scenario, {})
    assert cli.loss == 'focal'
    assert not cli.temperature_scaling


def test_seed_resolution_order():
    assert resolve_seeds({'seed': 1}, {'seeds': [42, 43, 44]}) == [42, 43, 44]
    assert resolve_seeds({'seed': 1}, {'seed': 5}) == [5]
    assert resolve_seeds({'seed': 1}, {}) == [1]
    assert resolve_seeds({}, {}) == [42]
    assert resolve_seeds({}, {'seeds': [42, 43]}, cli_seed=9) == [9]


def test_benchmark_scenarios_file_is_consistent():
    with open(Path(__file__).resolve().parents[1] / 'benchmark_scenarios.json', encoding='utf-8') as f:
        scenarios = json.load(f)['scenarios']
    main = scenarios['main']
    assert not main['optional']
    for model in ('v1', 'v2', 'v3'):
        assert main['models'][model]['seeds'] == [42, 43, 44]
    assert main['models']['v4']['seeds'] == [42, 43, 44, 45]
    assert main['models']['v5']['seeds'] == [42, 43, 44]
    assert main['models']['v4']['epochs'] >= 45 and main['models']['v4']['patience'] >= 8
    baseline = main['models']['v3']
    for name in ('ablation_no_sampler', 'ablation_no_focal', 'ablation_no_mixup'):
        ablation = scenarios[name]
        assert list(ablation['models']) == ['v3']
        for key in ('epochs', 'patience', 'batch_size', 'lr_stage1', 'lr_stage2'):
            assert ablation['models']['v3'][key] == baseline[key], f"{name}.{key} differs from main/v3 baseline"
        assert ablation['split_seed'] == main['split_seed']
    assert scenarios['ablation_no_sampler']['balanced_sampling'] is False
    assert scenarios['ablation_no_focal']['loss'] == 'ce'
    assert scenarios['ablation_no_mixup']['mixup_alpha'] == 0.0
    for legacy in ('standard', 'medium', 'low'):
        assert scenarios[legacy]['optional']


def test_rtx_5070_uses_12gb_tier():
    assert compute_adaptive_batch_strategy(11.9, 'v5') == {
        'micro_batch': 8, 'grad_accum_steps': 4, 'tier_gb': 12
    }
    assert compute_adaptive_batch_strategy(11.9, 'v1') == {
        'micro_batch': 32, 'grad_accum_steps': 1, 'tier_gb': 12
    }

