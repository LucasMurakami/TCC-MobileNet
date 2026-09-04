import importlib

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from train_timm_models import evaluate_dataset


@pytest.mark.parametrize('module_name', ['dataset', 'metrics', 'visualize', 'train_timm_models', 'main'])
def test_core_module_imports(module_name):
    assert importlib.import_module(module_name) is not None


def test_fp32_cpu_evaluation_path():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 7))
    inputs = torch.rand(4, 3, 2, 2)
    targets = torch.tensor([0, 1, 4, 4])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    result = evaluate_dataset(
        model, loader, torch.device('cpu'), torch.float32, False, autocast=False
    )
    assert result['all_probs'].shape == (4, 7)
    assert 0.0 <= result['mel_auc_roc'] <= 1.0
