import importlib

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from train_timm_models import evaluate_dataset, evaluation_accuracy_consistency
from visualize import PyTorchGradCAM, find_gradcam_target_layer


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
    assert 0.0 <= result['expected_calibration_error'] <= 1.0


def test_evaluation_accuracy_consistency_guard():
    small_delta, small_warning = evaluation_accuracy_consistency(0.81, 0.80)
    assert small_delta == pytest.approx(0.01)
    assert not small_warning
    delta, warning = evaluation_accuracy_consistency(0.50, 0.81)
    assert delta == pytest.approx(-0.31)
    assert warning


def test_tta_averages_probabilities_over_flip_views():
    class PositionModel(torch.nn.Module):
        def forward(self, inputs):
            logits = torch.zeros((len(inputs), 7), dtype=inputs.dtype)
            logits[:, 0] = inputs[:, 0, 0, 0] * 4
            logits[:, 1] = inputs[:, 0, -1, -1] * 4
            return logits

    model = PositionModel()
    inputs = torch.zeros(1, 3, 2, 2)
    inputs[0, 0, 0, 0] = 1
    loader = DataLoader(TensorDataset(inputs, torch.tensor([0])), batch_size=1)
    result = evaluate_dataset(model, loader, torch.device('cpu'), torch.float32, False, use_tta=True)
    views = [inputs, torch.flip(inputs, [-1]), torch.flip(inputs, [-2])]
    expected = torch.stack([torch.softmax(model(view), dim=1) for view in views]).mean(dim=0)

    torch.testing.assert_close(torch.tensor(result['all_probs']), expected)


def test_v5_gradcam_targets_pre_norm_projection():
    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(2, 2, 1)

    class FakeV5(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.msfa = torch.nn.Module()
            self.msfa.ffn = torch.nn.Module()
            self.msfa.ffn.pw_proj = Projection()
            self.msfa.norm = torch.nn.BatchNorm2d(2)

    model = FakeV5()
    assert find_gradcam_target_layer(model) is model.msfa.ffn.pw_proj.conv


def test_gradcam_localizes_synthetic_center_signal():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 1, 1, bias=False),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(1, 7, bias=False),
    )
    with torch.no_grad():
        model[0].weight.fill_(1.0)
        model[4].weight.zero_()
        model[4].weight[0].fill_(1.0)
    inputs = torch.zeros(1, 3, 16, 16)
    inputs[:, :, 6:10, 6:10] = 1.0
    cam = PyTorchGradCAM(model, target_layer=model[0])
    heatmap, probabilities = cam(inputs, target_class=0)
    cam.remove_hooks()
    row, column = divmod(int(torch.tensor(heatmap).argmax()), 16)

    assert heatmap.shape == (16, 16)
    assert torch.isfinite(torch.tensor(heatmap)).all()
    assert heatmap.std() > 0.05
    assert 5 <= row <= 10 and 5 <= column <= 10
    assert probabilities.shape == (7,)
