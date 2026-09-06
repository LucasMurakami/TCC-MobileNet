import importlib

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from train_timm_models import evaluate_dataset, evaluation_accuracy_consistency
from visualize import PyTorchGradCAM, OcclusionSensitivity, attribution_agreement, find_gradcam_target_layer


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


def test_logit_adjust_changes_decisions_not_probabilities():
    class FixedModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.tensor([[2.0, 1.8, 0, 0, 0, 0, 0]], dtype=inputs.dtype).repeat(len(inputs), 1)

    inputs = torch.zeros(2, 3, 2, 2)
    targets = torch.zeros(2, dtype=torch.long)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    base = evaluate_dataset(FixedModel(), loader, torch.device('cpu'), torch.float32, False)
    adjusted = evaluate_dataset(
        FixedModel(), loader, torch.device('cpu'), torch.float32, False,
        logit_adjust=0.5, class_priors=[0.8, 0.02, 0.036, 0.036, 0.036, 0.036, 0.036]
    )
    torch.testing.assert_close(torch.tensor(base['all_probs']), torch.tensor(adjusted['all_probs']))
    assert base['all_preds'][0] == 0
    assert adjusted['all_preds'][0] == 1
    assert adjusted['all_logits'].shape == (2, 7)


def test_temperature_scales_probabilities_but_keeps_logits_and_argmax():
    class FixedModel(torch.nn.Module):
        def forward(self, inputs):
            return torch.tensor([[3.0, 1.0, 0, 0, 0, 0, 0]], dtype=inputs.dtype).repeat(len(inputs), 1)

    inputs = torch.zeros(2, 3, 2, 2)
    loader = DataLoader(TensorDataset(inputs, torch.zeros(2, dtype=torch.long)), batch_size=2)
    base = evaluate_dataset(FixedModel(), loader, torch.device('cpu'), torch.float32, False)
    scaled = evaluate_dataset(FixedModel(), loader, torch.device('cpu'), torch.float32, False, temperature=2.0)
    torch.testing.assert_close(torch.tensor(base['all_logits']), torch.tensor(scaled['all_logits']))
    assert scaled['all_probs'][0, 0] < base['all_probs'][0, 0]
    assert scaled['all_probs'][0, 0] == pytest.approx(torch.softmax(torch.tensor([1.5, 0.5, 0, 0, 0, 0, 0]), 0)[0].item(), abs=1e-6)
    assert scaled['all_preds'][0] == base['all_preds'][0] == 0
    assert scaled['temperature'] == 2.0
    with pytest.raises(ValueError):
        evaluate_dataset(FixedModel(), loader, torch.device('cpu'), torch.float32, False, temperature=0.0)


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


def test_v5_gradcam_targets_last_stage_before_msfa():
    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(2, 2, 1)

    class FakeV5(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.Sequential(*[torch.nn.Sequential(torch.nn.Conv2d(2, 2, 3)) for _ in range(4)])
            self.msfa = torch.nn.Module()
            self.msfa.ffn = torch.nn.Module()
            self.msfa.ffn.pw_proj = Projection()
            self.msfa.norm = torch.nn.BatchNorm2d(2)

    model = FakeV5()
    assert find_gradcam_target_layer(model) is model.blocks[3]


def _center_sensitive_model():
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
    return model


def test_occlusion_sensitivity_localizes_synthetic_center_signal():
    model = _center_sensitive_model()
    inputs = torch.zeros(1, 3, 32, 32)
    inputs[:, :, 12:20, 12:20] = 4.0
    occ = OcclusionSensitivity(model, patch_frac=0.25, stride_frac=0.125, batch_size=8)
    heatmap, probabilities = occ(inputs, target_class=0)
    row, column = divmod(int(torch.tensor(heatmap).argmax()), 32)

    assert heatmap.shape == (32, 32)
    assert torch.isfinite(torch.tensor(heatmap)).all()
    assert heatmap.max() == pytest.approx(1.0)
    assert heatmap[0, 0] < 0.2
    assert 10 <= row <= 21 and 10 <= column <= 21
    assert probabilities.shape == (7,)


def test_attribution_agreement_bounds():
    a = torch.zeros(16, 16)
    a[4:8, 4:8] = 1.0
    same = attribution_agreement(a.numpy(), a.numpy())
    assert same['pearson'] == pytest.approx(1.0)
    assert same['top_iou'] == pytest.approx(1.0)
    b = torch.zeros(16, 16)
    b[10:14, 10:14] = 1.0
    disjoint = attribution_agreement(a.numpy(), b.numpy())
    assert disjoint['pearson'] < 0.0
    assert disjoint['top_iou'] == pytest.approx(0.0)


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
