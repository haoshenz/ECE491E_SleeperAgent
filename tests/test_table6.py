import copy
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import pytest
import torch
from torch import nn
import yaml
from src.baselines import (craft_baseline, clipped_trigger, greedy_assignment, stamp,
                           DEFAULTS)
from src.common import configure
from src.data import assert_integrity, normalize
from src.models import make_model, penultimate
from src.patch import make_patch
from src.run_trial import shared_directory
from src.table6 import cell_config, table_shared, reuse_check
from src.table6_progress import Dashboard
from scripts.make_table6 import build
from scripts.table6_preflight import estimate

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_cache_fingerprint_is_unchanged():
    cfg = yaml.safe_load((ROOT/'configs/cifar10_full.yaml').read_text())
    cfg.update(source=0, target=1, seed=17)
    assert shared_directory(cfg) == Path('results/full/shared/527ca756e205')
    from src.run_trial import ARM_ONLY_KEYS
    assert ARM_ONLY_KEYS == ('selection', 'patch_location', 'retraining_factor',
                            'craft_steps', 'craft_lr', 'craft_lr_units', 'output')


def test_new_attacks_share_surrogate_without_changing_old_fingerprints():
    cfg = yaml.safe_load((ROOT/'configs/table6_resnet18.yaml').read_text())
    directories = {table_shared(cell_config(cfg, attack)) for attack in ('htba', 'clba', 'full')}
    assert directories == {Path('results/table6/shared/527ca756e205')}


@pytest.mark.parametrize('arch,feature_dim', [('resnet18', 512), ('mobilenetv2', 1280), ('vgg11', 512)])
def test_feature_classifier_split(arch, feature_dim):
    configure()
    model = make_model(dict(architecture=arch, width=64, device='cpu')).eval()
    x = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        features = penultimate(model, x)
        assert features.shape == (2, feature_dim)
        torch.testing.assert_close(model(x), model.classifier(features), rtol=0, atol=0)
    assert model(x).shape == (2, 10)


@pytest.mark.parametrize('arch,filename,factory', [
    ('mobilenetv2', 'mobilenetv2.py', 'MobileNetV2'), ('vgg11', 'vgg.py', 'vgg11')])
def test_new_models_numerically_match_reference_topology(arch, filename, factory):
    reference = ROOT/'results/table6/reference/poisoning-benchmark/models'/filename
    if not reference.exists():
        pytest.skip('Optional pinned reference checkout absent')
    configure()
    spec = importlib.util.spec_from_file_location('table6_reference_'+arch, reference)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference_model = getattr(module, factory)().eval()
    model = make_model(dict(architecture=arch, width=64, device='cpu')).eval()
    kinds = (nn.Conv2d, nn.BatchNorm2d, nn.Linear)
    ours = [m for m in model.modules() if isinstance(m, kinds)]
    theirs = [m for m in reference_model.modules() if isinstance(m, kinds)]
    assert len(ours) == len(theirs)
    for current, original in zip(ours, theirs):
        assert type(current) is type(original)
        current.load_state_dict(original.state_dict())
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in reference_model.parameters())
    x = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(model(x), reference_model(normalize(x)), rtol=1e-5, atol=1e-6)


def test_exact_500_poison_budget_labels_and_method_specific_trigger_rules():
    x = torch.full((520, 3, 32, 32), .5)
    y = torch.ones(520, dtype=torch.long)
    indices = torch.arange(500)
    patch = make_patch(8, 18, 'cpu')
    altered = x.clone()
    altered[indices] += .01
    ordinary = assert_integrity(x, altered, y, y.clone(), indices, 1, 16/255, patch)
    assert ordinary['changed_images'] == 500 and ordinary['trigger_absent']
    with pytest.raises(ValueError, match='Only clba'):
        assert_integrity(x, altered, y, y, indices, 1, 16/255, patch, allow_trigger=True, arm='full')
    clba = altered.clone()
    clba[indices] = clipped_trigger(x[indices], altered[indices], patch, 16/255)
    result = assert_integrity(x, clba, y, y.clone(), indices, 1, 16/255, patch, allow_trigger=True, arm='clba')
    assert result['max_delta'] <= 16/255+1e-6
    assert bool((clba[indices, :, -8:, -8:] != altered[indices, :, -8:, -8:]).all())
    expected = .5 + (patch*2-1)*(16/255)
    torch.testing.assert_close(clba[0, :, -8:, -8:], expected)
    with pytest.raises(AssertionError, match='labels changed'):
        assert_integrity(x, clba, 1-y, y, indices, 1, 16/255, patch, allow_trigger=True, arm='clba')
    with pytest.raises(AssertionError, match='Incorrect poison count'):
        assert_integrity(x, clba, y, y, indices[:-1], 1, 16/255, patch, allow_trigger=True, arm='clba')


def test_htba_assignment_is_a_bijection_and_patch_placement_matches_settings():
    source = torch.tensor([[0.], [10.], [20.]])
    poison = torch.tensor([[19.], [.1], [9.]])
    assert greedy_assignment(source, poison).tolist() == [2, 0, 1]
    x = torch.full((2, 3, 32, 32), .5)
    patch = make_patch(8, 18, 'cpu')
    assert torch.equal(stamp(x, patch, 5)[0, :, 19:27, 19:27], patch)
    assert torch.equal(stamp(x, patch)[0, :, 24:32, 24:32], patch)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.Tanh(),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Linear(4, 10)

    def forward(self, x):
        return self.classifier(self.features(normalize(x)))


@pytest.mark.parametrize('attack', ['htba', 'clba'])
def test_baseline_resume_matches_uninterrupted_and_is_bounded(tmp_path, attack, monkeypatch):
    torch.manual_seed(7)
    model = Toy().eval()
    x, y = torch.rand(8, 3, 32, 32), torch.tensor([0]*4+[1]*4)
    idx, patch = torch.arange(4, 8), make_patch(8, 18, 'cpu')
    cfg = dict(attack=attack, seed=17, epsilon=16/255, checkpoint_every=1,
               baseline_settings={'steps': 4, 'batch_size': 2})
    if attack == 'htba':
        cfg['baseline_settings']['early_stop_loss'] = -1.
    args = (model, x, y, idx, x[:4], patch, cfg)
    expected, pre, _ = craft_baseline(*args, tmp_path/'whole')
    craft_baseline(*args, tmp_path/'resumed', step_limit=2)
    actual, pre_actual, _ = craft_baseline(*args, tmp_path/'resumed')
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(pre_actual, pre, rtol=0, atol=0)
    assert float(actual.abs().max()) <= 16/255+1e-6
    assert float(pre.abs().max()) <= 16/255+1e-6
    poisoned = x.clone()
    poisoned[idx] += actual
    result = assert_integrity(x, poisoned, y, y.clone(), idx, 1, 16/255, patch,
                              allow_trigger=attack == 'clba', arm=attack)
    assert result['changed_images'] == 4
    if attack == 'clba':
        torch.testing.assert_close(poisoned[idx], clipped_trigger(x[idx], x[idx]+pre, patch, 16/255))


def test_report_never_fills_missing_cells_or_partial_row_averages(tmp_path):
    output, report = tmp_path/'results', tmp_path/'report'
    cell = output/'cells'/'example'
    cell.mkdir(parents=True)
    row = dict(attack='full', architecture='resnet18', source=0, target=1, seed=17,
               metrics={'asr': .5, 'clean_accuracy': .9}, directory=str(cell), reused=False)
    (cell/'result.json').write_text(json.dumps(row))
    result = build(output, report)
    assert result['completed_cells'] == 1
    assert result['rows'][-1]['mean_asr'] is None
    assert 'not run' in (report/'table6.md').read_text()


def test_dashboard_writes_status_error_and_reused_result(tmp_path):
    row = {'metrics': {'asr': .999}, 'reused': True}
    dashboard = Dashboard(tmp_path, tmp_path/'results', {'full/resnet18': 0}, {'full/resnet18': row}, interval=.05)
    dashboard.start()
    time.sleep(.12)
    dashboard.close('CRASHED', 'RuntimeError: deliberate test')
    text = (tmp_path/'progress.md').read_text()
    assert 'Status: CRASHED' in text and 'deliberate test' in text and '99.90% (reused)' in text


def test_reuse_checks_every_setting_except_output():
    cfg = yaml.safe_load((ROOT/'configs/table6_resnet18.yaml').read_text())
    if not (ROOT/'results/full/0-1_s17_full_6e499e05ba36/result.json').exists():
        pytest.skip('Legacy local experiment not available')
    assert reuse_check(cfg)[0]
    cfg['epochs_victim'] = 79
    assert reuse_check(cfg) == (False, ['epochs_victim'])


def test_estimate_includes_full_schedules_and_margin():
    timings = {arch: {'seconds_per_epoch': 30} for arch in ('resnet18', 'mobilenetv2', 'vgg11')}
    crafting = {arch: {'htba_step_seconds': 1., 'clba_step_seconds': 1., 'full_step_seconds': 1.,
                       'selection_seconds': 100., 'adversarial_gradient_seconds': 40.}
                for arch in timings}
    result = estimate(timings, crafting, True)
    assert result['cells_seconds']['full/resnet18'] == 0
    assert result['cells_seconds']['htba/mobilenetv2'] == (120+80)*30+5000
    assert result['cells_seconds']['full/mobilenetv2'] == (160+80)*30+250+300
    assert result['estimated_hours'] == pytest.approx(result['raw_hours']*1.15)


def test_runner_refuses_missing_stale_or_over_budget_preflight(tmp_path, monkeypatch):
    from scripts import run_table6 as runner
    monkeypatch.setattr(runner, 'OUTPUT', tmp_path)
    monkeypatch.setattr(runner, 'implementation_digest', lambda: 'current')
    with pytest.raises(RuntimeError, match='Preflight gate'):
        runner.verify_gate()
    valid = dict(passed=True, tests_passed=True, smoke_cells=9,
                 estimate={'estimated_hours': 21.9}, implementation_digest='current')
    (tmp_path/'preflight.json').write_text(json.dumps(valid))
    assert runner.verify_gate() == valid
    valid['estimate']['estimated_hours'] = 22.1
    (tmp_path/'preflight.json').write_text(json.dumps(valid))
    with pytest.raises(RuntimeError, match='Preflight gate'):
        runner.verify_gate()
    valid['estimate']['estimated_hours'] = 21.
    valid['implementation_digest'] = 'outdated'
    (tmp_path/'preflight.json').write_text(json.dumps(valid))
    with pytest.raises(RuntimeError, match='changed since preflight'):
        runner.verify_gate()


def fake_runner(tmp_path, monkeypatch):
    from scripts import run_table6 as runner
    import src.data
    output = tmp_path/'results'
    output.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'OUTPUT', output)
    monkeypatch.setattr(runner, 'REPORT', tmp_path/'report')
    monkeypatch.setattr(runner.sys, 'argv', ['run_table6.py'])
    preflight = {'estimate': {'cells_seconds': {'htba/resnet18': 1.}},
                 'timings': {'epochs': {}, 'crafting': {}}}
    monkeypatch.setattr(runner, 'verify_gate', lambda: preflight)
    monkeypatch.setattr(runner, 'configs', lambda: {a: {} for a in ('resnet18', 'mobilenetv2', 'vgg11')})
    monkeypatch.setattr(runner, 'import_legacy', lambda cfg: {'reused': False})
    monkeypatch.setattr(runner, 'collect', lambda path: {})
    monkeypatch.setattr(runner, 'build', lambda *args: None)
    monkeypatch.setattr(src.data, 'load_data', lambda cfg: (None,)*4)
    return runner, output


def test_runner_honors_accumulated_budget_before_starting_next_cell(tmp_path, monkeypatch):
    runner, output = fake_runner(tmp_path, monkeypatch)
    (output/'run_state.json').write_text(json.dumps({'seconds_used': 22*3600}))
    def unexpected(*args, **kwargs):
        raise AssertionError('Should not start another cell after 22 hours')
    monkeypatch.setattr(runner, 'run_cell', unexpected)
    runner.main()
    assert 'Status: STOPPED' in (tmp_path/'progress.md').read_text()
    assert not (output/'runner.lock').exists()


def test_runner_records_failure_and_releases_lock(tmp_path, monkeypatch):
    runner, output = fake_runner(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, 'cell_config', lambda *args: {})
    monkeypatch.setattr(runner, 'cell_directory', lambda cfg: output/'cell')
    monkeypatch.setattr(runner, 'stages_for', lambda *args: [])
    def fail(*args, **kwargs):
        raise RuntimeError('simulated GPU failure')
    monkeypatch.setattr(runner, 'run_cell', fail)
    with pytest.raises(RuntimeError, match='simulated GPU failure'):
        runner.main()
    text = (tmp_path/'progress.md').read_text()
    assert 'Status: CRASHED' in text and 'RuntimeError: simulated GPU failure' in text
    assert not (output/'runner.lock').exists()
