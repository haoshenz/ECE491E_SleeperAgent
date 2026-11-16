import copy
import json
import pytest
import torch
from torch import nn
from src.patch import apply_patch, make_patch
from src.victim import evaluate
from scripts.rescore_table6 import (FIXED_POSITIONS, verify_random, tables, render_report,
                                   snapshot, verify_unchanged, trained_directory)


class Detector(nn.Module):
    def __init__(self, patch):
        super().__init__()
        self.patch = patch

    def forward(self, x):
        detected = x[:, :, 19:27, 19:27].eq(self.patch).flatten(1).all(1)
        return torch.stack([~detected, detected], dim=1).float()


def test_evaluation_default_stays_random_and_inset_position_is_distinct():
    patch = make_patch(8, 18, 'cpu')
    model = Detector(patch)
    x = torch.full((6, 3, 32, 32), .3)
    y = torch.tensor([0, 0, 0, 0, 1, 1])
    args = (model, x, y, 0, 1, patch)
    default = evaluate(*args, batch_size=2, seed=717)
    assert default == evaluate(*args, batch_size=2, seed=717, location='random')
    assert default['asr_definition'] == 'all held-out source images; uniformly random patch location'
    assert evaluate(*args, location=FIXED_POSITIONS['htba'])['asr'] == 1.
    assert evaluate(*args, location='fixed')['asr'] == 0.
    assert bool((x == .3).all())


def test_explicit_patch_coordinates_are_validated_and_do_not_mutate_input():
    x = torch.full((2, 3, 32, 32), .3)
    patch = make_patch(8, 18, 'cpu')
    stamped = apply_patch(x, patch, (19, 19))
    assert torch.equal(stamped[0, :, 19:27, 19:27], patch)
    assert torch.equal(apply_patch(x, patch, 'fixed'), apply_patch(x, patch, (24, 24)))
    assert bool((x == .3).all())
    with pytest.raises(ValueError, match='does not fit'):
        apply_patch(x, patch, (25, 24))


def test_random_mismatch_fails_even_for_one_image():
    original = {'source_test_count': 1000, 'asr': .3, 'clean_accuracy': .9}
    record = {'attack': 'htba', 'architecture': 'resnet18', 'metrics': original}
    verify_random(record, dict(original))
    changed = dict(original, asr=.301)
    with pytest.raises(RuntimeError, match='RANDOM asr mismatch'):
        verify_random(record, changed)
    changed = dict(original, clean_accuracy=.9001)
    with pytest.raises(RuntimeError, match='RANDOM clean_accuracy mismatch'):
        verify_random(record, changed)


def test_native_table_uses_fixed_baselines_and_random_sleeper():
    cells = []
    for attack, fixed, random in [('htba', .8, .1), ('clba', .6, .2), ('full', .9, .3)]:
        for arch in ('resnet18', 'mobilenetv2', 'vgg11'):
            cells.append({'attack': attack, 'architecture': arch,
                          'fixed': {'asr': fixed}, 'random': {'asr': random, 'clean_accuracy': .92}})
    result = tables(cells)
    assert [r['mean_asr'] for r in result['native']] == pytest.approx([.8, .6, .3])
    assert [r['mean_asr'] for r in result['random']] == pytest.approx([.1, .2, .3])
    report = render_report({'cells': cells, 'tables': result, 'elapsed_seconds': 1., 'completed_hawaii': 'test'})
    assert 'Sleeper Agent with the trigger FIXED at (24,24)' in report
    assert 'one pair (airplane->automobile), one seed; not a statistical reproduction' in report
    assert 'CLBA used a clean surrogate, not an adversarially trained surrogate.' in report


def test_protected_file_mutation_is_detected(tmp_path):
    path = tmp_path/'result.json'
    path.write_text('original')
    before = snapshot([path])
    verify_unchanged(before)
    path.write_text('modified')
    with pytest.raises(RuntimeError, match='Protected input files changed'):
        verify_unchanged(before)


def test_reused_victim_resolves_to_original_saved_checkpoint(tmp_path):
    record = {'directory': str(tmp_path/'new'), 'reused': True,
              'reused_from': str(tmp_path/'original'/'result.json')}
    assert trained_directory(record) == tmp_path/'original'
