import copy
import json
import torch
import pytest
from torch import nn
from torch.nn import functional as F
from src.craft import alignment_step, flat_grad, average_gradient, signed_adam_step
from src.data import augment, project, assert_integrity, STD
from src.patch import make_patch, apply_patch
from src.selection import select_poisons
from src.common import append_jsonl, read_jsonl, seed_all, configure
from src.victim import evaluate, train
from src.models import make_model
from analysis.summarize import summarize


def tiny():
    torch.manual_seed(3)
    return nn.Sequential(nn.Conv2d(3, 3, 3, padding=1), nn.Tanh(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2)).eval()


def test_two_pass_matches_direct_global_cosine_and_finite_difference():
    model = tiny().double()
    x = torch.rand(5, 3, 8, 8, dtype=torch.double)
    y = torch.tensor([1, 1, 1, 1, 1])
    delta = torch.zeros_like(x, requires_grad=True)
    a = flat_grad(model, x.flip(0), 1-y).detach()
    cosine, actual = alignment_step(model, x, delta, y, a, batch_size=2, seed=27)
    # Direct reference retains every graph, including an uneven final minibatch.
    generator = torch.Generator().manual_seed(27)
    total = 0
    for start in range(0, len(x), 2):
        sl = slice(start, start+2)
        xb = augment(x[sl] + delta[sl], generator)
        total = total + flat_grad(model, xb, y[sl], True) * (len(xb)/len(x))
    loss = 1-F.cosine_similarity(total[None], a[None])[0]
    expected = torch.autograd.grad(loss, delta)[0]
    assert abs(cosine - (1-loss.item())) < 1e-10
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-10)
    direction = torch.randn_like(delta)
    direction /= direction.norm()
    eps = 1e-4
    cp, _ = alignment_step(model, x, delta+eps*direction, y, a, 2, 27)
    cm, _ = alignment_step(model, x, delta-eps*direction, y, a, 2, 27)
    assert abs((cm-cp)/(2*eps) - float((actual*direction).sum())) < 1e-7


def test_gradient_average_is_global_not_per_layer():
    model = tiny()
    x, y = torch.rand(5, 3, 8, 8), torch.tensor([0, 1, 0, 1, 0])
    torch.testing.assert_close(average_gradient(model, x, y, 2), flat_grad(model, x, y), atol=1e-7, rtol=1e-5)


def test_signed_adam_normalized_updates_match_reference_coordinates():
    delta = torch.zeros(2, 3, 4, 4, requires_grad=True)
    z = torch.zeros_like(delta, requires_grad=True)
    scale = delta.new_tensor(STD)[None, :, None, None]
    pixel_opt = torch.optim.Adam([delta], lr=.025)
    normalized_opt = torch.optim.Adam([z], lr=.025)
    for _ in range(5):
        grad = torch.randn_like(delta)
        signed_adam_step(pixel_opt, delta, grad, units='normalized')
        normalized_opt.zero_grad(set_to_none=True)
        z.grad = (grad*scale).sign()
        normalized_opt.step()
        torch.testing.assert_close(delta, z*scale, rtol=1e-5, atol=1e-8)


def test_patch_overwrites_and_does_not_modify_input():
    x = torch.full((4, 3, 32, 32), .3)
    patch = make_patch(8, 7, 'cpu')
    out = apply_patch(x, patch, 'fixed')
    assert torch.equal(out[:, :, -8:, -8:], patch.expand(4, -1, -1, -1))
    assert int((out != x).sum()) == 4*3*8*8
    assert bool((x == .3).all())
    a = apply_patch(x, patch, 'random', torch.Generator().manual_seed(1))
    b = apply_patch(x, patch, 'random', torch.Generator().manual_seed(1))
    assert torch.equal(a, b)


def test_pixel_projection_integrity_and_rejections():
    x = torch.rand(5, 3, 32, 32)
    idx = torch.tensor([1, 3])
    y = torch.tensor([0, 1, 0, 1, 0])
    delta = torch.randn_like(x[idx])
    eps = 16/255
    project(delta, x[idx], eps)
    out = x.clone()
    out[idx] += delta
    patch = make_patch(8, 7, 'cpu')
    assert assert_integrity(x, out, y, y.clone(), idx, 1, eps, patch)['changed_images'] == 2
    with pytest.raises(AssertionError, match='labels changed'):
        assert_integrity(x, out, 1-y, y, idx, 1, eps, patch)
    bad = out.clone()
    bad[0, 0, 0, 0] += .01
    with pytest.raises(AssertionError, match='poison count'):
        assert_integrity(x, bad, y, y, idx, 1, eps, patch)
    # A matching trigger in an untouched image is caught by the exact-window scan.
    triggered = x.clone()
    triggered[0, :, 3:11, 4:12] = patch
    out = triggered.clone()
    out[idx] += delta
    with pytest.raises(AssertionError, match='Trigger'):
        assert_integrity(triggered, out, y, y, idx, 1, eps, patch)


def test_full_gradient_selection_uses_all_parameters():
    model = tiny()
    x, y = torch.rand(6, 3, 8, 8), torch.tensor([0, 1, 1, 0, 1, 1])
    chosen, _ = select_poisons(model, x, y, 1, 2, 'gradient', 4)
    candidates = torch.where(y == 1)[0]
    norms = torch.stack([flat_grad(model, x[i:i+1], y[i:i+1]).norm() for i in candidates])
    assert torch.equal(chosen, candidates[norms.topk(2).indices])


def test_jsonl_recovers_torn_final_record(tmp_path):
    p = tmp_path/'results.jsonl'
    append_jsonl({'a': 1}, p)
    with p.open('ab') as f:
        f.write(b'{"incomplete"')
    append_jsonl({'a': 2}, p)
    assert read_jsonl(p) == [{'a': 1}, {'a': 2}]


def test_paired_analysis_and_no_invented_uncertainty(tmp_path):
    rows = []
    for seed, a, b in [(1, .1, .5), (2, .7, .8)]:
        for arm, asr in [('random', a), ('full', b)]:
            rows.append({'source': 0, 'target': 1, 'seed': seed, 'arm': arm, 'trial_id': f'{seed}{arm}',
                         'metrics': {'asr': asr, 'clean_accuracy': .9},
                         'victim_history': [], 'alignment': []})
    result = summarize(rows, tmp_path)
    assert result['paired_differences']['full minus random']['mean'] == pytest.approx(25)
    assert result['arms']['full']['n'] == 2


def test_asr_denominator_is_all_source_images():
    class Constant(nn.Module):
        def forward(self, x):
            return torch.tensor([0., 1.]).expand(len(x), 2)
    x = torch.rand(4, 3, 32, 32)
    m = evaluate(Constant(), x, torch.tensor([0, 0, 1, 1]), 0, 1, make_patch(8, 2, 'cpu'))
    assert m['asr'] == 1 and m['source_test_count'] == 2
    assert m['asr_clean_correct'] is None


def test_epoch_resume_matches_uninterrupted_training(tmp_path):
    configure()
    cfg = dict(architecture='resnet9', width=2, device='cpu', lr=.02,
               weight_decay=.0004, milestones=[1], amp=False, batch_size=4, seed=7)
    seed_all(9)
    x, y = torch.rand(8, 3, 32, 32), torch.tensor([0, 1]*4)
    patch = make_patch(8, 7, 'cpu')
    kwargs = dict(cfg=cfg, x=x, y=y, test_x=x, test_y=y, seed=12, source=0, target=1, patch=patch)
    a, _, _ = train(**kwargs, epochs=2, directory=tmp_path/'continuous')
    train(**kwargs, epochs=1, directory=tmp_path/'resumed')
    b, _, _ = train(**kwargs, epochs=2, directory=tmp_path/'resumed')
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_real_resnet_cuda_double_backward_is_finite():
    configure()
    model = make_model(dict(architecture='resnet18', width=4, device='cuda')).eval()
    x = torch.rand(3, 3, 32, 32, device='cuda')
    y = torch.ones(3, dtype=torch.long, device='cuda')
    a = flat_grad(model, x, torch.zeros_like(y)).detach()
    c, g = alignment_step(model, x, torch.zeros_like(x), y, a, 2, 19)
    assert torch.isfinite(torch.tensor(c)) and bool(torch.isfinite(g).all())
    assert float(g.abs().max()) > 0


def test_crafting_resume_replays_updates_and_retraining(tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module('src.craft')
    cfg = dict(seed=7, epsilon=16/255, craft_lr=.01, craft_steps=6,
               retraining_factor=2, craft_batch_size=2, patch_location='random', checkpoint_every=1)
    x = torch.rand(6, 3, 8, 8)
    y = torch.tensor([0, 0, 0, 1, 1, 1])
    indices = torch.tensor([3, 4])
    patch = make_patch(2, 1, 'cpu')
    events = []
    def retrain(delta, step):
        events.append(step)
        return tiny()
    args = (x, y, indices, x[:3], 1, patch, cfg)
    expected, history = module.craft(tiny(), *args, tmp_path/'whole', retrain)
    assert events == [2, 4]
    original_save = module.atomic_save
    def interrupt(value, path):
        original_save(value, path)
        if value['step'] == 3:
            raise RuntimeError('simulated interruption')
    monkeypatch.setattr(module, 'atomic_save', interrupt)
    with pytest.raises(RuntimeError, match='simulated'):
        module.craft(tiny(), *args, tmp_path/'resumed', retrain)
    monkeypatch.setattr(module, 'atomic_save', original_save)
    actual, resumed_history = module.craft(tiny(), *args, tmp_path/'resumed', retrain)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert history == resumed_history
