import torch
from torch.nn import functional as F
from torchvision.datasets import CIFAR10

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def normalize(x):
    return (x - x.new_tensor(MEAN)[None, :, None, None]) / x.new_tensor(STD)[None, :, None, None]


def augment(x, generator=None):
    """Index-based crop/flip supports second derivatives (unlike grid_sample)."""
    n, _, h, w = x.shape
    shifts = torch.randint(0, 9, (n, 2), device=x.device, generator=generator)
    flip = torch.rand(n, device=x.device, generator=generator) < .5
    padded = F.pad(x, (4, 4, 4, 4))
    rows = shifts[:, 0, None, None] + torch.arange(h, device=x.device)[None, :, None]
    cols = shifts[:, 1, None, None] + torch.arange(w, device=x.device)[None, None, :]
    out = padded.permute(0, 2, 3, 1)[torch.arange(n, device=x.device)[:, None, None], rows, cols]
    out = out.permute(0, 3, 1, 2)
    return torch.where(flip[:, None, None, None], out.flip(-1), out)


def load_data(cfg):
    arrays = []
    for train, limit in [(True, cfg['train_per_class']), (False, cfg['test_per_class'])]:
        ds = CIFAR10(cfg['data_root'], train=train, download=True)
        x = torch.from_numpy(ds.data.copy()).permute(0, 3, 1, 2)
        y = torch.tensor(ds.targets)
        indices = []
        for c in cfg['classes']:
            found = torch.where(y == c)[0]
            indices.append(found if limit is None else found[:limit])
        indices = torch.cat(indices)
        arrays.extend([x[indices].to(cfg['device'], dtype=torch.float32).div_(255),
                       y[indices].to(cfg['device'])])
    return tuple(arrays)


def project(delta, clean, epsilon):
    with torch.no_grad():
        delta.clamp_(-epsilon, epsilon)
        delta.copy_((clean + delta).clamp(0, 1) - clean)


def assert_integrity(clean, poisoned, labels, original_labels, indices, target, epsilon, patch):
    assert torch.equal(labels, original_labels), 'Training labels changed'
    assert len(indices.unique()) == len(indices)
    assert bool((labels[indices] == target).all()), 'Poisons outside target class'
    delta = poisoned - clean
    assert float(delta.abs().max()) <= epsilon + 1e-6, 'Pixel-space budget exceeded'
    assert bool(((poisoned >= 0) & (poisoned <= 1)).all())
    changed = delta.flatten(1).ne(0).any(1)
    expected = torch.zeros_like(changed)
    expected[indices] = True
    assert torch.equal(changed, expected), 'Incorrect poison count or indices'
    # Bernoulli trigger is binary; only binary windows can equal it exactly.
    # Prefilter every spatial window without allocating N*C*H*W*P*P.
    p = patch.shape[-1]
    binary = ((poisoned == 0) | (poisoned == 1)).all(1, keepdim=True).float()
    windows = F.avg_pool2d(binary, p, stride=1).squeeze(1)
    for i, row, col in (windows == 1).nonzero().tolist():
        assert not torch.equal(poisoned[i, :, row:row+p, col:col+p], patch), 'Trigger in training data'
    return {'max_delta': float(delta.abs().max()), 'changed_images': int(changed.sum()),
            'labels_unchanged': True, 'trigger_absent': True}
