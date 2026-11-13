import torch


def make_patch(size, seed, device):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (3, size, size), generator=g).float().to(device)


def apply_patch(x, patch, location='random', generator=None):
    n, _, h, w = x.shape
    p = patch.shape[-1]
    assert p <= min(h, w)
    if location == 'random':
        rows = torch.randint(h-p+1, (n,), generator=generator, device=x.device)
        cols = torch.randint(w-p+1, (n,), generator=generator, device=x.device)
    elif location == 'fixed':
        rows = torch.full((n,), h-p, device=x.device)
        cols = torch.full((n,), w-p, device=x.device)
    else:
        raise ValueError(location)
    out = x.clone()
    rr = rows[:, None, None] + torch.arange(p, device=x.device)[None, :, None]
    cc = cols[:, None, None] + torch.arange(p, device=x.device)[None, None, :]
    out.permute(0, 2, 3, 1)[torch.arange(n, device=x.device)[:, None, None], rr, cc] = patch.permute(1, 2, 0)
    return out
