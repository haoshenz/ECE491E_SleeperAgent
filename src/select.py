import time
import torch
from torch.nn import functional as F


def select_poisons(model, x, y, target, count, method, seed, progress=None):
    candidates = torch.where(y == target)[0]
    assert 0 < count <= len(candidates)
    start = time.perf_counter()
    if progress:
        progress(0, len(candidates), 0)
    if method == 'random':
        order = torch.randperm(len(candidates), generator=torch.Generator().manual_seed(seed))
        chosen = candidates[order[:count].to(candidates.device)]
    elif method == 'gradient':
        # Full parameter gradient per example. No unvalidated last-layer proxy.
        model.eval()
        params = tuple(model.parameters())
        scores = []
        for j, idx in enumerate(candidates):
            loss = F.cross_entropy(model(x[idx:idx+1]), y[idx:idx+1])
            grads = torch.autograd.grad(loss, params)
            scores.append(torch.stack([g.detach().square().sum() for g in grads]).sum())
            if (j+1) % 500 == 0:
                print(f'full-gradient selection {j+1}/{len(candidates)}', flush=True)
                if progress:
                    progress(j+1, len(candidates), time.perf_counter()-start)
        chosen = candidates[torch.stack(scores).topk(count).indices]
    else:
        raise ValueError(method)
    if progress:
        progress(len(candidates), len(candidates), time.perf_counter()-start)
    return chosen, {'method': method, 'seconds': time.perf_counter()-start,
                    'candidate_count': len(candidates)}
