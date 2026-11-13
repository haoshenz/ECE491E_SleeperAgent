from scripts import overnight as night
from src.run_ablation import ARMS
import yaml

def test_exact_priority_and_noise_last():
    jobs=night.queue()
    assert [(j['pair']['name'],j['arm']) for j in jobs[:8]]==[
        ('A','full'),('B','full'),('A','fixed'),('A','random'),('A','select'),
        ('B','fixed'),('B','random'),('B','select')]
    assert all(j['kind']=='noise' for j in jobs[8:])

def test_settings_are_identical_to_pilot_per_arm():
    original=yaml.safe_load((night.ROOT/'configs/cifar10_full.yaml').read_text())
    for job in night.queue()[:8]:
        cfg,_=night.config(job)
        expected=dict(original,**ARMS[job['arm']])
        for k in expected:
            if k not in ('output','source','target','seed'): assert cfg[k]==expected[k]
        assert cfg['craft_steps']==250 and cfg['epochs_victim']==80
        assert cfg['epochs_retrain']==40

def test_recorded_draw_and_deadline():
    p=night.plan()
    pairs=[(x['source'],x['target']) for x in p['pairs']]
    assert len(set(pairs))==2 and (0,1) not in pairs
    assert all(s!=t for s,t in pairs)
    assert p['deadline_utc']=='2026-11-13T17:00:00+00:00'
