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
    assert p['deadline_utc']=='2026-11-13T18:30:00+00:00'

def test_dashboard_contains_all_pairs_and_all_twelve_setups():
    from scripts.overnight_display import render_dashboard,NAMES
    text=render_dashboard({'state':'waiting_for_pilot','jobs':night.queue()})
    assert '## Original: airplane → automobile' in text
    for pair in night.plan()['pairs']:
        assert f'## {pair["name"]}: {NAMES[pair["source"]]} → {NAMES[pair["target"]]}' in text
    assert text.count('| Full attack (selection + retraining) |') >= 3
    assert 'Victim training' in text and 'Practice-model refreshes' in text
    assert 'Reporting coverage' in text and 'Final ASR' in text
    assert 'Step-by-step time estimates' in text and 'Expected finish (HST)' in text

def test_step_estimates_include_prior_queue_work_and_partial_refresh():
    from scripts.overnight_display import estimate_schedule
    from datetime import datetime,timedelta
    from zoneinfo import ZoneInfo
    records=[]
    for pair in night.plan()['pairs']:
        exps=[]
        for arm in ('fixed','random','select','full'):
            exps.append({'arm':arm,'result':None,'selected':False,'craft':0,'victim':0,
                         'refresh':[0,0,0,0] if arm=='full' else []})
        records.append((pair,exps))
    a=records[0][1][-1]
    a.update(selected=True,craft=100,refresh=[40,10,0,0])
    jobs=night.queue(); jobs[0]['state']='running'
    now=datetime(2026,11,12,20,0,tzinfo=ZoneInfo('Pacific/Honolulu'))
    estimates=estimate_schedule({'jobs':jobs},records,now,30,2,60,'')
    first=estimates[('A','full','attack')]
    assert first['seconds']==6000
    assert first['finish']==now+timedelta(seconds=6000)
    assert next(p['seconds'] for p in first['phases'] if p['label']=='Practice-model refresh 2/4')==900
    assert estimates[('B','full','attack')]['finish']>first['finish']
