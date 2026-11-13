import json
import yaml
from scripts import run_study as study
from src.run_ablation import ARMS


def test_plan_pairs_are_distinct_and_valid():
    plan = study.load_plan()
    pairs = [(p['source'], p['target']) for p in plan['pairs']]
    assert len(set(pairs)) == len(pairs) == 3
    assert all(s != t and 0 <= s < 10 and 0 <= t < 10 for s, t in pairs)
    assert pairs[0] == (0, 1) and plan['pairs'][0]['seed'] == 17


def test_arm_settings_match_base_recipe():
    plan = study.load_plan()
    base = yaml.safe_load((study.ROOT/plan['base_config']).read_text())
    for arm in plan['arms']:
        cfg = study.trial_config(plan, arm)
        assert cfg == dict(base, **ARMS[arm], output=plan['output'])
        assert cfg['craft_steps'] == 250 and cfg['epochs_victim'] == 80 and cfg['epochs_retrain'] == 40


def test_frozen_controls_match_measurements():
    report = study.REPORT
    rows = json.loads((report/'measurements.json').read_text(encoding='utf-8'))
    controls = json.loads((report/'controls.json').read_text(encoding='utf-8'))
    pairs = [(p['source'], p['target'], p['seed']) for p in study.load_plan()['pairs']]
    assert [(c['source'], c['target'], c['seed']) for c in controls] == pairs
    for c in controls:
        full = next(r for r in rows if (r['source'], r['target'], r['arm']) == (c['source'], c['target'], 'full'))
        assert c['clean_model'] == full['clean_control']
        assert all(0 <= c[k]['asr'] < .05 for k in ('clean_model', 'random_noise', 'selected_noise'))


def test_controls_are_built_from_trial_rows():
    metric = lambda asr: {'asr': asr}
    rows = [{'source': 0, 'target': 1, 'seed': 5, 'arm': 'fixed',
             'clean_control': metric(.01), 'random_control': metric(.02)},
            {'source': 0, 'target': 1, 'seed': 5, 'arm': 'full',
             'clean_control': metric(.01), 'random_control': metric(.03)}]
    assert study.controls(rows) == [{'source': 0, 'target': 1, 'seed': 5, 'clean_model': metric(.01),
                                     'random_noise': metric(.02), 'selected_noise': metric(.03)}]
