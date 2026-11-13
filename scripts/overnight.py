"""Ordered overnight continuation; worker compute never calls dashboard code."""
import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.common import write_json, atomic_write_text, seed_all, fingerprint
from scripts.progress import read, paths
from src.run_ablation import ARMS

def plan():
    return read(ROOT/'configs/overnight_run.json')

def queue():
    p=plan(); a,b=p['pairs']
    jobs=[dict(pair=x,arm='full',kind='attack') for x in (a,b)]
    jobs += [dict(pair=x,arm=arm,kind='attack') for x in (a,b) for arm in ('fixed','random','select')]
    jobs += [dict(pair=x,arm=arm,kind='noise') for x in (a,b) for arm in ('full','fixed')]
    for i,j in enumerate(jobs): j.update(number=i+1,state='pending')
    return jobs

def config(job):
    cfg=yaml.safe_load((ROOT/'configs/cifar10_full.yaml').read_text())
    cfg.update(ARMS[job['arm']]); cfg['output']='results/overnight/trials'
    x=job['pair']; cfg.update(source=x['source'],target=x['target'],seed=x['seed'])
    name=f'{x["source"]}-{x["target"]}_s{x["seed"]}_{job["arm"]}_{fingerprint(cfg)}'
    return cfg,ROOT/cfg['output']/name

def worker(index):
    import torch
    import importlib
    from src.common import configure
    configure()
    # Completely remove dashboard callbacks from the compute worker.
    for name in ('src.victim','src.craft','src.run_trial'):
        importlib.import_module(name).report_progress=lambda *a,**k:None
    job=queue()[index]; cfg,directory=config(job); x=job['pair']
    if job['kind']=='attack':
        from src.run_trial import run_trial
        run_trial(cfg,x['source'],x['target'],x['seed'],job['arm'],
                  reference_shared=ROOT/plan()['reference_shared'],defer_noise=True)
        return
    from src.data import load_data, project
    from src.victim import train,evaluate
    state=torch.load(directory/'poisons.pt',map_location=cfg['device'],weights_only=False)
    images,labels,tx,ty=load_data(cfg); idx=state['indices'].to(cfg['device'])
    seed_all(x['seed']+800)
    noise=torch.empty_like(state['delta']).uniform_(-cfg['epsilon'],cfg['epsilon'])
    project(noise,images[idx],cfg['epsilon'])
    noisy=images.clone(); noisy[idx]+=noise
    key=fingerprint({'seed':x['seed'],'indices':idx.tolist(),'epsilon':cfg['epsilon']})
    model,_,_=train(cfg,noisy,labels,tx,ty,cfg['epochs_victim'],x['seed']+2000,
                   ROOT/'results/overnight/noise_cache'/key,x['source'],x['target'],state['patch'])
    metrics=evaluate(model,tx,ty,x['source'],x['target'],state['patch'],cfg['batch_size'],x['seed']+700)
    write_json({'pair':x,'selection':cfg['selection'],'metrics':metrics},
               ROOT/f'results/overnight/controls/{x["name"]}_{cfg["selection"]}.json')

def dashboard(watch):
    from scripts.overnight_display import render_dashboard
    while True:
        state={}
        try:
            state=read(ROOT/'results/overnight/queue.json',{'jobs':queue(),'state':'waiting_for_pilot'})
            atomic_write_text(render_dashboard(state),ROOT/'results/PROGRESS.md')
            write_json(state,ROOT/'results/progress.json')
        except Exception as error:
            print(f'Display deferred: {error}',flush=True)
        if not watch or state.get('state') in ('complete','failed'): return
        time.sleep(10)


def report():
    from analysis.summarize import summarize
    base=yaml.safe_load((ROOT/'configs/cifar10_full.yaml').read_text()); pilot=paths(base)[0]
    rows=[read(pilot['trials'][a][0]/'result.json') for a in ('full','fixed','random','select')]
    rows += [read(config(j)[1]/'result.json') for j in queue() if j['kind']=='attack']
    rows=[r for r in rows if r]
    keys=('architecture','width','batch_size','epochs_surrogate','epochs_retrain','epochs_victim','lr','milestones',
          'weight_decay','amp','epsilon','poison_count','patch_size','source_count','craft_steps','craft_lr','craft_lr_units','selection','patch_location','retraining_factor')
    signatures={}
    for r in rows:
        signature=json.dumps({k:r['config'].get(k) for k in keys},sort_keys=True)
        if r['arm'] in signatures and signatures[r['arm']]!=signature: raise ValueError('Incomparable settings; refusing to pool')
        signatures[r['arm']]=signature
    out=ROOT/'reports/overnight'; summary=summarize(rows,out)
    write_json(rows,out/'measurements.json'); write_json(plan(),out/'schedule.json')
    text='# Overnight multi-pair ablation\n\nClean baseline patched ASR on original pair: **1.50% (15/1,000)**. '
    text+='New pairs reuse the same clean baseline/surrogate weights, with newly evaluated pair-specific clean controls.\n\n'
    text+=(out/'table.md').read_text()+'\n'
    text+='**Unequal n:** each row reports its own completed pair count; arms with different n cover different pair sets. Only matched pairs enter each difference below. One pair has no across-pair error bar.\n\n'
    for name,s in summary['paired_differences'].items(): text+=f'- {name}: {s["mean"]:+.2f} pp; matched n={s["n"]}; 95% CI={s["ci95"]}.\n'
    text+='\n## Control status\n\n'
    seen=set()
    for r in rows:
        key=(r['source'],r['target'],r['seed'])
        if key not in seen:
            seen.add(key); c=r['clean_control']
            text+=f'- Clean baseline, pair {key[0]}â†’{key[1]}, seed {key[2]}: patched ASR **{c["asr"]:.2%}**, clean accuracy {c["clean_accuracy"]:.2%}.\n'
    for x in plan()['pairs']:
        for method in ('gradient','random'):
            c=read(ROOT/f'results/overnight/controls/{x["name"]}_{method}.json')
            text+=f'- Pair {x["name"]}, {method} noise control: '+(f'ASR {c["metrics"]["asr"]:.2%}' if c else '**incomplete / not reached**')+'.\n'
    state=read(ROOT/'results/overnight/queue.json',{})
    text+='\n## Queue outcomes\n\n'
    for j in state.get('jobs',[]): text+=f'- {j["number"]}: pair {j["pair"]["name"]}, {j["arm"]}, {j["kind"]}: {j["state"]}.\n'
    text+='\n![Gradient alignment](alignment.png)\n\nCompleted artifacts from the earlier pilot are preserved. No reduced-fidelity runs were added. The 28-test pilot verification is preserved; overnight changes have separate validation.\n'
    atomic_write_text(text,out/'README.md')
    main=ROOT/'README.md'; original=main.read_text(); start=original.index('<!-- RESULTS_START -->')+len('<!-- RESULTS_START -->'); end=original.index('<!-- RESULTS_END -->')
    atomic_write_text(original[:start]+'\n\n'+text.replace('(alignment.png)','(reports/overnight/alignment.png)')+'\n\n'+original[end:],main)

def bounded_report(end):
    child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--report'],cwd=ROOT,
                           stdout=sys.stdout,stderr=sys.stderr,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    try:
        return child.wait(timeout=max(1,(end-datetime.now(timezone.utc)).total_seconds()))==0
    except subprocess.TimeoutExpired:
        from scripts.run_four_hour import stop_compute
        stop_compute(child)
        return False

def run():
    from scripts.progress import process_alive
    pid_path=ROOT/'results/overnight/runner.pid'
    if pid_path.exists():
        old=pid_path.read_text().strip()
        if old.isdigit() and int(old)!=os.getpid() and process_alive(int(old)):
            raise RuntimeError('Overnight runner already active')
    atomic_write_text(str(os.getpid()),pid_path)
    p=plan(); end=datetime.fromisoformat(p['deadline_utc']); cutoff=end-timedelta(minutes=30)
    jobs=read(ROOT/'results/overnight/queue.json',{}).get('jobs',queue())
    def save(state):
        value={'state':state,'mode':'overnight','jobs':jobs,'updated_utc':datetime.now(timezone.utc).isoformat()}
        write_json(value,ROOT/'results/overnight/queue.json')
        if state!='waiting_for_pilot': write_json(value,ROOT/'results/run_status.json')
    save('waiting_for_pilot')
    base=yaml.safe_load((ROOT/'configs/cifar10_full.yaml').read_text()); pilot=paths(base)[0]
    while not all((pilot['trials'][a][0]/'result.json').exists() for a in ('full','fixed','random','select')):
        if datetime.now(timezone.utc)>=cutoff:
            save('reporting'); save('complete' if bounded_report(end) else 'failed'); return
        time.sleep(10)
    # Let the current runner finish its own report before taking over.
    while read(ROOT/'results/run_status.json',{}).get('state')=='running':
        if datetime.now(timezone.utc)>=cutoff: break
        time.sleep(5)
    save('running')
    atomic_write_text(str(os.getpid()),ROOT/'results/runner.pid')
    for i,j in enumerate(jobs):
        if j['state']=='complete': continue
        remaining=(cutoff-datetime.now(timezone.utc)).total_seconds()
        if remaining<=0: break
        j['state']='running'; save('running')
        child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(i)],cwd=ROOT,
                               stdout=sys.stdout,stderr=sys.stderr,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:
            code=child.wait(timeout=remaining); j['state']='complete' if code==0 else 'failed'
        except subprocess.TimeoutExpired:
            from scripts.run_four_hour import stop_compute
            stop_compute(child); j['state']='incomplete_at_cutoff'; save('reporting'); break
        save('running')
    for j in jobs:
        if j['state'] in ('pending','running'): j['state']='incomplete_not_reached'
    save('reporting'); save('complete' if bounded_report(end) else 'failed')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--worker',type=int); parser.add_argument('--dashboard',action='store_true'); parser.add_argument('--report',action='store_true')
    args=parser.parse_args()
    if args.worker is not None: worker(args.worker)
    elif args.dashboard: dashboard(True)
    elif args.report: report()
    else: run()
