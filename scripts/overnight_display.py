"""Read-only rendering of all three pairs; never loads model checkpoints."""
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import re
import yaml
import statistics
from scripts import overnight as night

NAMES=('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')
ARMS=('fixed','random','select','full')
LABELS={'fixed':'Fixed sticker','random':'Random sticker','select':'Random sticker + image selection',
        'full':'Full attack (selection + retraining)'}

def epoch(path):
    h=night.read(path/'history.json',[])
    return h[-1]['epoch'] if h else 0

def duration(seconds):
    minutes=max(1,round(seconds/60))
    return f'{minutes//60}h {minutes%60:02d}m' if minutes>=60 else f'{minutes} min'

def measured_rates(records):
    rates=[]; craft=[]; selection=[]
    for g,experiments in records:
        for e in experiments:
            r=e['result']; d=g['directories'][e['arm']]
            if r and r.get('victim_seconds',0)>0: rates.append(r['victim_seconds']/80)
            c=night.read(d/'craft_progress.json',{})
            if c.get('done',0)>0 and c.get('seconds',0)>0: craft.append(c['seconds']/c['done'])
            if r and r.get('selection',{}).get('method')=='gradient': selection.append(r['selection']['seconds'])
    # Observe small history files, never load GPU/model checkpoint files.
    cache_path=night.ROOT/'results/overnight/display_eta.json'
    old=night.read(cache_path,{})
    observations=[]; current={}
    for g,experiments in records:
        for e in experiments:
            d=g['directories'][e['arm']]
            for model in [d/'victim',*[d/f'retrain_{k}' for k in (50,100,150,200)]]:
                file=model/'history.json'
                if not file.exists(): continue
                n=epoch(model); stamp=file.stat().st_mtime; key=str(model)
                before=old.get('models',{}).get(key)
                if before and n>before['epoch']:
                    rate=(stamp-before['stamp'])/(n-before['epoch'])
                    if 5<rate<120: observations.append(rate)
                current[key]={'epoch':n,'stamp':stamp}
    recent=(old.get('rates',[])+observations)[-24:]
    try: night.write_json({'models':current,'rates':recent},cache_path)
    except OSError: pass
    return (statistics.median(recent or rates) if recent or rates else 28.4,
            statistics.median(craft) if craft else 1.6,
            statistics.median(selection) if selection else 90)

def estimate_schedule(state,records,now,epoch_seconds,craft_seconds,selection_seconds,latest):
    by_pair={g['name']:{e['arm']:e for e in es} for g,es in records}
    cursor=now; estimates={}
    for j in state['jobs']:
        e=by_pair[j['pair']['name']][j['arm']]
        phases=[]
        def add(label,seconds): phases.append({'label':label,'seconds':max(0,seconds)})
        if j['kind']=='attack' and not e['result']:
            add('Load shared models / prepare data',15 if j['state']=='pending' else 0)
            add('Choose training images',0 if e['selected'] else selection_seconds if j['arm'] in ('full','select') else 2)
            boundaries=[0,50,100,150,200,250] if j['arm']=='full' else [0,250]
            for index,(start,end) in enumerate(zip(boundaries,boundaries[1:]),1):
                add(f'Crafting block {index}',max(0,end-max(start,e['craft']))*craft_seconds)
                if j['arm']=='full' and index<=4:
                    add(f'Practice-model refresh {index}/4',(40-e['refresh'][index-1])*epoch_seconds)
            add('Victim training',(80-e['victim'])*epoch_seconds)
        elif j['kind']=='noise':
            method='gradient' if j['arm']=='full' else 'random'
            complete=night.read(night.ROOT/f'results/overnight/controls/{j["pair"]["name"]}_{method}.json')
            n=0
            if j['state']=='running':
                match=re.search(r'epoch (\d+)/80',latest)
                if match: n=int(match[1])
            add('Noise-control training',0 if complete else (80-n)*epoch_seconds)
        finished=j['state']=='complete' or (j['kind']=='attack' and bool(e['result']))
        terminal=j['state'] in ('failed','incomplete_at_cutoff','incomplete_not_reached')
        remaining=0 if finished or terminal else sum(p['seconds'] for p in phases)
        for phase in phases:
            if finished or terminal: phase['seconds']=0
            cursor+=timedelta(seconds=phase['seconds'])
            phase['finish']=cursor
        estimates[(j['pair']['name'],j['arm'],j['kind'])]={'seconds':remaining,'finish':cursor,
                    'phases':phases,'finished':finished,'terminal':terminal,'job':j}
    return estimates

def render_dashboard(state):
    root=night.ROOT; p=night.plan(); tz=ZoneInfo('Pacific/Honolulu'); now=datetime.now(tz)
    end=datetime.fromisoformat(p['deadline_utc']).astimezone(tz)
    left=max(0,int((end-now).total_seconds()))
    base=yaml.safe_load((root/'configs/cifar10_full.yaml').read_text())
    pilot=night.paths(base)[0]
    groups=[{'name':'Original','source':0,'target':1,'seed':17,
             'directories':{a:pilot['trials'][a][0] for a in ARMS}}]
    for pair in p['pairs']:
        groups.append(dict(pair,directories={a:night.config({'pair':pair,'arm':a})[1] for a in ARMS}))
    active=next((j for j in state['jobs'] if j['state']=='running'),None)
    log_path=root/('results/overnight/run.log' if state['state']!='waiting_for_pilot' else 'results/run.log')
    log=log_path.read_text(errors='replace') if log_path.exists() else ''
    log=log.rsplit('COMPLETED ',1)[-1] if 'COMPLETED ' in log else log
    latest=next((line for line in reversed(log.splitlines()) if re.search(r'epoch \d+/|craft \d+/|selection \d+/',line)), '')
    records=[]; counts={a:0 for a in ARMS}
    for g in groups:
        experiments=[]
        for arm,d in g['directories'].items():
            result=night.read(d/'result.json')
            victim=epoch(d/'victim'); refresh=[epoch(d/f'retrain_{step}') for step in (50,100,150,200)] if arm=='full' else []
            craft=night.read(d/'craft_progress.json',{}).get('done',0)
            craft=max([craft]+[step for step,e in zip((50,100,150,200),refresh) if e])
            if (d/'poisons.pt').exists() or result: craft=250
            running=bool(active and active['kind']=='attack' and active['pair']['name']==g['name'] and active['arm']==arm)
            if state['state']=='waiting_for_pilot' and g['name']=='Original' and not result and victim:
                running=True
            if running and not result:
                matches=re.findall(r'craft (\d+)/250',log)
                if matches: craft=max(craft,int(matches[-1]))
            status='Done' if result else ('RUNNING' if running else 'Queued')
            if not result and state['state'] in ('complete','failed'): status='Incomplete'
            if result: victim=80; counts[arm]+=1
            experiments.append({'arm':arm,'result':result,'victim':victim,'refresh':refresh,
                                'craft':craft,'status':status,'selected':(d/'selection.pt').exists()})
        records.append((g,experiments))
    done=sum(counts.values())
    epoch_seconds,craft_seconds,selection_seconds=measured_rates(records)
    estimates=estimate_schedule(state,records,now,epoch_seconds,craft_seconds,selection_seconds,latest)
    cutoff=end.replace(hour=6,minute=30,second=0,microsecond=0)
    def finish_text(estimate):
        if estimate['finished']: return 'Done'
        if estimate['terminal']: return 'Incomplete'
        value=estimate['finish'].strftime('%a %I:%M %p')
        return value+(' — past cutoff' if estimate['finish']>cutoff else '')
    lines=['# Experiment progress — all three pairs','',f'Updated: {now:%b %d, %I:%M:%S %p} Hawaii time. Refreshes every 10 seconds.','',
           f'**Attack experiments finished: {done}/12 ({done/12:.0%}).** Noise controls are counted separately below.',
           f'**Deadline: Friday 7:00 AM HST.** Compute stops at **6:30 AM**. Time remaining: **{left//3600}h {(left%3600)//60:02d}m**.','']
    if active:
        x=active['pair']; lines += [f'**Working now: Pair {x["name"]} — {NAMES[x["source"]]} → {NAMES[x["target"]]}, {LABELS[active["arm"]]}**'+(' — noise control.' if active['kind']=='noise' else '.'),'']
    elif state['state']=='waiting_for_pilot': lines+=['**Working now: finishing the original pair before starting A or B.**','']
    else: lines += [f'**Runner: {state["state"]}.**','']
    if latest: lines+=['Latest training/crafting update: `'+latest+'`','']
    lines += [f'**Timing basis:** {epoch_seconds:.1f} seconds/training round; '
              f'{craft_seconds:.2f} seconds/crafting update. Estimates refresh with measured progress.',
              'Finish times include earlier queued work in your exact order. Allow roughly **15–25% variation**; '
              '**“past cutoff” means the step may not finish or start before 6:30 AM**. Settings and queue order are unchanged.','']
    lines+=['Clean unpoisoned baseline on the original pair: **1.50% patched ASR (15/1,000)**.',
            'The clean baseline and initial practice model are reused for A and B. No new training is needed for those two initial models.','',
            '## At a glance','', '| Pair | Images being confused | Attack experiments finished |','|---|---|---:|']
    for g,exps in records:
        lines.append(f'| {g["name"]} | {NAMES[g["source"]]} → {NAMES[g["target"]]} ({g["source"]}→{g["target"]}) | {sum(bool(e["result"]) for e in exps)}/4 |')
    for g,exps in records:
        lines+=['',f'## {g["name"]}: {NAMES[g["source"]]} → {NAMES[g["target"]]}',f'Seed: `{g["seed"]}`.','',
                '| Setup | Status | Images chosen | Image-change updates | Practice-model refreshes | Victim training | Final ASR |',
                '|---|---|---|---:|---|---:|---:|']
        for e in exps:
            r=e['result']; ref='Not used' if not e['refresh'] else f'{sum(n==40 for n in e["refresh"])}/4 finished'
            asr=f'{r["metrics"]["asr"]:.2%}' if r else 'Pending'
            lines.append(f'| {LABELS[e["arm"]]} | {e["status"]} | {"Done" if e["selected"] else "Waiting"} | {e["craft"]}/250 | {ref} | {e["victim"]}/80 | {asr} |')
        full=next(e for e in exps if e['arm']=='full')
        lines+=['','Full attack — the four practice-model refreshes:', '', '| Refresh | Training rounds |','|---|---:|']
        for i,n in enumerate(full['refresh'],1): lines.append(f'| {i}/4 | {n}/40 |')
        if g['name']!='Original':
            lines += ['', 'Estimated remaining time for each setup:', '',
                      '| Setup | Compute still needed | Expected finish (HST) |','|---|---:|---|']
            for e in exps:
                estimate=estimates[(g['name'],e['arm'],'attack')]
                lines.append(f'| {LABELS[e["arm"]]} | '+('Done' if estimate['finished'] else duration(estimate['seconds']))+f' | {finish_text(estimate)} |')
        lines+=['','Noise controls (ordinary random pixel noise, **scheduled last** for new pairs):','',
                '| Image selection | Status | Final ASR |','|---|---|---:|']
        for method,arm in [('gradient','full'),('random','fixed')]:
            if g['name']=='Original':
                r=next(e['result'] for e in exps if e['arm']==arm)
                c={'metrics':r['random_control']} if r and r.get('random_control') else None
            else: c=night.read(root/f'results/overnight/controls/{g["name"]}_{method}.json')
            control_running=bool(active and active['kind']=='noise' and active['pair']['name']==g['name'] and active['arm']==arm)
            status='Done' if c else ('RUNNING — '+latest if control_running else 'Waiting until attacks finish')
            if not c and state['state'] in ('complete','failed'): status='Incomplete / not reached'
            lines.append(f'| {method} | {status} | '+(f'{c["metrics"]["asr"]:.2%}' if c else 'Pending')+' |')
    lines+=['','## Exact queue — unchanged priority','', '| # | Pair | Setup | Work | Status | Time still needed | Expected finish (HST) |','|---:|---|---|---|---|---:|---|']
    for j in state['jobs']:
        e=estimates[(j['pair']['name'],j['arm'],j['kind'])]
        lines.append(f'| {j["number"]} | {j["pair"]["name"]} | {LABELS[j["arm"]]} | {j["kind"]} | {j["state"]} | '+('Done' if e['finished'] else duration(e['seconds']))+f' | {finish_text(e)} |')
    lines += ['', '## Step-by-step time estimates', '', '| Queue # / pair | Step | Time remaining in step | Expected step finish (HST) |','|---|---|---:|---|']
    for e in estimates.values():
        j=e['job']
        for phase in e['phases']:
            if e['terminal']: time_text=finish='Incomplete'
            elif e['finished'] or not phase['seconds']: time_text=finish='Done'
            else:
                time_text=duration(phase['seconds']); finish=phase['finish'].strftime('%a %I:%M %p')
                if phase['finish']>cutoff: finish+=' — past cutoff'
            lines.append(f'| {j["number"]} / {j["pair"]["name"]} | {phase["label"]} | {time_text} | {finish} |')
    lines += ['| Final reporting | Analysis, figures, README | Up to 30 min | Fri 07:00 AM hard stop |']
    lines+=['','## Reporting coverage','', '| Setup | Completed pairs (n) |','|---|---:|']
    for arm in ARMS: lines.append(f'| {LABELS[arm]} | {counts[arm]}/3 |')
    lines+=['','Only completed attacks contribute results. Different n means different pair coverage; comparisons use matched pairs only.',
            'Crafting updates, refreshes, and victim rounds are separate work units, not equal chunks of time. A high crafting cosine is not the final attack success rate.',
            'At 6:30 AM, compute stops and analysis/reporting begins. Final report: `reports/overnight/README.md`.','']
    return '\n'.join(lines)
