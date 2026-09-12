"""Offline reports for immutable experiment evidence, with streamed chain checks.

This verifies recorded artifacts without importing a teacher or loading a neural
model. It never reruns evaluation or upgrades a recorded gate claim.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import statistics


def _json(text):
    def reject(value):raise ValueError('Nonfinite JSON value: '+value)
    return json.loads(text,parse_constant=reject)


def _digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def audit_journal(path):
    previous='0'*64;count=0;whole=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for line in stream:
            whole.update(line)
            if not line.endswith(b'\n'):raise ValueError('Truncated journal final line')
            row=_json(line);body={key:value for key,value in row.items() if key!='hash'}
            index=body.get('index',body.get('sequence'))
            prior=body.get('previous',body.get('previous_hash'))
            if index!=count or prior!=previous:raise ValueError('Journal sequence or chain mismatch')
            expected=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            if row.get('hash')!=expected:raise ValueError('Journal content checksum mismatch')
            previous=row['hash'];count+=1
    if _digest(path)!=whole.hexdigest():raise ValueError('Journal changed while reporting')
    return {'rows':count,'head':previous,'sha256':whole.hexdigest(),'chain_verified':True}


def audit_legacy_trials(path):
    """Historical controllability trials predate the chained journal schema."""
    checksum=_digest(path);count=0
    with Path(path).open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):raise ValueError('Truncated legacy trial line')
            row=_json(line)
            if not isinstance(row,dict) or not {'phase','target','repeat','seed','selected'}<=set(row):
                raise ValueError('Invalid legacy controllability row')
            count+=1
    if checksum!=_digest(path):raise ValueError('Legacy trials changed during report generation')
    return {'rows':count,'head':None,'sha256':checksum,'chain_verified':False,
            'scope':'Historical unchained trials: file checksum and JSON structure only'}


def evidence(run):
    root=Path(run).resolve()
    result_path=root/'result.json';manifest_path=root/'manifest.json'
    if not result_path.is_file() or not manifest_path.is_file():raise ValueError('Report requires result.json and manifest.json in a run directory')
    result=_json(result_path.read_text());manifest=_json(manifest_path.read_text())
    supported={'conditioning-result-v1','exact-transfer-result-v1','nfsp-training-v1','teacher-evaluation-v1','frozen-poker-evaluation-result-v1','controllability-result-v1'}
    if result.get('schema') not in supported:raise ValueError('Unsupported experiment report schema')
    manifest_hash=_digest(manifest_path)
    if result.get('manifest_sha256',manifest_hash)!=manifest_hash:raise ValueError('Result/manifest checksum mismatch')
    files={'result.json':_digest(result_path),'manifest.json':manifest_hash};journals={}
    for name in ('trials.jsonl','hands.jsonl','evaluation.jsonl','paired-deals.jsonl','events.jsonl'):
        path=root/name
        if path.is_file():
            journals[name]=audit_legacy_trials(path) if result['schema']=='controllability-result-v1' else audit_journal(path)
            files[name]=journals[name]['sha256']
    if 'journal_head' in result:
        if len(journals)!=1 or next(iter(journals.values()))['head']!=result['journal_head']:
            raise ValueError('Result does not describe the current complete journal')
    for name in ('preregistration.json','cues.json','teacher.json'):
        if (root/name).is_file():files[name]=_digest(root/name)
    if 'preregistration_sha256' in result and files.get('preregistration.json')!=result['preregistration_sha256']:
        raise ValueError('Preregistration checksum mismatch')
    expected_rows=result.get('operations',result.get('hands_completed'))
    if expected_rows is not None and sum(j['rows'] for j in journals.values())!=expected_rows:
        raise ValueError('Result journal count mismatch')
    rows=[]
    for item in result.get('seed_results',[]):
        seed=item['seed']
        for key,value in item.items():
            if '/' in key and isinstance(value,(int,float)) and not isinstance(value,bool):
                rows.append({'series':key,'seed':seed,'value':value,'unit':'accuracy'})
    summaries=[]
    for series in sorted({r['series'] for r in rows}):
        values=[r['value'] for r in rows if r['series']==series]
        summaries.append({'series':series,'n':len(values),'mean':statistics.mean(values),'minimum':min(values),'maximum':max(values),'unit':'accuracy'})
    if result['schema']=='controllability-result-v1':
        for phase in ('confirmatory','shuffled_input'):
            if phase in result:
                for action,value in enumerate(result[phase]['success']):
                    summaries.append({'series':phase+'/action-'+str(action),'n':sum(result[phase]['confusion'][action]),'mean':value,'unit':'success fraction'})
    for name,item in result.get('opponents',{}).items():
        summaries.append({'series':name,'n':item['paired_deals'],'mean':item['bb_per_hand'],
            'interval':item['suite_adjusted_bootstrap_ci'],'unit':'BB/hand','criterion':item.get('positive_lower_bound')})
    scope=result.get('scope') or ('Conventional teacher only; no fly learning claim' if result.get('mode')=='conventional-teacher-control' else 'Recorded experiment evidence; no new evaluation')
    status=result.get('status') or ('pass' if result.get('passes_fixed_suite') else 'fail' if 'passes_fixed_suite' in result else 'recorded')
    if result.get('profile')=='development' and (result.get('passes_fixed_suite') or result.get('development_criteria_met')):
        status='development-qualified; confirmation still required'
    for name,digest in files.items():
        if _digest(root/name)!=digest:raise ValueError('Run changed during report generation')
    return {'schema':'flyholdem-report-v1','run':str(root),'run_schema':result.get('schema'),'scope':scope,'status':status,
        'profile':result.get('profile',manifest.get('config',{}).get('profile','development')),
        'mode':result.get('mode',manifest.get('config',{}).get('mode','unspecified')),
        'learning_mode':result.get('learning_mode',manifest.get('config',{}).get('learning_mode','not applicable')),
        'recorded_learning_claim':bool(result.get('learning_claim',False)),
        'allowed_as_teacher':bool(result.get('allowed_as_teacher',False)),
        'execution_commit':manifest.get('commit'),'source_hash':manifest.get('source_hash'),
        'graph_hash':manifest.get('graph_hash'),'binary_hash':manifest.get('binary_hash'),
        'files':files,'journal_verification':journals,'summaries':summaries,'seed_values':rows,
        'paired_evidence':result.get('paired_evidence',{}),'elapsed_seconds':result.get('elapsed_seconds_this_invocation',result.get('elapsed_seconds')),
        'peak_rss_bytes':result.get('peak_rss_bytes'),
        'retention_criterion_met':result.get('retention_criterion_met'),
        'information_boundary_verified':result.get('information_boundary_verified'),
        'hands_completed':result.get('hands_completed'),'planned_hands':result.get('planned_hands'),
        'caution':'Artifact verification checks bytes and chain consistency; it does not replace numerical, leakage or statistical validation. No model is loaded and no new evaluation is run.'}


def _cell(value):return str(value).replace('|','\\|').replace('\n',' ')
def _number(value):return '—' if value is None else f'{value:.6g}'


def markdown(report):
    lines=['# FlyHoldem experiment report','',f"**{report['status']}** · {report['profile']} · {report['mode']} · {report['learning_mode']}",
        '',report['scope'],'',report['caution'],'']
    if report['summaries']:
        lines+=['| Series | N | Mean | Unit | Interval / range |','|---|---:|---:|---|---|']
        for row in report['summaries']:
            interval=row.get('interval',[row.get('minimum'),row.get('maximum')])
            lines.append(f"| {_cell(row['series'])} | {row['n']} | {_number(row['mean'])} | {row['unit']} | {' to '.join(_number(v) for v in interval)} |")
        lines+=['','Accuracy ranges are min/max across seeds; opponent intervals retain the registered suite-adjusted bootstrap confidence bounds.']
    if report['hands_completed'] is not None:lines+=['',f"Training hands: {report['hands_completed']} / {report['planned_hands']}."]
    if report['paired_evidence']:
        lines+=['','| Paired comparison | Mean difference | 95% seed bootstrap | One-sided sign-flip p |','|---|---:|---|---:|']
        for name,v in report['paired_evidence'].items():
            lines.append(f"| {_cell(name)} | {_number(v['mean'])} | {' to '.join(_number(x) for x in v['paired_seed_bootstrap_95'])} | {_number(v['one_sided_sign_flip_p'])} |")
    lines+=['',f"Runtime: {_number(report['elapsed_seconds'])} seconds; peak RSS: {_number(None if report['peak_rss_bytes'] is None else report['peak_rss_bytes']/1024**2)} MiB.",
        '',f"Recorded task learning claim: {report['recorded_learning_claim']}. Allowed as poker teacher: {report['allowed_as_teacher']}.",
        '',f"Execution commit: `{report['execution_commit']}`. Source hash: `{report['source_hash']}`.",
        '',f"Run: `{report['run']}`.",'','| Artifact | SHA-256 |','|---|---|']
    lines += [f'| {_cell(k)} | `{v}` |' for k,v in report['files'].items()]
    lines+=['','Journal checks: '+(', '.join(f"{k}: {v['rows']} rows, " + ("full chain verified" if v['chain_verified'] else "legacy file checksum only; no chain") for k,v in report['journal_verification'].items()) or 'No supported journal in this run.'),'']
    return '\n'.join(lines)


def html_report(report):
    esc=lambda v:html.escape(str(v))
    table=''
    if report['summaries']:
        table='<h2>Recorded results</h2><table><thead><tr><th>Series</th><th>N</th><th>Mean</th><th>Unit</th><th>Interval / range</th></tr></thead><tbody>'
        for row in report['summaries']:
            bounds=row.get('interval',[row.get('minimum'),row.get('maximum')])
            table+='<tr>'+''.join('<td>'+esc(value)+'</td>' for value in (row['series'],row['n'],_number(row['mean']),row['unit'],' to '.join(_number(v) for v in bounds)))+'</tr>'
        table+='</tbody></table><p class="muted">Accuracy ranges are min/max across seeds. Opponent intervals retain the registered suite-adjusted bootstrap confidence bounds.</p>'
    hashes=''.join('<tr><td>'+esc(k)+'</td><td><code>'+esc(v)+'</code></td></tr>' for k,v in report['files'].items())
    details=''.join('<div><dt>'+esc(k)+'</dt><dd>'+esc(v)+'</dd></div>' for k,v in {
        'Graph mode':report['mode'],'Learning mode':report['learning_mode'],'Profile':report['profile'],
        'Recorded task learning claim':report['recorded_learning_claim'],'Allowed as poker teacher':report['allowed_as_teacher'],
        'Runtime (s)':_number(report['elapsed_seconds']),'Peak RSS (MiB)':_number(None if report['peak_rss_bytes'] is None else report['peak_rss_bytes']/1024**2)}.items())
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FlyHoldem evidence report</title><style>body{margin:48px auto;max-width:1040px;padding:0 24px;background:#101a15;color:#deeadf;font:15px/1.6 system-ui}h1{font:46px Georgia,serif;margin:10px 0}h2{font:26px Georgia,serif;margin-top:36px}.eyebrow{letter-spacing:2px;font-size:11px;color:#b4dfc7}.status{display:inline-block;padding:6px 12px;border:1px solid #81967e;border-radius:4px}dl{display:grid;grid-template-columns:repeat(4,1fr);gap:20px;border-top:1px solid #34423a;padding-top:24px;margin-top:32px}dt,.muted{font-size:12px;color:#9db09f}dd{margin:6px 0;font-size:16px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:12px 8px;border-bottom:1px solid #34423a;vertical-align:top}th{color:#b4dfc7}code{overflow-wrap:anywhere;font-size:11px}.note{padding:16px;background:#1d2b22;border-left:3px solid #b4dfc7}.path{overflow-wrap:anywhere}@media(max-width:650px){body{margin:24px auto;padding:0 16px}h1{font-size:34px}dl{grid-template-columns:repeat(2,1fr)}td,th{padding:8px 4px;font-size:11px}}@media print{body{background:white;color:black;margin:0}.muted,dt{color:#444}.note{background:#eee}}</style></head><body><div class="eyebrow">FLYHOLDEM · RECORDED EVIDENCE</div><h1>Experiment report</h1><span class="status">'+esc(report['status'])+'</span><p>'+esc(report['scope'])+'</p><p class="note">'+esc(report['caution'])+'</p><dl>'+details+'</dl>'+table+'<h2>Provenance</h2><p>Execution commit <code>'+esc(report['execution_commit'])+'</code><br>Source hash <code>'+esc(report['source_hash'])+'</code></p><p class="path">'+esc(report['run'])+'</p><table><thead><tr><th>Artifact</th><th>SHA-256</th></tr></thead><tbody>'+hashes+'</tbody></table><p class="muted">Inspected '+str(sum(v['rows'] for v in report['journal_verification'].values()))+' recorded rows. Modern journals have full chain checks; historical controllability has file checksums only. Full paired statistics and exact metadata are included in REPORT.md and report.json.</p></body></html>'


def write_report(run,output=None):
    report=evidence(run);out=Path(output) if output else Path(run)/'report';out.mkdir(parents=True,exist_ok=True)
    text=markdown(report)
    from flyholdem.neural.checkpoint import atomic_json
    atomic_json(out/'report.json',report)
    (out/'REPORT.md').write_text(text)
    (out/'report.html').write_text(html_report(report))
    return {'report':str((out/'REPORT.md').resolve()),'json':str((out/'report.json').resolve()),'html':str((out/'report.html').resolve()),'status':report['status']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--output')
    args=parser.parse_args();print(json.dumps(write_report(args.run,args.output),indent=2))


if __name__=='__main__':main()
