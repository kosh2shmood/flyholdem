import copy
import hashlib
import json
import sys
from pathlib import Path
import pytest
from flyholdem.experiments.journal import Journal
from flyholdem.experiments.reports import evidence,write_report
from flyholdem.neural.checkpoint import atomic_json
from flyholdem.cli import main,parser,run_path


def run_fixture(tmp_path):
    root=tmp_path/'run';root.mkdir()
    atomic_json(root/'manifest.json',{'schema':'flyholdem-run-manifest-v1','commit':'test-commit','source_hash':'test-source','config':{}})
    journal=Journal(root/'trials.jsonl');journal.record(0,['seed','trial'],{'correct':True});journal.close()
    atomic_json(root/'result.json',{'schema':'exact-transfer-result-v1','status':'fail','profile':'confirmatory',
        'scope':'<script>Never execute evidence text</script>','mode':'circuit','learning_mode':'distilled-connectome',
        'journal_head':journal.rows[-1]['hash'],'operations':1,'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        'seed_results':[{'seed':1,'plastic/post':.7,'frozen/post':.5}], 'learning_claim':False})
    return root


def test_report_preserves_failure_scope_and_exact_artifacts_without_loading_teacher(tmp_path):
    root=run_fixture(tmp_path);before={p.name:p.read_bytes() for p in root.iterdir() if p.is_file()}
    loaded=set(sys.modules);out=write_report(root)
    report=json.loads(Path(out['json']).read_text())
    assert report['status']=='fail' and not report['recorded_learning_claim'] and not report['allowed_as_teacher']
    assert report['journal_verification']['trials.jsonl']['rows']==1
    assert report['journal_verification']['trials.jsonl']['chain_verified']
    assert report['summaries'][1]['mean']==.7
    assert '&lt;script&gt;' in Path(out['html']).read_text() and '<script>' not in Path(out['html']).read_text()
    assert {p.name:p.read_bytes() for p in root.iterdir() if p.is_file()}==before
    assert not any(name.startswith('flyholdem.teacher') for name in set(sys.modules)-loaded)


@pytest.mark.parametrize('mutation',['chain','truncation','result-head','manifest','nonfinite'])
def test_report_rejects_corrupt_or_inconsistent_evidence(tmp_path,mutation):
    root=run_fixture(tmp_path)
    if mutation=='chain':
        p=root/'trials.jsonl';p.write_text(p.read_text().replace('true','false'))
    elif mutation=='truncation':
        p=root/'trials.jsonl';p.write_bytes(p.read_bytes()[:-1])
    elif mutation=='result-head':
        p=root/'result.json';v=json.loads(p.read_text());v['journal_head']='bad';atomic_json(p,v)
    elif mutation=='manifest':(root/'manifest.json').write_text('{}')
    else:
        p=root/'result.json';p.write_text(p.read_text().replace('0.7','NaN'))
    with pytest.raises(ValueError):evidence(root)
    assert not (root/'report').exists()


def test_cli_report_and_resume_contract(tmp_path,capsys):
    root=run_fixture(tmp_path);main(['report','--run',str(root)]);value=json.loads(capsys.readouterr().out)
    assert value['status']=='fail' and Path(value['report']).is_file()
    args=parser().parse_args(['teacher','train','--config','configs/teacher_nfsp_double_v4.yaml','--resume',str(root)])
    assert run_path(args,'teacher')==(str(root),True)
    args.output=str(tmp_path/'different')
    with pytest.raises(ValueError,match='same run'):run_path(args,'teacher')
    for command in ('train','distill'):
        args=parser().parse_args([command,'--config','example.yaml','--learning-rate','0.1','--profile','confirmatory','--development-reference','dev.json'])
        assert args.learning_rate==.1 and args.development_reference=='dev.json'


def test_legacy_control_trials_never_claim_a_hash_chain(tmp_path):
    root=tmp_path/'legacy';root.mkdir();atomic_json(root/'manifest.json',{'config':{}})
    atomic_json(root/'result.json',{'schema':'controllability-result-v1','status':'fail'})
    (root/'trials.jsonl').write_text(json.dumps({'phase':'development','target':0,'repeat':0,'seed':1,'selected':0})+'\n')
    report=evidence(root);assert not report['journal_verification']['trials.jsonl']['chain_verified']
    out=write_report(root);assert 'no chain' in Path(out['report']).read_text()


def test_actual_self_play_cli_resume_export_and_report_counts(tmp_path,capsys):
    import yaml
    from flyholdem.teacher.self_play_regret import AGGREGATION,SCHEMA
    from flyholdem.teacher.regret import VERSION
    from flyholdem.teacher.self_play_policy import load_policy
    config={'schema':SCHEMA,'aggregation':AGGREGATION,'iterations':2,'stack_bb':2,
        'sampling_seed':913101,'deal_seed_start':991810000,'progress_iterations':100,
        'abstraction':{'feature_version':VERSION,'stack_bb':2,'equity_buckets':8,
            'equity_samples':2,'max_information_sets':10000,'max_nodes_per_traversal':10000}}
    config_path=tmp_path/'self-play.yaml';config_path.write_text(yaml.safe_dump(config))
    root=tmp_path/'self-play'
    main(['teacher','train-self-play-regret','--config',str(config_path),'--output',str(root),'--stop-after','1'])
    partial=json.loads(capsys.readouterr().out)
    assert partial['result']['status']=='interrupted'
    assert partial['report']['status']=='interrupted'
    assert evidence(root)['journal_verification']['iterations.jsonl']['rows']==1
    main(['teacher','train-self-play-regret','--config',str(config_path),'--resume',str(root)])
    completed=json.loads(capsys.readouterr().out)
    assert completed['result']['status']=='trained-unvalidated'
    report=json.loads(Path(completed['report']['json']).read_text())
    assert report['iterations_completed']==report['planned_iterations']==2
    assert report['traversals_completed']==4
    assert report['journal_verification']['iterations.jsonl']['rows']==2
    assert not report['allowed_as_teacher'] and not report['recorded_learning_claim']
    assert 'complete player traversals: 4' in Path(completed['report']['report']).read_text()
    assert 'Complete player traversals' in Path(completed['report']['html']).read_text()
    policy=tmp_path/'policy'
    main(['teacher','export-self-play-regret','--run',str(root),'--output',str(policy)])
    exported=json.loads(capsys.readouterr().out)
    assert exported==hashlib.sha256((policy/'manifest.json').read_bytes()).hexdigest()
    assert load_policy(policy)[1]['provenance']['iterations_completed']==2
    result=json.loads((root/'result.json').read_text())
    for change in ({'operations':2},{'iterations_completed':1,'operations':2},{'iterations_completed':True,'operations':2}):
        atomic_json(root/'result.json',{**result,**change})
        with pytest.raises(ValueError,match='traversals|journal count'):
            evidence(root)
    atomic_json(root/'result.json',result)
