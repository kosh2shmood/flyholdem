"""Stable entry points for the dashboard and registered research workflows."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def configuration(path):
    import yaml
    value=yaml.safe_load(Path(path).read_text())
    if not isinstance(value,dict):raise ValueError('A mapping configuration is required')
    return value


def run_path(args,prefix):
    if args.resume:
        if args.output and Path(args.output).resolve()!=Path(args.resume).resolve():
            raise ValueError('--output and --resume must name the same run')
        return args.resume,True
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    return args.output or f'runs/{prefix}-{stamp}',False


def _output_options(parser):
    parser.add_argument('--output',help='New run directory; defaults to a timestamped runs/ path')
    parser.add_argument('--resume',metavar='RUN',help='Resume this existing run with identical source/config/runtime')


def _learning_options(parser):
    _output_options(parser)
    parser.add_argument('--config',required=True)
    parser.add_argument('--profile',choices=['smoke','development','confirmatory'],default='development')
    parser.add_argument('--learning-rate',required=True,type=float,help='An already registered candidate rate')
    parser.add_argument('--development-reference')
    parser.add_argument('--stop-after',type=int,help='Deterministic recovery check at an operation boundary')


def parser():
    root=argparse.ArgumentParser(description='FlyHoldem play-chip research tools')
    sub=root.add_subparsers(dest='command',required=True)
    serve=sub.add_parser('serve',help='Run the labeled live or recorded neural dashboard')
    serve.add_argument('--port',type=int,default=8766);serve.add_argument('--seed',type=int,default=20260912)
    serve.add_argument('--replay');serve.add_argument('--interval',type=float,default=.9)
    serve.add_argument('--mode',choices=['fixture','circuit','full'],default='fixture')
    serve.add_argument('--preregistration',help='Checked native controllability artifact')
    serve.add_argument('--model',help='Teacher-disconnected frozen connectome model directory')
    serve.add_argument('--training-run',help='Completed curriculum child training arm to replay without neural execution')
    serve.add_argument('--graph',help='Exact prepared graph for the recorded training arm')
    serve.add_argument('--start-hand',type=int,default=0,help='Zero-based start within the recorded arm')
    serve.add_argument('--hands',type=int,default=16,help='Show up to 1–128 consecutive recorded training hands')
    record=sub.add_parser('record',help='Generate a deterministic compact fixture demonstration')
    record.add_argument('--hands',type=int,default=6);record.add_argument('--seed',type=int,default=20260912)
    record.add_argument('--output',default='examples/fixture-demo.jsonl')
    evaluate=sub.add_parser('evaluate',help='Measure a frozen native model on fixed paired deals; no learning or gate claim')
    evaluate.add_argument('--config',default='configs/frozen_poker_evaluation.yaml');evaluate.add_argument('--model')
    evaluate.add_argument('--stop-after',type=int);evaluate.add_argument('--disconnected',action='store_true',help='Evaluate an exported model in an isolated runtime with teacher access denied');_output_options(evaluate)
    report=sub.add_parser('report',help='Verify recorded artifact chains and generate Markdown/HTML/JSON reports')
    report.add_argument('--run',required=True);report.add_argument('--output')
    preregister=sub.add_parser('preregister',help='Run registered controllability candidates and freeze the first passing mapping')
    preregister.add_argument('--config',default='configs/controllability.yaml')
    preregister.add_argument('--mode',choices=['circuit','full'],default='circuit')
    preregister.add_argument('--failure-reference');_output_options(preregister)
    train=sub.add_parser('train',help='Run registered neural conditioning with matched controls')
    _learning_options(train)
    distill=sub.add_parser('distill',help='Run registered exact-cue local or bounded-edge surrogate transfer')
    _learning_options(distill)
    curriculum=sub.add_parser('curriculum',help='Run or reverify gated multi-seed native poker curricula')
    cs=curriculum.add_subparsers(dest='curriculum_command',required=True)
    run=cs.add_parser('run',help='Require Gate 2A, prior curriculum and development before the corresponding experiment')
    run.add_argument('--config',default='configs/poker_curricula.yaml');run.add_argument('--gate2a',required=True)
    run.add_argument('--stage',required=True,choices=['3','4','5','6'])
    run.add_argument('--profile',choices=['development','confirmatory'],default='development')
    run.add_argument('--learning-mode',required=True,choices=['bio-plastic','distilled-connectome'])
    run.add_argument('--previous');run.add_argument('--development');run.add_argument('--stop-after',type=int,help='Stop after this many complete phases')
    _output_options(run)
    verify=cs.add_parser('verify',help='Recompute all curriculum phases, native decisions, paired endpoints and prerequisites')
    verify.add_argument('--run',required=True);verify.add_argument('--require-pass',action='store_true')
    gates=sub.add_parser('gate',help='Reverify prerequisite evidence before biological poker training')
    gs=gates.add_subparsers(dest='gate_command',required=True)
    certify=gs.add_parser('certify-transfer',help='Require the full 20 BB teacher, reproduced corpus and actual cue transfer/removal')
    certify.add_argument('--policy',required=True);certify.add_argument('--confirmation',required=True)
    certify.add_argument('--corpus',required=True);certify.add_argument('--output',required=True)
    verify=gs.add_parser('verify-transfer',help='Repeat every dependency in an existing Gate 2A certificate')
    verify.add_argument('--certificate',required=True)
    cues=gs.add_parser('verify-cues',help='Recompute complete registered conditioning or exact-transfer confirmation evidence')
    cues.add_argument('--run',required=True);cues.add_argument('--config',required=True);cues.add_argument('--allow-failed',action='store_true')
    removal=gs.add_parser('verify-removal',help='Execute the actual cue model before and after deleting copied teaching files')
    removal.add_argument('--model',required=True);removal.add_argument('--graph',required=True);removal.add_argument('--transfer-run',required=True)
    teacher=sub.add_parser('teacher',help='Independent conventional teacher workflow; never fly inference')
    ts=teacher.add_subparsers(dest='teacher_command',required=True)
    self_play=ts.add_parser('train-self-play-regret',help='Train synchronous two-player external-sampling self-play')
    self_play.add_argument('--config',required=True);self_play.add_argument('--stop-after',type=int);_output_options(self_play)
    self_play_export=ts.add_parser('export-self-play-regret',help='Export only the completed average self-play policy, still unvalidated')
    self_play_export.add_argument('--run',required=True);self_play_export.add_argument('--output',required=True)
    regret=ts.add_parser('train-regret',help='Train a separate conventional external-sampling population response')
    regret.add_argument('--config',default='configs/teacher_external_regret_v10.yaml');regret.add_argument('--stop-after',type=int);_output_options(regret)
    export_regret=ts.add_parser('export-regret',help='Export the completed numeric regret-average policy, still unvalidated')
    export_regret.add_argument('--run',required=True);export_regret.add_argument('--output',required=True)
    current=ts.add_parser('export-final-regret',help='Export a separately registered fixed-final positive-regret candidate')
    current.add_argument('--config',required=True);current.add_argument('--output',required=True)
    small=ts.add_parser('train-shove-fold',help='Train the separate small tabular 10 BB reference')
    small.add_argument('--config',default='configs/shove_fold_teacher.yaml');small.add_argument('--stop-after',type=int);_output_options(small)
    small_export=ts.add_parser('export-shove-fold',help='Export the small tabular reference; does not qualify a full teacher')
    small_export.add_argument('--run',required=True);small_export.add_argument('--output',required=True)
    small_eval=ts.add_parser('evaluate-shove-fold',help='Evaluate the small tabular reference on disjoint paired deals')
    small_eval.add_argument('--policy',required=True);small_eval.add_argument('--training-run',required=True)
    small_eval.add_argument('--config',default='configs/shove_fold_teacher_evaluation.yaml')
    small_eval.add_argument('--profile',choices=['development','confirmatory'],default='development')
    small_eval.add_argument('--development-reference');_output_options(small_eval)
    boundary=ts.add_parser('export-potential-boundary',help='Apply the exact terminal fold value to a matched stack-potential Q candidate')
    boundary.add_argument('--config',required=True);boundary.add_argument('--output',required=True)
    q_export=ts.add_parser('export-best-response',help='Extract explicitly labeled final Q components from matched completed training')
    q_export.add_argument('--config',required=True);q_export.add_argument('--output',required=True)
    train=ts.add_parser('train',help='Train registered NFSP self-play')
    train.add_argument('--config',required=True);train.add_argument('--stop-after',type=int);_output_options(train)
    export=ts.add_parser('export',help='Export a numeric frozen average-policy checkpoint, still unvalidated')
    export.add_argument('--run',required=True);export.add_argument('--output',required=True);export.add_argument('--which',choices=['latest','best'],default='latest')
    evaluate=ts.add_parser('evaluate',help='Evaluate the entire registered held-out paired opponent suite')
    evaluate.add_argument('--policy',required=True);evaluate.add_argument('--config',default='configs/teacher_evaluation.yaml')
    evaluate.add_argument('--profile',choices=['development','confirmatory'],default='development');evaluate.add_argument('--development-reference');_output_options(evaluate)
    corpus=ts.add_parser('export-corpus',help='Export disjoint canonical states only from a confirmed validated teacher')
    corpus.add_argument('--policy',required=True);corpus.add_argument('--validation',required=True)
    corpus.add_argument('--config',required=True);_output_options(corpus)
    verified=ts.add_parser('verify-evaluation',help='Recompute teacher qualification from registered paired deals and numeric policy hashes')
    verified.add_argument('--run',required=True);verified.add_argument('--policy',required=True)
    verified.add_argument('--allow-development',action='store_true',help='Audit development evidence without authorizing a teacher')
    verify=ts.add_parser('verify-corpus',help='Check target legality, hashes and canonical split disjointness')
    verify.add_argument('--corpus',required=True)
    qualified=ts.add_parser('verify-qualified-corpus',help='Reproduce full PokerKit collection and targets from a confirmed teacher')
    qualified.add_argument('--corpus',required=True);qualified.add_argument('--policy',required=True)
    qualified.add_argument('--validation-run',required=True)
    return root


def dispatch(args):
    if args.command=='curriculum':
        from flyholdem.experiments.curriculum_protocol import run_curriculum,verify_curriculum
        if args.curriculum_command=='verify':return verify_curriculum(args.run,args.require_pass)
        output,resume=run_path(args,'poker-curriculum')
        return run_curriculum(configuration(args.config),output,args.stage,args.profile,args.learning_mode,args.gate2a,
            previous=args.previous,development=args.development,resume=resume,stop_after=args.stop_after)
    if args.command=='gate':
        if args.gate_command=='certify-transfer':
            from flyholdem.experiments.gates import certify_gate2a
            return certify_gate2a(args.policy,args.confirmation,args.corpus,args.output)
        if args.gate_command=='verify-transfer':
            from flyholdem.experiments.gates import verify_gate2a
            return verify_gate2a(args.certificate)
        if args.gate_command=='verify-cues':
            from flyholdem.experiments.cue_validation import verify_cue_evidence
            return verify_cue_evidence(args.run,configuration(args.config),require_pass=not args.allow_failed)
        from flyholdem.experiments.teacher_removal import verify_cue_teacher_removal
        return verify_cue_teacher_removal(args.model,args.graph,args.transfer_run)
    if args.command=='serve':
        import uvicorn
        from flyholdem.server.app import create_app
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        uvicorn.run(create_app(args.seed,args.replay,args.interval,f'runs/live-{stamp}/events.jsonl',
            args.mode,args.preregistration,args.model,args.training_run,args.graph,args.start_hand,args.hands),host='127.0.0.1',port=args.port)
        return None
    if args.command=='record':
        if args.hands<1:raise ValueError('At least one fixture hand is required')
        from flyholdem.server.events import Demo,dumps
        demo=Demo(args.seed);path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w') as stream:
            completed=0
            while completed<args.hands:
                event=demo.next_event();stream.write(dumps(event)+'\n');completed+=event['kind']=='reinforcement'
        return {'path':str(path),'fixture_hands':args.hands,'hash_chained_events':demo.sequence}
    if args.command=='report':
        from flyholdem.experiments.reports import write_report
        return write_report(args.run,args.output)
    if args.command=='preregister':
        from flyholdem.connectome.registry import ROOT
        from flyholdem.interface.preregister import run_gate
        out,resume=run_path(args,'controllability-'+args.mode)
        result=run_gate(ROOT/'connectome_data/malecns_v1'/('prepared-'+args.mode),configuration(args.config),out,resume,args.failure_reference)
    elif args.command=='evaluate':
        from flyholdem.experiments.evaluate import evaluate
        out,resume=run_path(args,'native-frozen-evaluation')
        if args.disconnected:
            from flyholdem.experiments.disconnected_evaluation import evaluate_disconnected
            from flyholdem.experiments.reports import write_report
            result=evaluate_disconnected(configuration(args.config),out,args.model,resume=resume,stop_after=args.stop_after)
            return {**result,'report':write_report(result['run'])}
        result=evaluate(configuration(args.config),out,args.model,resume,args.stop_after)
    elif args.command in ('train','distill'):
        config=configuration(args.config);out,resume=run_path(args,args.command+'-'+args.profile)
        if args.command=='train':
            if config.get('schema')!='conditioning-local-v1':raise ValueError('train currently requires a registered conditioning-local-v1 configuration')
            from flyholdem.experiments.conditioning import run
        else:
            if config.get('schema') not in ('exact-cue-local-transfer-v1','exact-cue-surrogate-transfer-v1'):
                raise ValueError('distill requires a registered exact-cue local or surrogate configuration')
            from flyholdem.experiments.exact_transfer import run
        result=run(config,out,args.profile,args.learning_rate,resume,args.development_reference,args.stop_after)
    elif args.command=='teacher':
        action=args.teacher_command
        if action=='export-self-play-regret':
            from flyholdem.teacher.self_play_policy import export_policy
            return export_policy(args.run,args.output)
        if action=='export-final-regret':
            from flyholdem.teacher.regret_current_policy import export_current
            return export_current(configuration(args.config),args.output)
        if action=='export-regret':
            from flyholdem.teacher.regret_policy import export_policy
            return export_policy(args.run,args.output)
        if action=='export-shove-fold':
            from flyholdem.teacher.shove_fold_training import export_training
            return export_training(args.run,args.output)
        if action=='export-potential-boundary':
            from flyholdem.teacher.potential_policy import export_boundary
            return export_boundary(configuration(args.config),args.output)
        if action=='export-best-response':
            from flyholdem.teacher.q_policy import export_components
            return export_components(configuration(args.config),args.output)
        if action=='export':
            from flyholdem.teacher.export import export_training
            return export_training(args.run,args.output,args.which)
        if action=='verify-evaluation':
            from flyholdem.teacher.validation import verify_evaluation
            return verify_evaluation(args.run,args.policy,require_confirmatory=not args.allow_development)
        if action=='verify-qualified-corpus':
            from flyholdem.teacher.corpus_validation import verify_qualified_corpus
            return verify_qualified_corpus(args.corpus,args.policy,args.validation_run)
        if action=='verify-corpus':
            from flyholdem.teacher.corpus import verify_corpus
            return verify_corpus(args.corpus)
        out,resume=run_path(args,'teacher-'+action)
        config=configuration(args.config)
        if action=='train-self-play-regret':
            from flyholdem.teacher.self_play_training import train
            result=train(config,out,resume,args.stop_after)
        elif action=='train-regret':
            from flyholdem.teacher.regret_training import train
            result=train(config,out,resume,args.stop_after)
        elif action=='train-shove-fold':
            from flyholdem.teacher.shove_fold_training import train
            result=train(config,out,resume,args.stop_after)
        elif action=='evaluate-shove-fold':
            from flyholdem.teacher.shove_fold_evaluation import evaluate
            result=evaluate(args.policy,config,out,args.profile,resume,args.development_reference,training_run=args.training_run)
        elif action=='train':
            from flyholdem.teacher.training import train
            result=train(config,out,resume,args.stop_after)
        elif action=='evaluate':
            from flyholdem.teacher.evaluation import evaluate
            result=evaluate(args.policy,config,out,args.profile,resume,args.development_reference)
        else:
            from flyholdem.teacher.corpus import export_corpus
            return export_corpus(args.policy,args.validation,config,out,resume)
    else:raise ValueError('Unknown command')
    from flyholdem.experiments.reports import write_report
    return {'run':str(Path(out).resolve()),'result':result,'report':write_report(out)}


def main(argv=None):
    command_parser=parser();args=command_parser.parse_args(argv)
    try:value=dispatch(args)
    except (ValueError,FileNotFoundError,FileExistsError) as error:command_parser.error(str(error))
    if value is not None:print(json.dumps(value,indent=2))


if __name__=='__main__':main()
