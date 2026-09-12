"""Private entry point for an isolated, teacher-free native evaluation process."""
import argparse
import importlib.abc
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--resume',action='store_true');parser.add_argument('--stop-after',type=int)
    args=parser.parse_args();root=Path(args.root).resolve();runtime=root/'runtime'
    contract=json.loads((root/'isolation.json').read_text());project=Path(contract['project']).resolve()
    graph=Path(contract['graph']).resolve();environment=Path(sys.prefix).resolve()
    allowed=(root,graph,environment,Path(contract['git_directory']).resolve())
    if (runtime/'src/flyholdem/teacher').exists():raise ValueError('Teacher code must not be present in this runtime')
    class NoTeacher(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname=='flyholdem.teacher' or fullname.startswith('flyholdem.teacher.'):
                raise ImportError('Teacher modules are unavailable in frozen evaluation')
    sys.meta_path.insert(0,NoTeacher())
    def protect(event,values):
        if event!='open' or not values or not isinstance(values[0],(str,bytes)):return
        path=Path(values[0].decode() if isinstance(values[0],bytes) else values[0]).resolve()
        if path.is_relative_to(project) and not any(path.is_relative_to(item) for item in allowed):
            raise PermissionError('Project training artifacts are unavailable to isolated evaluation')
    sys.addaudithook(protect)
    # Demonstrate both boundaries before loading the frozen neural model.
    try:__import__('flyholdem.teacher')
    except ImportError:pass
    else:raise AssertionError('Teacher import was not denied')
    try:open(project/'src/flyholdem/teacher/__init__.py').close()
    except PermissionError:pass
    else:raise AssertionError('External project training files were not denied')
    from flyholdem.experiments.evaluate import evaluate
    from flyholdem.neural.checkpoint import atomic_json
    import yaml
    config=yaml.safe_load((runtime/'config.yaml').read_text())
    result=evaluate(config,root/'evaluation',runtime/'model',args.resume,args.stop_after)
    if any(name.startswith('flyholdem.teacher') for name in sys.modules):raise AssertionError('Teacher module was imported')
    atomic_json(root/'isolation-result.json',{'schema':'disconnected-native-evaluation-v1','teacher_import_denied':True,
        'external_training_files_denied':True,'teacher_modules_loaded':False,'hands_completed':result['hands_completed'],
        'weights_unchanged':result['weights_unchanged'],'status':result['status'],'learning_claim':False})
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
