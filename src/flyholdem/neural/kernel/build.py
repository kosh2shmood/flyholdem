"""Build only the reviewed source and lock source, flags, compiler and binary."""
import json
from pathlib import Path
import platform
import subprocess
import sys
from flyholdem.connectome.registry import digest, ROOT

SOURCE=Path(__file__).with_name('doomfly_lif.cpp')
LIBRARY=ROOT/'runs/build'/('liblif.dylib' if sys.platform=='darwin' else 'liblif.so')


def build():
    LIBRARY.parent.mkdir(parents=True,exist_ok=True)
    compiler=subprocess.check_output(['c++','--version'],text=True).splitlines()[0]
    flags=['-std=c++17','-O3','-fPIC','-shared','-ffp-contract=off','-fno-fast-math']
    temporary=LIBRARY.with_suffix(LIBRARY.suffix+'.partial')
    subprocess.run(['c++',*flags,str(SOURCE),'-o',str(temporary)],check=True)
    temporary.replace(LIBRARY)
    record={'schema':'native-build-v1','source_sha256':digest(SOURCE),'binary_sha256':digest(LIBRARY),
            'compiler':compiler,'flags':flags,'platform':platform.platform()}
    LIBRARY.with_suffix(LIBRARY.suffix+'.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def verify_build():
    record=json.loads(LIBRARY.with_suffix(LIBRARY.suffix+'.json').read_text())
    if digest(SOURCE)!=record['source_sha256'] or digest(LIBRARY)!=record['binary_sha256']:
        raise ValueError('Native source/binary changed; run make build-kernel')
    return record


if __name__=='__main__':print(json.dumps(build(),indent=2))
