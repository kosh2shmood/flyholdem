"""Explicit, resumable official-data fetch with pre-existing hash expectations."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
LOCK = ROOT / 'data-provenance/malecns_v1/source.lock.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def registry(path=LOCK):
    data = json.loads(Path(path).read_text())
    if set(data) != {'annotations.feather','neurotransmitters.feather','edges.feather'}:
        raise ValueError('Unexpected official source set')
    for name, item in data.items():
        if not item['url'].startswith('https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'):
            raise ValueError('Unexpected source origin or release')
        if item['bytes'] <= 0 or len(item['sha256']) != 64 or any(c not in '0123456789abcdef' for c in item['sha256']):
            raise ValueError('Invalid source integrity metadata')
    return data


def verify(path, expected):
    path=Path(path)
    if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
        raise ValueError(f'Source checksum/size mismatch: {path.name}; expected registry is never rewritten')
    return {'bytes':path.stat().st_size,'sha256':expected['sha256']}


def fetch_one(path, expected):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        return verify(path,expected)
    partial=path.with_suffix(path.suffix+'.partial')
    offset=partial.stat().st_size if partial.exists() else 0
    if offset > expected['bytes']:
        raise ValueError('Oversized partial download; inspect/remove it explicitly')
    if offset < expected['bytes']:
        request=urllib.request.Request(expected['url'],headers={'Range':f'bytes={offset}-','Accept-Encoding':'identity'})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status == 206:
                if not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                    raise ValueError('Server returned an unexpected resume range')
                mode='ab'
            elif response.status == 200:
                mode='wb'  # Range ignored: restart, never append duplicate bytes.
            else:
                raise ValueError(f'Unexpected download status {response.status}')
            with partial.open(mode) as stream:
                while chunk:=response.read(4*1024*1024):
                    stream.write(chunk)
                    if stream.tell()>expected['bytes']:
                        raise ValueError('Download exceeds locked byte count')
                stream.flush()
                os.fsync(stream.fileno())
    result=verify(partial,expected)
    partial.replace(path)
    return result


def fetch_all(root=None):
    root=Path(root or ROOT/'connectome_data/malecns_v1')
    result={}
    for name, item in registry().items():
        print(f'Verifying/downloading {name}: {item["bytes"]:,} bytes',flush=True)
        result[name]=fetch_one(root/name,item)
        print(f'Verified {name}: {result[name]["sha256"]}',flush=True)
    return result


if __name__=='__main__':
    print(json.dumps(fetch_all(),indent=2))
