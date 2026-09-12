import copy
import hashlib
import json
import pytest
from flyholdem.connectome.registry import registry, verify, fetch_one
from flyholdem.provenance import identity, assert_compatible


def test_official_lock_is_exact_and_never_trusts_changed_files(tmp_path):
    locked=registry()
    assert sum(e['bytes'] for e in locked.values())==1109008094
    path=tmp_path/'source'
    path.write_bytes(b'original')
    expected={'bytes':8,'sha256':hashlib.sha256(b'original').hexdigest()}
    assert verify(path,expected)['bytes']==8
    path.write_bytes(b'changed!')
    with pytest.raises(ValueError,match='checksum'):
        verify(path,expected)
    with pytest.raises(ValueError,match='checksum'):
        fetch_one(path,expected)


def test_hashes_are_canonical_and_resume_rejects_each_changed_identity():
    assert identity({'a':1,'b':2})==identity({'b':2,'a':1})
    saved={k:'fixed' for k in ('config_hash','source_hash','graph_hash','encoder_hash','decoder_hash','binary_hash')}
    saved['environment']={'python':'locked'}
    assert_compatible(saved,copy.deepcopy(saved))
    for key in saved:
        changed=copy.deepcopy(saved);changed[key]='different'
        with pytest.raises(ValueError):
            assert_compatible(saved,changed)


def test_invalid_origin_fails_without_network(tmp_path):
    data=registry()
    data['edges.feather']['url']='https://example.com/unregistered'
    path=tmp_path/'bad.json';path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='origin'):
        registry(path)
