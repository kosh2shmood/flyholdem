import copy
import json
import numpy as np
import pytest
from flyholdem.neural.checkpoint import Checkpoints


def identity():
    out={k:'fixed' for k in ('config_hash','source_hash','graph_hash','encoder_hash','decoder_hash','binary_hash')}
    out['environment']={'python':'locked'};return out


def test_atomic_round_trip_retains_latest_three_plus_best(tmp_path):
    checkpoints=Checkpoints(tmp_path)
    best=checkpoints.save({'voltage':np.array([1.,2.]),'spikes':np.array([0,1],dtype=np.int32)},identity(),{'rng':{'state':123},'hand':None},0,best=True)
    for step in range(1,6):checkpoints.save({'voltage':np.array([step,step+1.])},identity(),{'step':step},step)
    dirs=[p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(dirs)==4 and (tmp_path/best).exists()
    arrays,extra,meta=checkpoints.load(identity())
    np.testing.assert_array_equal(arrays['voltage'],[5,6]);assert extra=={'step':5}
    arrays,extra,meta=checkpoints.load(identity(),'best')
    np.testing.assert_array_equal(arrays['spikes'],[0,1]);assert extra['rng']['state']==123


def test_corruption_and_identity_fail_before_restore(tmp_path):
    checkpoints=Checkpoints(tmp_path)
    name=checkpoints.save({'voltage':np.array([1.,2.])},identity(),{},0)
    altered=copy.deepcopy(identity());altered['graph_hash']='other'
    with pytest.raises(ValueError,match='identity mismatch'):checkpoints.load(altered)
    path=tmp_path/name/'voltage.npy';data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
    with pytest.raises(ValueError,match='checksum'):checkpoints.load(identity())


def test_failed_generation_keeps_last_checkpoint(tmp_path):
    c=Checkpoints(tmp_path);first=c.save({'v':np.array([1.])},identity(),{},0)
    with pytest.raises(ValueError):c.save({'v':np.array([float('nan')])},identity(),{},1)
    assert json.loads((tmp_path/'latest.json').read_text())['generation']==first
    assert not list(tmp_path.glob('.writing-*'))
