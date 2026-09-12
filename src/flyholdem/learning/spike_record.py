"""Lossless sparse readout-window activity for recorded training inspection."""
import base64
import binascii
import hashlib
import zlib
import numpy as np

SCHEMA='sparse-native-spike-counts-v1'
CODEC='zlib-base64-little-endian-int32-index-count-pairs-v1'


def record_spikes(counts):
    counts=np.asarray(counts)
    if counts.ndim!=1 or counts.dtype!=np.dtype('int32') or not 0<len(counts)<=10000000 or np.any(counts<0):
        raise ValueError('A nonnegative native int32 spike vector is required')
    indices=np.flatnonzero(counts)
    pairs=np.empty((len(indices),2),dtype='<i4');pairs[:,0]=indices;pairs[:,1]=counts[indices]
    return {'schema':SCHEMA,'neuron_count':len(counts),'dtype':counts.dtype.str,'codec':CODEC,
        'nonzero':len(indices),'payload':base64.b64encode(zlib.compress(pairs.tobytes(),6)).decode('ascii'),
        'total':int(counts.sum(dtype=np.int64))}


def restore_spikes(record,expected_hash,neuron_count=None):
    """Validate/reconstruct recorded bytes; never recompute or infer activity."""
    if (record.get('schema')!=SCHEMA or type(record.get('neuron_count')) is not int
            or record['neuron_count']<1 or record['neuron_count']>10000000
            or record.get('dtype') not in ('<i4','>i4') or record.get('codec')!=CODEC
            or neuron_count is not None and record['neuron_count']!=neuron_count
            or type(record.get('nonzero')) is not int or not 0<=record['nonzero']<=record['neuron_count']):
        raise ValueError('Invalid recorded native spike shape, dtype or codec')
    size=record['nonzero']*8;payload=record.get('payload')
    if not isinstance(payload,str) or len(payload)>4*(size+1024):
        raise ValueError('Invalid bounded compressed native activity')
    try:
        compressed=base64.b64decode(payload,validate=True);decoder=zlib.decompressobj()
        raw=decoder.decompress(compressed,size+1)
    except (binascii.Error,zlib.error,ValueError) as error:
        raise ValueError('Invalid compressed native activity') from error
    if len(raw)!=size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError('Invalid bounded compressed native activity')
    pairs=np.frombuffer(raw,dtype='<i4').reshape(-1,2);indices=pairs[:,0];counts=pairs[:,1]
    n=record['neuron_count']
    if (np.any(indices<0) or np.any(indices>=n) or np.any(np.diff(indices.astype(np.int64))<=0)
            or np.any(counts<=0) or type(record.get('total')) is not int
            or record['total']!=int(counts.sum(dtype=np.int64))):
        raise ValueError('Invalid sparse native spike indices, values or total')
    restored=np.zeros(n,dtype=record['dtype']);restored[indices]=counts
    if hashlib.sha256(restored.tobytes()).hexdigest()!=expected_hash:
        raise ValueError('Recorded activity differs from the original native spike checksum')
    return restored
