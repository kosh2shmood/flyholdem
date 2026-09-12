"""Loss-accounted official MaleCNS import; no topology threshold or ID rounding.

exact_ids/index_edges adapted from nftechie/doomfly, revision
71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33, doom/connectome.py.
Copyright (c) 2026 nftechie and DOOMFLY contributors, MIT.
Full notice: licenses/DOOMFLY-MIT.txt. Streaming orchestration is FlyHoldem.
"""
import json
from pathlib import Path
import os
import uuid
import numpy as np
from .registry import ROOT, registry, verify, digest


def exact_ids(values) -> np.ndarray:
    """Never round 64-bit biological IDs through floating point or JavaScript."""
    items = np.asarray(values) if hasattr(values, "dtype") else np.asarray(values, dtype=object)
    if items.dtype.kind == "f":
        raise ValueError("Neuron IDs must be integers or decimal strings, never floats.")
    if items.dtype.kind in "iu":
        if np.any(items < 0):
            raise ValueError("Neuron IDs cannot be negative.")
        return items.astype(np.uint64)
    text = [str(value) for value in items]
    if any(not value.isascii() or not value.isdecimal() for value in text):
        raise ValueError("Neuron IDs must be nonnegative decimal integers.")
    return np.asarray(text, dtype=np.uint64)


def index_edges(ids: np.ndarray, pre, post, counts):
    """Return retained edges plus a mask accounting for every excluded row."""
    if not len(ids) or np.any(ids[1:] <= ids[:-1]):
        raise ValueError("Node IDs must be nonempty, unique, and sorted.")
    pre, post = exact_ids(pre), exact_ids(post)
    counts = np.asarray(counts)
    if len(pre) != len(post) or len(pre) != len(counts):
        raise ValueError("Edge columns have different lengths.")
    if (not np.all(np.isfinite(counts)) or np.any(counts < 1)
            or np.any(counts != np.floor(counts)) or np.any(counts > 2**32 - 1)):
        raise ValueError("Synapse counts must be positive uint32-compatible integers.")
    i, j = np.searchsorted(ids, pre), np.searchsorted(ids, post)
    keep = (i < len(ids)) & (j < len(ids))
    keep &= ids[np.minimum(i, len(ids) - 1)] == pre
    keep &= ids[np.minimum(j, len(ids) - 1)] == post
    return i[keep].astype(np.uint32), j[keep].astype(np.uint32), counts[keep].astype(np.uint32), keep



def normalize_nodes(frame, nt_frame):
    """Preserve source annotations, exact IDs and a reason for every exclusion."""
    import pandas as pd
    frame=frame.copy()
    frame['source_id']=exact_ids(frame.bodyId)
    if frame.source_id.duplicated().any():raise ValueError('Duplicate annotation IDs')
    nt=nt_frame[['body','consensus_nt']].copy();nt['body']=exact_ids(nt.body)
    if nt.body.duplicated().any():raise ValueError('Duplicate neurotransmitter IDs')
    glia=frame.status.eq('Glia').fillna(False)
    assigned=frame.superclass.notna() & frame.superclass.ne('').fillna(False)
    frame['retained']=assigned & ~glia
    frame['exclusion_reason']=np.where(glia,'explicit_glia',np.where(assigned,'retained','unassigned_superclass'))
    frame['cell_type']=frame['type']
    frame['neurotransmitter']=frame.source_id.map(nt.set_index('body').consensus_nt)
    frame=frame.sort_values('source_id',ignore_index=True)
    nodes=frame.loc[frame.retained].reset_index(drop=True)
    nodes.insert(0,'node_index',np.arange(len(nodes),dtype=np.uint32))
    if not len(nodes):raise ValueError('No retained neurons')
    return frame,nodes


def import_graph(root=None, output=None):
    import pyarrow as pa
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc
    root=Path(root or ROOT/'connectome_data/malecns_v1');output=Path(output or root/'normalized')
    sources={name:verify(root/name,item) for name,item in registry().items()}
    if output.exists():
        report=json.loads((output/'report.json').read_text())
        if report['sources']!=sources or report['importer_sha256']!=digest(__file__):raise ValueError('Normalized source/importer mismatch; use a new output directory')
        for name,sha in report['files'].items():
            if digest(output/name)!=sha:raise ValueError('Normalized artifact hash mismatch')
        return report
    temporary=output.with_name(output.name+'.partial-'+uuid.uuid4().hex[:8]);temporary.mkdir(parents=True)
    annotations=feather.read_table(root/'annotations.feather').to_pandas()
    nt=feather.read_table(root/'neurotransmitters.feather',columns=['body','consensus_nt']).to_pandas()
    catalog,nodes=normalize_nodes(annotations,nt)
    feather.write_feather(catalog,temporary/'catalog.feather');feather.write_feather(nodes,temporary/'neurons.feather')
    ids=exact_ids(nodes.source_id);np.save(temporary/'ids.npy',ids,allow_pickle=False)
    incoming=np.zeros(len(ids),dtype=np.int64);outgoing=np.zeros(len(ids),dtype=np.int64);degree=np.zeros(len(ids),dtype=np.int64)
    stats={key:0 for key in ('source_edge_rows','retained_edge_rows','source_synaptic_contacts','retained_synaptic_contacts','retained_weight_one_edges','retained_self_edges')}
    reader=ipc.open_file(pa.memory_map(str(root/'edges.feather'),'r'))
    schema=pa.schema([('pre_index',pa.uint32()),('post_index',pa.uint32()),('synapse_count',pa.uint32())])
    with pa.OSFile(str(temporary/'edges.arrow'),'wb') as sink,ipc.new_file(sink,schema) as writer:
        for number in range(reader.num_record_batches):
            batch=reader.get_batch(number)
            pre,post,weights=[batch.column(batch.schema.get_field_index(c)).to_numpy(zero_copy_only=False) for c in ('body_pre','body_post','weight')]
            i,j,count,keep=index_edges(ids,pre,post,weights)
            stats['source_edge_rows']+=len(pre);stats['retained_edge_rows']+=len(i)
            stats['source_synaptic_contacts']+=int(weights.sum(dtype=np.uint64));stats['retained_synaptic_contacts']+=int(count.sum(dtype=np.uint64))
            stats['retained_weight_one_edges']+=int(np.count_nonzero(count==1));stats['retained_self_edges']+=int(np.count_nonzero(i==j))
            np.add.at(incoming,j,count);np.add.at(outgoing,i,count);np.add.at(degree,i,1)
            writer.write_batch(pa.record_batch([pa.array(i),pa.array(j),pa.array(count)],schema=schema))
    stats['excluded_edge_rows']=stats['source_edge_rows']-stats['retained_edge_rows']
    stats['excluded_synaptic_contacts']=stats['source_synaptic_contacts']-stats['retained_synaptic_contacts']
    if int(incoming.sum())!=int(outgoing.sum()) or int(incoming.sum())!=stats['retained_synaptic_contacts']:raise ValueError('Contact accounting mismatch')
    for name,value in [('incoming',incoming),('outgoing',outgoing),('degree',degree)]:np.save(temporary/(name+'.npy'),value,allow_pickle=False)
    report={'schema':'malecns-normalized-v1','dataset':'MaleCNS v1.0','sources':sources,'importer_sha256':digest(__file__),
        'source_nodes':len(catalog),'nodes':len(nodes),'graph':stats,'excluded_objects':catalog.loc[~catalog.retained].exclusion_reason.value_counts().to_dict(),
        'superclasses':nodes.superclass.value_counts().to_dict(),'classes':nodes['class'].fillna('unknown').value_counts().to_dict(),
        'isolated_nodes':int(np.count_nonzero(incoming+outgoing==0)),
        'files':{p.name:digest(p) for p in sorted(temporary.iterdir()) if p.is_file()}}
    (temporary/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    for path in temporary.iterdir():
        with path.open('rb') as f:os.fsync(f.fileno())
    temporary.rename(output)
    return report


if __name__=='__main__':
    report=import_graph();print(json.dumps({k:report[k] for k in ('nodes','graph','classes')},indent=2))
