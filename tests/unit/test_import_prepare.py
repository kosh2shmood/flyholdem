import numpy as np
import pytest
pd=pytest.importorskip('pandas')
from flyholdem.connectome.import_malecns import exact_ids,index_edges,normalize_nodes
from flyholdem.connectome.prepare import compile_csr,transmitter_signs


def test_integer_identity_and_edge_loss_accounting():
    ids=exact_ids(['9007199254740993','9007199254740995'])
    assert list(ids)==[9007199254740993,9007199254740995]
    for bad in ([float(2**53)],[-1],['1e3']):
        with pytest.raises(ValueError):exact_ids(bad)
    pre,post,count,keep=index_edges(ids,[ids[0],ids[0],ids[1],9],[ids[0],ids[1],ids[0],ids[0]],[1,3,2,50])
    assert keep.tolist()==[True,True,True,False]
    assert list(zip(pre,post,count))==[(0,0,1),(0,1,3),(1,0,2)]
    with pytest.raises(ValueError):index_edges(ids,[ids[0]],[ids[1]],[1.5])


def test_node_policy_retains_tbc_and_exact_annotation_fields():
    source=pd.DataFrame({'bodyId':[4,1,3,2], 'superclass':['tbc','central',None,'central'],
      'status':['Untraced','Traced','Traced','Glia'],'type':[None,'KC',None,None],
      'somaLocation':[[1,2,3],[4,5,6],None,None]})
    nt=pd.DataFrame({'body':[1,4],'consensus_nt':['acetylcholine','dopamine']})
    catalog,nodes=normalize_nodes(source,nt)
    assert nodes.source_id.tolist()==[1,4]
    assert nodes.somaLocation.tolist()==[[4,5,6],[1,2,3]]
    assert catalog.loc[~catalog.retained].exclusion_reason.tolist()==['explicit_glia','unassigned_superclass']
    with pytest.raises(ValueError):normalize_nodes(source,pd.concat([nt,nt]))


def test_sign_proxy_ambiguity_never_silently_drops_edges():
    values=['acetylcholine','gaba','glutamate','histamine','dopamine',None,'gaba, acetylcholine','gaba, dopamine']
    plus,u=transmitter_signs(values);minus,_=transmitter_signs(values,-1)
    assert plus.tolist()==[1,-1,-1,-1,1,1,1,-1]
    assert u.tolist()==[False,False,False,False,True,True,True,False]
    assert minus[u].tolist()==[-1,-1,-1]


def test_batch_size_independent_all_edge_csr(tmp_path):
    nodes=pd.DataFrame({'source_id':[1,2,3],'neurotransmitter':['acetylcholine','gaba',None]})
    pre=np.array([2,0,2,1,0,2]);post=np.array([0,0,2,2,1,1]);counts=np.array([7,1,2,3,4,1],dtype=np.uint32)
    def stream(size):
        return lambda: ((pre[i:i+size],post[i:i+size],counts[i:i+size]) for i in range(0,len(pre),size))
    a,b=tmp_path/'a',tmp_path/'b'
    compile_csr(nodes,stream(2),a);compile_csr(nodes,stream(5),b)
    for name in ['ptr','post','contacts','weight','ids','signs','uncertain']:
        assert (a/(name+'.npy')).read_bytes()==(b/(name+'.npy')).read_bytes()
    assert np.load(a/'ptr.npy').tolist()==[0,2,3,6]
    assert np.load(a/'post.npy').tolist()==[0,1,2,0,2,1]
    assert np.load(a/'contacts.npy').sum()==counts.sum()
    assert np.load(a/'weight.npy')[2]<0


def test_prepared_manifest_tamper_detection(tmp_path):
    import hashlib,json
    from flyholdem.connectome.prepare import load_graph
    from flyholdem.connectome.registry import digest
    nodes=pd.DataFrame({'source_id':[1,2],'neurotransmitter':['acetylcholine','gaba']})
    compile_csr(nodes,lambda:iter([(np.array([0,1]),np.array([1,0]),np.array([1,2],dtype=np.uint32))]),tmp_path)
    h=hashlib.sha256()
    for name in ('ptr','post','weight'):h.update(memoryview(np.load(tmp_path/(name+'.npy'))))
    manifest={'files':{p.name:digest(p) for p in tmp_path.iterdir()},'graph_hash':h.hexdigest()}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    graph,_=load_graph(tmp_path)
    assert graph['ids'].tolist()==[1,2]
    path=tmp_path/'weight.npy';data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
    with pytest.raises(ValueError,match='hash mismatch'):load_graph(tmp_path)
