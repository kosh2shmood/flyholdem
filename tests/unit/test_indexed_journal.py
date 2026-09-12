from pathlib import Path
import tracemalloc
import pytest
from flyholdem.experiments.journal import Journal
from flyholdem.experiments.poker_evidence import iter_journal


def test_indexed_journal_is_byte_identical_and_replays_saved_tail(tmp_path,monkeypatch):
    plain=Journal(tmp_path/'plain.jsonl');indexed=Journal(tmp_path/'indexed.jsonl',indexed=True)
    values=[{'unicode':'♠','spikes':[i]*32,'decision':i} for i in range(8)]
    for i,value in enumerate(values):
        plain.record(i,['hand',i],value);indexed.record(i,['hand',i],value)
    assert indexed.rows[:]==plain.rows
    assert indexed.rows[-1]==plain.rows[-1]
    plain.close();indexed.close()
    assert (tmp_path/'plain.jsonl').read_bytes()==(tmp_path/'indexed.jsonl').read_bytes()
    def no_whole_file(*args,**kwargs):raise AssertionError('Resume must stream detailed rows')
    monkeypatch.setattr(Path,'read_text',no_whole_file)
    recovered=Journal(tmp_path/'indexed.jsonl',resume=True,indexed=True)
    before=(tmp_path/'indexed.jsonl').read_bytes()
    for i in range(5,8):recovered.record(i,['hand',i],values[i])
    assert (tmp_path/'indexed.jsonl').read_bytes()==before
    with pytest.raises(ValueError,match='re-execution differs'):
        recovered.record(7,['hand',7],{'changed':True})
    with pytest.raises(ValueError,match='sequence gap'):recovered.record(9,[],{})
    recovered.record(8,['next'],{'new':True});recovered.close()
    with pytest.raises(IndexError):_ = recovered.rows[9]
    assert len(list(iter_journal(tmp_path/'indexed.jsonl')))==9


@pytest.mark.parametrize('damage',['content','newline','order'])
def test_indexed_resume_refuses_damaged_journals(tmp_path,damage):
    path=tmp_path/'hands.jsonl';journal=Journal(path,indexed=True)
    journal.record(0,['first'],{'value':10});journal.record(1,['next'],{'value':20});journal.close()
    data=path.read_bytes()
    if damage=='content':data=data.replace(b'"value": 10',b'"value": 11')
    elif damage=='newline':data=data.rstrip(b'\n')
    else:data=b''.join(reversed(data.splitlines(keepends=True)))
    path.write_bytes(data)
    with pytest.raises(ValueError):Journal(path,resume=True,indexed=True)


def test_indexed_access_detects_changes_after_resume(tmp_path):
    path=tmp_path/'hands.jsonl';journal=Journal(path,indexed=True)
    journal.record(0,[],{'value':10});journal.close()
    path.write_bytes(path.read_bytes().replace(b'"value": 10',b'"value": 11'))
    with pytest.raises(ValueError,match='bytes changed'):_=journal.rows[0]


def test_stream_rejects_replacement_after_initial_audit(tmp_path):
    path=tmp_path/'hands.jsonl';journal=Journal(path,indexed=True)
    for i in range(3):journal.record(i,[],{'value':i})
    journal.close();records=iter_journal(path);next(records)
    replacement=tmp_path/'replacement.jsonl';replacement.write_bytes(path.read_bytes()+b'\n');replacement.replace(path)
    with pytest.raises(ValueError,match='changed while consuming'):list(records)


def test_large_indexed_training_log_has_bounded_live_memory(tmp_path):
    # Eight MiB of detailed payload stays on disk, including after recovery.
    path=tmp_path/'large.jsonl';payload='x'*65536
    tracemalloc.start()
    try:
        journal=Journal(path,indexed=True)
        for i in range(128):journal.record(i,[i],{'activity_record':payload,'hand':i})
        journal.close();del journal
        journal=Journal(path,resume=True,indexed=True)
        assert len(journal.rows)==128 and journal.rows[-1]['value']['hand']==127
        _,peak=tracemalloc.get_traced_memory();journal.close()
    finally:tracemalloc.stop()
    assert path.stat().st_size>8*1024*1024
    assert peak<2*1024*1024
