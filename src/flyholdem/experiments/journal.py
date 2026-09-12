"""Hash-chained append journal with deterministic replay of a checkpoint tail."""
from array import array
from collections.abc import Sequence
import json
import os
from pathlib import Path
from flyholdem.provenance import identity


class _IndexedRows(Sequence):
    """Keep offsets and hashes in memory; decode detailed rows only on access."""
    def __init__(self,path):
        self.path=path;self.offsets=array('Q');self.hashes=bytearray()

    def __len__(self):return len(self.offsets)

    def hash_at(self,index):
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        return self.hashes[32*index:32*(index+1)].hex()

    def append_at(self,offset,checksum):
        self.offsets.append(offset);self.hashes.extend(bytes.fromhex(checksum))

    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self)))]
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        with self.path.open('rb') as stream:
            stream.seek(self.offsets[index]);line=stream.readline()
        if not line.endswith(b'\n'):raise ValueError('Truncated journal final line')
        row=json.loads(line);body={key:value for key,value in row.items() if key!='hash'}
        previous=self.hash_at(index-1) if index else '0'*64
        if (body.get('index')!=index or body.get('previous')!=previous
                or row.get('hash')!=self.hash_at(index) or identity(body)!=self.hash_at(index)):
            raise ValueError('Indexed journal bytes changed after verification')
        return row


class Journal:
    def __init__(self, path, resume=False, *, indexed=False):
        self.path = Path(path)
        self.rows = _IndexedRows(self.path) if indexed else []
        previous = '0' * 64
        if resume:
            # A truncated final write is a detected failure, never silently erased.
            with self.path.open('rb') as reader:
                offset=0
                for line in reader:
                    if not line.endswith(b'\n'):raise ValueError('Truncated journal final line')
                    row = json.loads(line)
                    body = {key: value for key, value in row.items() if key != 'hash'}
                    if body['index'] != len(self.rows) or body['previous'] != previous or identity(body) != row['hash']:
                        raise ValueError('Experiment journal checksum/order mismatch')
                    if indexed:self.rows.append_at(offset,row['hash'])
                    else:self.rows.append(row)
                    previous = row['hash'];offset+=len(line)
        self.stream = self.path.open('a' if resume else 'x')

    def record(self, index, label, value):
        if not 0<=index<=len(self.rows):raise ValueError('Journal sequence gap')
        prior='0'*64
        if index:
            prior=self.rows.hash_at(index-1) if isinstance(self.rows,_IndexedRows) else self.rows[index-1]['hash']
        body = {'index': index, 'previous': prior,
                'label': label, 'value': value}
        row = {**body, 'hash': identity(body)}
        if index < len(self.rows):
            if row != self.rows[index]:
                raise ValueError(f'Checkpoint-tail re-execution differs at operation {index}')
        elif index == len(self.rows):
            offset=self.stream.tell()
            self.stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            self.stream.flush()
            os.fsync(self.stream.fileno())
            if isinstance(self.rows,_IndexedRows):self.rows.append_at(offset,row['hash'])
            else:self.rows.append(row)
        else:
            raise ValueError('Journal sequence gap')
        return value

    def close(self):
        self.stream.close()
