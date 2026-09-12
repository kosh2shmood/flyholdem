"""Hash-chained append journal with deterministic replay of a checkpoint tail."""
import json
import os
from pathlib import Path
from flyholdem.provenance import identity


class Journal:
    def __init__(self, path, resume=False):
        self.path = Path(path)
        self.rows = []
        previous = '0' * 64
        if resume:
            # A truncated final write is a detected failure, never silently erased.
            for line in self.path.read_text().splitlines():
                row = json.loads(line)
                body = {key: value for key, value in row.items() if key != 'hash'}
                if body['index'] != len(self.rows) or body['previous'] != previous or identity(body) != row['hash']:
                    raise ValueError('Experiment journal checksum/order mismatch')
                self.rows.append(row)
                previous = row['hash']
        self.stream = self.path.open('a' if resume else 'x')

    def record(self, index, label, value):
        body = {'index': index, 'previous': self.rows[index - 1]['hash'] if index else '0' * 64,
                'label': label, 'value': value}
        row = {**body, 'hash': identity(body)}
        if index < len(self.rows):
            if row != self.rows[index]:
                raise ValueError(f'Checkpoint-tail re-execution differs at operation {index}')
        elif index == len(self.rows):
            self.stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.rows.append(row)
        else:
            raise ValueError('Journal sequence gap')
        return value

    def close(self):
        self.stream.close()
