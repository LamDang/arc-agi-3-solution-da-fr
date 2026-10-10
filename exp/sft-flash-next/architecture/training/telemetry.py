"""Small scalar events and immediately flushed TensorBoard records."""
import json
from pathlib import Path
import time


class Logger:
    def __init__(self, output):
        from torch.utils.tensorboard import SummaryWriter
        self.root = Path(output)
        self.writer = SummaryWriter(str(self.root/'tensorboard'), max_queue=1, flush_secs=10)

    def __call__(self, group, step, values):
        for key,value in values.items():
            if isinstance(value, (float, int)):
                self.writer.add_scalar(group+'/'+key, value, step)
        self.writer.flush()
        row = dict(group=group, step=step, time=time.time(), **values)
        with (self.root/'training-events.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n');stream.flush()
        print(json.dumps(row), flush=True)

    def close(self):
        self.writer.close()
