"""Production settings; architecture flags remain shared with qualification."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from config import Config, Optimizations


@dataclass(frozen=True)
class Training:
    dataset: str
    processor: str
    folds: str
    validation_fold: int = 0
    target_tokens_per_update: int = 49152
    learning_rate: float = 1e-4
    epochs: int = 5
    weight_decay: float = 0.0
    betas: tuple = (.9, .999)
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    lookahead: int = 2
    resume: str | None = None
    shard_bytes: int = 64 << 20
    ack_timeout: int = 86400
    accuracy_tokens: int = 128
    accuracy_vocab: int = 8192

    def validate(self):
        for name in ('target_tokens_per_update', 'epochs', 'shard_bytes', 'ack_timeout',
                     'accuracy_tokens', 'accuracy_vocab'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError('Expected positive integer: '+name)
        if self.epochs > 5 or self.lookahead not in (1, 2) or self.validation_fold != 0:
            raise ValueError('Require at most five epochs, bounded prefetch and validation fold 0')
        if not 8192 <= self.shard_bytes <= (256 << 20):
            raise ValueError('Checkpoint shards must be between 8 KiB and 256 MiB')
        for name in ('learning_rate', 'eps', 'max_grad_norm'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('Expected positive finite '+name)
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('Invalid weight decay')
        if len(self.betas) != 2 or any(not 0 <= b < 1 for b in self.betas):
            raise ValueError('Invalid AdamW betas')


def load(path, architecture=None, overrides=None):
    value = json.loads(Path(path).read_text())
    settings = Training(**value.pop('training'))
    settings.validate()
    value.setdefault('samples', [])
    value['dataset'] = settings.dataset
    value['learning_rate'] = settings.learning_rate
    value['weight_decay'] = settings.weight_decay
    value['target_tokens_per_update'] = settings.target_tokens_per_update
    value['epochs'] = settings.epochs
    value['optimizations'] = Optimizations(**{**value.get('optimizations', {}), **(overrides or {})})
    if architecture is not None:
        value['architecture'] = architecture
    config = Config(**value)
    config.validate('train')
    if config.max_tokens != 130000:
        raise ValueError('Production trajectory cap is exactly 130000 input + output tokens')
    return config, settings


def identity_config(config, settings):
    """Transport paths and requested resume location do not change the objective."""
    arch = config.as_dict()
    for name in ('output', 'dispatch_commit', 'model', 'dataset'):
        arch.pop(name, None)
    train = asdict(settings)
    for name in ('dataset', 'processor', 'folds', 'resume', 'ack_timeout'):
        train.pop(name, None)
    return json.loads(json.dumps(dict(architecture=arch, training=train)))
