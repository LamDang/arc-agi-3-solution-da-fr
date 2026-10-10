"""Validated, serializable architecture and execution configuration."""
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


@dataclass(frozen=True)
class Optimizations:
    head: str = 'native'
    direct_attention_bias: bool = False
    disk_ple: bool = False
    bf16_lora: bool = False
    bf16_activations: bool = False
    liger_rmsnorm: bool = False
    liger_swiglu: bool = False
    offload_routed_experts: bool = False
    lora_routed_experts: bool = False
    expert_chunking: bool = False
    qsa_chunking: bool = False
    hyperconnection_chunking: bool = False
    ple_chunking: bool = False
    chunk_tokens: int = 8192

    def validate(self):
        if self.head not in {'native', 'target', 'cce_exact', 'liger_flce'}:
            raise ValueError('Unknown head implementation')
        for field in fields(self):
            if field.name not in {'head','chunk_tokens'} and type(getattr(self, field.name)) is not bool:
                raise ValueError('Optimization flags must be booleans: '+field.name)
        if type(self.chunk_tokens) is not int or self.chunk_tokens < 1:
            raise ValueError('chunk_tokens must be a positive integer')
        if self.expert_chunking and not self.offload_routed_experts:
            raise ValueError('Expert chunks currently require CPU canonical experts')
        if self.qsa_chunking and not self.direct_attention_bias:
            raise ValueError('QSA chunks require direct bias selection')
        if self.ple_chunking and not self.disk_ple:
            raise ValueError('PLE chunks require prepared full-context embeddings')
        if any((self.expert_chunking,self.qsa_chunking,self.hyperconnection_chunking,self.ple_chunking)) and not self.bf16_activations:
            raise ValueError('Chunk components currently require the BF16 activation reference')


@dataclass(frozen=True)
class Config:
    architecture: str
    model: str
    samples: list[str]
    output: str
    optimizations: Optimizations
    adapter: str | None = None
    prompt_tokens: int | None = None
    seed: int = 20261009
    checkpointing: bool = True
    save_on_cpu: bool = True
    max_tokens: int = 130000
    learning_rate: float = 0.0002
    weight_decay: float = 0.0
    target_tokens_per_update: int = 32768
    epochs: int = 1
    expected_sha256: dict | None = None
    baseline: dict | None = None
    dispatch_commit: str | None = None
    comparison_mode: str = 'report'
    diagnostic_initialization: bool = False
    benchmark_repeats: int = 2
    benchmark_tokens: int | None = None
    fla_numeric_profile: str | None = None

    @property
    def adapter_tensors(self):
        return 744 + (48*256*3*2 if self.optimizations.lora_routed_experts else 0)

    def validate(self, mode):
        if mode not in {'test','train','benchmark'}:raise ValueError('Unknown execution mode')
        self.optimizations.validate()
        if self.fla_numeric_profile not in {None,'reference-v0'}:
            raise ValueError('Unknown FLA numerical profile')
        if self.comparison_mode not in {'report','exact'}:
            raise ValueError('Comparison mode must be report or exact')
        if self.architecture not in {'native','reference', 'optimized'}:
            raise ValueError('Architecture must be native, reference or optimized')
        if self.architecture == 'native' and self.optimizations != Optimizations():
            raise ValueError('Historical native architecture cannot enable flags')
        if self.architecture == 'reference' and self.optimizations != reference_options():
            raise ValueError('Reference settings are fixed; use optimized for variations')
        if self.optimizations.offload_routed_experts and not self.checkpointing:
            raise ValueError('Expert prefetch requires non-reentrant layer checkpointing')
        if self.optimizations.offload_routed_experts and self.optimizations.bf16_lora:
            raise ValueError('CPU expert masters and gradients must remain FP32')
        if not self.samples or len(set(self.samples)) != len(self.samples):
            raise ValueError('Require distinct, complete encoded sample paths')
        if mode in {'test','benchmark'} and len(self.samples) != 1:
            raise ValueError('Capture/benchmark requires exactly one sample')
        if self.optimizations.disk_ple and self.epochs != 1:
            raise ValueError('Disk PLE loader currently supports one complete epoch')
        if mode == 'train' and self.adapter is not None:
            raise ValueError('Production training starts fresh; diagnostic adapters are test-only')
        if self.diagnostic_initialization and (mode not in {'test','benchmark'} or self.adapter is not None):
            raise ValueError('Nonzero diagnostic initialization is test-only and excludes an adapter file')
        if type(self.benchmark_repeats) is not int or self.benchmark_repeats<1:
            raise ValueError('benchmark_repeats must be a positive integer')
        if mode == 'benchmark':
            if type(self.benchmark_tokens) is not int or not 1<self.benchmark_tokens<=self.max_tokens:
                raise ValueError('Benchmark requires an exact token count within max_tokens')
            if self.prompt_tokens is not None:raise ValueError('Benchmark requires explicit interleaved labels')
            if not self.baseline or not self.baseline.get('initial_adapter'):
                raise ValueError('Benchmark requires the saved reference initialization pin')
        if self.prompt_tokens is not None and self.prompt_tokens < 1:
            raise ValueError('Invalid diagnostic prompt length')
        if min(self.max_tokens, self.target_tokens_per_update, self.epochs) < 1:
            raise ValueError('Invalid positive run limit')
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError('Invalid optimizer settings')

    def as_dict(self):
        return asdict(self)


def reference_options():
    return Optimizations(bf16_activations=True,disk_ple=True,
        offload_routed_experts=True,lora_routed_experts=True)


def load_config(path, mode, architecture=None, overrides=None):
    value = json.loads(Path(path).read_text())
    selected=architecture or value['architecture']
    options = asdict(reference_options()) if selected=='reference' else {}
    options.update(value.pop('optimizations', {}))
    options.update(overrides or {})
    value['optimizations'] = Optimizations(**options)
    if architecture is not None:
        value['architecture'] = architecture
    config = Config(**value)
    config.validate(mode)
    return config
