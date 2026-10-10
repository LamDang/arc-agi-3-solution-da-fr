"""Shared entry-point parsing; validate configuration before CUDA imports."""
import argparse
import os
from pathlib import Path
import traceback

from config import load_config
from .fla_environment import configure_fla_environment


def main(mode):
    parser = argparse.ArgumentParser(description='Explicit reference/optimized architecture '+mode)
    parser.add_argument('--config',required=True)
    parser.add_argument('--architecture',choices=['native','reference','optimized'])
    parser.add_argument('--head',choices=['native','target','cce_exact','liger_flce'])
    flags=['direct_attention_bias','disk_ple','bf16_lora','bf16_activations','liger_rmsnorm','liger_swiglu','offload_routed_experts','lora_routed_experts','expert_chunking','qsa_chunking','hyperconnection_chunking','ple_chunking']
    parser.add_argument('--chunk-tokens',type=int)
    for name in flags:
        parser.add_argument('--'+name.replace('_','-'),action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--validate-only',action='store_true')
    args = parser.parse_args()
    overrides = {name:getattr(args,name) for name in ['head','chunk_tokens',*flags] if getattr(args,name) is not None}
    production = mode == 'train' and 'training' in __import__('json').loads(Path(args.config).read_text())
    if production:
        from training.config import load
        config,settings = load(args.config,args.architecture,overrides)
    else:
        config = load_config(args.config,mode,args.architecture,overrides)
    if args.validate_only:
        value = config.as_dict()
        if production:value['training'] = settings.__dict__
        print(__import__('json').dumps(value,indent=2));return
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    configure_fla_environment(os.environ,mode,config.fla_numeric_profile,Path(__file__).resolve().parents[1])
    try:
        if production:
            from training.run import run
            run(config,settings)
        else:
            from .loop import run
            run(config,mode)
    except BaseException as exc:
        output = Path(config.output)
        if output.exists():
            from .evidence import write
            write(output/'failure.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
        raise
