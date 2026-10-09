"""Shared entry-point parsing; validate configuration before CUDA imports."""
import argparse
import os
from pathlib import Path
import traceback

from config import load_config


def main(mode):
    parser = argparse.ArgumentParser(description='Explicit reference/optimized architecture '+mode)
    parser.add_argument('--config',required=True)
    parser.add_argument('--architecture',choices=['reference','optimized'])
    parser.add_argument('--head',choices=['native','target','cce_exact','liger_flce'])
    for name in ['direct_attention_bias','disk_ple','bf16_lora','bf16_activations','liger_rmsnorm','liger_swiglu']:
        parser.add_argument('--'+name.replace('_','-'),action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--validate-only',action='store_true')
    args = parser.parse_args()
    overrides = {name:getattr(args,name) for name in ['head','direct_attention_bias','disk_ple','bf16_lora','bf16_activations','liger_rmsnorm','liger_swiglu'] if getattr(args,name) is not None}
    config = load_config(args.config,mode,args.architecture,overrides)
    if args.validate_only:
        print(__import__('json').dumps(config.as_dict(),indent=2));return
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    from .loop import run
    try:
        run(config,mode)
    except BaseException as exc:
        output = Path(config.output)
        if output.exists():
            from .evidence import write
            write(output/'failure.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
        raise
