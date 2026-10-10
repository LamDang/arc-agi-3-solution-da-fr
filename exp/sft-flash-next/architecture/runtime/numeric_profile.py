"""Verify FLA's native exact-key config file and actual backward selection.

Uses FLA_CACHE_MODE=strict/FLA_CONFIG_DIR, without replacing callables or editing
package sources. The sole profile entry fixes the reference reverse scan's
reduction configuration; other kernel keys retain native autotuning.
"""
import importlib
import inspect
import json
import os
from pathlib import Path

from .evidence import sha,write

CUMSUM_SHA='0405701c46cee331088bfee395b3bf37f8384829dace0aa3a55a509844bcf4cb'
KEY=(1,48,64,False,True,'torch.float32','torch.float32')
SELECTED={'num_warps':1,'num_ctas':1,'num_stages':3}


def verify_profile(config,output,required=False):
    if config.fla_numeric_profile is None:return
    path=Path(__file__).resolve().parents[1]/'configs/kernels-reference-v0'
    if os.environ.get('FLA_CACHE_MODE')!='strict' or Path(os.environ.get('FLA_CONFIG_DIR','')).resolve()!=path:
        raise RuntimeError('FLA reference profile must be selected before package imports')
    cumsum=importlib.import_module('fla.ops.utils.cumsum')
    cache=importlib.import_module('fla.ops.utils.cache')
    if sha(inspect.getfile(cumsum))!=CUMSUM_SHA:raise RuntimeError('Unsupported FLA cumsum source')
    if cache.FLA_CACHE_MODE.value!='strict':raise RuntimeError('FLA imported with incorrect cache mode')
    kernel=cumsum.chunk_local_cumsum_scalar_kernel;auto=kernel.fn
    if type(kernel).__name__!='Heuristics' or type(auto).__name__!='CachedAutotuner':
        raise RuntimeError('Unsupported reverse scan wrapper')
    if auto.keys!=['B','H','BT','IS_VARLEN','REVERSE']:raise RuntimeError('Reverse scan key schema changed')
    entry=cache.load_cached_config(auto.kernel_name,cache.AutotuneKey(KEY))
    if entry!={'kwargs':{},**SELECTED}:raise RuntimeError('Reference profile entry differs')
    actual=auto.cache.get(KEY)
    selected=actual.all_kwargs() if actual is not None else None
    if selected is not None and selected!=SELECTED:raise RuntimeError('Reverse scan configuration differs from reference')
    if required and selected is None:raise RuntimeError('Reference reverse scan key was not executed')
    write(Path(output)/'numeric-profile.json',dict(profile=config.fla_numeric_profile,
        mode='native FLA strict exact-key configuration',autotune_key=KEY,
        required_configuration=SELECTED,observed_configuration=selected,
        backward_verified=required and selected==SELECTED,cumsum_sha256=CUMSUM_SHA,
        config_sha256=sha(path/'chunk_local_cumsum_scalar_kernel.json'),
        other_keys='native autotuning; no fuzzy/default profile entry',package_sources_modified=False))
