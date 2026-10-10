"""Build one architecture explicitly. No global patches, registries or source execution."""
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoRoundConfig
from transformers.models.qwen4_exp import modeling_qwen4_exp as native
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

from components.common import adopt, replace_components

def targets(all_experts=False):
    routed=r'|mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)' if all_experts else ''
    return (r'^model\.language_model\.layers\.\d+\.(?:'
           r'self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|'
           r'linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|'
           r'mlp\.shared_expert\.(?:gate_proj|up_proj|down_proj)'+routed+r')$')
NATIVE_SHA256 = '0154ba57593c79330a97aefa2a909390f5f1f949aaacc6cbc8586174cd301206'


def resident_device_map(model_path):
    excluded = 'model.language_model.layers.1.ple.ple_embedding.ngram_embedding'.split('.')
    keys = json.loads((Path(model_path)/'model.safetensors.index.json').read_text())['weight_map']
    placement = {}
    for name in keys:
        if name.startswith('mtp.'):
            continue
        parts = name.split('.')
        for index, (actual, omitted) in enumerate(zip(parts, excluded)):
            if actual != omitted:
                placement['.'.join(parts[:index+1])] = 0
                break
    if not placement or '' in placement:
        raise ValueError('Invalid resident placement')
    return placement


class DiskPLEModel(native.Qwen4ExpForConditionalGeneration):
    """HF constructs on meta during loading. Remove table before materialization."""
    def __init__(self, config):
        super().__init__(config)
        from components.ple import DiskTablePlaceholder
        module = self.model.language_model.layers[1].ple.ple_embedding
        if module.ngram_embedding.weight.device.type != 'meta':
            raise RuntimeError('Disk PLE loader requires meta construction')
        module.ngram_embedding = DiskTablePlaceholder()


def compose(base, options):
    from components import precision as p
    from components.normalization import LigerRMSNorm, LigerGatedRMSNorm
    from components.mlp import LigerMLP, LigerExperts
    from components.attention import DirectBiasIndexer, DirectBiasTextModel
    from components.qsa_chunks import ChunkedAttention,WindowIndexer
    from components.hyperconnection_chunks import ChunkedResidual,ChunkedDecoder
    from components.ple_chunks import ChunkedPLE
    inventory = []
    if options.liger_swiglu:
        from auto_round.modeling.fused_moe import moe_experts_interface as moe
        from transformers.integrations.moe import _default_apply_gate
        for module in base.modules():
            if type(module) is native.Qwen4ExpTextExperts:
                if module.config._experts_implementation != moe.LINEAR_LOOP_IMPL or type(module)._apply_gate is not _default_apply_gate:
                    raise RuntimeError('Unsupported expert dispatch')
                if type(module.act_fn).__name__ != 'SiLUActivation':
                    raise RuntimeError('Unsupported expert activation')
    def choose(name, module):
        cls = type(module)
        selected = None
        if options.direct_attention_bias and cls is native.Qwen4ExpTextQSAIndexer:
            selected = DirectBiasIndexer
        if options.liger_rmsnorm:
            selected = {native.Qwen4ExpTextRMSNorm:LigerRMSNorm,
                        native.Qwen4ExpTextRMSNormGated:LigerGatedRMSNorm}.get(cls,selected)
        if options.liger_swiglu:
            selected = {native.Qwen4ExpTextMLP:LigerMLP, native.Qwen4ExpTextExperts:LigerExperts}.get(cls,selected)
        if options.bf16_activations:
            if selected in p.LIGER_BOUNDARIES:
                selected = p.LIGER_BOUNDARIES[selected]
            elif cls in p.NATIVE_BOUNDARIES:
                selected = p.NATIVE_BOUNDARIES[cls]
        if options.qsa_chunking:
            selected={native.Qwen4ExpTextAttention:ChunkedAttention,
                      native.Qwen4ExpTextQSAIndexer:WindowIndexer}.get(cls,selected)
        if options.hyperconnection_chunking:
            selected={native.Qwen4ExpTextGatedResidual:ChunkedResidual,
                      native.Qwen4ExpTextDecoderLayer:ChunkedDecoder}.get(cls,selected)
        if options.ple_chunking and cls is native.Qwen4ExpTextPLELayer:selected=ChunkedPLE
        return selected
    replace_components(base, choose, inventory)
    for module in base.modules():
        if isinstance(module,(ChunkedAttention,ChunkedResidual,ChunkedDecoder,ChunkedPLE)):
            module.chunk_tokens=options.chunk_tokens
    if options.direct_attention_bias:
        lm = base.model.language_model
        if lm.config._attn_implementation != 'sdpa':
            raise ValueError('Direct bias requires SDPA')
        base.model.language_model = adopt(lm, DirectBiasTextModel)
        inventory.append(dict(name='model.language_model',original=type(lm).__name__,implementation='DirectBiasTextModel'))
    return inventory


@dataclass
class Architecture:
    model: object
    config: object
    inventory: list
    loading: dict
    ple_manifest: dict | None = None
    expert_stager: object | None = None

    def loss(self, batch, labels):
        from components.head import objective
        return objective(self.model, batch, labels, self.config.optimizations.head)

    def activate(self, prepared):
        if not self.config.optimizations.disk_ple:
            return
        from components.ple import PreparedNGramEmbedding
        module = self.model.get_base_model().model.language_model.layers[1].ple.ple_embedding
        if type(module) is native.Qwen4ExpTextNGramEmbedding:
            module = adopt(module, PreparedNGramEmbedding)
            self.model.get_base_model().model.language_model.layers[1].ple.ple_embedding = module
        module.prepared_payload = prepared['ple_embeddings']
        module.prepared_input_ids = prepared['batch']['input_ids']


def build(config, output):
    native_path = Path(inspect.getfile(native))
    if hashlib.sha256(native_path.read_bytes()).hexdigest() != NATIVE_SHA256:
        raise RuntimeError('Installed native model differs from the pinned reference')
    from auto_round.modeling.fused_moe import moe_experts_interface as moe
    moe_path = Path(inspect.getfile(moe))
    if hashlib.sha256(moe_path.read_bytes()).hexdigest() != '2df52654daddd827cf7501ad862caf8e588be3430e5e2ce658a10ba493af161c':
        raise RuntimeError('Installed AutoRound expert implementation differs')
    cfg = AutoConfig.from_pretrained(config.model,local_files_only=True,trust_remote_code=False)
    if cfg.text_config.num_experts != 256:
        raise ValueError('Require the pinned 256-expert model')
    path, manifest = config.model, None
    loader = AutoModelForImageTextToText
    if config.optimizations.disk_ple:
        from components.ple import checkpoint_view
        path, manifest = checkpoint_view(config.model,Path(output)/'model-view')
        loader = DiskPLEModel
    base, loading = loader.from_pretrained(str(path),dtype=torch.bfloat16,
        device_map=resident_device_map(config.model),local_files_only=True,trust_remote_code=False,
        output_loading_info=True,quantization_config=AutoRoundConfig(backend='auto_round:tritonv2_zp'))
    if any(loading.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise RuntimeError('Non-exact checkpoint loading: '+repr(loading))
    for module in base.modules():
        for name, value in module._buffers.items():
            if value is not None and value.device.type != 'cuda':
                module._buffers[name] = value.to('cuda')
    inventory = compose(base,config.optimizations) if config.architecture != 'native' else []
    base.requires_grad_(False)
    factory=LoraConfig
    if config.optimizations.lora_routed_experts:
        from components.adapters import configuration
        factory=configuration
    model = get_peft_model(base,factory(r=16,lora_alpha=32,lora_dropout=0.,
        target_modules=targets(config.optimizations.lora_routed_experts),bias='none',task_type='CAUSAL_LM'))
    parameters = {name:value for name,value in model.named_parameters() if value.requires_grad}
    if len(parameters) != config.adapter_tensors or any('.lora_' not in name for name in parameters):
        raise RuntimeError('Unexpected trainable parameter inventory')
    if config.optimizations.bf16_lora:
        for parameter in parameters.values():
            parameter.data = parameter.data.to(torch.bfloat16)
    if config.adapter:
        state = torch.load(config.adapter,map_location='cpu',weights_only=True)
        if set(state) != set(get_peft_model_state_dict(model)):
            raise ValueError('Diagnostic adapter keys differ')
        set_peft_model_state_dict(model,state)
        for name,value in get_peft_model_state_dict(model).items():
            if not torch.equal(value.cpu(),state[name].to(value.dtype)):
                raise RuntimeError('Diagnostic adapter loading differs')
    elif not config.diagnostic_initialization:
        if any(torch.count_nonzero(value).item() != 0 for name,value in parameters.items() if '.lora_B.' in name):
            raise RuntimeError('Fresh training requires zero B')
    if config.diagnostic_initialization:
        # Test fixture only: deterministic per-name CPU generators; production
        # keeps PEFT random-A/zero-B. Both A/B are nonzero to exercise all VJPs.
        with torch.no_grad():
            for name,parameter in parameters.items():
                seed=int.from_bytes(hashlib.sha256((str(config.seed)+':'+name).encode()).digest()[:8],'little')%(2**63-1)
                generator=torch.Generator(device='cpu').manual_seed(seed)
                value=torch.randn(parameter.shape,dtype=torch.float32,generator=generator)*.002
                parameter.copy_(value)
    stager=None
    if config.optimizations.offload_routed_experts:
        from components.expert_offload import install
        stager=install(model,inventory,config.optimizations.chunk_tokens if config.optimizations.expert_chunking else 0)
        del parameters
        parameters={name:p for name,p in model.named_parameters() if p.requires_grad}
        torch.cuda.empty_cache()
    if not config.optimizations.bf16_lora and any(p.dtype!=torch.float32 for p in parameters.values()):
        raise RuntimeError('Reference LoRA master dtype must be FP32')
    if any(p.device.type!='cuda' for p in parameters.values()):
        raise RuntimeError('All trainable LoRA masters must remain on GPU')
    model.train()
    if config.checkpointing:
        kwargs={'use_reentrant':False}
        if stager:kwargs['context_fn']=stager.checkpoint_contexts
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
    return Architecture(model,config,inventory,loading,manifest,stager)
