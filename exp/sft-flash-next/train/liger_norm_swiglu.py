"""Opt6: targeted Liger arithmetic, preserving routing, weights and norm conventions.

Grouped RMSNorm uses one call per group because each group has distinct affine
weights. Native FP32 reductions remain FP32 inside Liger; its saved inverse RMS
also remains FP32. Gated RMSNorm retains the native FP32 SiLU/gate multiply.
"""
import ast
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path

import torch
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

MODEL_SHA256 = '0154ba57593c79330a97aefa2a909390f5f1f949aaacc6cbc8586174cd301206'
MOE_SHA256 = '2df52654daddd827cf7501ad862caf8e588be3430e5e2ce658a10ba493af161c'


def rms_forward(module, x):
    """Qwen4: normalize FP32, multiply (1+weight) FP32, return input dtype."""
    group = module.group_size
    if group is None:
        return LigerRMSNormFunction.apply(x, module.weight, module.eps, 1.0, 'gemma', False)
    shaped = x.reshape(*x.shape[:-1], -1, group)
    weights = module.weight.reshape(-1, group)
    outputs = [LigerRMSNormFunction.apply(shaped[..., i, :], weights[i],
               module.eps, 1.0, 'gemma', False) for i in range(weights.shape[0])]
    return torch.stack(outputs, dim=-2).reshape_as(x)


def gated_rms_forward(module, hidden_states, gate):
    """Keep native norm-before-gate ordering and the FP32 gate calculation."""
    if module.activation != 'silu':
        raise ValueError('Opt6 gated norm requires native SiLU activation')
    normed = LigerRMSNormFunction.apply(hidden_states, module.weight,
        module.variance_epsilon, 0.0, 'llama', False)
    return (normed * torch.nn.functional.silu(gate.float())).to(hidden_states.dtype)


def fused_swiglu(gate, up):
    if gate.dtype != up.dtype or gate.shape != up.shape:
        raise ValueError('Opt6 SwiGLU expects matching projection tensors')
    return LigerSiLUMulFunction.apply(gate, up)


def replace_expert_activation(source):
    """Replace exactly one AST node; all native dispatch/routing stays intact."""
    tree = ast.parse(source)
    wanted = ast.parse('''if hasattr(self, "_apply_gate"):
    gate_up_out = torch.cat([gate_out, up_out], dim=-1)
    gated_out = self._apply_gate(gate_up_out)
else:
    gated_out = self.act_fn(gate_out) * up_out
''').body[0]
    matches = []
    class Replace(ast.NodeTransformer):
        def visit_If(self, node):
            if ast.dump(node, include_attributes=False) == ast.dump(wanted, include_attributes=False):
                matches.append(node)
                return ast.copy_location(ast.parse('gated_out = _opt6_swiglu(gate_out, up_out)').body[0], node)
            return self.generic_visit(node)
    result = ast.fix_missing_locations(Replace().visit(tree))
    if len(matches) != 1:
        raise ValueError('Unreviewed expert activation structure')
    return result


def install(launch_dir, report_path):
    import peft
    import transformers.models.qwen4_exp.modeling_qwen4_exp as model_source
    import auto_round.modeling.fused_moe.moe_experts_interface as moe
    from transformers.integrations.moe import _default_apply_gate
    if model_source.Qwen4ExpTextExperts._apply_gate is not _default_apply_gate:
        raise RuntimeError('Opt6 requires the reviewed default expert gate')
    launch = Path(launch_dir)
    report = dict(implementation='opt6_liger_rmsnorm_swiglu', finalized=False,
        rms_casting_mode='gemma', rms_offset=1.0, rms_in_place_backward=False,
        grouped_norm='separate group calls with distinct original affine slices',
        gated_norm='llama offset0; native FP32 SiLU and gate multiply retained',
        expert_change='default SiLU(gate)*up block replaced; redundant gate/up cat removed',
        routing_and_projections_unchanged=True, statistics_blanket_cast=False,
        module_counts={}, calls={}, kernel_sources={}, native_sources={})
    for label, module, expected in [('model', model_source, MODEL_SHA256), ('moe', moe, MOE_SHA256)]:
        source = Path(inspect.getfile(module)).read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        if digest != expected:
            raise RuntimeError('Unreviewed native '+label+' source')
        (launch/('opt6-native-'+label+'.py')).write_bytes(source)
        report['native_sources'][label] = digest
    for label, cls in [('rms_norm', LigerRMSNormFunction), ('swiglu', LigerSiLUMulFunction)]:
        source = Path(inspect.getfile(cls)).read_bytes()
        (launch/('opt6-liger-'+label+'.py')).write_bytes(source)
        report['kernel_sources'][label] = hashlib.sha256(source).hexdigest()
    calls = Counter()
    def norm(self, x):
        calls['grouped_rms' if self.group_size else 'rms'] += 1
        return rms_forward(self, x)
    def gated(self, hidden_states, gate):
        calls['gated_rms'] += 1
        return gated_rms_forward(self, hidden_states, gate)
    def swiglu(gate, up):
        calls['swiglu'] += 1
        return fused_swiglu(gate, up)
    def expert_swiglu(gate, up):
        calls['expert_swiglu'] += 1
        return swiglu(gate, up)
    def mlp(self, x):
        calls['shared_mlp'] += 1
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))
    model_source.Qwen4ExpTextRMSNorm.forward = norm
    model_source.Qwen4ExpTextRMSNormGated.forward = gated
    model_source.Qwen4ExpTextMLP.forward = mlp
    expert_source = inspect.getsource(moe.linear_loop_experts_forward)
    tree = replace_expert_activation(expert_source)
    generated = ast.unparse(tree)+'\n'
    (launch/'opt6-generated-experts.py').write_text(generated)
    report['generated_experts_sha256'] = hashlib.sha256(generated.encode()).hexdigest()
    namespace = {**vars(moe), '_opt6_swiglu': expert_swiglu}
    exec(compile(tree, str(launch/'opt6-generated-experts.py'), 'exec'), namespace)
    replacement = namespace['linear_loop_experts_forward']
    moe.linear_loop_experts_forward = replacement
    moe.ALL_EXPERTS_FUNCTIONS._global_mapping[moe.LINEAR_LOOP_IMPL] = replacement
    old_factory = peft.get_peft_model
    def factory(*args, **kwargs):
        model = old_factory(*args, **kwargs)
        modules = list(model.get_base_model().named_modules())
        kinds = Counter(type(m).__name__ for _, m in modules)
        expected = {'Qwen4ExpTextRMSNorm':148, 'Qwen4ExpTextRMSNormGated':36,
                    'Qwen4ExpTextMLP':48, 'Qwen4ExpTextExperts':48}
        report['module_counts'] = {key:kinds[key] for key in expected}
        if report['module_counts'] != expected:
            raise RuntimeError('Unexpected Opt6 module inventory')
        for name, module in modules:
            if type(module).__name__ == 'Qwen4ExpTextExperts':
                if type(module)._apply_gate is not _default_apply_gate or getattr(module.config, '_experts_implementation', None) != moe.LINEAR_LOOP_IMPL:
                    raise RuntimeError('Unreviewed expert gate/dispatch '+name)
            if type(module).__name__ in ('Qwen4ExpTextMLP','Qwen4ExpTextExperts'):
                if type(module.act_fn).__name__ != 'SiLUActivation':
                    raise RuntimeError('Unreviewed activation '+name)
        return model
    peft.get_peft_model = factory
    def finalize():
        report['calls'] = dict(calls)
        report['finalized'] = True
        Path(report_path).write_text(json.dumps(report, indent=2)+'\n')
        assert all(calls[k] > 0 for k in ('rms','grouped_rms','gated_rms','shared_mlp','expert_swiglu','swiglu'))
        return report
    return finalize
