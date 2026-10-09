"""BF16 activation ports as component classes; native FP32 statistics stay native."""
import torch
from torch.utils._pytree import tree_map
from transformers.models.qwen4_exp import modeling_qwen4_exp as native
from .normalization import LigerRMSNorm, LigerGatedRMSNorm
from .mlp import LigerMLP, LigerExperts


def cast_boundary(value):
    def cast(tensor):
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point() and tensor.ndim:
            return tensor.to(torch.bfloat16)
        return tensor
    return tree_map(cast, value)


class BF16Boundary:
    activation_role = 'hidden'
    def forward(self, *args, **kwargs):
        if args:
            args = (cast_boundary(args[0]), *args[1:])
        else:
            kwargs = dict(kwargs)
            key = next((key for key in ('hidden_states','x','hyper_input') if key in kwargs), None)
            if key is None:
                raise ValueError('Missing component hidden input')
            kwargs[key] = cast_boundary(kwargs[key])
        output = super().forward(*args, **kwargs)
        if self.activation_role == 'attention':
            return (cast_boundary(output[0]), *output[1:])
        if self.activation_role == 'residual' and isinstance(output, tuple):
            return (cast_boundary(output[0]), cast_boundary(output[1]), *output[2:])
        return cast_boundary(output)


class BF16Decoder(BF16Boundary, native.Qwen4ExpTextDecoderLayer): pass
class BF16GDN(BF16Boundary, native.Qwen4ExpTextGatedDeltaNet): pass
class BF16Attention(BF16Boundary, native.Qwen4ExpTextAttention): activation_role = 'attention'
class BF16Residual(BF16Boundary, native.Qwen4ExpTextGatedResidual): activation_role = 'residual'
class BF16MoE(BF16Boundary, native.Qwen4ExpTextSparseMoeBlock): pass
class BF16PLE(BF16Boundary, native.Qwen4ExpTextPLELayer): pass
class BF16RMS(BF16Boundary, native.Qwen4ExpTextRMSNorm): pass
class BF16GatedRMS(BF16Boundary, native.Qwen4ExpTextRMSNormGated): pass
class BF16MLP(BF16Boundary, native.Qwen4ExpTextMLP): pass
class BF16Experts(BF16Boundary, native.Qwen4ExpTextExperts): pass
class BF16LigerRMS(BF16Boundary, LigerRMSNorm): pass
class BF16LigerGatedRMS(BF16Boundary, LigerGatedRMSNorm): pass
class BF16LigerMLP(BF16Boundary, LigerMLP): pass
class BF16LigerExperts(BF16Boundary, LigerExperts): pass

NATIVE_BOUNDARIES = {
    native.Qwen4ExpTextDecoderLayer:BF16Decoder, native.Qwen4ExpTextGatedDeltaNet:BF16GDN,
    native.Qwen4ExpTextAttention:BF16Attention, native.Qwen4ExpTextGatedResidual:BF16Residual,
    native.Qwen4ExpTextSparseMoeBlock:BF16MoE, native.Qwen4ExpTextPLELayer:BF16PLE,
    native.Qwen4ExpTextRMSNorm:BF16RMS, native.Qwen4ExpTextRMSNormGated:BF16GatedRMS,
    native.Qwen4ExpTextMLP:BF16MLP, native.Qwen4ExpTextExperts:BF16Experts,
}
LIGER_BOUNDARIES = {LigerRMSNorm:BF16LigerRMS, LigerGatedRMSNorm:BF16LigerGatedRMS,
                    LigerMLP:BF16LigerMLP, LigerExperts:BF16LigerExperts}
