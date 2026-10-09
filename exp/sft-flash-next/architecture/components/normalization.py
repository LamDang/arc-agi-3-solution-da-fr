"""Liger normalization components; no changes to native classes or factories."""
from transformers.models.qwen4_exp import modeling_qwen4_exp as native
from transformers.activations import ACT2FN
import torch


class LigerRMSNorm(native.Qwen4ExpTextRMSNorm):
    def forward(self, x):
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction
        if self.group_size is None:
            return LigerRMSNormFunction.apply(x, self.weight, self.eps, 1.0, 'gemma', False)
        shaped = x.reshape(*x.shape[:-1], -1, self.group_size)
        weights = self.weight.reshape(-1, self.group_size)
        output = [LigerRMSNormFunction.apply(shaped[..., i, :], weights[i], self.eps, 1.0, 'gemma', False)
                  for i in range(weights.shape[0])]
        return torch.stack(output, dim=-2).reshape_as(x)


class LigerGatedRMSNorm(native.Qwen4ExpTextRMSNormGated):
    def forward(self, hidden_states, gate):
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction
        normalized = LigerRMSNormFunction.apply(hidden_states, self.weight,
            self.variance_epsilon, 0.0, 'llama', False)
        return (normalized * ACT2FN[self.activation](gate.float())).to(hidden_states.dtype)
