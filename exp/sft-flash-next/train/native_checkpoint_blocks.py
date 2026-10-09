"""Token-local checkpoint blocks that reuse native HF autograd exactly.

These are candidates for measurement, not claims of numerical equivalence.
No hand-derived backward formula or recurrent state truncation is used.
"""
import torch
from torch.utils.checkpoint import checkpoint


class NativeLayerGroup(torch.nn.Module):
    """Outer checkpoint around unchanged HF layers, including inner checkpoints."""
    def __init__(self,layers):
        super().__init__()
        self.layers=torch.nn.ModuleList(layers)

    def forward(self,hidden_states,**kwargs):
        if kwargs.get('past_key_values') is not None:
            raise ValueError('Checkpoint groups are for cache-free training')
        def run(hidden):
            for layer in self.layers:
                hidden=layer(hidden,**kwargs)
            return hidden
        return checkpoint(run,hidden_states,use_reentrant=False)


def rms_rows(self, x):
    shape=x.shape
    flat=x.reshape(-1,shape[-1])
    outputs=[checkpoint(self.native_forward,part,use_reentrant=False)
             for part in flat.split(self.native_block,0)]
    return torch.cat(outputs,0).reshape(shape)


def gated_rows(self,x,gate):
    shape=x.shape
    outputs=[checkpoint(self.native_forward,a,b,use_reentrant=False)
             for a,b in zip(x.reshape(-1,shape[-1]).split(self.native_block,0),
                            gate.reshape(-1,shape[-1]).split(self.native_block,0))]
    return torch.cat(outputs,0).reshape(shape)


def hyper_rows(self,x):
    shape=x.shape;flat=x.reshape(-1,shape[-1])
    if self.block_inject_weight is None:
        outputs=[checkpoint(self.native_forward,part,use_reentrant=False)
                 for part in flat.split(self.native_block,0)]
        return torch.cat(outputs,0).reshape(*shape[:-1],self.hidden_size)
    def local(part):
        mixed,_,injection=self.native_forward(part)
        return mixed,injection
    outputs=[checkpoint(local,part,use_reentrant=False)
             for part in flat.split(self.native_block,0)]
    return (torch.cat([o[0] for o in outputs],0).reshape(*shape[:-1],self.hidden_size),
            x,torch.cat([o[1] for o in outputs],0).reshape(*shape[:-1],self.hc_count))
