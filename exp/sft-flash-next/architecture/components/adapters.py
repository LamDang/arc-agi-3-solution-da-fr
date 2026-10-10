"""Per-config PEFT integration for frozen AutoRound projections; no global patch."""
import torch
from peft import LoraConfig
from peft.tuners.lora.layer import Linear


class QuantizedLoRALinear(Linear):
    """Use PEFT's standard LoRA forward around the native quantized projection."""
    def _get_in_out_features(self,module):
        return module.infeatures,module.outfeatures

    def _move_adapter_to_device_of_base_layer(self,adapter_name,device=None):
        device=self.get_base_layer().qweight.device if device is None else device
        for name in ('lora_A','lora_B'):
            layers=getattr(self,name)
            if adapter_name in layers:layers[adapter_name].to(device=device,dtype=torch.float32)

    def merge(self,*args,**kwargs):
        raise ValueError('Merging into packed frozen expert weights requires separate requantization')


def configuration(**kwargs):
    from auto_round_extension.triton.qlinear_tritonv2_zp import QuantLinear
    config=LoraConfig(**kwargs)
    config._register_custom_module({QuantLinear:QuantizedLoRALinear})
    return config
