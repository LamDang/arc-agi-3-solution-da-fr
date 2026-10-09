"""Real CUDA quantized-expert fixture; enabled only for expert-offload captures."""
import os
import unittest


@unittest.skipUnless(os.environ.get('FLASH_NEXT_CHECK_EXPERT_OFFLOAD')=='1','Enabled for CPU expert reference jobs only')
class ExpertOffloadTests(unittest.TestCase):
    def test_quantized_prefetch_preserves_loss_and_fp32_adapter_gradients(self):
        import copy,json
        from pathlib import Path
        import torch
        from torch import nn
        from torch.utils.checkpoint import checkpoint
        from peft import LoraConfig,get_peft_model
        from auto_round_extension.triton.qlinear_tritonv2_zp import QuantLinear
        from auto_round.modeling.fused_moe.moe_experts_interface import linear_loop_experts_forward
        from components.common import adopt
        from components.expert_offload import ExpertStage,LayerPrefetch
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__();self.num_experts=2;self.act_fn=nn.SiLU()
                self.register_parameter('_device_anchor',nn.Parameter(torch.zeros(1),requires_grad=False))
                for index in range(2):
                    expert=nn.Module()
                    for name,ins,outs in [('gate_proj',128,64),('up_proj',128,64),('down_proj',64,128)]:
                        projection=QuantLinear(4,32,ins,outs,False)
                        projection.qweight.random_(0,2**30);projection.qzeros.zero_();projection.scales.fill_(.002)
                        setattr(expert,name,projection)
                    self.add_module(str(index),expert)
            def forward(self,x,indices,weights):
                return linear_loop_experts_forward(self,x,indices,weights).bfloat16()
        class Staged(ExpertStage,Tiny):pass
        torch.manual_seed(8643)
        canonical=get_peft_model(Tiny(),LoraConfig(r=4,lora_alpha=8,lora_dropout=0.,target_modules=['gate_proj','up_proj','down_proj'])).get_base_model()
        for p in canonical.parameters():
            if p.requires_grad:
                self.assertEqual(p.dtype,torch.float32)
                with torch.no_grad():p.normal_(0,.01)
        resident=copy.deepcopy(canonical).cuda()
        staged=adopt(canonical,Staged);manager=LayerPrefetch([staged])
        x=torch.randn(32,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        y=x.detach().clone().requires_grad_(True)
        indices=torch.tensor([[0,1]]*32,device='cuda');weights=torch.full((32,2),.5,device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            expected=resident(x,indices,weights);ref_loss=expected.float().square().mean()
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                actual=checkpoint(staged,y,indices,weights,use_reentrant=False,context_fn=manager.checkpoint_contexts)
                loss=actual.float().square().mean()
            loss.backward()
        ref_loss.backward()
        self.assertTrue(torch.equal(expected,actual));self.assertTrue(torch.equal(x.grad,y.grad))
        reference=dict(resident.named_parameters());count=0
        for name,p in staged.named_parameters():
            if not p.requires_grad:continue
            self.assertEqual(p.device.type,'cpu');self.assertEqual(p.grad.dtype,torch.float32)
            self.assertTrue(torch.equal(p.grad,reference[name].grad.cpu()),name);count+=1
        manager.close()
        report=dict(passed=True,quantized_projection=True,loss_bitwise_equal=bool(torch.equal(loss,ref_loss)),
            output_and_input_gradient_bitwise_equal=True,fp32_cpu_adapter_gradient_tensors=count,
            checkpoint_recomputation=True,max_staged_layers=manager.max_staged_layers)
        if os.environ.get('FLASH_NEXT_EXPERT_CHECK_RESULT'):
            Path(os.environ['FLASH_NEXT_EXPERT_CHECK_RESULT']).write_text(json.dumps(report,indent=2)+'\n')
