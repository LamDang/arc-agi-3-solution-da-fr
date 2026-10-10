"""Real CUDA quantized-expert fixture; enabled only for expert-offload captures."""
import os
import unittest


@unittest.skipUnless(os.environ.get('FLASH_NEXT_CHECK_EXPERT_OFFLOAD')=='1','Enabled for frozen expert offload jobs only')
class ExpertOffloadTests(unittest.TestCase):
    def test_quantized_prefetch_preserves_loss_and_fp32_adapter_gradients(self):
        self.check_quantized(0)

    def test_chunked_quantized_experts_recompute_all_gradients(self):
        self.check_quantized(7)

    def test_real_size_split_expert_preserves_forward_and_gradients(self):
        self.check_quantized(8192,tokens=9705,hidden=2560,intermediate=640,experts=1)

    def test_unsplit_uneven_routing_isolates_custom_dispatch(self):
        self.check_quantized(8192,tokens=32,hidden=128,intermediate=64,experts=4,uneven=True)

    def check_quantized(self,chunk_tokens,tokens=32,hidden=128,intermediate=64,experts=None,uneven=False):
        import copy,json
        from pathlib import Path
        import torch
        from torch import nn
        from torch.utils.checkpoint import checkpoint
        from peft import get_peft_model
        from components.adapters import configuration
        from auto_round_extension.triton.qlinear_tritonv2_zp import QuantLinear
        from auto_round.modeling.fused_moe.moe_experts_interface import linear_loop_experts_forward
        from components.common import adopt
        from components.expert_offload import ExpertStage,LayerPrefetch
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__();self.num_experts=experts or (4 if chunk_tokens else 2);self.act_fn=nn.SiLU()
                self.register_parameter('_device_anchor',nn.Parameter(torch.zeros(1),requires_grad=False))
                for index in range(self.num_experts):
                    expert=nn.Module()
                    for name,ins,outs in [('gate_proj',hidden,intermediate),('up_proj',hidden,intermediate),('down_proj',intermediate,hidden)]:
                        projection=QuantLinear(4,128 if tokens>8192 else 32,ins,outs,False)
                        projection.qweight.random_(0,2**30);projection.qzeros.zero_();projection.scales.fill_(.002)
                        if tokens>8192:
                            projection.qweight.random_(-2**31,2**31-1);projection.qzeros.fill_(0x77777777)
                        setattr(expert,name,projection)
                    self.add_module(str(index),expert)
            def forward(self,x,indices,weights):
                return linear_loop_experts_forward(self,x,indices,weights).bfloat16()
        class Staged(ExpertStage,Tiny):pass
        torch.manual_seed(8643)
        rank=16 if tokens>8192 else 4
        canonical=get_peft_model(Tiny(),configuration(r=rank,lora_alpha=2*rank,lora_dropout=0.,target_modules=['gate_proj','up_proj','down_proj'])).get_base_model()
        for p in canonical.parameters():
            if p.requires_grad:
                self.assertEqual(p.dtype,torch.float32)
                with torch.no_grad():p.normal_(0,.002 if tokens>8192 else .01)
        resident=copy.deepcopy(canonical).cuda()
        staged=adopt(canonical.cuda(),Staged);manager=LayerPrefetch([staged]);self.addCleanup(manager.close);staged.chunk_tokens=chunk_tokens
        x=torch.randn(tokens,hidden,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        y=x.detach().clone().requires_grad_(True)
        k=canonical.num_experts
        indices=torch.arange(k,device='cuda').expand(tokens,-1);weights=torch.full((tokens,k),1/k,device='cuda',requires_grad=True);other_weights=weights.detach().clone().requires_grad_()
        if uneven:
            # Expert3 receives no rows; experts0/1/2 receive unequal counts.
            indices=torch.tensor([[0,1],[0,2],[1,2],[0,1]],device='cuda').repeat(tokens//4,1)
            weights=torch.rand(tokens,2,device='cuda');weights=(weights/weights.sum(-1,keepdim=True)).requires_grad_()
            other_weights=weights.detach().clone().requires_grad_()
        cotangent=torch.randn(tokens,hidden,device='cuda',dtype=torch.bfloat16)/tokens
        with torch.autocast('cuda',dtype=torch.bfloat16):
            expected=resident(x,indices,weights);ref_loss=(expected.float()*cotangent.float()).sum()
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                actual=checkpoint(staged,y,indices,other_weights,use_reentrant=False,context_fn=manager.checkpoint_contexts)
                loss=(actual.float()*cotangent.float()).sum()
            loss.backward()
        ref_loss.backward()
        if tokens>8192:
            def relative(a,b):
                return float((a.double()-b.double()).norm()/b.double().norm())
            diagnostic=dict(tokens=tokens,chunk_tokens=chunk_tokens,hidden=hidden,intermediate=intermediate,
                lora_rank=rank,output_bitwise_equal=bool(torch.equal(actual,expected)),
                output_relative_l2=relative(actual,expected),input_gradient_relative_l2=relative(y.grad,x.grad),
                differing_output_elements=int(torch.count_nonzero(actual!=expected)),
                max_output_absolute_difference=float((actual.float()-expected.float()).abs().max()))
            if not diagnostic['output_bitwise_equal']:
                def chunk_projection(projection,values):
                    pieces=[]
                    for start in range(0,tokens,chunk_tokens):
                        local=values[start:start+chunk_tokens];n=local.shape[0]
                        if n<chunk_tokens:local=torch.nn.functional.pad(local,(0,0,0,chunk_tokens-n))
                        pieces.append(projection(local)[:n])
                    return torch.cat(pieces)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    expert=getattr(resident,'0')
                    gate=expert.gate_proj(x);up=expert.up_proj(x)
                    activated=resident.act_fn(gate)*up
                    diagnostic['projection_differences']={}
                    for name,values in [('gate_proj',x),('up_proj',x),('down_proj',activated)]:
                        projection=getattr(expert,name)
                        full=projection(values);part=chunk_projection(projection,values)
                        base=projection.base_layer(values);base_part=chunk_projection(projection.base_layer,values)
                        diagnostic['projection_differences'][name]=dict(output_relative_l2=relative(part,full),
                            output_bitwise_equal=bool(torch.equal(part,full)),base_relative_l2=relative(base_part,base),
                            base_bitwise_equal=bool(torch.equal(base_part,base)))
            if os.environ.get('FLASH_NEXT_EXPERT_CHECK_RESULT'):
                Path(os.environ['FLASH_NEXT_EXPERT_CHECK_RESULT']).with_name('expert-split-check.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
            print('real_size_expert_split_diagnostic='+json.dumps(diagnostic),flush=True)
            self.assertTrue(diagnostic['output_bitwise_equal'],
                'Real-size expert split forward must be bitwise equal before full-model routing')
        if chunk_tokens:
            if uneven:self.assertTrue(torch.equal(actual,expected),'Unsplit native GEMM shapes must preserve forward')
            torch.testing.assert_close(actual,expected,rtol=.03,atol=1e-5)
            torch.testing.assert_close(y.grad,x.grad,rtol=.05,atol=1e-5)
            torch.testing.assert_close(other_weights.grad,weights.grad,rtol=.05,atol=1e-5)
        else:
            self.assertTrue(torch.equal(expected,actual));self.assertTrue(torch.equal(x.grad,y.grad))
            self.assertTrue(torch.equal(other_weights.grad,weights.grad))
        reference=dict(resident.named_parameters());count=0
        grad_pairs=[]
        for name,p in staged.named_parameters():
            if not p.requires_grad:continue
            self.assertEqual(p.device.type,'cuda');self.assertEqual(p.grad.device.type,'cuda');self.assertEqual(p.grad.dtype,torch.float32)
            reference_grad=reference[name].grad
            if reference_grad is None:
                self.assertTrue(uneven and name.startswith('3.'),'Only the unused expert may lack a native gradient')
                reference_grad=torch.zeros_like(p)
            if chunk_tokens:torch.testing.assert_close(p.grad,reference_grad,rtol=.05,atol=1e-5,msg=name)
            else:self.assertTrue(torch.equal(p.grad,reference_grad),name)
            grad_pairs.append((p.grad,reference_grad))
            count+=1
        if chunk_tokens:
            error=sum(float((a.double().cpu()-b.double().cpu()).square().sum()) for a,b in grad_pairs)
            norm=sum(float(b.double().cpu().square().sum()) for a,b in grad_pairs)
            adapter_relative_l2=(error/norm)**.5
            input_relative_l2=float((y.grad.double()-x.grad.double()).norm()/x.grad.double().norm())
            routing_relative_l2=float((other_weights.grad.double()-weights.grad.double()).norm()/weights.grad.double().norm())
            print('expert_gradient_diagnostic='+json.dumps(dict(adapter_relative_l2=adapter_relative_l2,
                input_relative_l2=input_relative_l2,routing_relative_l2=routing_relative_l2)),flush=True)
            self.assertLess(adapter_relative_l2,.01)
        manager.close()
        report=dict(passed=True,quantized_projection=True,loss_bitwise_equal=bool(torch.equal(loss,ref_loss)),
            output_and_input_gradient_bitwise_equal=True,fp32_cuda_adapter_gradient_tensors=count,
            checkpoint_recomputation=True,max_staged_layers=manager.max_staged_layers)
        if not chunk_tokens and os.environ.get('FLASH_NEXT_EXPERT_CHECK_RESULT'):
            Path(os.environ['FLASH_NEXT_EXPERT_CHECK_RESULT']).write_text(json.dumps(report,indent=2)+'\n')
