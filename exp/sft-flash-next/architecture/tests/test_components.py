"""Component state/dispatch isolation and numerical boundary tests with real HF classes."""
import importlib.util
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
HAS_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH,'Torch is available in the pinned Kaggle runtime')
class ComponentsTests(unittest.TestCase):
    def test_optimized_off_changes_no_modules_parameters_or_global_factories(self):
        import torch,peft
        from transformers.models.qwen4_exp import modeling_qwen4_exp as native
        from model import compose
        from config import Optimizations
        root = torch.nn.Module();root.norm = native.Qwen4ExpTextRMSNorm(16,group_size=4)
        before = root.norm;factory = peft.get_peft_model;forward = type(before).forward
        values = torch.randn(2,3,16)
        expected = before(values)
        self.assertEqual(compose(root,Optimizations()),[])
        self.assertIs(root.norm,before);self.assertIs(peft.get_peft_model,factory)
        self.assertIs(type(before).forward,forward)
        self.assertTrue(torch.equal(root.norm(values),expected))

    def test_bf16_component_keeps_parameter_objects_and_checkpoint_names(self):
        import torch
        from transformers.models.qwen4_exp import modeling_qwen4_exp as native
        from model import compose
        from config import Optimizations
        root = torch.nn.Module();root.norm = native.Qwen4ExpTextRMSNorm(16,group_size=4).bfloat16()
        with torch.no_grad():root.norm.weight.uniform_(-.2,.2)
        old = root.norm;keys = list(root.state_dict());parameter = old.weight
        values = torch.randn(2,3,16,dtype=torch.float32,requires_grad=True)
        ref = old(values.bfloat16())
        compose(root,Optimizations(bf16_activations=True))
        actual = root.norm(values)
        self.assertEqual(list(root.state_dict()),keys);self.assertIs(root.norm.weight,parameter)
        self.assertTrue(torch.equal(actual,ref));self.assertEqual(actual.dtype,torch.bfloat16)
        grad = torch.randn_like(actual)
        a = torch.autograd.grad(actual,values,grad,retain_graph=True)[0]
        b = torch.autograd.grad(ref,values,grad)[0]
        self.assertTrue(torch.equal(a,b))
        self.assertIs(type(old),native.Qwen4ExpTextRMSNorm)

    def test_interleaved_target_selection_and_native_ce_gradient(self):
        import torch
        from components.head import target_positions,target_logits,target_cross_entropy
        hidden = torch.randn(1,7,8,requires_grad=True);weight = torch.randn(11,8)
        labels = torch.tensor([[-100,2,-100,4,5,-100,7]])
        positions,targets = target_positions(labels)
        self.assertEqual(positions.tolist(),[0,2,3,5])
        selected,_ = target_logits(hidden,weight,labels)
        candidate = target_cross_entropy(selected,labels,positions,targets)
        logits = torch.nn.functional.linear(hidden,weight).float()
        shift = torch.nn.functional.pad(labels,(0,1),value=-100)[...,1:]
        reference = torch.nn.functional.cross_entropy(logits.reshape(-1,11),shift.reshape(-1),ignore_index=-100)
        self.assertTrue(torch.equal(candidate,reference))
        a = torch.autograd.grad(candidate,hidden,retain_graph=True)[0]
        b = torch.autograd.grad(reference,hidden)[0]
        self.assertTrue(torch.equal(a,b))

    def test_bf16_residual_preserves_fp32_auxiliary_coefficient(self):
        import torch
        from components.precision import BF16Boundary
        class Original(torch.nn.Module):
            def forward(self,x):return x.float(),x.float(),torch.tensor([.123456789],dtype=torch.float32)
        class Boundary(BF16Boundary,Original):activation_role='residual'
        result = Boundary()(torch.randn(2,3))
        self.assertEqual(result[0].dtype,torch.bfloat16);self.assertEqual(result[1].dtype,torch.bfloat16)
        self.assertEqual(result[2].dtype,torch.float32)
        self.assertTrue(torch.equal(result[2],Original()(torch.zeros(2,3))[2]))

if __name__ == '__main__':unittest.main()
