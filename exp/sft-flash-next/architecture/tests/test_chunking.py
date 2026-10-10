"""Chunk-boundary, full-context and overlap VJP checks in the pinned runtime."""
import importlib.util
import unittest
HAS_TORCH=importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH,'Torch available in pinned runtime')
class ChunkingTests(unittest.TestCase):
    def test_production_size_native_route_and_unroute(self):
        import torch
        from components.expert_chunks import route_sum_native,unroute_native
        if not torch.cuda.is_available():self.skipTest('Production-size CUDA launch fixture')
        previous=torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            torch.manual_seed(552)
            tokens,k,hidden,size=16249,10,2560,8192
            values=torch.randn(tokens,k,hidden,device='cuda',dtype=torch.bfloat16)
            weights=torch.rand(tokens,k,device='cuda',dtype=torch.bfloat16)
            expected=(values*weights[:,:,None]).sum(dim=1)
            actual=route_sum_native(values.cpu(),weights,'cuda',size)
            self.assertTrue(torch.equal(actual,expected))
            del expected,actual,weights
            # Alternating signs exercise cancellation across repeated gathers.
            values[:,1::2].neg_()
            source=torch.zeros(tokens,hidden,device='cuda',dtype=torch.bfloat16,requires_grad=True)
            rows=torch.arange(tokens,device='cuda')[:,None].expand(-1,k).reshape(-1)
            expected=torch.autograd.grad(source[rows],source,values.flatten(0,1))[0]
            actual=unroute_native(values.cpu(),'cuda',size)
            self.assertTrue(torch.equal(actual,expected))
        finally:torch.use_deterministic_algorithms(previous)

    def close_gradients(self,actual,expected):
        import torch
        aa=sum(float(x.double().square().sum()) for x in expected)
        ee=sum(float((x.double()-y.double()).square().sum()) for x,y in zip(actual,expected))
        self.assertLess((ee/aa)**.5, .01)
        self.assertTrue(all(bool(torch.isfinite(x).all()) for x in actual))

    def test_route_windows_preserve_native_topk_reduction_bitwise(self):
        import torch
        from components.expert_chunks import route_sum_native
        device='cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(552)
        for tokens in (6,7,8,23):
            values=torch.randn(tokens,10,2560,device=device,dtype=torch.bfloat16)
            weights=torch.rand(tokens,10,device=device,dtype=torch.bfloat16)
            with torch.autocast(device,dtype=torch.bfloat16):
                expected=(values*weights[:,:,None]).sum(dim=1).bfloat16()
                actual=route_sum_native(values.cpu(),weights,device,7)
            self.assertTrue(torch.equal(actual,expected))

    def test_unroute_windows_preserve_native_gather_backward_bitwise(self):
        import torch
        from components.expert_chunks import unroute_native
        device='cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(419)
        for tokens in (6,7,8,23):
            x=torch.zeros((tokens,2560),device=device,dtype=torch.bfloat16,requires_grad=True)
            rows=torch.arange(tokens,device=device).unsqueeze(1).expand(-1,10).reshape(-1)
            slot_grad=torch.randn(tokens,10,2560,device=device,dtype=torch.bfloat16)
            expected=torch.autograd.grad(x[rows],x,slot_grad.flatten(0,1))[0]
            actual=unroute_native(slot_grad.cpu(),device,7)
            self.assertTrue(torch.equal(actual,expected))

    def test_qsa_windows_preserve_selection_full_kv_and_gradients(self):
        import copy,torch
        from types import SimpleNamespace
        from components.common import adopt,replace_components
        from components.precision import BF16Attention,NATIVE_BOUNDARIES
        from components.attention import DirectBiasIndexer,LazyCausalMask
        from components.qsa_chunks import ChunkedAttention,WindowIndexer
        from peft import LoraConfig,get_peft_model
        from transformers.models.qwen4_exp import modeling_qwen4_exp as native
        device='cuda' if torch.cuda.is_available() else 'cpu'
        cfg=SimpleNamespace(hidden_size=32,num_attention_heads=2,num_key_value_heads=1,head_dim=16,
            attention_dropout=0.,attention_bias=False,rms_norm_eps=1e-6,_attn_implementation='sdpa',
            indexer_n_heads=2,indexer_kv_heads=1,indexer_head_dim=16,indexer_budget=8,indexer_compress_ratio=4)
        torch.manual_seed(737)
        original=native.Qwen4ExpTextAttention(cfg,0).to(device=device,dtype=torch.bfloat16)
        replace_components(original,lambda n,m:NATIVE_BOUNDARIES.get(type(m)),[])
        original=adopt(original,BF16Attention)
        original.indexer=adopt(original.indexer,DirectBiasIndexer)
        original=get_peft_model(original,LoraConfig(r=4,lora_alpha=8,lora_dropout=0.,target_modules=['q_proj','k_proj','v_proj','o_proj'])).get_base_model()
        for p in original.parameters():
            if p.requires_grad:
                with torch.no_grad():p.normal_(0,.01)
        candidate=adopt(copy.deepcopy(original),ChunkedAttention);candidate.chunk_tokens=7
        candidate.indexer=adopt(candidate.indexer,WindowIndexer)
        for tokens in (6,7,8,23):
            x=torch.randn(1,tokens,32,device=device,dtype=torch.bfloat16,requires_grad=True)
            y=x.detach().clone().requires_grad_();angle=torch.randn(1,tokens,16,device=device,dtype=torch.bfloat16)
            pos=(angle.cos(),angle.sin());mask=LazyCausalMask(tokens,device)
            with torch.autocast(device,dtype=torch.bfloat16):
                full=original.indexer(x,pos,mask,None)
                iq,ik=candidate.indexer.project(y,pos)
                windows=[candidate.indexer.bias(iq[:,s:s+7],ik,pos,s,x.dtype) for s in range(0,tokens,7)]
                self.assertTrue(torch.equal(torch.cat(windows,dim=2),full))
                expected=original(x,pos,mask)[0];actual=candidate(y,pos,mask)[0]
                torch.testing.assert_close(actual,expected,rtol=.01,atol=.002)
                probe=torch.randn_like(expected)
                a=torch.autograd.grad(actual,(y,*[p for p in candidate.parameters() if p.requires_grad]),probe)
                b=torch.autograd.grad(expected,(x,*[p for p in original.parameters() if p.requires_grad]),probe)
                self.close_gradients(a,b)
            self.assertLessEqual(candidate.chunk_stats['max_query_tokens'],7)
            self.assertEqual(candidate.chunk_stats['key_tokens'],tokens)

    def test_hyperconnection_mixing_and_injection_gradients(self):
        import copy,torch
        from types import SimpleNamespace
        from components.common import adopt,replace_components
        from components.precision import BF16Residual,NATIVE_BOUNDARIES
        from components.hyperconnection_chunks import ChunkedResidual,inject
        from transformers.models.qwen4_exp import modeling_qwen4_exp as native
        from torch.utils.checkpoint import checkpoint
        device='cuda' if torch.cuda.is_available() else 'cpu'
        cfg=SimpleNamespace(hidden_size=16,hc_count=4,hc_lowrank=8,rms_norm_eps=1e-6)
        torch.manual_seed(914)
        for combine in (False,True):
            ref=native.Qwen4ExpTextGatedResidual(cfg,use_combine=combine).to(device=device,dtype=torch.bfloat16)
            replace_components(ref,lambda n,m:NATIVE_BOUNDARIES.get(type(m)),[])
            ref=adopt(ref,BF16Residual).requires_grad_(False)
            new=adopt(copy.deepcopy(ref),ChunkedResidual);new.chunk_tokens=7
            for tokens in (6,7,8,23):
                x=torch.randn(1,tokens,64,device=device,dtype=torch.bfloat16,requires_grad=True)
                y=x.detach().clone().requires_grad_()
                with torch.autocast(device,dtype=torch.bfloat16):
                    a=ref(x);b=new(y)
                    if combine:
                        self.assertTrue(all(torch.equal(u,v) for u,v in zip(a,b)))
                        cotangents=tuple(torch.randn_like(v) for v in a)
                        self.close_gradients(torch.autograd.grad(b,y,cotangents,retain_graph=True),
                            torch.autograd.grad(a,x,cotangents,retain_graph=True))
                        # Independent block with trainable input: mixing/injection VJPs.
                        block=torch.randn_like(a[0],requires_grad=True);other=block.detach().clone().requires_grad_()
                        expected=(a[1]+(block.unsqueeze(-2)*a[2].unsqueeze(-1)).flatten(-2)).bfloat16()
                        actual=torch.cat([checkpoint(inject,other[:,s:s+7],b[1][:,s:s+7],b[2][:,s:s+7],use_reentrant=False) for s in range(0,tokens,7)],dim=1)
                        aa=(y,other);bb=(x,block)
                    else:expected=a;actual=b;aa=(y,);bb=(x,)
                    torch.testing.assert_close(actual,expected,rtol=.01,atol=.002)
                    probe=torch.randn_like(expected)
                    self.close_gradients(torch.autograd.grad(actual,aa,probe),torch.autograd.grad(expected,bb,probe))

    def test_ple_windows_use_full_ngram_payload_and_sum_halo_gradients(self):
        import copy,torch
        from types import SimpleNamespace
        from components.common import adopt,replace_components
        from components.precision import BF16PLE,NATIVE_BOUNDARIES
        from components.ple_chunks import ChunkedPLE
        from components.ple import PreparedNGramEmbedding
        from transformers.models.qwen4_exp import modeling_qwen4_exp as native
        device='cuda' if torch.cuda.is_available() else 'cpu'
        cfg=SimpleNamespace(hidden_size=8,hc_count=4,ple_embed_dim=16,ngram_size=3,heads_per_ngram=2,
            vocab_size=31,ngram_vocab_size_base=37,seed=42,eos_token_id=0,make_ngram_vocab_size_divisible_by=8,
            ple_conv_kernel_size=4,rms_norm_eps=1e-6)
        torch.manual_seed(513)
        ref=native.Qwen4ExpTextPLELayer(cfg,1,0).to(device=device,dtype=torch.bfloat16)
        replace_components(ref,lambda n,m:NATIVE_BOUNDARIES.get(type(m)),[])
        ref=adopt(ref,BF16PLE).requires_grad_(False)
        new=adopt(copy.deepcopy(ref),ChunkedPLE);new.chunk_tokens=7
        new.ple_embedding=adopt(new.ple_embedding,PreparedNGramEmbedding)
        for tokens in (6,7,8,23):
            ids=torch.randint(1,31,(1,tokens),device=device);ids[:,5]=0
            with torch.no_grad():payload=ref.ple_embedding(ids,None).cpu()
            new.ple_embedding.prepared_payload=payload;new.ple_embedding.prepared_input_ids=ids.cpu()
            x=torch.randn(1,tokens,32,device=device,dtype=torch.bfloat16,requires_grad=True)
            y=x.detach().clone().requires_grad_()
            with torch.autocast(device,dtype=torch.bfloat16):
                expected=ref(x,ids,None);actual=new(y,ids,None)
                torch.testing.assert_close(actual,expected,rtol=.01,atol=.002)
                probe=torch.randn_like(expected)
                a=torch.autograd.grad(actual,y,probe)[0];b=torch.autograd.grad(expected,x,probe)[0]
                self.close_gradients((a,),(b,))
            self.assertEqual(new.chunk_stats['halo_tokens'],9)
            self.assertLessEqual(new.chunk_stats['max_window_tokens'],16)
