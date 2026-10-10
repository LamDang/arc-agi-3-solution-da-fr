"""Full-sequence indexed attention versus checkpointed token-local projections."""
import torch


def check():
    import backend
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(91)
        config = backend.rm.mq.Qwen4ExpTextConfig(hidden_size=2560,num_attention_heads=24,
            num_key_value_heads=2,head_dim=256,indexer_n_heads=4,indexer_kv_heads=1,
            indexer_head_dim=128,indexer_budget=128,indexer_compress_ratio=4,
            num_hidden_layers=1,layer_types=['full_attention'],
            rope_parameters={'rope_type':'default','rope_theta':10000000,'partial_rotary_factor':.25})
        module = backend.rm.mq.Qwen4ExpTextAttention(config,0).to(device='cuda',dtype=torch.bfloat16)
        module.requires_grad_(False)
        for name in ('q_proj','k_proj','v_proj','o_proj'):
            setattr(module,name,backend.LoRALinear(getattr(module,name),rank=16,alpha=32))
        with torch.no_grad():
            for name,p in module.named_parameters():
                if p.requires_grad:p.normal_(0,.003)
        module.query_block,module.index_block,module.key_block = 16,256,32
        module.train_attention_backend = 'triton'
        tokens,size = 2049,512
        x = torch.randn(1,tokens,2560,device='cuda',dtype=torch.bfloat16)
        angle = torch.randn(1,tokens,32,device='cuda')
        angle = torch.cat((angle,angle),dim=-1)
        positions = angle.cos().bfloat16(),angle.sin().bfloat16()
        selected = [0,1,size,size+1,tokens-2,tokens-1]
        weights = torch.randn(1,len(selected),2560,device='cuda')
        parameters = [(n,p) for n,p in module.named_parameters() if p.requires_grad]
        def run(block):
            module.train_projection_block = block
            local = x.detach().clone().requires_grad_(True)
            out,_ = backend.training_attention(module,local,positions)
            gradients = torch.autograd.grad((out[:,selected].float()*weights).mean(),[local,*[p for _,p in parameters]])
            return [out.detach(),*[g.detach() for g in gradients]]
        ref,got = run(0),run(size)
        errors = [float((a.float()-r.float()).norm()/r.float().norm().clamp_min(1e-12)) for a,r in zip(got,ref)]
        if not all(torch.isfinite(t).all() for t in got) or max(errors) > .01:
            raise RuntimeError(f'Attention projection block parity failed: {errors}')
        return dict(tokens=tokens,block_tokens=size,query_heads=24,kv_heads=2,head_dim=256,
            output_relative_error=errors[0],input_gradient_relative_error=errors[1],
            adapter_gradient_relative_errors=dict(zip([n for n,_ in parameters],errors[2:])),
            reference='Same full-sequence indexed attention; independent unblocked projections/gating/output',
            tolerance=.01)
