"""Capture clean PEFT default A/zero-B before exposure to any trajectory data."""
import argparse
from pathlib import Path
import torch
from peft import get_peft_model_state_dict
from dataset import file_hash, write_json
from trajectory_run import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=20261009)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(args.seed)
    model = load_model(args.model,None)
    state = {n:v.detach().cpu().clone() for n,v in get_peft_model_state_dict(model).items()}
    if any(torch.count_nonzero(v) for n,v in state.items() if 'lora_B' in n):
        raise ValueError('PEFT initialization was not zero-B')
    torch.save(state,out/'initial-adapter.pt')
    write_json(out/'provenance.json',dict(purpose='clean-production-initialization',
        optimizer_updates=0,validation_exposed=False,data_loaded=False,seed=args.seed,
        model_config_sha256=file_hash(Path(args.model)/'config.json'),
        adapter_sha256=file_hash(out/'initial-adapter.pt'),
        script_sha256=file_hash(__file__)))


if __name__=='__main__':
    main()
