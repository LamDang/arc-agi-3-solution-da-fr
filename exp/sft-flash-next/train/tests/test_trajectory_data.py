import copy
import pytest

from trajectory_data import reconstruct, plan_token_updates
import trajectory_encode


def fixtures():
    system={'role':'system','content':'rules'}
    user={'role':'user','content':'initial frame'}
    a={'role':'assistant','reasoning_content':'historical','tool_calls':[{'id':'a0','function':{'name':'python','arguments':{'code':'act()'}}}]}
    obs={'role':'tool','tool_call_id':'a0','content':'actual observation'}
    b={'role':'assistant','reasoning_content':'other','content':'done'}
    rows=[dict(game='g',request_index=0,messages=[system,user,a]),
          dict(game='g',request_index=1,messages=copy.deepcopy([system,a,obs,b]))]
    generated={f'g_p0#{i}':dict(status='ok',thinking=f'generated {i}') for i in range(2)}
    return rows,generated


def test_trimmed_history_reconstructs_exact_chronology_and_replaces_every_thought():
    rows,generated=fixtures()
    result=reconstruct(rows,generated)
    assert [m['role'] for m in result['messages']]==['system','user','assistant','tool','assistant']
    assert [m['reasoning_content'] for m in result['messages'] if m['role']=='assistant']==['generated 0','generated 1']
    assert result['messages'][3]['content']=='actual observation'
    assert [s['source_id'] for s in result['source_turns']]==['g_p0#0','g_p0#1']
    assert rows[0]['messages'][-1]['reasoning_content']=='historical'


@pytest.mark.parametrize('mutation',['gap','observation','code','thinking','unowned'])
def test_incomplete_or_unverified_sources_fail_closed(mutation):
    rows,generated=fixtures()
    if mutation=='gap': rows[1]['request_index']=2
    if mutation=='observation': rows[1]['messages'][1]['role']='tool'
    if mutation=='code': rows[1]['messages'][1]['tool_calls'][0]['function']['arguments']['code']='invented()'
    if mutation=='thinking': generated.pop('g_p0#1')
    if mutation=='unowned': rows[1]['messages'].insert(-1,{'role':'assistant','content':'unowned'})
    with pytest.raises(ValueError): reconstruct(rows,generated)


def test_planner_exact_once_counts_overshoot_and_tail():
    rows=[dict(sample_id=str(i),target_tokens=n) for i,n in enumerate([6,7,25,3])]
    plans=plan_token_updates(rows,10)
    assert [p['target_tokens'] for p in plans]==[13,25,3]
    assert [p['overshoot_tokens'] for p in plans]==[3,15,0]
    assert [p['tail'] for p in plans]==[False,False,True]
    assert [sid for p in plans for sid in p['sample_ids']]==['0','1','2','3']
    assert plans==plan_token_updates(rows,10)
    with pytest.raises(ValueError): plan_token_updates(rows,0)


class IDs:
    def __init__(self,values): self.values=values
    def __getitem__(self,index): return self
    def tolist(self): return self.values


def test_all_assistant_positions_ignore_intervening_observations(monkeypatch):
    sample=dict(chat_template_kwargs={'preserve_thinking':True},messages=[
        {'role':'user'},{'role':'assistant','reasoning_content':'a'},
        {'role':'tool'},{'role':'assistant','reasoning_content':'b'}])
    def fake_encode(processor,prefix):
        n=4 if len(prefix['messages'])==2 else 9
        positions=[2,3] if n==4 else [7,8]
        return {'input_ids':IDs(list(range(n)))},dict(positions=positions,target_ids=positions,
            target_tokens=2,labels=[0,0],category_counts={'thinking':1,'tool_code':1})
    monkeypatch.setattr(trajectory_encode,'encode_reply',fake_encode)
    enc,annotation=trajectory_encode.encode(None,sample)
    assert annotation['positions']==[2,3,7,8]
    assert annotation['target_ids']==[2,3,7,8]
    assert annotation['target_tokens']==4
    assert annotation['total_tokens']==9
    assert annotation['non_target_tokens']==5
    assert 'prompt_tokens' not in annotation
    assert annotation['category_counts']['thinking']==2


@pytest.mark.parametrize('mutation',['boolean','duplicate','tools','template','system','initial-system','call-id'])
def test_identity_and_conditioning_are_immutable(mutation):
    rows,generated=fixtures()
    if mutation=='boolean': rows[0]['request_index']=False
    if mutation=='duplicate': rows[1]['request_index']=0
    if mutation=='tools': rows[1]['tools']=[{'changed':True}]
    if mutation=='template': rows[1]['chat_template_kwargs']={'preserve_thinking':False}
    if mutation=='system': rows[1]['messages'][0]['content']='changed'
    if mutation=='initial-system': rows[0]['messages'][0]['role']='user'
    if mutation=='call-id': rows[1]['messages'][-1]['tool_calls']=copy.deepcopy(rows[0]['messages'][-1]['tool_calls'])
    with pytest.raises(ValueError): reconstruct(rows,generated)


def test_real_processor_two_assistant_turns_with_image():
    import base64, io
    from pathlib import Path
    from PIL import Image
    from dataset import load_processor
    processor_path=Path('/workspace/sft-validation-inputs/processor')
    if not processor_path.exists(): pytest.skip('Pinned processor assets unavailable')
    processor=load_processor(processor_path)
    image=Image.new('RGB',(32,32),'red'); stream=io.BytesIO();image.save(stream,format='PNG')
    sample=dict(chat_template_kwargs={'preserve_thinking':True},messages=[
        {'role':'system','content':'You solve puzzles.'},
        {'role':'user','content':'begin'},
        {'role':'assistant','reasoning_content':'First reasoning','content':'First answer'},
        {'role':'user','content':[{'type':'text','text':'Observation'},
         {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()}}]},
        {'role':'assistant','reasoning_content':'Second reasoning','content':'Final answer'}])
    enc,annotation=trajectory_encode.encode(processor,sample)
    ids=enc['input_ids'][0].tolist()
    assert annotation['images']==1
    assert len(annotation['assistant_turns'])==2
    assert annotation['target_ids']==[ids[p] for p in annotation['positions']]
    assert annotation['category_counts']['thinking']>0
    a,b=annotation['assistant_turns']
    assert a['positions'][-1]+1 < b['positions'][0]
    import torch
    labels=torch.full_like(enc['input_ids'],-100)
    labels[0,annotation['positions']]=enc['input_ids'][0,annotation['positions']]
    assert labels[0,1:].ne(-100).nonzero().flatten().tolist()==[p-1 for p in annotation['positions']]
    sample['chat_template_kwargs']['preserve_thinking']=False
    with pytest.raises(ValueError): trajectory_encode.encode(processor,sample)
    sample['chat_template_kwargs']['preserve_thinking']=True
    processor.chat_template=processor.chat_template.replace('set reasoning_content = message.reasoning_content', "set reasoning_content = ''")
    with pytest.raises(ValueError): trajectory_encode.encode(processor,sample)


def test_preparation_split_manifest_corruption_and_atomic_failure(tmp_path,monkeypatch):
    import argparse,json
    import prepare_trajectories as prep
    import trajectory_data
    from dataset import digest
    processor=tmp_path/'processor';processor.mkdir();(processor/'config.json').write_text('{}')
    folds=tmp_path/'folds.json';folds.write_text(json.dumps({'folds':[{'fold':0,'game_ids':['v']},{'fold':1,'game_ids':['g']}]}))
    source=tmp_path/'input.jsonl'; generated_dir=tmp_path/'generated';generated_dir.mkdir()
    rows,generated=fixtures()
    rows += [dict(copy.deepcopy(rows[0]),game='v')]
    for row in rows: row.update(chat_template_kwargs={'preserve_thinking':True},tools=[{'function':{'name':'python','parameters':{'properties':{'code':{}},'required':['code']}}}])
    source.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    generated['v_p0#0']=dict(status='ok',thinking='v generated')
    generated_dir.joinpath('thoughts.jsonl').write_text('\n'.join(json.dumps(dict(v,key=k)) for k,v in generated.items()))
    def fake_encode(processor,sample):
        return {},dict(total_tokens=9,target_tokens=4,images=0,positions=[2,3,7,8],
            positions_sha256=digest([2,3,7,8]),category_counts={'thinking':2},input_sha256=digest(list(range(9))),target_sha256='target',target_ids=[2,3,7,8],
            vision_sha256={'pixel_values':None,'image_grid_thw':None})
    monkeypatch.setattr(prep,'load_processor',lambda p:None)
    monkeypatch.setattr(prep,'encode',fake_encode)
    import torch
    monkeypatch.setattr(prep,'encode_input',lambda p,s:{'input_ids':torch.arange(9).reshape(1,-1)})
    monkeypatch.setattr(trajectory_data,'load_processor',lambda p:None)
    monkeypatch.setattr(trajectory_data,'encode_input',lambda processor,sample:{'input_ids':IDs(list(range(9)))})
    args=argparse.Namespace(input=str(source),generated_dir=str(generated_dir),processor=str(processor),
        folds=str(folds),validation_fold=0,max_tokens=10,token_budget=5,out=str(tmp_path/'out'))
    manifest=prep.prepare(args)
    assert {r['game']:r['split'] for r in manifest['rows']}=={'g':'train','v':'validation'}
    with pytest.raises(ValueError,match='complete original logs'): trajectory_data.Trajectories(args.out)
    manifest['source_completeness']='complete original logs with verified game completion'
    manifest['sha256']=digest({k:v for k,v in manifest.items() if k!='sha256'})
    (tmp_path/'out'/'manifest.json').write_text(json.dumps(manifest))
    loader=trajectory_data.Trajectories(args.out)
    assert loader.get(loader.rows[0])[1]['positions']==[2,3,7,8]
    monkeypatch.setattr(trajectory_data,'encode_input',lambda processor,sample:{'input_ids':IDs(list(range(9))), 'pixel_values':torch.ones(1,3)})
    with pytest.raises(ValueError,match='vision processor'): loader.get(loader.rows[0])
    target=tmp_path/'out'/'trajectories.jsonl';target.write_bytes(target.read_bytes()+b' ')
    with pytest.raises(ValueError,match='file changed'): trajectory_data.Trajectories(args.out)
    args.out=str(tmp_path/'failed');args.max_tokens=8
    with pytest.raises(ValueError,match='blocked'):prep.prepare(args)
    assert not (tmp_path/'failed').exists()
    assert len(json.loads((tmp_path/'failed.audit.json').read_text())['failed_games'])==2
    assert not list(tmp_path.glob('failed.preparing-*'))


def test_vision_hashes_bind_pixels_grid_shape_and_dtype():
    import torch
    from trajectory_encode import vision_hashes
    enc={'pixel_values':torch.zeros(2,3),'image_grid_thw':torch.tensor([[1,2,3]])}
    original=vision_hashes(enc)
    enc['pixel_values'][0,0]=1
    assert vision_hashes(enc)['pixel_values'] != original['pixel_values']
    enc['image_grid_thw'][0,1]=4
    assert vision_hashes(enc)['image_grid_thw'] != original['image_grid_thw']
    assert vision_hashes({'pixel_values':torch.zeros(3,2)})['pixel_values'] != original['pixel_values']
    assert vision_hashes({'pixel_values':torch.zeros(2,3,dtype=torch.float64)})['pixel_values'] != original['pixel_values']
