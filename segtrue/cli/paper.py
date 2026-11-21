"""Portable training and FP32 evaluation for the archived paper protocol."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import cv2
import torch

from segtrue.constants import PROJECT_ROOT
from segtrue.data.datasets import DatasetSpec, SegmentationDataset, discover_samples, filter_samples_by_image_names
from segtrue.data.loaders import LoaderConfig, build_dataloader
from segtrue.models.factory import build_model
from segtrue.preprocess.registry import INPUT_CONFIG_REGISTRY
from segtrue.analysis.paper_eval import SeedInputs, CONDITIONS, evaluate_planes, scheme_channel_indices, rule_key
from segtrue.engine.losses import build_loss
from segtrue.engine.trainer import TrainerConfig, train_run
from segtrue.utils.seed import set_global_seed

SEEDS10=[42,2024,12345,7,123,456,789,1000,2000,3000]
SEEDS5=SEEDS10[:5]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()

def dump(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')

def environment():
    packages={}
    for name in ['numpy','Pillow','PyYAML','scikit-image','segmentation-models-pytorch','torch','torchvision','scipy','pandas']:
        packages[name]=importlib.metadata.version(name)
    return {'python':platform.python_version(),'platform':platform.platform(),'packages':packages,'opencv_runtime':cv2.__version__,'torch_build':torch.__version__,'cuda':torch.version.cuda}

def get_samples(args):
    data=args.data_root.resolve()
    p=PROJECT_ROOT/'data'/'splits'/f'{args.dataset}_groups_seed{args.seed}.json'
    split=json.loads(p.read_text(encoding='utf8'))
    spec=DatasetSpec(args.dataset,'TA2' if args.dataset=='tio' else '316L',data/args.dataset/'micrographs',data/args.dataset/'masks')
    samples=discover_samples(spec)
    return {role:filter_samples_by_image_names(samples,split[role+'_images']) for role in ['train','val','test']},p

def setup(args,training=False):
    if args.preprocess=='gray' and args.arch!='unet_r34':
        raise ValueError('The Gray control was evaluated only with U-Net in this paper.')
    if args.seed not in (SEEDS10 if args.arch=='unet_r34' else SEEDS5):
        raise ValueError('Seed is outside the archived paper schedule.')
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is not available; use --device cpu.')
    torch.set_num_threads(4);cv2.setNumThreads(2)
    set_global_seed(args.seed,deterministic=True,cudnn_benchmark=False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=training
    return device

def fresh_output(path):
    path=path.resolve()
    if path.exists():raise FileExistsError(f'Refusing to overwrite existing output: {path}')
    path.mkdir(parents=True)
    return path

def train(args):
    device=setup(args,True)
    samples,split=get_samples(args)
    if args.smoke:samples={k:v[:2] for k,v in samples.items()}
    epochs=1 if args.smoke else args.epochs
    batch=args.batch_size or (8 if args.arch=='unet_r34' else 4)
    if args.smoke:batch=2
    pretrained=not (args.no_pretrained or args.smoke)
    out=fresh_output(args.output)
    loaders={role:build_dataloader(SegmentationDataset(values,preprocess=args.preprocess),LoaderConfig(batch_size=batch,shuffle=(role=='train'),drop_last=False,num_workers=0,pin_memory=False)) for role,values in samples.items()}
    model=build_model(args.arch,len(scheme_channel_indices(args.preprocess)),pretrained=pretrained).to(device)
    protocol_ok=not args.smoke and epochs==50 and pretrained and batch==(8 if args.arch=='unet_r34' else 4) and device.type=='cuda'
    config={'dataset':args.dataset,'architecture':args.arch,'preprocess':args.preprocess,'seed':args.seed,'epochs':epochs,'batch_size':batch,'pretrained':pretrained,'learning_rate':0.001,'weight_decay':0.0001,'amp':device.type=='cuda','cudnn_allow_tf32_training':True,'matmul_allow_tf32':False,'split_sha256':sha(split),'device':str(device),'smoke_only':args.smoke,'paper_training_settings':protocol_ok,'environment':environment(),'checkpoint_selection':'maximum clean-validation interior Dice; strict greater-than; earliest tie','samples':{k:[x.image_name for x in v] for k,v in samples.items()}}
    dump(out/'config.json',config)
    train_run(model,loaders['train'],loaders['val'],loaders['test'],build_loss(),device,out,TrainerConfig(epochs,0.001,0.0001,threshold=0.5,amp=device.type=='cuda'))
    dump(out/'training.complete.json',{'epochs_completed':epochs,'best_sha256':sha(out/'best.ckpt'),'paper_training_settings':protocol_ok,'smoke_only':args.smoke})
    print(json.dumps({'output':str(out),'epochs_completed':epochs,'paper_training_settings':protocol_ok,'next':'Use evaluate for the corrected FP32/TF32-disabled 21-condition results.'}))

def evaluate(args):
    device=setup(args,False)
    samples,split=get_samples(args);test=samples['test']
    if args.max_images:test=test[:args.max_images]
    digest=sha(args.checkpoint)
    with (PROJECT_ROOT/'provenance/checkpoints.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
    ref=next(r for r in rows if r['dataset']==args.dataset and r['architecture']==args.arch and r['preprocess']==args.preprocess and int(r['seed'])==args.seed)
    matched=digest==ref['checkpoint_sha256']
    complete=args.checkpoint.parent/'training.complete.json';own50=False
    if complete.exists():
        meta=json.loads(complete.read_text(encoding='utf8'))
        own50=meta.get('epochs_completed')==50 and meta.get('best_sha256')==digest
    if not matched and not own50 and not args.allow_unregistered_checkpoint:
        raise ValueError('Checkpoint does not match the paper manifest or a verified 50-epoch completion record. Do not substitute the archived 1-epoch weights. For an intentional non-paper smoke run, use --allow-unregistered-checkpoint.')
    chosen=[c for c in CONDITIONS if args.conditions is None or c[1] in args.conditions]
    if args.conditions and len(chosen)!=len(set(args.conditions)):raise ValueError('Unknown condition. Available conditions: '+', '.join(c[1] for c in CONDITIONS))
    out=fresh_output(args.output)
    model=build_model(args.arch,len(scheme_channel_indices(args.preprocess)),pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint,map_location=device,weights_only=True),strict=True)
    inputs=SeedInputs(test)
    batch=args.batch_size or (8 if args.arch=='unet_r34' else 4)
    records=[]
    try:
        for degradation,condition,level,params in chosen:
            planes=inputs.condition_planes(degradation,params,base_seed=0)
            masks=out/'masks'/condition if args.save_masks else None
            result=evaluate_planes(model,planes,scheme_channel_indices(args.preprocess),inputs.mask_tensor,inputs.gt_components,device,batch_size=batch,mask_output_dir=masks,image_names=inputs.names)
            for name,row in zip(inputs.names,result):
                item={'dataset':args.dataset,'architecture':args.arch,'preprocess':args.preprocess,'seed':args.seed,'condition':condition,'degradation':degradation,'level':level,'image_id':name,**{k:v for k,v in row.items() if k!='grain'}}
                for rule,values in row['grain'].items():
                    item.update({rule_key(*rule)+'_'+k:v for k,v in zip(['N_pred','N_gt','area_pred','area_gt','d_pred','d_gt'],values)})
                item['checkpoint_sha256']=digest;records.append(item)
            print(condition,len(result),'images',flush=True)
            del planes
    finally:inputs.close()
    tag=f'{args.dataset}__{args.arch}__{args.preprocess}__seed_{args.seed}'
    with gzip.open(out/(tag+'.csv.gz'),'wt',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    dump(out/'evaluation.json',{'dataset':args.dataset,'architecture':args.arch,'preprocess':args.preprocess,'seed':args.seed,'split_sha256':sha(split),'checkpoint_sha256':digest,'matches_archived_paper_checkpoint':matched,'verified_50_epoch_completion':own50,'partial_evaluation':len(chosen)!=21 or len(test)!=len(samples['test']),'n_images':len(test),'conditions':[x[1] for x in chosen],'threshold':0.5,'amp':False,'tf32':False,'degradation_seed':0,'environment':environment(),'mask_semantics':'0 boundary, 255 interior; filenames correspond to image_id stems'} )
    print('Saved',len(records),'per-image records to',out)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    for name in ['train','evaluate']:
        p=subs.add_parser(name)
        p.add_argument('--dataset',choices=['316l','tio'],required=True)
        p.add_argument('--arch',choices=['unet_r34','deeplabv3plus_r34'],default='unet_r34')
        p.add_argument('--preprocess',choices=list(INPUT_CONFIG_REGISTRY),required=True)
        p.add_argument('--seed',type=int,default=42)
        p.add_argument('--data-root',type=Path,default=PROJECT_ROOT/'data',help='Image/mask directory containing 316l/ and/or tio/. Split files are bundled with this package.')
        p.add_argument('--output',type=Path,required=True)
        p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
        p.add_argument('--batch-size',type=int)
    p=subs.choices['train'];p.add_argument('--epochs',type=int,default=50);p.add_argument('--no-pretrained',action='store_true');p.add_argument('--smoke',action='store_true')
    p=subs.choices['evaluate'];p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--conditions',nargs='+');p.add_argument('--max-images',type=int,default=0);p.add_argument('--save-masks',action='store_true');p.add_argument('--allow-unregistered-checkpoint',action='store_true')
    args=parser.parse_args()
    if args.batch_size is not None and args.batch_size<1:parser.error('--batch-size must be positive')
    if args.command=='train':
        if args.epochs<1:parser.error('--epochs must be positive')
        train(args)
    else:
        if args.max_images<0:parser.error('--max-images cannot be negative')
        evaluate(args)

if __name__=='__main__':main()
