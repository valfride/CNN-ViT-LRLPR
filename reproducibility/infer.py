"""One-track CNN–ViT inference with the exact original EMA state; no table metrics.

CPU FP32 demonstration only. It does not reproduce the paper's CUDA FP16 runtime.
Outputs are local/private and never belong in the source repository.
"""
import argparse
import json
import os
from pathlib import Path

from .common import ROOT, assets, check_sources, new_output, require_hash, write_new


def verified_inputs(checkpoint, output):
    asset = next(a for a in assets(1) if a['id'] == 'ours')
    checkpoint = require_hash(checkpoint, asset['checkpoint_sha256'], 'original CNN–ViT EMA epoch58')
    require_hash(ROOT / asset['config'], asset['config_sha256'], 'original paper config')
    check_sources()
    destination = new_output(output)
    destination.chmod(0o700)
    return asset, checkpoint, destination


def infer(checkpoint, output, image=None, frames=1, fusion='bayes'):
    if frames not in (1,3,5) or fusion not in ('bayes','average','majority','char_majority'):
        raise ValueError('Use F=1/3/5 and one of the four original fusion methods')
    # Explicit CPU isolation precedes any torch import; never initialize a GPU.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    asset, checkpoint, destination = verified_inputs(checkpoint, output)
    import numpy as np
    import torch
    import yaml
    from PIL import Image
    import datasets
    import models
    import test as evaluator
    torch.set_num_threads(2)
    evaluator.utils.setup_seed(42)
    config = yaml.safe_load((ROOT / asset['config']).read_text())
    model = models.make(config['model_g'])
    # Only the trusted original, after complete file/config SHA checks above.
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    tensors, source_key = evaluator.state_dict_from_checkpoint(state)
    evaluator.load_model_state(model, tensors, checkpoint.name)
    del state, tensors
    model.eval()
    if image:
        with Image.open(image) as original:
            pixels = np.asarray(original.convert('RGB')).copy()
    else:
        y,x = np.mgrid[:13,:37]
        pixels = np.stack(((x*7+y*3)%256,(x*3+y*11)%256,(x*13+y*5)%256),axis=-1).astype(np.uint8)
    class SingleImage:
        def __len__(self):return 1
        def __getitem__(self,index):
            return dict(img_path='local/input/image',name='input',img_raw=pixels,gt='')
    spec = config['val_dataset']['wrapper']
    wrapper = datasets.make(dict(name=spec['name'],args=dict(spec['args'],dataset=SingleImage(),test=True,in_images=frames)))
    batch = wrapper.collate_fn([wrapper[0]])
    converter = evaluator.strLabelConverter(config.get('alphabet','0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    scores=[]
    def observe(stage,tensor):
        if tensor.ndim!=3 or tensor.device.type!='cpu' or not torch.isfinite(tensor).all():
            raise RuntimeError('Unexpected or nonfinite CPU inference scores')
        scores.append(dict(stage=stage,shape=list(tensor.shape),dtype=str(tensor.dtype),device=str(tensor.device),finite=True))
    with torch.no_grad():
        prediction,raw,confidence = evaluator.infer_sequence(model,batch['lr_seq'],str(config['cls_loss']).upper(),converter,
                                                              fusion=fusion,use_fp16=False,tta=False,observe=observe)
    require_hash(checkpoint,asset['checkpoint_sha256'],'unchanged original EMA')
    record=dict(model='CNN–ViT',checkpoint_sha256=asset['checkpoint_sha256'],config_sha256=asset['config_sha256'],
                checkpoint_source_key=source_key,device='cpu',precision='FP32',frames=frames,fusion=fusion,
                input_shape=list(batch['lr_seq'].shape),synthetic=image is None,repeated_single_image=frames>1,
                scores=scores,prediction=prediction,confidence=float(confidence),strict_load=True,
                limitation='Usability demonstration only; no ground truth, recognition rate, cohort evaluation or paper GPU precision claim')
    write_new(destination/'inference.json',record)
    return destination/'inference.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True,help='NEW private output directory outside checkout')
    parser.add_argument('--image',help='Your authorized RGB-readable local crop; absent means synthetic input')
    parser.add_argument('--frames',type=int,choices=[1,3,5],default=1,help='Repeat the same input; not real temporal evaluation')
    parser.add_argument('--fusion',choices=['bayes','average','majority','char_majority'],default='bayes')
    args=parser.parse_args()
    print(infer(args.checkpoint,args.output,args.image,args.frames,args.fusion))


if __name__=='__main__':
    main()
