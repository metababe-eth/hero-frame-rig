"""LTX camera adapters and native-image crop guides. No cloud submission."""
from pathlib import Path
import math
import numpy as np
from PIL import Image,ImageOps

# Slide left was run and visually reviewed in Brief13 (with hand/glass drift).
CHOICES=['None','Push in','Pull out','Slide left','Slide right (untested)','Rise (untested)','Lower (untested)']
ADAPTERS=dict(zip(CHOICES[1:],['dolly-in','dolly-out','dolly-left','dolly-right','jib-up','jib-down']))

def apply(graph,choice):
    if choice not in CHOICES:raise ValueError('Choose a listed camera move.')
    if choice=='None':return {'choice':choice,'adapter':None,'strength':0}
    adapter=f'ltx-2-19b-lora-camera-control-{ADAPTERS[choice]}.safetensors'
    graph['camera:adapter']={'class_type':'LoraLoaderModelOnly','inputs':{'model':['320:285',0],'lora_name':adapter,'strength_model':1.0}}
    for n in ['320:282','320:314']:graph[n]['inputs']['model']=['camera:adapter',0]
    return {'choice':choice,'adapter':adapter,'strength':1.0}

def crop_guide(source,directory,amount=25,matte=None):
    amount=float(amount)
    if not math.isfinite(amount) or not 10<=amount<=35:raise ValueError('Camera amount must be 10–35%.')
    image=ImageOps.exif_transpose(Image.open(source)).convert('RGB');w,h=image.size
    cx,cy=w/2,h/2;basis='frame centre'
    if matte:
        mask=np.asarray(Image.open(matte).convert('L'),dtype=np.float64)/255
        # A normalized centroid maps the fitted still matte back to native upload size.
        if mask.sum()>0:
            ys,xs=np.indices(mask.shape);cx=float((xs*mask).sum()/mask.sum()/mask.shape[1]*w)
            cy=float((ys*mask).sum()/mask.sum()/mask.shape[0]*h);basis='subject matte centroid'
    cw,ch=round(w*(1-amount/100)),round(h*(1-amount/100))
    left=min(w-cw,max(0,round(cx-cw/2)));top=min(h-ch,max(0,round(cy-ch/2)))
    rect=[left,top,left+cw,top+ch];directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/'native_crop.png';image.crop(tuple(rect)).save(path)
    return str(path),{'amount_percent':amount,'fraction_removed_from_each_dimension':amount/100,
        'crop_rect_xyxy':rect,'native_size':[w,h],'centroid_xy':[cx,cy],'centroid_basis':basis,
        'source':'original image before canvas fit','crop_file':path.name}
