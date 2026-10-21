"""Continuous depth weights for normalized background gather, never a subject cutout.

Near-focus colours must not bleed into a defocused, farther surface. The source
weight excludes a narrow mixed-depth edge; the destination weight only changes
far-side gather. Original depth-of-field blend weights still determine sharpness.
"""
from pathlib import Path
import cv2,numpy as np
import video_masks

VERSION='normalized-far-gather-v1'

def weights(disparity,focus):
    # A proportional transition also works when the focus plane is distant.
    margin=max(.025,float(focus)*.15)
    t=np.clip((focus-margin-disparity)/margin,0,1)
    source=t*t*(3-2*t)
    source=cv2.erode(source,np.ones((7,7),np.uint8))
    # One lossless mask code value is a neutral fallback where no farther samples
    # exist. It prevents division into black and tends to ordinary gather there.
    source=np.maximum(source,1/255)
    t=np.clip((focus-.01-disparity)/.04,0,1)
    target=t*t*(3-2*t)
    return source,target

def prepare(disparities,focus,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if len(disparities)!=len(focus):raise ValueError('Focus length mismatch.')
    out=[]
    for index,name in enumerate(['gather_source','gather_target']):
        tmp=directory/(name+'.npy')
        values=np.lib.format.open_memmap(tmp,mode='w+',dtype='float32',shape=disparities.shape)
        for i,(d,f) in enumerate(zip(disparities,focus)):values[i]=weights(d,f)[index]
        values.flush();p=directory/(name+'.mp4')
        video_masks.encode_masks(values,p,lambda x:x)
        del values;tmp.unlink();out.append(str(p))
    return out

def attach(graph,source_name,target_name):
    """Add normalized convolution to each existing round-video gather level.

    Multiply/Divide preserve HDR values in the existing open Radiance package.
    Both numerator and denominator use the identical round Gaussian kernel.
    """
    if 'video_round_scale' not in graph:return graph
    graph['41']['_meta']['title']=VERSION
    def node(t,**kw):return {'class_type':t,'inputs':kw}
    for tag,name in [('occ_source',source_name),('occ_target',target_name)]:
        graph[tag+'_load']=node('LoadVideo',file=name)
        graph[tag]=node('GetVideoComponents',video=[tag+'_load',0])
    graph['occ_target_mask']=node('ImageToMask',image=['occ_target',0],channel='red')
    graph['occ_premultiply']=node('◎ RadianceBlendComposite',base=['5',0],blend=['occ_source',0],mode='Multiply',opacity=1.)
    scale=dict(graph['video_round_scale']['inputs'])
    for tag,ref in [('occ_weight_low',['occ_source',0]),('occ_colour_low',['occ_premultiply',0])]:
        graph[tag]=node('ImageScale',**{**scale,'image':ref})
    for key,value in list(graph.items()):
        if not key.startswith('video_round_') or not key.endswith('_blur'):continue
        stem=key[:-5];params=value['inputs']
        for tag,ref in [('weight',['occ_weight_low',0]),('colour',['occ_colour_low',0])]:
            graph[stem+'_occ_'+tag]=node('ImageBlur',**{**params,'image':ref})
        graph[stem+'_occ_divide']=node('◎ RadianceBlendComposite',base=[stem+'_occ_colour',0],blend=[stem+'_occ_weight',0],mode='Divide',opacity=1.)
        graph[stem+'_occ_up']=node('ImageScale',**{**graph[stem+'_up']['inputs'],'image':[stem+'_occ_divide',0]})
        graph[stem+'_occ_mix']=node('ImageCompositeMasked',destination=[stem+'_up',0],source=[stem+'_occ_up',0],mask=['occ_target_mask',0],x=0,y=0,resize_source=False)
        graph[stem+'_mix']['inputs']['source']=[stem+'_occ_mix',0]
    return graph
