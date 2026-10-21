"""Apply the still round lens and Guide to a clip using per-frame mask videos.

Mask inputs are lossless RGB MP4 sequences. They select the same five Gaussian
levels as the still renderer, so focus can vary per frame without 121 lens graphs.
"""
import rig_workflow as rig
import round_lens,guide_eye
from animation_workflow import _output_closure
VERSION='round-video-mask-sequence-v1'

def _sequence_masks(nodes,names):
    for node,entry in list(nodes.items()):
        if entry['class_type']=='LoadImage':
            filename=entry['inputs']['image']
            if filename not in names:raise ValueError('Missing mask video.')
            loader=node+'_video'
            nodes[loader]={'class_type':'LoadVideo','inputs':{'file':names[filename]}}
            nodes[node]={'class_type':'GetVideoComponents','inputs':{'video':[loader,0]}}

def graph(sharp,weights,width,height,blur,guide=0,guide_mask=None,stock='digital',fps=25):
    if len(weights)!=5:raise ValueError('Five weight videos required.')
    wf={'1':{'class_type':'LoadVideo','inputs':{'file':sharp}},
        '2':{'class_type':'GetVideoComponents','inputs':{'video':['1',0]}}}
    wf.update(rig._to_linear('5',['2',0]))
    nodes,out=round_lens.graph(['5',0],width,height,blur,[f'weight{i}.png' for i in range(5)],'video_round')
    _sequence_masks(nodes,{f'weight{i}.png':p for i,p in enumerate(weights)});wf.update(nodes)
    if guide:
        if not guide_mask:raise ValueError('Guide mask video required.')
        nodes,out=guide_eye.graph(out,'guide.png',guide,'video')
        _sequence_masks(nodes,{'guide.png':guide_mask});wf.update(nodes)
    wf.update(rig._finish(('31','32','33','34'),out,stock))
    wf['40']={'class_type':'CreateVideo','inputs':{'images':['34',0],'fps':fps}}
    wf['41']={'class_type':'SaveVideo','_meta':{'title':VERSION},'inputs':{
        'video':['40',0],'filename_prefix':'rig/hero_clip','format':'mp4','format.codec':'h264',
        'format.codec.encoding':'re-encode','format.codec.encoding.crf':12}}
    return _output_closure(wf,'41')
