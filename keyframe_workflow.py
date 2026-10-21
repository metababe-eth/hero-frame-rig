"""Bounded three-keyframe LTX study, with guides at start, midpoint and end.

The default app builder remains limited to 121 frames. This explicit study
builder permits 81, 97, 121 or 193, updating both video and audio latent lengths.
Middle/end references are appended in both stages and cropped before upscaling
or final decoding. No paid work occurs here.
"""
import copy
import animation_workflow
import endframe_workflow as end

MIDDLE_IMAGE_NODE='keyframe:middle'
VERSION='ltx-three-keyframes-v1'

def build(first,middle,last,prompt,seed,frames=193):
    if type(frames) is not int or frames not in (81,97,121,193):
        raise ValueError('Three-frame study permits 81, 97, 121 or 193 output frames.')
    if not isinstance(middle,str) or not middle.strip() or '\x00' in middle:
        raise ValueError('A middle reference filename is required.')
    graph=end.build_endframe_workflow(first,last,seed,prompt)
    graph['320:295']['inputs']['length']=frames
    graph['320:305']['inputs']['frames_number']=frames
    graph[MIDDLE_IMAGE_NODE]={'class_type':'LoadImage','inputs':{'image':middle}}
    for stage,node in [(1,end.STAGE_ONE_GUIDE_NODE),(2,end.STAGE_TWO_GUIDE_NODE)]:
        midpoint=f'keyframe:middle_guide_{stage}'
        middle_node=copy.deepcopy(graph[node])
        middle_node['inputs'].update(image=[MIDDLE_IMAGE_NODE,0],frame_idx=(frames-1)//2)
        graph[midpoint]=middle_node
        graph[node]['inputs'].update(positive=[midpoint,0],negative=[midpoint,1],
                                     latent=[midpoint,2],frame_idx=frames-1)
    return animation_workflow._output_closure(graph,end.OUTPUT_NODE)
