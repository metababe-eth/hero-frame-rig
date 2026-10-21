"""BiRefNet top-third focus targeting; mattes never composite the video."""
import numpy as np,cv2
import rig_workflow,rack_focus,optics

def graph(sharp='sharp.mp4',with_depth=False):
    g=rig_workflow.clip_depth(sharp) if with_depth else {
        '1':{'class_type':'LoadVideo','inputs':{'file':sharp}},
        '2':{'class_type':'GetVideoComponents','inputs':{'video':['1',0]}}}
    nodes=rig_workflow._matte_nodes(['2',0],22)
    nodes.pop('25');g.update(nodes)
    g['25']={'class_type':'SaveImageAdvanced','inputs':{'images':['24',0],
        'filename_prefix':'rig/clip_head_matte','format':'avif','format.bit_depth':'8-bit YUV420',
        'format.input_color_space':'sRGB','format.crf':1,'format.save_mode':'animated',
        'format.save_mode.fps':25,'format.save_mode.loop_count':0}}
    return g

def targets(path,disparities):
    n,h,w=disparities.shape
    with rack_focus._as_sequence(path) as seq:
        masks=np.clip((rack_focus._read_y(seq,8,n)-16)/219,0,1)
    if masks.shape!=disparities.shape:raise ValueError('Subject matte dimensions differ from the clip.')
    values=[];regions=[];found=[]
    for d,m in zip(disparities,masks):
        labels,ids=optics.subject_components(m,min_frac=.01)
        chosen=None
        if ids:
            chosen=labels==ids[0];ys,xs=np.where(chosen)
            # BiRefNet is foreground extraction, not a semantic person detector.
            # Reject tiny/flat silhouettes; a substantial character uses its top third.
            if ys.max()-ys.min()<h*.15:chosen=None
        if chosen is not None:
            top=int(ys.min());bottom=int(ys.max())+1;cut=top+max(1,(bottom-top)//3)
            chosen[cut:]=False
            if chosen.sum()<32:chosen=None
        if chosen is None:
            value=np.percentile(d[round(h*.2):round(h*.8),round(w*.3):round(w*.7)],90)
            regions.append(None);found.append(False)
        else:
            value=np.median(d[chosen]);regions.append([int(xs.min()),top,int(xs.max())+1,cut]);found.append(True)
        values.append(float(value))
    return {'targets':values,'subject_found':found,'top_third_boxes':regions,
            'rule':'Median disparity within the largest substantial foreground matte, top third of its bounding box; nearest central percentile90 when absent. No face recognition or video hold.'}
