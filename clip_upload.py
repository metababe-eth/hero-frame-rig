"""Local phone-clip conform. No cloud client, generation or audio retention."""
from pathlib import Path
import hashlib,json,math,subprocess,tempfile,shutil
from fractions import Fraction
import clip_focus

MAX_FRAMES=193
FPS=25

def inspect_upload(path, max_frames=97):
    if not path or Path(path).suffix.lower() not in ('.mp4','.mov'):
        raise ValueError('Choose an MP4 or MOV clip.')
    if Path(path).stat().st_size>1_000_000_000:raise ValueError('Choose a clip smaller than 1 GB.')
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))
    streams=[s for s in data['streams'] if s['codec_type']=='video' and not s.get('disposition',{}).get('attached_pic')]
    if len(streams)!=1:raise ValueError('Choose a clip with one video stream.')
    s=streams[0];duration=float(s.get('duration') or data['format']['duration'])
    if not math.isfinite(duration) or duration<9/FPS:raise ValueError('Choose a clip at least 0.36 seconds long.')
    w,h=int(s['width']),int(s['height']);rotation=float(s.get('tags',{}).get('rotate',0))
    for item in s.get('side_data_list',[]):rotation=float(item.get('rotation',rotation))
    sar=s.get('sample_aspect_ratio','1:1')
    try:ratio=float(Fraction(sar.replace(':','/')))
    except (ValueError,ZeroDivisionError):ratio=1
    dw,dh=round(w*ratio),h
    if abs(round(rotation/90))%2:dw,dh=dh,dw
    # DA3 resizes to its own 14-pixel inference grid and returns the original
    # size. Only YUV420 needs even output sides; LTX's 64-pixel grid does not
    # apply to uploaded clips. Keep small clips native and cap existing work.
    scale=min(1,1280/max(dw,dh),math.sqrt(704*1280/(dw*dh)))
    fitted=[max(2,math.ceil(side*scale/2)*2) for side in (dw,dh)]
    canvas=[max(2,math.floor(side*scale/2)*2) for side in (dw,dh)]
    return {'duration_seconds':duration,'source_size':[w,h],'display_size':[dw,dh],
            'rotation_degrees':rotation,'sample_aspect_ratio':sar,'canvas_size':canvas,'fit_mode':'scale_and_crop',
            # FFmpeg's YUV420 crop origin is aligned to two pixels. Record that
            # actual origin (a two-pixel trim falls on the right/bottom edges).
            'fitted_size':fitted,'offset_xy':[-((fitted[i]-canvas[i])//4*2) for i in (0,1)],
            'max_start_seconds':round(max(0,duration-9/FPS),3),'show_trim':duration>max_frames/FPS,
            'source_video_stream':s['index']}

def prepare(path,start=0,lens=24,max_frames=97):
    if type(max_frames) is not int or max_frames not in (81,97,121,193):
        raise ValueError('Unsupported clip length.')
    meta=inspect_upload(path,max_frames);start=float(start)
    if not math.isfinite(start) or not 0<=start<=meta['max_start_seconds']+0.001:
        raise ValueError('Start at must leave at least nine frames.')
    w,h=meta['canvas_size'];fw,fh=meta['fitted_size'];x,y=meta['offset_xy']
    digest=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    directory=Path(tempfile.mkdtemp(prefix='rig_upload_'))
    try:
        target=directory/'conformed.mp4'
        # ffmpeg autorotates the decoded image before the explicitly recorded fit.
        subprocess.run(['ffmpeg','-v','error','-y','-ss',str(start),'-i',str(path),
            '-map',f"0:{meta['source_video_stream']}",'-vf',f'scale={fw}:{fh}:flags=lanczos,crop={w}:{h}:{-x}:{-y},setsar=1,fps=25',
            '-frames:v',str(max_frames),'-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-map_metadata','-1','-movflags','+faststart',str(target)],check=True)
        state=clip_focus.initialize(target,lens,1.0)
    finally:shutil.rmtree(directory)
    state['provenance']={'kind':'uploaded clip','upload_name':Path(path).name,'upload_sha256':digest.hexdigest(),
        'fit':meta,'trim':{'start_seconds':start,'frames':state['frames'],'duration_seconds':state['frames']/FPS,
                         'maximum_frames':max_frames},'lens_mm':float(lens),'audio':'dropped','crop':meta['canvas_size']!=meta['fitted_size']}
    clip_focus.save(state);return state
