"""Two muted, twice-played local exports. No cloud or generative processing."""
from pathlib import Path
import math,subprocess,uuid,json
from PIL import Image,ImageDraw,ImageFont

def crop_geometry(width,height,point):
    if width<2 or height<2 or len(point)!=2 or not all(math.isfinite(float(v)) for v in point):raise ValueError('Invalid share crop.')
    scale=max(1080/width,960/height)
    sw=math.ceil(width*scale/2)*2;sh=math.ceil(height*scale/2)*2
    x=min(sw-1080,max(0,round(float(point[0])*sw/width-540)))
    y=min(sh-960,max(0,round(float(point[1])*sh/height-480)))
    return {'scaled':[sw,sh],'crop':[x,y,1080,960],'focus_end':list(point)}

def export(state):
    if not state.get('focused') or not state.get('sharp'):raise ValueError('Render focus on a clip before sharing it.')
    # Use the treatment's actual endpoint, even if the aiming controls changed later.
    point=(state.get('plan') or {}).get('points',state['points'])[-1]
    if point is None:raise ValueError('The rendered clip has no focus endpoint.')
    geometry=crop_geometry(state['width'],state['height'],point)
    folder=Path(state['dir'])/('share_'+uuid.uuid4().hex[:8]);folder.mkdir()
    directed=folder/'directed_twice.mp4';split=folder/'directed_original_1080x1920.mp4'
    base=['ffmpeg','-v','error','-y']
    enc=['-an','-c:v','libx264','-crf','18','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart','-threads','2']
    subprocess.run(base+['-stream_loop','1','-i',state['focused'],'-map','0:v:0','-vf','fps=25,setsar=1']+enc+[str(directed)],check=True)
    sw,sh=geometry['scaled'];x,y,_,_=geometry['crop']
    common=f'fps=25,scale={sw}:{sh}:flags=lanczos,crop=1080:960:{x}:{y},setsar=1'
    # Raster labels avoid platform-specific ffmpeg/fontconfig failures.
    labels=[]
    for text in ['Directed','Original']:
        label=Image.new('RGBA',(160,52),(0,0,0,128))
        ImageDraw.Draw(label).text((10,8),text,fill='white',font=ImageFont.load_default(size=28))
        path=folder/(text+'.png');label.save(path);labels.append(str(path))
    filters=f'[0:v]{common}[top];[1:v]{common}[bottom];[top][2:v]overlay=28:28[t];[bottom][3:v]overlay=28:28[b];[t][b]vstack=inputs=2[out]'
    subprocess.run(base+['-stream_loop','1','-i',state['focused'],'-stream_loop','1','-i',state['sharp'],
        '-i',labels[0],'-i',labels[1],'-filter_complex',filters,'-map','[out]']+enc+[str(split)],check=True)
    (folder/'share.json').write_text(json.dumps({**geometry,'frames':state['frames']*2,'fps':25,'muted':True,'order':['Directed','Original'],'directed_native_size':[state['width'],state['height']]},indent=2),encoding='utf-8')
    return [str(directed),str(split)]
