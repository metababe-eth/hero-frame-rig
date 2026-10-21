"""Plain-language presentation for the clip director; recipe values stay stable."""
import base64,html,json,subprocess
from pathlib import Path
import clip_focus

MIDDLE=[('Follow the face',clip_focus.HEAD),("Follow what's closest",clip_focus.FOLLOW),('Stay put',clip_focus.HOLD)]

def aperture(blur):
    value=float(blur)
    if not 0<=value<=100:raise ValueError('Blur must be between Subtle and Strong.')
    return round(5.6-4.2*value/100,8)

def blur(aperture_value):
    return (5.6-float(aperture_value))/4.2*100

def cost(state):
    n=clip_focus.job_count(state)
    return 'Cached depth · local focus render · no Cloud job' if n==0 else f'{n} backend '+('job' if n==1 else 'jobs')+' · focus runs locally'

def plot(plan):
    text=clip_focus.plot(plan)
    for old,new in [('Estimated-depth focus schedule, nearer upward','Focus over time, closer upward'),
                    ('focus depth','focus'),('follow head','follow face'),('follow nearest','follow closest'),
                    ('hold start','start'),('hold end','end'),('>pull<','>change focus<')]:
        text=text.replace(old,new)
    return text

def progress_text(message):
    if message.startswith(('Reading cached','Estimating')):return 'Reading depth…'
    if message.startswith('Preparing and verifying'):return 'Preparing focus…'
    if message.startswith('Rendering focus'):return 'Rendering focus…'
    if message.startswith('Finding the subject'):return 'Finding the subject…'
    return message

def progress_html(message,done=False):
    # Stages are known; a percent-complete estimate for Cloud execution is not.
    bar=' value="1" max="1"' if done else ''
    return f'<div class="clip-progress" role="status"><span>{html.escape(progress_text(message))}</span><progress{bar} aria-label="Clip processing"></progress></div>'

def poster(state):
    if not state.get('focused'):return ''
    target=Path(state['focused']).with_suffix('.poster.jpg')
    if not target.exists():
        subprocess.run(['ffmpeg','-v','error','-y','-i',state['focused'],'-vf',
            f"select=eq(n\\,{state['frames']-1}),scale=640:640:force_original_aspect_ratio=decrease",
            '-frames:v','1','-q:v','2',str(target)],check=True)
    return 'data:image/jpeg;base64,'+base64.b64encode(target.read_bytes()).decode('ascii')

def metadata(state):
    data={k:state[k] for k in ('id','width','height','points')}
    data.update(labels=['Start','End'],has_focused=bool(state.get('focused')),result=Path(state['focused']).name if state.get('focused') else '',poster=poster(state))
    return f'<span id="rig-clip-data" data-clip="{html.escape(json.dumps(data),quote=True)}"></span>'

def fresh_study(recipe):
    state=clip_focus.import_recipe(recipe)
    state.update(points=[None,None],focused=None,plan=None,study='Arrival')
    state['settings']={'middle':clip_focus.HEAD,'landing':85,'f_stop':1.4,'guide':60,'edge_cleanup':.5}
    clip_focus.save(state)
    return state
