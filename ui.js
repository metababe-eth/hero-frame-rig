() => {
  const preview = __NOLA_PREVIEW__;
  __DOWNLOAD_HELPER__
  const downloads=createRigDownloads(); downloads.install();
  document.addEventListener('click',event=>{
    const button=event.target.closest?.('#clip-share-button');
    if(!button || button.hasAttribute('data-rig-file-download'))return;
    const root=document.getElementById('clip-result-actions');
    let status=root.querySelector('.rig-download-status');
    if(!status){status=document.createElement('p');status.className='rig-download-status';status.setAttribute('role','status');root.appendChild(status);}
    status.textContent='Preparing your share video… This can take a minute.';
  },true);
  // The hosted mini-header injects these independently of the app theme. Removing
  // them keeps active styles local; the browser may already have requested the CSS.
  let removedFontLinks = 0;
  function removeSpaceHeaderFonts() {
    for (const link of document.head.querySelectorAll('link[rel="stylesheet"]')) {
      try {
        const url = new URL(link.href, document.baseURI), family = url.searchParams.get('family') || '';
        if (url.hostname === 'fonts.googleapis.com' &&
            (family.startsWith('Source Sans Pro:') || family.startsWith('IBM Plex Mono:'))) {
          link.remove(); removedFontLinks++;
        }
      } catch {}
    }
  }
  new MutationObserver(removeSpaceHeaderFonts).observe(document.head, {childList:true, subtree:true, attributes:true, attributeFilter:['href','rel']});
  removeSpaceHeaderFonts();
  // Gradio 6.30 scopes nested CSS below .contain, excluding its own outer shell.
  // Keep the shell correction narrow; all app styling remains in ui.css.
  if (!document.getElementById('rig-mobile-shell')) {
    const style = document.createElement('style'); style.id='rig-mobile-shell';
    style.textContent='@media(max-width:700px){.gradio-container{width:100%!important;min-width:0!important;box-sizing:border-box!important;padding:10px 16px 24px!important}.gradio-container>.main{min-width:0!important;width:100%!important;box-sizing:border-box!important;padding:0!important}} #huggingface-space-header{position:static!important;inset:auto!important;width:max-content!important;max-width:calc(100vw - 32px)!important;margin:12px auto!important;}';
    document.head.appendChild(style);
  }
  let pending = false, observer, currentImage, imageResize;
  function refresh() {
    pending = false;
    const root = document.getElementById('hero-rig');
    const stage = document.getElementById('stage-image');
    if (!root) return;
    root.dataset.workflow=[...document.querySelectorAll('#workflow-tabs [role="tab"]')].find(t=>t.getAttribute('aria-selected')==='true')?.textContent.trim() || 'Use my clip';
    clipUI.refresh(); downloads.refresh();
    if (!stage) return;
    document.documentElement.classList.add('dark');
    let data = null;
    const meta = document.getElementById('rig-stage-data');
    if (meta) { try { data = JSON.parse(meta.dataset.stage); } catch {} }
    if (data) {
      root.classList.add('rig-loaded'); root.classList.remove('rig-empty');
      root.classList.toggle('rig-portrait', data.orientation === 'portrait');
      root.classList.toggle('rig-landscape', data.orientation !== 'portrait');
      root.style.setProperty('--frame-ratio', String(data.aspect_ratio));
    }
    const builder = document.getElementById('frame-builder');
    root.classList.toggle('rig-editing', !!builder && !!builder.offsetParent && !root.classList.contains('rig-loading'));
    root.classList.toggle('rig-animation-open',root.dataset.workflow==='Make a clip');
    if (!stage.querySelector('.rig-loading-layer')) {
      const loading = document.createElement('div'); loading.className='rig-loading-layer';
      loading.innerHTML='<img class="rig-preview" alt="Saved NOLA study preview"><div class="rig-loading-note" role="status">Loading full-resolution frame…</div>';
      loading.querySelector('img').src=preview; stage.appendChild(loading);
      const ring = document.createElement('div'); ring.className='rig-focus-ring'; ring.setAttribute('aria-hidden','true'); stage.appendChild(ring);
      const label = document.createElement('div'); label.className='rig-focus-label'; stage.appendChild(label);
    }
    const img = [...stage.querySelectorAll('img:not(.rig-preview)')].find(x => x.naturalWidth > 100) || stage.querySelector('img:not(.rig-preview)');
    if (img && currentImage !== img) {
      currentImage=img; img.addEventListener('load', schedule);
      if (imageResize) imageResize.disconnect();
      imageResize=new ResizeObserver(schedule); imageResize.observe(img);
    }
    const ready = !!data && img?.complete && img.naturalWidth > 100;
    if (ready) {
      root.classList.remove('rig-loading'); stage.querySelector('.rig-loading-layer').classList.remove('visible');
      if (window.heroRig?.started && !window.heroRig.fullPaintMs) window.heroRig.fullPaintMs=performance.now()-window.heroRig.started;
    }
    const ring = stage.querySelector('.rig-focus-ring'), label = stage.querySelector('.rig-focus-label');
    if (ready && data?.point) {
      const rect=img.getBoundingClientRect(), bounds=stage.getBoundingClientRect();
      const scale=Math.min(rect.width/img.naturalWidth,rect.height/img.naturalHeight,1);
      const w=img.naturalWidth*scale, h=img.naturalHeight*scale;
      const x=rect.left-bounds.left+(rect.width-w)/2+data.point[0]*(img.naturalWidth-1)*scale;
      const y=rect.top-bounds.top+(rect.height-h)/2+data.point[1]*(img.naturalHeight-1)*scale;
      ring.style.left=x+'px'; ring.style.top=y+'px'; ring.style.display='block';
      label.style.left=Math.max(3,Math.min(x+13,bounds.width-105))+'px'; label.style.top=Math.max(2,y-9)+'px'; label.style.display='block';
      if(label.textContent!==data.point_label) label.textContent=data.point_label;
    } else { ring.style.display='none'; label.style.display='none'; }
  }
  function schedule() { if (!pending) { pending=true; requestAnimationFrame(refresh); } }
  const clipUI = (() => {
    const bound = new WeakSet(), observedImages = new WeakSet();
    let active='Sharp', clipId=null, resultId=null, lock=false, tick=null;
    const videos=()=>[document.querySelector('#clip-sharp video'),document.querySelector('#clip-comparison video')];
    const current=()=>videos()[active==='Sharp'?0:1];
    function marker() {
      const video=current(), svg=document.querySelector('#clip-plot svg');
      if (video && svg) {
        const duration=Number(svg.dataset.duration)||video.duration||1;
        const time=video.currentTime||0, x=40+820*Math.min(1,time/duration);
        const line=svg.querySelector('.clip-playhead'), dot=svg.querySelector('.clip-playhead-dot');
        if(line) {line.setAttribute('x1',x);line.setAttribute('x2',x);}
        if(dot) dot.setAttribute('cx',x);
        const label=document.querySelector('.clip-playback-time');
        const text=`${time.toFixed(2)} s / ${(video.duration||duration).toFixed(2)} s`;
        if(label && label.textContent!==text) label.textContent=text;
      }
    }
    function sync(playState=true) {
      if(lock)return;
      const source=current(), target=videos().find(v=>v && v!==source);
      if(!source || !target || target.readyState<1)return;
      lock=true;
      if(Math.abs(target.currentTime-source.currentTime)>.025 && !target.seeking) target.currentTime=source.currentTime;
      target.playbackRate=source.playbackRate;
      if(playState) {
        if(source.paused)target.pause();
        else if(target.paused)target.play().catch(()=>{});
      }
      lock=false;marker();
    }
    function animateMarker() {
      tick=null;marker();sync(false);
      if(current() && !current().paused)tick=requestAnimationFrame(animateMarker);
    }
    function flip(view) {
      const next=(view==='Focus-directed'||view==='After')?'Focus-directed':'Sharp';
      const stage=document.getElementById('clip-videos');
      const source=current(), target=videos()[next==='Sharp'?0:1];
      if(next!==active && source && target) {
        const time=source.currentTime, paused=source.paused;
        if(target.readyState<1) {
          target.addEventListener('loadedmetadata',()=>flip(next),{once:true});return;
        }
        lock=true;
        if(Math.abs(target.currentTime-time)>.025)target.currentTime=time;
        target.playbackRate=source.playbackRate;
        if(paused)target.pause();else target.play().catch(()=>{});
        active=next;lock=false;
      } else active=next;
      if(stage)stage.dataset.view=active;
      for(const id of ['clip-sharp','clip-comparison']) {
        const el=document.getElementById(id), shown=(id==='clip-sharp')===(active==='Sharp');
        if(el){el.style.opacity=shown?'1':'0';el.style.visibility=shown?'visible':'hidden';el.style.pointerEvents=shown?'auto':'none';}
      }
      marker();
      if(current() && !current().paused && tick===null)tick=requestAnimationFrame(animateMarker);
    }
    function aim(stage,img,point,text) {
      if(!stage || !img)return;
      if(!stage.querySelector('.rig-focus-ring')) {
        const ring=document.createElement('div');ring.className='rig-focus-ring';ring.setAttribute('aria-hidden','true');stage.appendChild(ring);
        const label=document.createElement('div');label.className='rig-focus-label';stage.appendChild(label);
      }
      if(!observedImages.has(img)) {
        observedImages.add(img);img.addEventListener('load',schedule);new ResizeObserver(schedule).observe(img);
      }
      const ring=stage.querySelector('.rig-focus-ring'), label=stage.querySelector('.rig-focus-label');
      if(!point || !img.naturalWidth){ring.style.display=label.style.display='none';return;}
      const r=img.getBoundingClientRect(), b=stage.getBoundingClientRect();
      const scale=Math.min(r.width/img.naturalWidth,r.height/img.naturalHeight);
      const x=r.left-b.left+(r.width-img.naturalWidth*scale)/2+point[0]*scale;
      const y=r.top-b.top+(r.height-img.naturalHeight*scale)/2+point[1]*scale;
      ring.style.left=x+'px';ring.style.top=y+'px';ring.style.display='block';
      label.style.left=Math.max(3,Math.min(x+13,b.width-145))+'px';label.style.top=Math.max(2,y-10)+'px';label.style.display='block';
      if(label.textContent!==text)label.textContent=text;
      stage.dataset.focusX=String(point[0]);stage.dataset.focusY=String(point[1]);
    }
    function refreshClip() {
      const meta=document.getElementById('rig-clip-data');if(!meta)return;
      let data;try{data=JSON.parse(meta.dataset.clip);}catch{return;}
      const layout=document.getElementById('clip-workbench');
      layout?.style.setProperty('--clip-ratio',String(data.width/data.height));
      document.getElementById('hero-rig')?.classList.add('rig-clip-loaded');
      for(const [i,id] of ['clip-start','clip-end'].entries()) {
        const el=document.getElementById(id);aim(el,el?.querySelector('img'),data.points[i],data.labels[i]);
      }
      for(const video of videos()) {
        if(!video)continue;
        if(video===videos()[1] && data.poster && video.dataset.rigPoster!==data.result) {
          video.poster=data.poster;video.dataset.rigPoster=data.result;
        }
        video.muted=true;video.defaultMuted=true;video.loop=true;video.playsInline=true;
        if(bound.has(video))continue;bound.add(video);
        video.addEventListener('play',()=>{if(video===current()&&!lock){sync();if(tick===null)tick=requestAnimationFrame(animateMarker);}});
        video.addEventListener('pause',()=>{if(video===current()&&!lock)sync();});
        for(const name of ['seeked','ratechange'])video.addEventListener(name,()=>{if(video===current()&&!lock)sync();});
        video.addEventListener('loadedmetadata',()=>{sync();marker();});
        video.addEventListener('timeupdate',()=>{if(video===current()){marker();sync(false);}});
      }
      flip(document.querySelector('#clip-toggle input:checked')?.value || 'Sharp');
      if(clipId!==data.id) {
        const share=document.getElementById('clip-share-button');
        share?.removeAttribute('data-rig-file-download');share?.removeAttribute('data-rig-download-url');
        clipId=data.id;resultId=null;
        document.getElementById('hero-rig')?.classList.remove('rig-changing-clip');
        document.getElementById('clip-aim-heading')?.scrollIntoView({block:'start',behavior:'instant'});
      }
      if(data.result && data.result!==resultId) {
        const share=document.getElementById('clip-share-button');
        share?.removeAttribute('data-rig-file-download');share?.removeAttribute('data-rig-download-url');
        const result=document.getElementById('clip-result');
        if(result?.offsetParent) {resultId=data.result;result.scrollIntoView({block:'start',behavior:'instant'});}
      }
    }
    return {refresh:refreshClip,flip};
  })();
  window.heroRig = {
    changeClip() {
      document.getElementById('hero-rig')?.classList.add('rig-changing-clip');
      document.getElementById('clip-source-heading')?.scrollIntoView({block:'start',behavior:'instant'});
    },
    saveShare(files) {
      const file=(files||[]).find(f=>(f.url||'').includes('directed_original_1080x1920.mp4'));
      if(file){
        const button=document.getElementById('clip-share-button');
        button.dataset.rigFileDownload='';button.dataset.rigDownloadUrl=file.url;
        downloads.save(file.url,button).then(()=>{
          const root=document.getElementById('clip-result-actions');
          if(root.dataset.downloadStatus==='ready')root.querySelector('.rig-download-status').textContent+=' If nothing saved, press Share again.';
        });
      }
    },
    showWorkflow(name) {
      const tab=[...document.querySelectorAll('#workflow-tabs [role="tab"]')].find(t=>t.textContent.trim()===name);
      tab?.click();
    },
    startStudy() {
      const root=document.getElementById('hero-rig'), stage=document.getElementById('stage-image');
      this.started=performance.now(); this.fullPaintMs=null;
      if (!root || !stage) return;
      root.classList.add('rig-loading','rig-portrait'); root.classList.remove('rig-landscape');
      root.style.setProperty('--frame-ratio','.5625');
      const layer=stage.querySelector('.rig-loading-layer');
      if(layer) { layer.classList.add('visible'); requestAnimationFrame(() => {this.previewPaintMs=performance.now()-this.started;}); }
    }, refresh: schedule, flipClip:clipUI.flip, fontAudit: () => ({removedHeaderStylesheetLinks: removedFontLinks})
  };
  observer=new MutationObserver(schedule);
  observer.observe(document.body,{subtree:true,childList:true,attributes:true,attributeFilter:['src','data-stage','data-clip','aria-expanded','aria-selected','hidden']});
  window.addEventListener('resize',schedule); schedule();
}
