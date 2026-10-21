// One authenticated path for every exported file. No navigation to a private URL.
function createRigDownloads(env = globalThis) {
  const pending = new WeakSet();
  function fileURL(href) {
    const url = new URL(href, env.document.baseURI);
    if (url.origin !== new URL(env.document.baseURI).origin || !url.pathname.includes('/gradio_api/file='))
      throw new Error('This download is not an app file.');
    return url;
  }
  function filename(url) {
    return decodeURIComponent(url.pathname).split('/').pop().replace(/[\\\r\n]/g, '_') || 'download';
  }
  async function fetchValidated(href, progress = () => {}) {
    const url = fileURL(href), name = filename(url);
    const response = await env.fetch(url.href, {credentials:'include'});
    if (!response.ok) throw new Error(`Download failed (HTTP ${response.status}). Try creating the export again.`);
    const type = response.headers.get('content-type') || '';
    if (/text\/html/i.test(type)) throw new Error(`Download failed (HTTP ${response.status}): the server returned a page instead of ${name}.`);
    const length = response.headers.get('content-length');
    const expected = length !== null && !response.headers.get('content-encoding') ? Number(length) : null;
    const chunks = []; let received = 0;
    if (response.body?.getReader) {
      const reader = response.body.getReader();
      while (true) {
        const {done,value} = await reader.read(); if (done) break;
        chunks.push(value); received += value.byteLength; progress(received,expected);
      }
    } else {
      const data = await response.arrayBuffer(); chunks.push(data); received = data.byteLength;
    }
    const blob = new Blob(chunks,{type});
    if (!blob.size || (expected !== null && expected !== blob.size))
      throw new Error(`Download incomplete (HTTP ${response.status}): received ${blob.size} of ${expected ?? 'unknown'} bytes. Retry the download.`);
    const bytes = new Uint8Array(await blob.slice(0,16).arrayBuffer());
    const starts = (...values) => values.every((v,i) => bytes[i] === v);
    const ext = name.split('.').pop().toLowerCase();
    const valid = ext === 'zip' ? starts(80,75) : ext === 'png' ? starts(137,80,78,71,13,10,26,10) :
      ext === 'mp4' ? String.fromCharCode(...bytes.slice(4,8)) === 'ftyp' : true;
    if (!valid) throw new Error(`Download failed (HTTP ${response.status}): invalid ${ext.toUpperCase()} file. Create the export again.`);
    if (ext === 'json') {
      try { JSON.parse(await blob.text()); }
      catch { throw new Error(`Download failed (HTTP ${response.status}): invalid JSON file.`); }
    }
    return {blob,name,serverBytes:expected};
  }
  function statusFor(control) {
    let root = control.closest('.rig-download-source') || control.parentElement;
    if (root.matches('a,button')) root = root.parentElement;
    let status = root.querySelector(':scope > .rig-download-status');
    if (!status) {
      status = env.document.createElement('p'); status.className='rig-download-status';
      status.setAttribute('role','status'); status.setAttribute('aria-live','polite'); root.appendChild(status);
    }
    return {root,status};
  }
  async function save(href,control) {
    if (pending.has(control)) return;
    pending.add(control);
    const {root,status}=statusFor(control);
    root.dataset.downloadStatus='fetching'; delete root.dataset.downloadBytes;
    status.textContent='Preparing download…'; control.setAttribute('aria-busy','true');
    try {
      const {blob,name,serverBytes}=await fetchValidated(href,(bytes,total) => {
        status.textContent=`Downloading ${total ? Math.min(100,Math.round(bytes/total*100))+'%' : (bytes/1048576).toFixed(1)+' MB'}…`;
      });
      const url=env.URL.createObjectURL(blob), anchor=env.document.createElement('a');
      anchor.href=url; anchor.download=name; anchor.hidden=true;
      env.document.body.appendChild(anchor); anchor.click(); anchor.remove();
      env.setTimeout(()=>env.URL.revokeObjectURL(url),120000);
      root.dataset.downloadStatus='ready'; root.dataset.downloadBytes=String(blob.size);
      root.dataset.downloadName=name; root.dataset.serverBytes=serverBytes === null ? '' : String(serverBytes);
      status.textContent=`${name} ready · ${blob.size.toLocaleString()} bytes. Check your browser downloads.`;
    } catch(error) {
      root.dataset.downloadStatus='error';
      status.textContent=error instanceof TypeError ? 'Download failed: network connection unavailable. Retry the download.' : error.message || 'Download failed. Retry the download.';
    } finally { pending.delete(control); control.removeAttribute('aria-busy'); }
  }
  function refresh() {
    let hasFocused=false;
    try { hasFocused=JSON.parse(env.document.getElementById('rig-clip-data')?.dataset.clip || '{}').has_focused === true; } catch {}
    for (const button of env.document.querySelectorAll('[data-rig-video-download]')) {
      const video=env.document.querySelector(`#${button.dataset.rigVideoDownload} video`);
      button.disabled=!(video?.currentSrc || video?.getAttribute('src')) || (button.dataset.rigVideoDownload === 'clip-comparison' && !hasFocused);
    }
    for (const video of env.document.querySelectorAll('#hero-rig video')) video.setAttribute('controlslist','nodownload');
  }
  function install() {
    env.document.addEventListener('click',event => {
      const button=event.target.closest?.('[data-rig-video-download],[data-rig-file-download],#hero-download');
      if (button) {
        event.preventDefault(); event.stopImmediatePropagation();
        const video=button.dataset.rigVideoDownload ? env.document.querySelector(`#${button.dataset.rigVideoDownload} video`) : null;
        const href=button.dataset.rigDownloadUrl || video?.currentSrc || video?.getAttribute('src');
        if (href) save(href,button);
        else statusFor(button).status.textContent='Download unavailable. Select the saved take again.';
        return;
      }
      const link=event.target.closest?.('#hero-rig a[href]');
      if (!link || !link.href.includes('/gradio_api/file=')) return;
      event.preventDefault(); event.stopImmediatePropagation();
      if (!link.closest('.rig-input-file')) save(link.href,link);
    },true);
  }
  return {install,refresh,fetchValidated,save};
}
