/* 結果ビューア: 波形 / MIDI ノート / 文字ボックスのタイムライン + カラオケ歌詞.
   createViewer(rootEl, {base}) — base は result.json / peaks.json / audio.wav / click.wav がある URL のプレフィックス.
   utavista2 の Timeline (左ヘッダ列 + スクロール本体、ルーラ、プレイヘッド) と同じ構成. */
(function () {
  const I = {
    play: '<svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="6 3 20 12 6 21 6 3"/></svg>',
    pause: '<svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor" stroke="none"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>',
    back: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="19 20 9 12 19 4 19 20"/><line x1="5" y1="19" x2="5" y2="5"/></svg>',
  };
  const COL = { wave: '#4f7dd9', note: ['#3da08a', '#2f7f6d', '#5cc4ac', '#2a6b5c'], char: '#e5a842', charBad: '#e2483d', midi: '#3da08a', ctc: '#e2483d', grid: '#2e333c', text: '#e2e4e8', label: '#8b919e', ruler: '#22262e', head: '#e2483d' };

  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === 'class') el.className = v; else if (k === 'html') el.innerHTML = v; else if (k.startsWith('on')) el[k] = v; else el.setAttribute(k, v);
    }
    for (const k of kids) el.append(k);
    return el;
  }

  async function createViewer(root, opts) {
    const base = (opts && opts.base) || './';
    root.innerHTML = '';
    let res, peaks;
    try {
      res = await (await fetch(base + 'result.json', { cache: 'no-store' })).json();
      peaks = await (await fetch(base + 'peaks.json', { cache: 'no-store' })).json();
    } catch (e) {
      root.append(h('div', { class: 'empty' }, '結果がまだありません。「入力」タブで解析を実行してください。'));
      return null;
    }
    const duration = peaks.max.length / peaks.px_per_sec;
    const m = res.meta;
    const RULER = 22, H = { wave: [RULER, RULER + 80], notes: [RULER + 84, RULER + 184], chars: [RULER + 188, RULER + 278] };
    const CANVAS_H = H.chars[1];

    // ---- DOM
    const audio = h('audio', { preload: 'auto' });
    const playBtn = h('button', { class: 'tp-btn tp-btn--play', title: '再生 / 停止 (Space)', html: I.play });
    const homeBtn = h('button', { class: 'tp-btn', title: '先頭へ', html: I.back });
    const time = h('span', { class: 'vw-time' }, '0.000 / 0.000');
    const seek = h('input', { class: 'vw-seek', type: 'range', min: 0, max: 1000, value: 0 });
    const rate = h('select', {}, ...[0.5, 0.75, 1].map((r) => h('option', r === 1 ? { selected: '' } : {}, String(r))));
    const src = h('select', {}, h('option', { value: 'audio.wav' }, 'ボーカル'), h('option', { value: 'click.wav' }, 'ボーカル + クリック'));
    const off = h('input', { type: 'range', min: -200, max: 200, step: 5, value: 0, style: 'width:90px' });
    const offv = h('span', { class: 'mono' }, '0');
    const follow = h('input', { type: 'checkbox', checked: '' });
    const zooms = [50, 120, 250, 500];
    let pxs = 120;
    const zoomBox = h('div', { class: 'vw-zoom' });
    const zoomBtns = zooms.map((z) => h('button', { 'data-active': String(z === pxs), title: `${z} px/s`, onclick: () => setZoom(z) }, z >= 250 ? '+' + (z === 500 ? '+' : '') : (z === 50 ? '−' : '=')));
    zoomBtns.forEach((b) => zoomBox.append(b));
    const transport = h('div', { class: 'vw-transport' },
      homeBtn, playBtn, time, seek,
      h('label', {}, '速度', rate), h('label', {}, '音源', src), h('label', {}, 'オフセット', off, offv, 'ms'),
      h('label', {}, follow, '追従'), zoomBox);
    const legend = h('div', { class: 'vw-legend' },
      h('span', { html: `<span class="sw" style="background:${COL.wave}"></span>波形` }),
      h('span', { html: `<span class="sw" style="background:${COL.note[0]}"></span>MIDI ノート (フレーズごとに濃淡)` }),
      h('span', { html: `<span class="sw" style="background:${COL.char}"></span>文字 (start–end)` }),
      h('span', { html: `<span class="sw" style="background:${COL.charBad}"></span>MIDI と CTC の差が大 (&gt;150ms)` }),
      h('span', { html: `<span class="sw" style="border:1px dashed ${COL.char};background:transparent"></span>漢字読み按分 (推定)` }),
      h('span', {}, '▲ CTC 検出 / | MIDI onset'));
    const headers = h('div', { class: 'vw-headers' },
      h('div', { style: `top:0;height:${RULER}px` }, ''),
      h('div', { style: `top:${H.wave[0]}px;height:${H.wave[1] - H.wave[0]}px` }, '波形'),
      h('div', { style: `top:${H.notes[0]}px;height:${H.notes[1] - H.notes[0]}px` }, 'MIDI'),
      h('div', { style: `top:${H.chars[0]}px;height:${H.chars[1] - H.chars[0]}px` }, '文字'));
    const cv = h('canvas', { height: CANVAS_H });
    const scroll = h('div', { class: 'vw-scroll' }, cv);
    const timeline = h('div', { class: 'vw-timeline', style: `height:${CANVAS_H + 2}px` }, headers, scroll);
    const info = h('div', { class: 'vw-info' }, 'タイムラインをクリックでシーク。文字をクリックでその位置へ。');
    const lyr = h('div', { class: 'vw-lyrics' });
    const wrap = h('div', { class: 'vw' }, transport, legend, timeline, info, lyr, audio);
    root.append(wrap);
    const ctx = cv.getContext('2d');
    let playOffset = 0;
    audio.src = base + src.value;

    // ---- 歌詞パネル
    const charEls = new Map();
    res.chars.forEach((c) => {
      if (c.char === '\n') { lyr.appendChild(document.createElement('br')); return; }
      const s = document.createElement('span');
      s.textContent = c.char;
      s.className = 'c' + (c.sung ? '' : ' unsung') + (c.split_estimated ? ' est' : '') + (c.confidence !== undefined && c.confidence < 0.5 ? ' low' : '');
      if (c.sung) { s.title = `${c.start.toFixed(3)} – ${c.end.toFixed(3)}s`; s.onclick = () => seekTo(c.start - 0.3); }
      lyr.appendChild(s); charEls.set(c.index, s);
    });
    const sungChars = res.chars.filter((c) => c.sung).sort((a, b) => a.start - b.start);

    // ---- 描画
    function x(t) { return t * pxs; }
    function resize() { cv.width = Math.ceil(duration * pxs); draw(); }
    function draw() {
      const left = scroll.scrollLeft, w = scroll.clientWidth;
      const t0 = Math.max(0, left / pxs - 1), t1 = Math.min(duration, (left + w) / pxs + 1);
      ctx.clearRect(x(t0), 0, x(t1) - x(t0) + 2, cv.height);
      // ルーラ
      ctx.fillStyle = COL.ruler; ctx.fillRect(x(t0), 0, x(t1) - x(t0) + 2, RULER);
      ctx.strokeStyle = COL.grid; ctx.fillStyle = COL.label; ctx.font = '10px JetBrains Mono, SF Mono, Menlo, monospace';
      const step = pxs < 100 ? 5 : 1;
      for (let s = Math.floor(t0); s <= t1; s++) {
        ctx.beginPath(); ctx.moveTo(x(s) + 0.5, s % step === 0 ? RULER - 8 : RULER - 4); ctx.lineTo(x(s) + 0.5, cv.height); ctx.stroke();
        if (s % step === 0) ctx.fillText(fmt(s), x(s) + 3, 11);
      }
      ctx.fillStyle = COL.grid; ctx.fillRect(x(t0), RULER - 0.5, x(t1) - x(t0) + 2, 1);
      // 波形
      const [wy0, wy1] = H.wave, mid = (wy0 + wy1) / 2, amp = (wy1 - wy0) / 2 - 4;
      ctx.fillStyle = COL.wave;
      const bpp = peaks.px_per_sec / pxs;
      for (let px = Math.floor(x(t0)); px < x(t1); px++) {
        const b0 = Math.floor(px * bpp), b1 = Math.max(b0 + 1, Math.floor((px + 1) * bpp));
        let mn = 1, mx = -1;
        for (let b = b0; b < b1 && b < peaks.max.length; b++) { if (peaks.max[b] > mx) mx = peaks.max[b]; if (peaks.min[b] < mn) mn = peaks.min[b]; }
        if (mx < mn) continue;
        ctx.fillRect(px, mid - mx * amp, 1, Math.max(1, (mx - mn) * amp));
      }
      ctx.fillStyle = COL.grid; ctx.fillRect(x(t0), H.wave[1] + 1.5, x(t1) - x(t0) + 2, 1); ctx.fillRect(x(t0), H.notes[1] + 1.5, x(t1) - x(t0) + 2, 1);
      // ノート
      const [ny0, ny1] = H.notes;
      const pitches = res.notes.map((n) => n.pitch), pmin = Math.min(...pitches) - 1, pmax = Math.max(...pitches) + 1;
      const ph = (ny1 - ny0) / (pmax - pmin + 1);
      ctx.font = '11px Inter, Noto Sans JP, sans-serif';
      for (const n of res.notes) {
        if (n.end < t0 || n.start > t1) continue;
        const y = ny1 - (n.pitch - pmin + 1) * ph;
        ctx.fillStyle = COL.note[n.phrase % COL.note.length];
        ctx.fillRect(x(n.start), y, Math.max(2, x(n.end) - x(n.start) - 1), Math.max(2, ph - 1));
        if (pxs >= 100) { ctx.fillStyle = COL.text; ctx.fillText(n.mora_ids.map((i) => res.moras[i].kana).join(''), x(n.start) + 2, y - 2); }
      }
      // 文字
      const [cy0] = H.chars;
      ctx.textBaseline = 'middle';
      for (const c of sungChars) {
        if (c.end < t0 || c.start > t1) continue;
        const mora = res.moras[c.mora_ids[0]];
        const disagree = mora.ctc_start != null && Math.abs(mora.ctc_start - mora.midi_start) > 0.15;
        ctx.fillStyle = disagree ? COL.charBad : COL.char;
        const bw = Math.max(2, x(c.end) - x(c.start) - 1);
        ctx.fillRect(x(c.start), cy0 + 28, bw, 38);
        if (c.split_estimated) { ctx.strokeStyle = '#1a1d23'; ctx.setLineDash([3, 2]); ctx.strokeRect(x(c.start) + 0.5, cy0 + 28.5, bw - 1, 37); ctx.setLineDash([]); }
        ctx.fillStyle = '#0d1117'; ctx.font = (pxs >= 200 ? 16 : 13) + 'px Inter, Noto Sans JP, sans-serif';
        ctx.fillText(c.char, x(c.start) + 2, cy0 + 47);
        for (const mi of c.mora_ids) {
          const mo = res.moras[mi];
          if (!mo.sung) continue;
          if (mo.share_index === 0 && mo.midi_start != null) { ctx.fillStyle = COL.midi; ctx.fillRect(x(mo.midi_start), cy0 + 16, 1.5, 12); }
          if (mo.ctc_start != null) { ctx.fillStyle = COL.ctc; ctx.beginPath(); ctx.moveTo(x(mo.ctc_start), cy0 + 68); ctx.lineTo(x(mo.ctc_start) - 4, cy0 + 76); ctx.lineTo(x(mo.ctc_start) + 4, cy0 + 76); ctx.fill(); }
        }
      }
      ctx.textBaseline = 'alphabetic';
      drawHead();
    }
    let lastHeadX = -1;
    function drawHead() {
      const hx = Math.round(x(now())) + 0.5;
      ctx.strokeStyle = COL.head; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(hx, 0); ctx.lineTo(hx, cv.height); ctx.stroke(); ctx.lineWidth = 1;
      ctx.fillStyle = COL.head; ctx.beginPath(); ctx.moveTo(hx - 5, 0); ctx.lineTo(hx + 5, 0); ctx.lineTo(hx, 7); ctx.fill();
      lastHeadX = hx;
    }
    function fmt(s) { const mm = Math.floor(s / 60), ss = s % 60; return `${mm}:${ss < 10 ? '0' : ''}${ss}`; }
    function now() { return audio.currentTime + playOffset; }
    function seekTo(t) { audio.currentTime = Math.max(0, Math.min(duration, t - playOffset)); }
    function setZoom(z) { const t = now(); pxs = z; zoomBtns.forEach((b, i) => b.setAttribute('data-active', String(zooms[i] === z))); resize(); scroll.scrollLeft = x(t) - scroll.clientWidth / 2; draw(); }

    // ---- 操作
    playBtn.onclick = () => { if (audio.paused) audio.play(); else audio.pause(); };
    homeBtn.onclick = () => seekTo(0);
    audio.onplay = () => { playBtn.innerHTML = I.pause; };
    audio.onpause = () => { playBtn.innerHTML = I.play; };
    rate.onchange = () => { audio.playbackRate = parseFloat(rate.value); };
    src.onchange = () => { const t = audio.currentTime, p = !audio.paused; audio.src = base + src.value; audio.currentTime = t; if (p) audio.play(); };
    off.oninput = () => { playOffset = parseInt(off.value, 10) / 1000; offv.textContent = off.value; };
    seek.oninput = () => { seekTo(parseInt(seek.value, 10) / 1000 * duration); };
    scroll.onscroll = () => draw();
    cv.onclick = (e) => { const r = cv.getBoundingClientRect(); seekTo((e.clientX - r.left) / pxs); };
    cv.onmousemove = (e) => {
      const r = cv.getBoundingClientRect(); const t = (e.clientX - r.left) / pxs;
      const c = sungChars.find((c) => c.start <= t && t < c.end);
      const n = res.notes.find((n) => n.start <= t && t < n.end);
      let s = `t=${t.toFixed(3)}s`;
      if (c) { const mo = res.moras[c.mora_ids[0]]; s += `  文字「${c.char}」 ${c.start.toFixed(3)}–${c.end.toFixed(3)}  モーラ ${c.mora_ids.map((i) => res.moras[i].kana).join('')}  source=${mo.source} end=${mo.end_kind}  midi=${mo.midi_start?.toFixed(3)} ctc=${mo.ctc_start?.toFixed(3) ?? '-'}`; }
      if (n) s += `  ノート#${n.id} pitch=${n.pitch} ${n.start.toFixed(3)}–${n.end.toFixed(3)} phrase=${n.phrase}`;
      info.textContent = s;
    };
    const onKey = (e) => {
      if (!document.body.contains(wrap)) { document.removeEventListener('keydown', onKey); return; }
      if (e.code === 'Space' && !['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) { e.preventDefault(); playBtn.click(); }
    };
    document.addEventListener('keydown', onKey);
    new ResizeObserver(() => draw()).observe(scroll);

    // ---- アニメーション
    let alive = true;
    function frame() {
      if (!alive || !document.body.contains(wrap)) { alive = false; audio.pause(); return; }
      const t = now();
      time.textContent = `${t.toFixed(3)} / ${duration.toFixed(3)}`;
      if (document.activeElement !== seek) seek.value = String(Math.round(t / duration * 1000));
      if (follow.checked && !audio.paused) {
        const hx = x(t);
        if (hx < scroll.scrollLeft + 40 || hx > scroll.scrollLeft + scroll.clientWidth - 80) scroll.scrollLeft = hx - 80;
      }
      if (Math.round(x(t)) + 0.5 !== lastHeadX) draw();
      for (const c of sungChars) {
        const el = charEls.get(c.index);
        let cls = 'c';
        if (t >= c.start && t < c.end) cls += ' now'; else if (t >= c.end) cls += ' done'; else if (c.start - t < 0.5) cls += ' soon';
        if (c.split_estimated) cls += ' est';
        if (c.confidence !== undefined && c.confidence < 0.5) cls += ' low';
        if (el.className !== cls) el.className = cls;
      }
      requestAnimationFrame(frame);
    }
    resize();
    requestAnimationFrame(frame);
    return { meta: m, destroy: () => { alive = false; audio.pause(); } };
  }
  window.createViewer = createViewer;
})();
