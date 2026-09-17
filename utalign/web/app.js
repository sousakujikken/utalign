/* utalign app (フレームワーク不使用). 画面: 入力 / 結果 / 書き出し / 設定 */
(function () {
  const $ = (id) => document.getElementById(id);
  const ICON = {
    folder: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/></svg>',
    file: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/></svg>',
    up: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m5 12 7-7 7 7"/><path d="M12 19V5"/></svg>',
    x: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>',
    check: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>',
    dl: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 15V3"/><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/></svg>',
    spin: '<svg class="ut-icon--spin" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>',
    trash: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>',
  };
  const KIND_LABEL = { audio: 'ボーカル音声', lyrics: '歌詞テキスト', midi: 'ボーカル MIDI' };
  const KIND_HINT = { audio: 'MP3 / WAV / M4A / FLAC (ボーカルのみのトラック)', lyrics: 'UTF-8 テキスト。括弧・タグ内と記号は非発音として扱います', midi: 'ボーカルのメロディを採譜した MIDI (ドラム系トラックは自動除外)' };

  // ---------------------------------------------------------------- utils
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === undefined || v === null || v === false) continue;
      if (k === 'class') el.className = v; else if (k === 'html') el.innerHTML = v; else if (k.startsWith('on')) el[k] = v; else if (k === 'checked' || k === 'disabled' || k === 'selected') el[k] = !!v; else el.setAttribute(k, v === true ? '' : v);
    }
    for (const k of kids) { if (Array.isArray(k)) el.append(...k.filter((x) => x !== null && x !== undefined)); else if (k !== null && k !== undefined) el.append(k); }
    return el;
  }
  async function api(method, path, body, isForm) {
    const opt = { method, headers: {} };
    if (body !== undefined) { if (isForm) opt.body = body; else { opt.headers['Content-Type'] = 'application/json'; opt.body = JSON.stringify(body); } }
    const r = await fetch('/api/' + path, opt);
    let data = null;
    try { data = await r.json(); } catch (e) { /* empty */ }
    if (!r.ok) throw new Error((data && data.error) || `${r.status} ${r.statusText}`);
    return data;
  }
  function toast(msg, kind = 'info', ms = 4500) {
    const el = h('div', { class: 'toast', 'data-kind': kind }, h('span', { style: 'flex:1;white-space:pre-wrap' }, msg),
      h('button', { class: 'ut-icon-btn', style: 'width:20px;height:20px', html: ICON.x, onclick: () => el.remove() }));
    $('toasts').append(el);
    if (ms) setTimeout(() => el.remove(), ms);
  }
  const fmtMs = (s) => s == null ? '–' : `${Math.round(s * 1000)} ms`;
  const fmtPct = (x) => x == null ? '–' : `${(x * 100).toFixed(1)} %`;
  const fmtSize = (n) => n >= 1e9 ? (n / 1e9).toFixed(2) + ' GB' : n >= 1e6 ? (n / 1e6).toFixed(1) + ' MB' : n >= 1e3 ? (n / 1e3).toFixed(0) + ' KB' : n + ' B';
  const basename = (p) => (p || '').split('/').pop();
  const enc = encodeURIComponent;
  function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

  // ---------------------------------------------------------------- state
  const S = { cfg: null, projects: [], slug: null, project: null, models: [], mode: 'input', viewer: null, jobs: {}, running: {} };

  // ---------------------------------------------------------------- file browser modal
  function pickPath({ kind = '', title = 'ファイルを選択', start = '' } = {}) {
    return new Promise((resolve) => {
      let cur = start || '', selected = null;
      const input = h('input', { type: 'text', placeholder: 'パスを直接入力して Enter' });
      const roots = h('div', { class: 'fs-roots' });
      const list = h('div', { class: 'fs-list' });
      const okBtn = h('button', { class: 'ut-btn-sm ut-btn-accent', disabled: true }, kind === 'dir' ? 'このフォルダを選択' : '選択');
      const upBtn = h('button', { class: 'ut-btn-sm', html: ICON.up + ' 上へ' });
      const close = (v) => { bd.remove(); document.removeEventListener('keydown', onKey); resolve(v); };
      const onKey = (e) => { if (e.key === 'Escape') { e.stopPropagation(); close(null); } };
      const bd = h('div', { class: 'backdrop', onpointerdown: (e) => { if (e.target === bd) close(null); } },
        h('div', { class: 'modal', role: 'dialog', 'aria-modal': 'true' },
          h('div', { class: 'modal-head' }, h('strong', {}, title), h('span', { class: 'spacer' }), h('button', { class: 'ut-icon-btn', html: ICON.x, onclick: () => close(null) })),
          h('div', { class: 'fs-bar' }, upBtn, input),
          h('div', { class: 'fs-body' }, roots, list),
          h('div', { class: 'actions', style: 'justify-content:flex-end' }, h('button', { class: 'ut-btn-sm', onclick: () => close(null) }, 'キャンセル'), okBtn)));
      document.body.append(bd); document.addEventListener('keydown', onKey);
      input.onkeydown = (e) => { if (e.key === 'Enter') { const v = input.value.trim(); if (kind !== 'dir' && /\.[a-z0-9]+$/i.test(v)) close(v); else load(v); } };
      okBtn.onclick = () => close(kind === 'dir' ? (selected || cur) : selected);
      async function load(p) {
        try {
          const d = await api('GET', `fs?path=${encodeURIComponent(p)}&kind=${kind}`);
          cur = d.path; selected = null; input.value = cur; okBtn.disabled = kind !== 'dir';
          upBtn.disabled = !d.parent; upBtn.onclick = () => d.parent && load(d.parent);
          roots.innerHTML = '';
          for (const r of d.roots) roots.append(h('div', { class: 'fs-item', 'data-active': String(r.path === cur), onclick: () => load(r.path) }, h('span', { html: ICON.folder }), h('span', { class: 'nowrap', style: 'overflow:hidden;text-overflow:ellipsis' }, r.name)));
          list.innerHTML = '';
          if (!cur) { list.append(h('div', { class: 'empty' }, '左の場所から選ぶか、パスを入力してください')); return; }
          for (const dir of d.dirs) {
            const row = h('div', { class: 'fs-item', ondblclick: () => load(dir.path), onclick: () => { if (kind === 'dir') { selected = dir.path; mark(row); } else load(dir.path); } }, h('span', { html: ICON.folder }), h('span', {}, dir.name));
            list.append(row);
          }
          for (const f of d.files) {
            const row = h('div', { class: 'fs-item', onclick: () => { selected = f.path; okBtn.disabled = false; mark(row); }, ondblclick: () => close(f.path) }, h('span', { html: ICON.file }), h('span', {}, f.name), h('span', { class: 'sz' }, fmtSize(f.size)));
            list.append(row);
          }
          if (!d.dirs.length && !d.files.length) list.append(h('div', { class: 'empty' }, '該当するファイルがありません'));
        } catch (e) { toast(e.message, 'error'); }
      }
      function mark(row) { list.querySelectorAll('.fs-item').forEach((r) => r.setAttribute('data-active', 'false')); row.setAttribute('data-active', 'true'); }
      load(cur);
    });
  }

  // ---------------------------------------------------------------- jobs
  function watchJob(job, { onLog, onDone, onError } = {}) {
    let from = 0;
    S.jobs[job.id] = job;
    const tick = async () => {
      try {
        const j = await api('GET', `jobs/${job.id}?from=${from}`);
        if (j.log.length && onLog) onLog(j.log);
        from = j.log_len;
        if (j.status === 'done') { onDone && onDone(j); return; }
        if (j.status === 'error') { onError && onError(j); return; }
        setTimeout(tick, 700);
      } catch (e) { onError && onError({ error: e.message, log: [] }); }
    };
    tick();
  }

  // ---------------------------------------------------------------- projects (left pane)
  async function loadProjects(keep = true) {
    S.projects = await api('GET', 'projects');
    if (keep && S.slug && !S.projects.find((p) => p.slug === S.slug)) S.slug = null;
    if (!S.slug && S.projects.length) S.slug = S.projects[0].slug;
    renderProjectList();
    await loadProject();
  }
  function renderProjectList() {
    const box = $('proj-list'); box.innerHTML = '';
    if (!S.projects.length) { box.append(h('div', { class: 'empty' }, 'プロジェクトがありません。「＋ 新規」で作成してください。')); return; }
    for (const p of S.projects) {
      const st = S.running[p.slug] ? 'run' : p.has_result ? 'ok' : (Object.values(p.inputs_exist).every(Boolean) ? 'warn' : '');
      box.append(h('div', { class: 'proj-item', 'data-active': String(p.slug === S.slug), onclick: () => selectProject(p.slug) },
        h('div', { class: 'name' }, h('span', { class: 'dot ' + st }), p.name),
        h('div', { class: 'sub' }, h('span', {}, S.running[p.slug] ? '解析中…' : p.has_result ? `残差 ${fmtMs(p.last_run && p.last_run.residual_median)}` : '未解析'), h('span', {}, (p.updated || '').slice(0, 10)))));
    }
  }
  async function selectProject(slug) { await flushAllSaves(); S.slug = slug; renderProjectList(); await loadProject(); }
  async function loadProject() {
    S.project = S.slug ? await api('GET', `projects/${enc(S.slug)}`) : null;
    $('top-project').textContent = S.project ? S.project.name : '';
    $('top-run').disabled = !S.project;
    renderMode();
  }
  $('proj-new').onclick = async () => {
    const name = prompt('プロジェクト名', '新しい曲');
    if (name === null) return;
    const p = await api('POST', 'projects', { name });
    S.slug = p.slug; await loadProjects(); setMode('input');
  };

  // ---------------------------------------------------------------- mode tabs
  function setMode(m) { S.mode = m; renderMode(); }
  $('mode-tabs').querySelectorAll('.ut-mode-tab').forEach((b) => { b.onclick = async () => { await flushAllSaves(); setMode(b.dataset.mode); }; });
  function renderMode() {
    $('mode-tabs').querySelectorAll('.ut-mode-tab').forEach((b) => b.setAttribute('data-active', String(b.dataset.mode === S.mode)));
    for (const m of ['input', 'result', 'export', 'settings']) $(`page-${m}`).hidden = m !== S.mode;
    if (S.viewer && S.mode !== 'result') { S.viewer.destroy(); S.viewer = null; $('page-result').innerHTML = ''; }
    ({ input: renderInput, result: renderResult, export: renderExport, settings: renderSettings })[S.mode]();
  }
  $('top-run').onclick = () => runAlign();

  // ---------------------------------------------------------------- 入力
  // 遅延保存: 編集時の slug ごとに patch を統合して保持し (options / inputs は深いマージ)、400ms 後または flush 時に送る。
  // 送信中の PUT は slug ごとに inflight に保持し、後続の保存はその完了後に直列に送る。flush は pending と inflight の両方を待つ。
  const pendingSaves = new Map();   // slug -> { patch, timer }
  const inflight = new Map();       // slug -> Promise (送信中の PUT チェーン)
  function saveProject(patch) {
    const slug = S.slug; if (!slug) return;
    const cur = pendingSaves.get(slug) || { patch: {}, timer: null };
    for (const [k, v] of Object.entries(patch)) cur.patch[k] = (k === 'options' || k === 'inputs') ? { ...(cur.patch[k] || {}), ...v } : v;
    clearTimeout(cur.timer);
    cur.timer = setTimeout(() => flushSave(slug), 400);
    pendingSaves.set(slug, cur);
  }
  function flushSave(slug) {
    const cur = pendingSaves.get(slug);
    if (cur) {
      clearTimeout(cur.timer); pendingSaves.delete(slug);
      const prev = inflight.get(slug) || Promise.resolve();
      const run = prev.then(async () => {
        try {
          const p = await api('PUT', `projects/${enc(slug)}`, cur.patch);
          if (S.slug === slug) S.project = p;
          await loadProjects();
        } catch (e) { toast(e.message, 'error'); }
      }).finally(() => { if (inflight.get(slug) === run) inflight.delete(slug); });
      inflight.set(slug, run);
    }
    return inflight.get(slug) || Promise.resolve();
  }
  async function flushAllSaves() {
    // pending を送信に移し、送信中のものも含めて全て完了するまで待つ (待機中に新たな保存が入れば繰り返す)
    while (pendingSaves.size || inflight.size) {
      for (const slug of [...pendingSaves.keys()]) flushSave(slug);
      await Promise.all([...inflight.values()]);
    }
  }
  window.addEventListener('beforeunload', () => { for (const [slug, cur] of pendingSaves) { clearTimeout(cur.timer); navigator.sendBeacon && fetch(`/api/projects/${enc(slug)}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cur.patch), keepalive: true }); } pendingSaves.clear(); });
  function renderInput() {
    const page = $('page-input'); page.innerHTML = '';
    if (!S.project) { page.append(h('div', { class: 'empty' }, '左の一覧からプロジェクトを選ぶか、新規作成してください。')); return; }
    const p = S.project, o = p.options || {};
    // ---- 名前
    const nameIn = h('input', { type: 'text', value: p.name, class: 'field-w', oninput: (e) => saveProject({ name: e.target.value }) });
    page.append(h('div', { class: 'card' }, h('div', { class: 'card-title' }, 'プロジェクト', h('span', { class: 'spacer' }),
      h('button', { class: 'ut-btn-sm ut-btn-danger', html: ICON.trash + ' 削除', onclick: async () => { if (!confirm(`「${p.name}」を削除しますか? (出力ファイルも消えます)`)) return; await api('DELETE', `projects/${enc(p.slug)}`); S.slug = null; await loadProjects(); toast('削除しました', 'success'); } })),
      h('label', { class: 'row' }, h('span', {}, '名前'), nameIn),
      h('label', { class: 'row' }, h('span', {}, '保存場所'), h('span', { class: 'mono secondary', style: 'font-size:11px;word-break:break-all' }, p.path))));
    // ---- 入力ファイル
    const inputsCard = h('div', { class: 'card' }, h('div', { class: 'card-title' }, '入力ファイル'));
    for (const kind of ['audio', 'lyrics', 'midi']) {
      const path = p.inputs[kind]; const exists = p.inputs_exist[kind];
      const fileIn = h('input', { type: 'file', hidden: true, accept: { audio: '.mp3,.wav,.m4a,.flac,.aif,.aiff,.ogg', lyrics: '.txt,.md', midi: '.mid,.midi' }[kind], onchange: (e) => e.target.files[0] && upload(kind, e.target.files[0]) });
      const slot = h('div', { class: 'slot', 'data-state': path ? (exists ? 'ok' : 'missing') : '' },
        h('div', { class: 'slot-head' }, h('span', {}, KIND_LABEL[kind]), path ? h('span', { class: 'chip ' + (exists ? '' : 'err') }, exists ? basename(path) : 'ファイルが見つかりません') : h('span', { class: 'chip dim' }, '未設定')),
        h('div', { class: 'slot-path' }, path || ''),
        h('div', { class: 'slot-actions' },
          h('button', { class: 'ut-btn-sm', html: ICON.folder + ' 参照…', onclick: async () => { const v = await pickPath({ kind, title: `${KIND_LABEL[kind]}を選択`, start: path ? path.replace(/\/[^/]*$/, '') : (S.cfg.recent_dirs || [])[0] || '' }); if (v) setInput(kind, v); } }),
          h('button', { class: 'ut-btn-sm', html: ICON.up + ' アップロード', onclick: () => fileIn.click() }), fileIn,
          path ? h('button', { class: 'ut-btn-sm ut-btn-ghost', onclick: () => setInput(kind, '') }, 'クリア') : null,
          h('span', { class: 'hint', style: 'margin:0 0 0 auto' }, KIND_HINT[kind])));
      slot.ondragover = (e) => { e.preventDefault(); slot.setAttribute('data-drag', 'true'); };
      slot.ondragleave = () => slot.setAttribute('data-drag', 'false');
      slot.ondrop = (e) => { e.preventDefault(); slot.setAttribute('data-drag', 'false'); const f = e.dataTransfer.files[0]; if (f) upload(kind, f); };
      inputsCard.append(slot);
    }
    page.append(inputsCard);
    async function setInput(kind, v) { try { S.project = await api('PUT', `projects/${enc(S.slug)}`, { inputs: { [kind]: v } }); if (v) rememberDir(v.replace(/\/[^/]*$/, '')); await loadProjects(); renderInput(); } catch (e) { toast(e.message, 'error'); } }
    async function upload(kind, file) {
      const fd = new FormData(); fd.append('file', file, file.name);
      try { await api('POST', `projects/${enc(S.slug)}/upload/${kind}`, fd, true); toast(`${file.name} をプロジェクトに取り込みました`, 'success'); await loadProjects(); renderInput(); } catch (e) { toast(e.message, 'error'); }
    }
    // ---- 解析オプション
    const models = S.models.filter((m) => m.kind === 'hf_ctc');
    const modelSel = h('select', { class: 'field-w', onchange: (e) => opt({ model: e.target.value }) },
      ...models.map((m) => h('option', { value: m.id, selected: m.id === o.model }, `${m.id}  [${m.license}${m.downloaded ? '' : ' / 未取得'}]`)),
      ...(models.find((m) => m.id === o.model) ? [] : [h('option', { value: o.model, selected: true }, o.model)]));
    const curModel = S.models.find((m) => m.id === o.model);
    const modelState = h('div', { class: 'hint' }, curModel ? (curModel.downloaded ? `保管済み: ${curModel.path}` : '未取得 — 解析開始時に自動で取得します (「設定」で事前取得もできます)') : 'モデル情報なし');
    page.append(h('div', { class: 'card' }, h('div', { class: 'card-title' }, '解析オプション'),
      h('div', { class: 'grid2' },
        h('div', {},
          h('label', { class: 'row' }, h('span', {}, 'モデル'), modelSel), modelState,
          h('label', { class: 'row' }, h('span', {}, 'デバイス'), h('select', { onchange: (e) => opt({ device: e.target.value }) }, ...['auto', 'mps', 'cpu', 'cuda'].map((d) => h('option', { value: d, selected: (o.device || 'auto') === d }, d))), h('span', { class: 'hint', style: 'margin:0' }, `auto = ${S.cfg._paths.device}`)),
          h('label', { class: 'row' }, h('input', { type: 'checkbox', checked: !!o.star, onchange: (e) => opt({ star: e.target.checked }) }), h('span', {}, '行境界に star トークン (歌詞にない発声を吸収)'))),
        h('div', {},
          h('label', { class: 'row' }, h('span', {}, 'フレーズ分割の休符'), h('input', { type: 'number', class: 'num-w', step: 0.05, min: 0.05, value: o.min_rest ?? 0.2, onchange: (e) => opt({ min_rest: parseFloat(e.target.value) || 0.2 }) }), h('span', { class: 'secondary' }, '秒')),
          h('label', { class: 'row' }, h('span', {}, 'MIDI トラック'), h('input', { type: 'text', class: 'num-w', placeholder: '全部', value: o.midi_tracks || '', onchange: (e) => opt({ midi_tracks: e.target.value }) }), h('span', { class: 'hint', style: 'margin:0' }, 'カンマ区切りの番号、空欄で全トラック')),
          h('label', { class: 'row' }, h('input', { type: 'checkbox', checked: !!o.keep_drums, onchange: (e) => opt({ keep_drums: e.target.checked }) }), h('span', {}, 'ドラム系トラック / ch10 を除外しない'))))));
    function opt(patch) { S.project.options = { ...S.project.options, ...patch }; saveProject({ options: patch }); }
    // ---- 読み上書き
    page.append(h('div', { class: 'card' }, h('div', { class: 'card-title' }, '読みの上書き', h('span', { class: 'hint', style: 'margin:0 0 0 8px;font-weight:400' }, '1 行 1 語: 「表層 読み」 例: 言葉 こと|ば   (| で文字ごとに区切る)')),
      h('textarea', { rows: 4, placeholder: '景色 け|しき\n言葉 こと|ば', oninput: (e) => saveProject({ readings: e.target.value }) }, p.readings || '')));
    // ---- 実行 + ログ
    const ready = Object.values(p.inputs_exist).every(Boolean);
    const runBtn = h('button', { class: 'ut-btn-sm ut-btn-accent', disabled: !ready || !!S.running[p.slug], onclick: () => runAlign() }, S.running[p.slug] ? '解析中…' : '解析を実行');
    const log = h('div', { class: 'log', id: 'align-log' }, S.running[p.slug] ? S.running[p.slug].join('\n') : (p.last_run ? `前回: ${p.last_run.generated}  ${p.last_run.model}  残差 中央値 ${fmtMs(p.last_run.residual_median)} p90 ${fmtMs(p.last_run.residual_p90)}` : ''));
    page.append(h('div', { class: 'card' }, h('div', { class: 'card-title' }, '解析', h('span', { class: 'spacer' }), ready ? null : h('span', { class: 'chip warn' }, '入力ファイルが揃っていません'), runBtn), log));
    page.append(h('div', { class: 'card' }, h('div', { class: 'card-title' }, '仕組み'),
      h('div', { class: 'hint', style: 'line-height:1.6' }, '歌詞 → 読み付与・モーラ分割 (fugashi/unidic) → 音声の CTC 強制アライメントでモーラ開始時刻を検出 → MIDI ノートと DP で突合 (1 ノート複数モーラ・メリスマ・skip) → フレーズ先頭は MIDI、フレーズ内は CTC を MIDI±100ms でクランプして時刻確定。結果は「結果」タブで波形・ピアノロール・カラオケ表示で確認し、「書き出し」で UTAVISTA 用 JSON にします。')));
  }
  function rememberDir(d) { if (!d) return; const rd = (S.cfg.recent_dirs || []).filter((x) => x !== d); rd.unshift(d); S.cfg.recent_dirs = rd.slice(0, 10); api('PUT', 'config', { recent_dirs: S.cfg.recent_dirs }).catch(() => {}); }
  function appendLog(lines) {
    const el = $('align-log'); if (!el) return;
    for (const l of lines) { const cls = /^\[warn\]/.test(l) ? 'l-warn' : /Traceback|Error/.test(l) ? 'l-err' : /^\[done\]/.test(l) ? 'l-done' : ''; el.append(h('div', { class: cls }, l)); }
    el.scrollTop = el.scrollHeight;
  }
  async function runAlign() {
    if (!S.project || S.running[S.slug]) return;
    const slug = S.slug;
    try {
      await flushAllSaves();
      const job = await api('POST', `projects/${enc(slug)}/align`);
      S.running[slug] = []; renderProjectList(); if (S.mode === 'input') renderInput(); else setMode('input');
      $('top-status').textContent = '解析中…';
      watchJob(job, {
        onLog: (lines) => { S.running[slug].push(...lines); if (S.slug === slug) appendLog(lines); },
        onDone: async (j) => { delete S.running[slug]; $('top-status').textContent = ''; toast(`解析完了: 残差 中央値 ${fmtMs(j.result.residual_median)} / p90 ${fmtMs(j.result.residual_p90)}`, 'success', 8000); await loadProjects(); if (S.slug === slug) setMode('result'); },
        onError: async (j) => { delete S.running[slug]; $('top-status').textContent = ''; toast('解析に失敗しました: ' + j.error, 'error', 12000); if (S.slug === slug) appendLog(j.log || []); await loadProjects(); },
      });
    } catch (e) { toast(e.message, 'error'); }
  }

  // ---------------------------------------------------------------- 結果
  async function renderResult() {
    const page = $('page-result'); page.innerHTML = '';
    if (!S.project) { page.append(h('div', { class: 'empty' }, 'プロジェクトを選んでください。')); return; }
    if (!S.project.has_result) { page.append(h('div', { class: 'empty' }, '結果がまだありません。「入力」タブで解析を実行してください。')); return; }
    const r = S.project.last_run || {};
    const stats = h('div', { class: 'stat-row', style: 'padding:8px 12px 0' },
      stat('残差 中央値', fmtMs(r.residual_median), r.residual_median > 0.04), stat('残差 p90', fmtMs(r.residual_p90), r.residual_p90 > 0.15),
      stat('150ms 超', fmtPct(r.residual_over150), r.residual_over150 > 0.05), stat('MIDI→音声', fmtMs(r.offset)),
      stat('未割当モーラ', String(r.n_mora_skipped ?? '–'), r.n_mora_skipped > 0), stat('歌われないノート', String(r.n_note_skipped ?? '–')),
      stat('繰返し行の不一致', `${r.repeat_mismatches ?? '–'} / ${r.repeated_lines ?? '–'}`),
      h('div', { class: 'stat', style: 'flex:1;min-width:200px' }, h('div', { class: 'k' }, 'モデル'), h('div', { style: 'font-size:11px', class: 'mono' }, `${r.model || ''}`)));
    if (r.warnings && r.warnings.length) stats.append(h('div', { class: 'stat bad', style: 'flex-basis:100%' }, h('div', { class: 'k' }, '警告'), ...r.warnings.map((w) => h('div', { class: 'warn', style: 'font-size:12px' }, w))));
    page.append(stats);
    const host = h('div', { style: 'flex:1;min-height:0' });
    page.append(host);
    S.viewer = await createViewer(host, { base: `/projects/${enc(S.slug)}/out/` });
  }
  function stat(k, v, bad) { return h('div', { class: 'stat' + (bad ? ' bad' : '') }, h('div', { class: 'k' }, k), h('div', { class: 'v' }, v)); }

  // ---------------------------------------------------------------- 書き出し
  function renderExport() {
    const page = $('page-export'); page.innerHTML = '';
    if (!S.project) { page.append(h('div', { class: 'empty' }, 'プロジェクトを選んでください。')); return; }
    const p = S.project, o = p.options || {}, ex = p.last_export;
    page.append(h('div', { class: 'card' }, h('div', { class: 'card-title' }, 'UTAVISTA 歌詞タイミング (lyrics-timing/1.0)'),
      h('div', { class: 'hint', style: 'margin-bottom:8px' }, 'phrase = 歌詞の 1 行、word = 行内の空白区切り、char = 1 文字。UTAVISTA の Lyrics タブ →「歌詞読込」でこの JSON を指定します。歌われない文字 (記号・括弧内) は隣接文字の端 10ms を借りた区間で出力し sung:false を付けます。'),
      h('label', { class: 'row' }, h('input', { type: 'checkbox', checked: o.export_ruby !== false, onchange: (e) => opt({ export_ruby: e.target.checked }) }), h('span', {}, '漢字にルビ (読み) を付与する')),
      h('label', { class: 'row' }, h('input', { type: 'checkbox', checked: !!o.export_strip_spaces, onchange: (e) => opt({ export_strip_spaces: e.target.checked }) }), h('span', {}, 'phrase.text から空白を除く (UTAVISTA の「空白を無視する」既定と同じ形)')),
      h('div', { class: 'actions' },
        h('button', { class: 'ut-btn-sm ut-btn-accent', disabled: !p.has_result, onclick: runExport }, '書き出す'),
        p.has_export ? h('a', { class: 'ut-btn-sm', href: `/projects/${enc(p.slug)}/out/utavista-lyrics-timing.json`, download: `${p.name}-lyrics-timing.json`, html: ICON.dl + ' JSON をダウンロード' }) : null,
        p.has_export ? h('span', { class: 'mono secondary', style: 'font-size:11px;word-break:break-all' }, `${p.path}/out/utavista-lyrics-timing.json`) : null,
        p.has_result ? null : h('span', { class: 'chip warn' }, '先に解析を実行してください'))));
    if (ex) {
      const issues = ex.issues || [];
      page.append(h('div', { class: 'card' }, h('div', { class: 'card-title' }, '前回の書き出し', h('span', { class: 'hint', style: 'margin:0 0 0 8px;font-weight:400' }, ex.generated || '')),
        h('div', { class: 'stat-row' }, stat('phrase', String(ex.phrases)), stat('word', String(ex.words)), stat('char', String(ex.chars)), stat('自己検査', issues.length ? `${issues.length} 件` : 'OK', issues.length > 0)),
        issues.length ? h('div', { class: 'log', style: 'height:120px' }, issues.join('\n')) : null,
        h('div', { class: 'card-title', style: 'margin-top:8px' }, 'UTAVISTA 本体の検証器'),
        ex.validator ? h('div', {}, h('span', { class: 'chip ' + (ex.validator.ok ? '' : 'err') }, ex.validator.ok ? '検証 OK' : '検証 NG'), h('div', { class: 'log', style: 'height:160px;margin-top:6px' }, ex.validator.output || ''))
          : h('div', { class: 'hint' }, '「設定」で utavista2 のリポジトリを指定すると、書き出し時に validateLyricsTiming / インポータで自動検査します。')));
    }
    const log = h('div', { class: 'log', id: 'export-log', style: 'height:140px' });
    page.append(h('div', { class: 'card' }, h('div', { class: 'card-title' }, 'ログ'), log));
    function opt(patch) { S.project.options = { ...S.project.options, ...patch }; saveProject({ options: patch }); }
    async function runExport() {
      try {
        await flushAllSaves();
        const job = await api('POST', `projects/${enc(p.slug)}/export`);
        watchJob(job, {
          onLog: (lines) => { const el = $('export-log'); if (el) { el.append(...lines.map((l) => h('div', {}, l))); el.scrollTop = el.scrollHeight; } },
          onDone: async () => { toast('書き出しました', 'success'); await loadProjects(); if (S.mode === 'export') renderExport(); },
          onError: (j) => toast('書き出しに失敗: ' + j.error, 'error', 10000),
        });
      } catch (e) { toast(e.message, 'error'); }
    }
  }

  // ---------------------------------------------------------------- 設定
  async function loadConfig() { S.cfg = await api('GET', 'config'); S.models = (await api('GET', 'models')).models; }
  function renderSettings() {
    const page = $('page-settings'); page.innerHTML = '';
    const c = S.cfg; const pending = {};
    const pathRow = (key, label, hint, isDir = true) => {
      const inp = h('input', { type: 'text', class: 'field-w', value: c[key] || '', oninput: (e) => { pending[key] = e.target.value; e.target.classList.add('pending'); } });
      return h('div', {}, h('label', { class: 'row' }, h('span', {}, label), inp, isDir ? h('button', { class: 'ut-btn-sm', html: ICON.folder, title: 'フォルダを選択', onclick: async () => { const v = await pickPath({ kind: 'dir', title: `${label}を選択`, start: c._paths[key] || c[key] || '' }); if (v) { inp.value = v; pending[key] = v; inp.classList.add('pending'); } } }) : null),
        h('div', { class: 'hint', style: 'margin-left:116px' }, hint));
    };
    const saveBtn = h('button', { class: 'ut-btn-sm ut-btn-accent', onclick: async () => { try { S.cfg = await api('PUT', 'config', pending); S.models = (await api('GET', 'models')).models; toast('設定を保存しました', 'success'); renderSettings(); } catch (e) { toast(e.message, 'error'); } } }, '保存');
    const hfModels = S.models.filter((m) => m.kind === 'hf_ctc');
    page.append(h('div', { class: 'card' }, h('div', { class: 'card-title' }, '場所', h('span', { class: 'spacer' }), saveBtn),
      h('div', { class: 'hint', style: 'margin-bottom:8px' }, `設定ファイル: ${c._paths.config}  (環境変数 UTALIGN_HOME で場所を変えられます)`),
      pathRow('models_dir', 'モデル保管場所', `現在: ${c._paths.models_dir} — ここに HF スナップショットを丸ごと保存し、以後は Hub に接続しません`),
      pathRow('projects_dir', 'プロジェクト置き場', `現在: ${c._paths.projects_dir} — 各プロジェクトの入力コピー・出力・キャッシュ`),
      pathRow('utavista_dir', 'utavista2 リポジトリ', '任意。指定すると書き出し時に UTAVISTA 本体の検証器 (tsx) で検査します')));
    page.append(h('div', { class: 'card' }, h('div', { class: 'card-title' }, '既定値 (新規プロジェクトに適用)'),
      h('div', { class: 'grid2' },
        h('div', {},
          h('label', { class: 'row' }, h('span', {}, 'モデル'), h('select', { class: 'field-w', onchange: (e) => { pending.model = e.target.value; } }, ...hfModels.map((m) => h('option', { value: m.id, selected: m.id === c.model }, m.id)))),
          h('label', { class: 'row' }, h('span', {}, 'デバイス'), h('select', { onchange: (e) => { pending.device = e.target.value; } }, ...['auto', 'mps', 'cpu', 'cuda'].map((d) => h('option', { value: d, selected: c.device === d }, d))), h('span', { class: 'hint', style: 'margin:0' }, `auto = ${c._paths.device}`))),
        h('div', {},
          h('label', { class: 'row' }, h('span', {}, 'ポート'), h('input', { type: 'number', class: 'num-w', value: c.port, onchange: (e) => { pending.port = parseInt(e.target.value, 10) || 8791; } }), h('span', { class: 'hint', style: 'margin:0' }, '次回起動から')),
          h('label', { class: 'row' }, h('span', {}, 'フレーズ分割の休符'), h('input', { type: 'number', class: 'num-w', step: 0.05, value: c.min_rest, onchange: (e) => { pending.min_rest = parseFloat(e.target.value) || 0.2; } }), h('span', { class: 'secondary' }, '秒')),
          h('label', { class: 'row' }, h('input', { type: 'checkbox', checked: c.export_ruby !== false, onchange: (e) => { pending.export_ruby = e.target.checked; } }), h('span', {}, '書き出しでルビを付与')),
          h('label', { class: 'row' }, h('input', { type: 'checkbox', checked: !!c.export_strip_spaces, onchange: (e) => { pending.export_strip_spaces = e.target.checked; } }), h('span', {}, '書き出しで空白を除く'))))));
    // ---- モデル
    const tbl = h('table', { class: 'tbl' }, h('thead', {}, h('tr', {}, h('th', {}, '状態'), h('th', {}, 'モデル'), h('th', {}, 'ライセンス'), h('th', {}, '語彙'), h('th', {}, 'サイズ'), h('th', {}, '備考'), h('th', {}, ''))));
    const tb = h('tbody'); tbl.append(tb);
    for (const m of S.models) {
      const busy = S.jobs[`dl:${m.id}`];
      const dlBtn = h('button', { class: 'ut-btn-sm', disabled: !!busy, html: (busy ? ICON.spin : ICON.dl) + (busy ? ' 取得中' : ' 取得'), onclick: async () => {
        try { const job = await api('POST', 'models/download', { id: m.id }); S.jobs[`dl:${m.id}`] = job; renderSettings(); toast(`${m.id} を取得しています (${m.size_mb} MB)`);
          watchJob(job, { onDone: async () => { delete S.jobs[`dl:${m.id}`]; S.models = (await api('GET', 'models')).models; toast('取得しました', 'success'); if (S.mode === 'settings') renderSettings(); }, onError: (j) => { delete S.jobs[`dl:${m.id}`]; toast('取得に失敗: ' + j.error, 'error', 10000); if (S.mode === 'settings') renderSettings(); } }); } catch (e) { toast(e.message, 'error'); } } });
      const exBtn = h('button', { class: 'ut-btn-sm', title: '別フォルダにコピー (バックアップ)', onclick: async () => { const d = await pickPath({ kind: 'dir', title: 'コピー先フォルダ' }); if (!d) return; try { const r = await api('POST', 'models/export', { id: m.id, path: d }); toast('コピーしました: ' + r.path, 'success'); } catch (e) { toast(e.message, 'error'); } } }, 'バックアップ');
      const rmBtn = h('button', { class: 'ut-btn-sm ut-btn-danger', html: ICON.trash, title: '保管場所から削除', onclick: async () => { if (!confirm(`${m.id} をローカルから削除しますか?`)) return; await api('POST', 'models/delete', { id: m.id }); S.models = (await api('GET', 'models')).models; renderSettings(); } });
      tb.append(h('tr', {},
        h('td', {}, h('span', { class: 'chip ' + (m.downloaded ? '' : 'dim') }, m.downloaded ? '保管済み' : '未取得')),
        h('td', { title: m.downloaded ? m.path : '' }, h('div', { class: 'mono', style: 'font-size:11px' }, m.id)),
        h('td', {}, h('span', { class: 'chip ' + (m.commercial ? '' : 'warn') }, m.license)),
        h('td', {}, m.vocab), h('td', { class: 'mono nowrap' }, `${m.downloaded ? m.local_size_mb : m.size_mb} MB`), h('td', { class: 'secondary', style: 'font-size:11px' }, m.note),
        h('td', {}, h('div', { style: 'display:flex;gap:4px;justify-content:flex-end' }, m.downloaded ? [exBtn, rmBtn] : [dlBtn]))));
    }
    const importBtn = h('button', { class: 'ut-btn-sm', html: ICON.folder + ' フォルダから取り込み…', onclick: async () => {
      const d = await pickPath({ kind: 'dir', title: 'モデルフォルダ (config.json と safetensors があるもの)' }); if (!d) return;
      const id = prompt('モデル ID (HF の repo id 形式。空欄ならフォルダ名から)', basename(d).replace('--', '/')); if (id === null) return;
      try { const r = await api('POST', 'models/import', { path: d, id: id || null }); S.models = (await api('GET', 'models')).models; toast('取り込みました: ' + r.id, 'success'); renderSettings(); } catch (e) { toast(e.message, 'error'); } } });
    const customIn = h('input', { type: 'text', placeholder: 'HF repo id (例: TKU410410103/wav2vec2-base-japanese-asr)', style: 'width:360px' });
    const customBtn = h('button', { class: 'ut-btn-sm', html: ICON.dl + ' 取得', onclick: async () => { const id = customIn.value.trim(); if (!id) return; try { const job = await api('POST', 'models/download', { id }); toast(`${id} を取得しています`); watchJob(job, { onDone: async () => { S.models = (await api('GET', 'models')).models; toast('取得しました', 'success'); if (S.mode === 'settings') renderSettings(); }, onError: (j) => toast('取得に失敗: ' + j.error, 'error', 10000) }); } catch (e) { toast(e.message, 'error'); } } });
    page.append(h('div', { class: 'card' }, h('div', { class: 'card-title' }, 'モデル', h('span', { class: 'hint', style: 'margin:0 0 0 8px;font-weight:400' }, '保管場所に置いたモデルはオフラインで使えます。「バックアップ」で別フォルダにコピー、「取り込み」でコピーから復元できます。'), h('span', { class: 'spacer' }), importBtn),
      h('div', { style: 'overflow:auto;max-height:50vh' }, tbl),
      h('div', { class: 'actions' }, h('span', { class: 'secondary', style: 'font-size:12px' }, '登録簿にないモデル:'), customIn, customBtn),
      h('div', { class: 'hint' }, 'CTC ヘッド付きで語彙が音素 (OpenJTalk 系) またはかなの Wav2Vec2ForCTC / HubertForCTC のみ対応。漢字語彙のモデルは使えません。')));
  }

  // ---------------------------------------------------------------- init
  (async () => {
    try { await loadConfig(); await loadProjects(); } catch (e) { toast('初期化に失敗: ' + e.message, 'error', 0); }
    try { const act = await api('GET', 'jobs'); for (const j of act) if (j.kind === 'align' && j.project) { S.running[j.project] = []; watchJob(j, { onLog: (l) => { S.running[j.project] && S.running[j.project].push(...l); if (S.slug === j.project) appendLog(l); }, onDone: async () => { delete S.running[j.project]; await loadProjects(); }, onError: async () => { delete S.running[j.project]; await loadProjects(); } }); } renderProjectList(); } catch (e) { /* ignore */ }
  })();
})();
