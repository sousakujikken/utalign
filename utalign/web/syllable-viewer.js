/* Multilingual result viewer. Source syllables remain distinct from proxy kana. */
(function () {
  window.createSyllableViewer = function (root, opts) {
    const data = opts.result, base = opts.base || './';
    const el = (tag, cls, text) => {
      const node = document.createElement(tag);
      if (cls) node.className = cls;
      if (text != null) node.textContent = text;
      return node;
    };
    const wrap = el('div', 'sv'), header = el('div', 'sv-header');
    const audio = el('audio'); audio.controls = true; audio.preload = 'metadata'; audio.src = base + 'audio.wav';
    const controls = el('div', 'sv-controls'), stop = el('button', 'ut-btn-sm', '再生を止める');
    stop.onclick = () => audio.pause();
    const clock = el('span', 'mono', '0.00 s'), followLabel = el('label'), follow = el('input');
    follow.type = 'checkbox'; followLabel.append(follow, ' 再生箇所を追う');
    const download = el('a', 'ut-btn-sm', '解析JSONを保存'); download.href = base + 'result.json'; download.download = 'syllable-timing.json';
    const detail = el('div', 'sv-detail'), notes = el('div', 'sv-notes');
    controls.append(stop, clock, followLabel, download); header.append(audio, controls, notes, detail);
    const explanation = el('p', 'hint', '原文の下に、各音節に対応する照合用の近似読みを表示します。数字は単語内の音節位置です。点線は音声との一致が弱い箇所です。');
    const lyrics = el('div', 'sv-lyrics'); wrap.append(header, explanation, lyrics); root.replaceChildren(wrap);
    const views = [], wordViews = [];
    for (const [li, text] of data.text.split('\n').entries()) {
      const words = data.words.filter(w => w.line === li); if (!words.length) continue;
      const row = el('div', 'sv-line'), num = el('span', 'sv-number', li + 1), body = el('div');
      row.dataset.line = li + 1;
      const raw = el('div', 'sv-raw', text), boxes = el('div', 'sv-words'); body.append(raw, boxes); row.append(num, body); lyrics.append(row);
      for (const word of words) {
        const box = el('div', 'sv-word'), spelling = el('span', 'sv-spelling', word.text);
        const sounds = el('div', 'sv-sounds'); box.append(spelling, el('small', 'secondary', word.language), sounds); boxes.append(box);
        wordViews.push({ word, box });
        for (const id of word.syllable_ids) {
          const s = data.syllables[id], button = el('button', 'sv-syllable' + (s.status === 'low_support' ? ' low' : ''));
          button.dataset.syllable = id; button.dataset.word = word.text;
          button.title = '原語の発音仮説: ' + s.reading;
          button.append(el('span', '', s.proxy_reading || s.reading), el('small', '', `${s.syllable_index + 1}/${word.syllable_ids.length} · ${s.start.toFixed(2)}s`));
          button.onclick = () => { audio.currentTime = s.start; audio.play().catch(() => {}); update(); };
          sounds.append(button); views.push({ s, button, row });
        }
      }
    }
    let last = -1, animation = null, alive = true;
    function update() {
      if (!alive) return;
      const t = audio.currentTime; clock.textContent = t.toFixed(2) + ' s'; let active = null;
      for (const v of views) { const yes = v.s.start <= t && t < v.s.end; v.button.classList.toggle('now', yes); if (yes) active = v; }
      for (const v of wordViews) v.box.classList.toggle('now', v.word.start <= t && t < v.word.end);
      if (!active) { last = -1; detail.textContent = 'この時刻に割り当てられた音節はありません。'; notes.replaceChildren(); return; }
      if (last === active.s.id) return;
      last = active.s.id;
      const s = active.s, w = data.words[s.word_id];
      detail.textContent = `${w.text} の ${s.syllable_index + 1}/${w.syllable_ids.length}音節（照合音「${s.proxy_reading || s.reading}」） ${s.start.toFixed(2)}–${s.end.toFixed(2)}秒 · 母音核 ${s.nucleus_start.toFixed(2)}秒${s.status === 'low_support' ? ' · 音声支持が弱い' : ''}`;
      notes.replaceChildren();
      const ids = s.note_ids;
      for (const n of data.notes.slice(Math.max(0, ids[0] - 2), ids[ids.length - 1] + 3)) {
        notes.append(el('span', 'sv-note' + (ids.includes(n.id) ? ' now' : ''), `#${n.id + 1} MIDI ${n.pitch} ${n.start.toFixed(2)}–${n.end.toFixed(2)}`));
      }
      if (follow.checked) active.row.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }
    function tick() {
      if (!alive || !document.body.contains(wrap)) { destroy(); return; }
      update(); animation = audio.paused ? null : requestAnimationFrame(tick);
    }
    function destroy() { alive = false; if (animation !== null) cancelAnimationFrame(animation); animation = null; audio.pause(); }
    audio.addEventListener('play', () => { if (animation === null) tick(); });
    audio.addEventListener('pause', () => { if (animation !== null) cancelAnimationFrame(animation); animation = null; update(); });
    audio.addEventListener('seeked', update); audio.addEventListener('timeupdate', update); update();
    return { meta: data.meta, destroy };
  };
})();
