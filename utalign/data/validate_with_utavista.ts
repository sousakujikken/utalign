/**
 * utalign が出力した lyrics-timing/1.0 JSON を UTAVISTA 本体の検証器・インポータで検査する。
 * 使い方 (utavista2-refactor のリポジトリで実行):
 *   ./node_modules/.bin/tsx <utalign>/tools/validate_with_utavista.ts <json> <歌詞txt> [audioDurationMs]
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
async function main(): Promise<void> {
  const repo = process.cwd();
  const { validateLyricsTiming } = await import(resolve(repo, 'src/lyricsTiming/validate.ts'));
  const { normalizeLyricsTiming } = await import(resolve(repo, 'src/lyricsTiming/normalize.ts'));
  const { parseLyricsJson } = await import(resolve(repo, 'src/app/renderer/lyricsJsonImport.ts'));
  const [jsonPath, lyricsPath, durArg] = process.argv.slice(2);
  if (!jsonPath || !lyricsPath) { console.error('usage: validate_with_utavista.ts <json> <lyrics.txt> [audioDurationMs]'); process.exit(2); }
  const raw = readFileSync(jsonPath, 'utf8');
  const doc = JSON.parse(raw);
  const lines = readFileSync(lyricsPath, 'utf8').split('\n').map((l: string) => l.replace(/\s+/g, '')).filter((l: string) => l !== '');
  const r = validateLyricsTiming(doc, lines, durArg ? { audioDurationMs: Number(durArg) } : undefined);
  console.log('validateLyricsTiming ok =', r.ok);
  if (!r.ok) { for (const i of r.issues.slice(0, 30)) console.log(' ', i.path, i.code, i.message); console.log(' total issues', r.issues.length); process.exit(1); }
  const norm = normalizeLyricsTiming(r.value);
  const viaAi = parseLyricsJson(JSON.stringify({ phrases: norm.phrases }));
  const direct = parseLyricsJson(raw);   // 歌詞読込 ボタンでファイルをそのまま読ませた場合
  console.log('parseLyricsJson (AI 経路) phrases =', viaAi.phrases.length, 'warnings =', viaAi.warnings);
  console.log('parseLyricsJson (歌詞読込 直接) phrases =', direct.phrases.length, 'warnings =', direct.warnings);
  const p = direct.phrases[0]!;
  console.log(' sample:', p.text, p.startMs, '-', p.endMs, '| words', p.words.length, '| chars', p.words[0]!.chars.map((c: any) => c.char + (c.ruby ? `(${c.ruby})` : '') + ':' + c.startMs + '-' + c.endMs).join(' '));
}

main().catch((e) => { console.error(e); process.exit(1); });
