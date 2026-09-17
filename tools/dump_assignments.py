"""out/result.json の行ごとのモーラ割当をテキストで表示する (デバッグ用).

使い方: python tools/dump_assignments.py [out/result.json] [行番号 ...]
表記: 文字@開始秒  s{i}/{n}=ノート共有 (i 番目/n モーラ)  m{k}=メリスマ (k ノート)  [SKIP]=未歌唱
"""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".json") else "out/result.json"
lines = {int(x) for x in sys.argv[1:] if x.isdigit()}
r = json.load(open(path, encoding="utf-8"))
m = r["meta"]
print(f"offset={m['offset']:.3f}s resid median={m['residual_median']*1000:.0f}ms p90={m['residual_p90']*1000:.0f}ms "
      f"mora_skipped={m['n_mora_skipped']} note_skipped={m['n_note_skipped']}")
cur = -1
for mo in r["moras"]:
    if lines and mo["line"] not in lines:
        continue
    if mo["line"] != cur:
        cur = mo["line"]; print(f"\nL{cur:2d} ", end="")
    if not mo["sung"]:
        print(f"[{mo['text']}:SKIP]", end=" "); continue
    tag = f"s{mo['share_index']}/{mo['share_count']}" if mo["share_count"] > 1 else (f"m{len(mo['note_ids'])}" if len(mo["note_ids"]) > 1 else "")
    print(f"{mo['text']}@{mo['start']:.2f}{tag}", end=" ")
print()
