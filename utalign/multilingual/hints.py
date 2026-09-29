"""Plain-text project inputs; the CLI also accepts structured JSON hints."""
def language_hints(text):
    spans=[]
    for line in text.splitlines():
        if not line.strip():continue
        language,sep,source=line.partition(':')
        if not sep or not source.strip():
            raise ValueError('言語ヒントは「fr: Bonjour」の形式で指定してください')
        spans.append(dict(language=language.strip(),text=source.strip()))
    return spans


def proxy_readings(text):
    spans=[]
    for line in text.splitlines():
        if not line.strip():continue
        source,sep,reading=line.partition('=')
        if not sep or not source.strip() or not reading.strip():
            raise ValueError('近似読みは「knock out = ノッ / キャウ」の形式で指定してください')
        spans.append(dict(text=source.strip(),syllables=[[s.strip() for s in word.split('|')] for word in reading.split('/')]))
    return spans
