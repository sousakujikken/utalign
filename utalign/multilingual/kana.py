"""Presentation of Japanese model phonemes as an approximate kana reading."""

def kana_hint(groups):
    """Approximate display: a coda kana does NOT insert its vowel in the model input."""
    vowels={'a':'ア','i':'イ','u':'ウ','e':'エ','o':'オ','I':'イ','U':'ウ'}
    rows={
        'k':'カキクケコ','g':'ガギグゲゴ','s':'サシスセソ','z':'ザジズゼゾ',
        't':['タ','ティ','トゥ','テ','ト'],
        'n':'ナニヌネノ','h':'ハヒフヘホ','m':'マミムメモ','r':'ラリルレロ',
        'b':'バビブベボ','p':'パピプペポ',
        'v':['ヴァ','ヴィ','ヴ','ヴェ','ヴォ'],'f':['ファ','フィ','フ','フェ','フォ'],
        'sh':['シャ','シ','シュ','シェ','ショ'],'ch':['チャ','チ','チュ','チェ','チョ'],
        'ky':['キャ','キ','キュ','キェ','キョ'],'gy':['ギャ','ギ','ギュ','ギェ','ギョ'],
        'j':['ジャ','ジ','ジュ','ジェ','ジョ'],'w':['ワ','ウィ','ウ','ウェ','ウォ'],
        'y':['ヤ','イ','ユ','イェ','ヨ'],'d':['ダ','ディ','ドゥ','デ','ド']}
    coda={'N':'ン','n':'ン','cl':'ッ','k':'ク','g':'グ','s':'ス','z':'ズ','t':'ト','d':'ド',
          'r':'ル','v':'ヴ','b':'ブ','p':'プ','f':'フ','sh':'シュ','ch':'チ','j':'ジ','m':'ム','h':'フ','y':'イ','w':'ウ'}
    rendered=[]
    for group in groups:
        result=[];i=0
        while i<len(group):
            p=group[i]
            if p in rows and i+1<len(group) and group[i+1].lower() in 'aiueo' and len(group[i+1])==1:
                result.append(rows[p]['aiueo'.index(group[i+1].lower())]);i+=2
            else:
                result.append(vowels.get(p,coda.get(p,f'〈{p}〉')));i+=1
        rendered.append(''.join(result))
    return '|'.join(rendered)
