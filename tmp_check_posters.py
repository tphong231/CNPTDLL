import json
from pathlib import Path
import urllib.request

path = Path('project/app/movie_posters.json')
print('exists', path.exists())
with path.open('r', encoding='utf-8') as f:
    data = json.load(f)
print('count', len(data))
for k, v in data.items():
    try:
        req = urllib.request.Request(v, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            print(k, r.status, r.getheader('Content-Type'))
    except Exception as e:
        print(k, 'ERR', e, v)
