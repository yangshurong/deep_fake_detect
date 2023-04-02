import json
path1 = '/mnt/f/home/CADDM/test_FF/ldm.json'
with open(path1, 'r', encoding='utf-8') as f:
    s = f.read()
    res = json.loads(s)

path1 = '/mnt/f/home/CADDM/ldm.json'
with open(path1, 'r', encoding='utf-8') as f:
    s = f.read()
    res = json.loads(s)
