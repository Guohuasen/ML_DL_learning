#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# sync-to-github.py — 把 LearningPrompt 的论文精读提示词等推送到 GitHub（Guohuasen/ML_DL_learning）
# 用法:  $env:GITHUB_TOKEN="ghp_xxx"; python sync-to-github.py [-m "v4.6: 说明"]
# 原理:  本机沙箱对 D:\.git 有 DENY 写权限，无法 git push；改用 GitHub Contents API 上传（每次 PUT = 一次真实 commit）。
import os, sys, json, base64, urllib.request, urllib.error, urllib.parse

TOKEN = os.environ.get('GITHUB_TOKEN')
if not TOKEN:
    sys.exit('请先设置环境变量 GITHUB_TOKEN')

# 提交说明
MESSAGE = 'sync: prompt update'
args = sys.argv[1:]
if args:
    if args[0] in ('-m', '--message') and len(args) > 1:
        MESSAGE = args[1]
    else:
        MESSAGE = ' '.join(args)

SLUG = 'Guohuasen/ML_DL_learning'
ROOT = os.path.dirname(os.path.abspath(__file__))  # LearningPrompt 目录
FILES = ['LearningPrompt/论文精读提示词.md', 'LearningPrompt/sync-to-github.py', 'CHANGELOG.md', '.gitignore']

def call(url, method='GET', body=None):
    headers = {'Authorization': 'Bearer ' + TOKEN, 'User-Agent': 'codex-sync', 'Accept': 'application/vnd.github+json'}
    data = None
    if body is not None:
        data = json.dumps(body).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()

for f in FILES:
    local = os.path.normpath(os.path.join(ROOT, '..', *f.split('/')))
    if not os.path.exists(local):
        print('skip (not found):', f)
        continue
    content = base64.b64encode(open(local, 'rb').read()).decode('ascii')
    url = 'https://api.github.com/repos/%s/contents/%s' % (SLUG, '/'.join(urllib.parse.quote(s) for s in f.split('/')))
    st, b = call(url)
    body = {'message': MESSAGE, 'branch': 'main', 'content': content}
    if st == 200:
        body['sha'] = json.loads(b)['sha']
    st2, b2 = call(url, 'PUT', body)
    print(f, '->', st2, 'OK' if st2 in (200, 201) else b2[:200])
print('sync done.')
