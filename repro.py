#!/usr/bin/env python3
"""Prompt-eval time vs number of new tokens against a running llama-server.

Hybrid/recurrent models save a context checkpoint per request. On a Vulkan UMA device the
checkpoint is read back from write-combined memory, which adds a fixed cost to every request
that shows up most clearly on short prompts over a cached prefix.

usage: repro.py [BASE_URL]   (default http://127.0.0.1:8099)
"""
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8099'


def call(prompt, cache=True):
    body = {'prompt': prompt, 'n_predict': 1, 'cache_prompt': cache, 'temperature': 0}
    req = urllib.request.Request(f'{BASE}/completion', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    t = json.load(urllib.request.urlopen(req))['timings']
    return t['prompt_n'], t['prompt_ms']


words = ('apples pears plums apricots grapes ' * 400).split()
prefix = ' '.join(words[:170])
for _ in range(3):
    call(prefix)

print('cached prefix + N new words   (prompt_n tokens evaluated, prompt_ms)')
for n in (1, 4, 16, 51, 200, 800):
    call(prefix)  # put the slot back on the prefix
    tokens, ms = call(prefix + ' ' + ' '.join(f'x{i}' for i in range(n)))
    print(f'  N={n:<4} {tokens:>5} tok  {ms:8.1f} ms')

print('cold, cache_prompt=false')
for n in (16, 51, 222, 1000):
    tokens, ms = call(' '.join(words[:n]), cache=False)
    print(f'  N={n:<4} {tokens:>5} tok  {ms:8.1f} ms')
