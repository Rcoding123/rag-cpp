"""Uncovered regions per agent header, from `llvm-cov export -format=text`.

A region is uncovered when its execution count, summed over every function
(template instantiation) that maps it, is 0.
"""
import json
import os
import sys
from collections import defaultdict

data = json.load(open(sys.argv[1]))
W = "/home/user/rag-cpp/.claude/worktrees/agent-a281e366cb7a121d6/include/rag/agent/"
counts = defaultdict(int)  # (file, l1, c1, l2, c2, kind) -> summed count
for export in data["data"]:
    for fn in export["functions"]:
        files = fn["filenames"]
        for r in fn["regions"]:
            l1, c1, l2, c2, count, file_id, exp_file_id, kind = r
            f = files[file_id]
            if not f.startswith(W):
                continue
            if kind != 0:  # 0 = code region; skip expansion/skipped/gap
                continue
            counts[(os.path.basename(f), l1, c1, l2, c2)] += count

src = {}
by_file = defaultdict(list)
for (f, l1, c1, l2, c2), n in sorted(counts.items()):
    if n == 0:
        by_file[f].append((l1, c1, l2, c2))
for f in sorted(by_file):
    lines = open(W + f).read().split("\n")
    print("==", f)
    for l1, c1, l2, c2 in by_file[f]:
        text = lines[l1 - 1]
        snippet = text[c1 - 1 : (c2 - 1 if l2 == l1 else len(text))].strip()
        print(f"  {l1}:{c1}-{l2}:{c2}  {snippet[:110]}")
