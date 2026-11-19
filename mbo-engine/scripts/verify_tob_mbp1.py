#!/usr/bin/env python3
"""Phase 2 acceptance: our per-order book's top of book vs Databento mbp-1.
Usage: verify_tob_mbp1.py TOB_DUMP MBP1_DBN

mbp-1 is NOT a 1:1 projection of the MBO records: it publishes only records that touch the top
levels, reorders/merges records inside a multi-record event, and shows some cancels as side N.
So records cannot be paired one by one. Instead both streams are grouped by packet
(ts_recv, sequence) and the top of book AFTER THE LAST record of each group is compared:
best bid/ask price, total size and order count. Groups present in mbp-1 are checked against
our state at the end of the same group (our dump has every A/C/M/T record; F/N do not change
the book). The same comparison is also reported for groups whose last mbp-1 record is F_LAST.
"""
import sys

import databento as db
import numpy as np

ROW = np.dtype([("ts_recv", "<u8"), ("sequence", "<u4"), ("action", "u1"), ("side", "u1"), ("flags", "u1"), ("p0", "u1"),
                ("price", "<i8"), ("size", "<u4"), ("bid_ct", "<u4"), ("bid_px", "<i8"), ("ask_px", "<i8"),
                ("bid_sz", "<u4"), ("ask_sz", "<u4"), ("ask_ct", "<u4"), ("p1", "<u4")])
assert ROW.itemsize == 64
UND = 9223372036854775807

o = np.fromfile(sys.argv[1], dtype=ROW)
m = db.DBNStore.from_file(sys.argv[2]).to_ndarray()
print(f"ours {len(o):,} A/C/M/T records in window, mbp-1 {len(m):,} records")
def last_of_group(ts, seq):
    g = np.lexsort((np.arange(len(ts)), seq, ts))          # stable by (ts, seq), file order inside
    ts_s, seq_s = ts[g], seq[g]
    end = np.ones(len(g), bool); end[:-1] = (ts_s[1:] != ts_s[:-1]) | (seq_s[1:] != seq_s[:-1])
    return g[end], ts_s[end], seq_s[end]
o_all, m_all = o, m
io, to, so = last_of_group(o["ts_recv"], o["sequence"])
im, tm, sm = last_of_group(m["ts_recv"], m["sequence"])
# map mbp-1 group -> our group by (ts, seq)
ours_key = {(int(a), int(b)): int(i) for a, b, i in zip(to, so, io)}
pairs = [(ours_key.get((int(a), int(b))), int(i)) for a, b, i in zip(tm, sm, im)]
found = [(x, y) for x, y in pairs if x is not None]
print(f"mbp-1 packet groups: {len(im):,}; found in our stream: {len(found):,}")
if len(found) != len(im):
    miss = [(int(a), int(b)) for (x, _), a, b in zip(pairs, tm, sm) if x is None][:3]
    print("  groups missing from our stream (first 3):", miss)
oi = np.array([x for x, _ in found]); mi = np.array([y for _, y in found])
o, m = o[oi], m[mi]
n = len(m)
mb = np.where(m["bid_px_00"] == UND, 0, m["bid_px_00"]); ma = np.where(m["ask_px_00"] == UND, 0, m["ask_px_00"])
last = (m["flags"] & 128) != 0
chk = {
    "bid_px": o["bid_px"] == mb, "ask_px": o["ask_px"] == ma,
    "bid_sz": o["bid_sz"] == m["bid_sz_00"], "ask_sz": o["ask_sz"] == m["ask_sz_00"],
    "bid_ct": o["bid_ct"] == m["bid_ct_00"], "ask_ct": o["ask_ct"] == m["ask_ct_00"],
}
allok = np.logical_and.reduce(list(chk.values()))
crossed = (mb != 0) & (ma != 0) & (mb >= ma)
for name, mask in (("all packet groups", np.ones(n, bool)), ("groups whose last mbp-1 record is F_LAST", last)):
    print(f"{name}: {int(mask.sum()):,}  full match (px+size+count, both sides): {int((allok & mask).sum()):,} "
          f"({100 * (allok & mask).sum() / max(1, mask.sum()):.4f}%)")
    for k, v in chk.items():
        print(f"   {k:7s} mismatches: {int((~v & mask).sum()):,}")
print(f"mbp-1 groups whose top of book is crossed/locked: {int(crossed.sum()):,}; of those, ours match: {int((allok & crossed).sum()):,}")
bad = np.nonzero(~allok)[0]
print("first mismatching groups (ts_recv, last action, ours bid/ask px,sz,ct vs mbp-1):")
for i in bad[:6]:
    print(" ", int(o["ts_recv"][i]), chr(o["action"][i]), (int(o["bid_px"][i]), int(o["bid_sz"][i]), int(o["bid_ct"][i]), int(o["ask_px"][i]), int(o["ask_sz"][i]), int(o["ask_ct"][i])),
          "vs", (int(mb[i]), int(m["bid_sz_00"][i]), int(m["bid_ct_00"][i]), int(ma[i]), int(m["ask_sz_00"][i]), int(m["ask_ct_00"][i])))

# ---- residual classification (is each leftover difference a known mbp-1 publication artefact?) ----
def state_o(r):  return (int(r["bid_px"]), int(r["bid_sz"]), int(r["bid_ct"]), int(r["ask_px"]), int(r["ask_sz"]), int(r["ask_ct"]))
def state_m(r):
    b = int(r["bid_px_00"]); a = int(r["ask_px_00"])
    return (0 if b == UND else b, int(r["bid_sz_00"]), int(r["bid_ct_00"]), 0 if a == UND else a, int(r["ask_sz_00"]), int(r["ask_ct_00"]))
mfull = m_all[im]   # mbp-1 last-of-group rows, all groups (found or not)
# (0) the book is only defined at event boundaries (F_LAST). Where does each mismatch occur?
our_boundary = (o["flags"] & 128) != 0   # our last record of the packet group carries F_LAST
mm_boundary = int((~allok & our_boundary).sum()); mm_mid = int((~allok & ~our_boundary).sum())
print(f"groups ending at an F_LAST boundary in the MBO stream: {int(our_boundary.sum()):,}; mismatches there: {mm_boundary}")
print(f"groups ending MID-event (last MBO record has no F_LAST): {int((~our_boundary).sum()):,}; mismatches there: {mm_mid}")
# (a) mismatching groups: does our end-of-group state equal mbp-1's state at the NEXT mbp-1 group?
pos_found = np.array([k for k, (x, _) in enumerate(pairs) if x is not None])  # index into mfull of found groups
resolved_a = 0; unresolved = []
for i in bad:
    k = pos_found[i]
    nxt = state_m(mfull[k + 1]) if k + 1 < len(mfull) else None
    if nxt is not None and state_o(o[i]) == nxt: resolved_a += 1
    else: unresolved.append(i)
# (b) mbp-1 groups absent from our dump (records we do not dump, e.g. N): compare to our latest row at or before ts
o_ts = o_all["ts_recv"]; resolved_b = 0; unresolved_b = []
for k, (x, _) in enumerate(pairs):
    if x is not None: continue
    j = np.searchsorted(o_ts, mfull[k]["ts_recv"], side="right") - 1
    if j >= 0 and state_o(o_all[j]) == state_m(mfull[k]): resolved_b += 1
    else: unresolved_b.append(k)
print(f"residual (a) mismatching groups: {len(bad)}; explained (ours == mbp-1 state at next group): {resolved_a}; unexplained: {len(unresolved)}")
print(f"residual (b) mbp-1 groups not in our dump: {len(pairs) - len(found)}; ours equal at that time: {resolved_b}; unexplained: {len(unresolved_b)}")
for i in unresolved[:5]:
    print("  unexplained (a):", int(o["ts_recv"][i]), state_o(o[i]), "vs", state_m(m[i]))
for k in unresolved_b[:5]:
    print("  unexplained (b):", int(mfull[k]["ts_recv"]), state_m(mfull[k]))
sys.exit(0 if (mm_boundary == 0 and not unresolved_b) else 1)
