"""Independent check of data/features/levels/session_levels.parquet.
Sessions come from the EXCHANGE'S scheduled open times in the status feed (no timezone logic); RTH is
[open+15.5h, open+22h) = 08:30-15:00 CT because sessions never straddle a DST change (it happens Sunday 02:00).
Trades come straight from the Python databento reader. Compares count, volume, open/high/low/close, vwap for
ETH and RTH per (session, instrument); also invariants."""
import glob
import sys

import databento as db
import numpy as np
import pandas as pd

H = 3600 * 10**9
st = np.concatenate([db.DBNStore.from_file(f).to_ndarray() for f in sorted(glob.glob("/srv/quant/data/raw/es-status/*/*.status.dbn.zst"))])
o = st[(st["reason"] == 1) & (st["action"] == 7) & (st["is_trading"] == b"Y") & ((st["ts_recv"].astype("int64") % (86400 * 10**9)) > 5 * 10**9)]
ot = np.sort(o["ts_recv"].astype("int64"))
opens = ot[np.concatenate([[True], np.diff(ot) > 3600 * 10**9])]  # one per session (instruments open within seconds of each other)
# The status record arrives slightly AFTER the scheduled 17:00:00.000 (the opening trade prints first): the schedule is exact
# to the minute, so floor to the minute. (No timezone arithmetic is involved.)
opens = (opens // (60 * 10**9)) * (60 * 10**9)
print(len(opens), "scheduled session opens")

ts, px, sz, iid = [], [], [], []
for f in sorted(glob.glob("/srv/quant/data/raw/es-mbo/*/*.mbo.dbn.zst")):
    a = db.DBNStore.from_file(f).to_ndarray()
    a = a[a["action"] == b"T"]
    ts.append(a["ts_recv"].astype("int64")); px.append(a["price"] / 1e9); sz.append(a["size"].astype("int64")); iid.append(a["instrument_id"].astype("int64"))
ts, px, sz, iid = map(np.concatenate, (ts, px, sz, iid))
# early close = a scheduled stop (is_trading N, reason Scheduled) earlier than the normal 16:00 CT, i.e. < 23h after the open
nt = st[(st["reason"] == 1) & (st["is_trading"] == b"N") & ((st["ts_recv"].astype("int64") % (86400 * 10**9)) > 5 * 10**9)]["ts_recv"].astype("int64")
nsess = np.searchsorted(opens, nt, side="right") - 1
early = set(int(k) for k, a in zip(nsess, nt - opens[np.clip(nsess, 0, None)]) if 0 <= k and 3 * H < a < 22.5 * H)
print("sessions with a scheduled early stop (no RTH expected):", sorted(early))
sess = np.searchsorted(opens, ts, side="right") - 1          # -1 = before the first scheduled open
age = ts - opens[np.clip(sess, 0, None)]
rth = (age >= int(15.5 * H)) & (age < 22 * H)
inhalt = age >= 23 * H                                        # trades after the 16:00 CT close (shouldn't exist)
print("trades before first open:", int((sess < 0).sum()), "| trades >=23h after an open (in the halt):", int(inhalt.sum()))

L = pd.read_parquet("/srv/quant/data/features/levels/session_levels.parquet").sort_values(["trade_date", "instrument_id"]).reset_index(drop=True)
bad = 0
rows = []
for s_i in np.unique(sess[sess >= 0]):
    for i in np.unique(iid[sess == s_i]):
        m = (sess == s_i) & (iid == i) & ~inhalt
        rows.append((s_i, int(i), m))
rows.sort(key=lambda r: (r[0], r[1]))
print(len(rows), "independent session groups vs", len(L), "table rows")
if len(rows) != len(L):
    bad += 1
for (s_i, i, m), (_, r) in zip(rows, L.iterrows()):
    if i != r.instrument_id:
        print("instrument order mismatch", s_i, i, r.instrument_id); bad += 1; continue
    for pre, mm in (("eth_", m), ("rth_", m & rth & (s_i not in early))):
        p, v = px[mm], sz[mm]
        exp = dict(volume=int(v.sum()), open=p[0] if len(p) else np.nan, high=p.max() if len(p) else np.nan,
                   low=p.min() if len(p) else np.nan, close=p[-1] if len(p) else np.nan,
                   vwap=float((p * v).sum() / v.sum()) if v.sum() else np.nan)
        for k, e in exp.items():
            g = r[pre + k]
            if not (np.isclose(g, e, rtol=0, atol=1e-9, equal_nan=True) if k != "vwap" else np.isclose(g, e, rtol=1e-12, equal_nan=True)):
                print("MISMATCH", str(r.trade_date)[:10], i, pre + k, g, e); bad += 1
    # volume-profile POC independent brute force (ETH)
    p, v = px[m], sz[m]
    prof = {}
    for a_, b_ in zip(np.round(p / 0.25).astype(int), v): prof[a_] = prof.get(a_, 0) + b_
    top = max(prof.values()); vw = (p * v).sum() / v.sum()
    poc = min((k * 0.25 for k, val in prof.items() if val == top), key=lambda x: abs(x - vw))
    if abs(poc - r.eth_poc) > 1e-9:
        print("POC MISMATCH", str(r.trade_date)[:10], poc, r.eth_poc); bad += 1
    # 70% value area by brute force definition: area must hold >=70% of volume, be contiguous around POC
    if not (r.eth_val <= r.eth_poc <= r.eth_vah):
        print("VA does not contain POC", str(r.trade_date)[:10]); bad += 1
    inside = sum(val for k, val in prof.items() if r.eth_val - 1e-9 <= k * 0.25 <= r.eth_vah + 1e-9)
    if inside < 0.70 * v.sum() - 1e-9:
        print("VA holds <70%", str(r.trade_date)[:10], inside / v.sum()); bad += 1
# invariants
ok = (L.eth_low <= L.eth_vwap) & (L.eth_vwap <= L.eth_high) & (L.eth_low <= L.eth_poc) & (L.eth_poc <= L.eth_high)
rr = L.dropna(subset=["rth_high"])
ok2 = (rr.rth_low >= rr.eth_low) & (rr.rth_high <= rr.eth_high) & (rr.rth_low <= rr.rth_vwap) & (rr.rth_vwap <= rr.rth_high)
print("ETH invariants ok:", bool(ok.all()), "| RTH within ETH & vwap within range:", bool(ok2.all()))
print("incomplete sessions:", [(str(r.trade_date)[:10], int(r.instrument_id), r.first_trade_ct[11:16], r.last_trade_ct[11:16]) for _, r in L[~L.complete].iterrows()])
print("RTH-less sessions:", [str(d)[:10] for d in L[L.rth_volume == 0].trade_date])
print("ALL MATCH" if bad == 0 and ok.all() and ok2.all() else f"{bad} PROBLEMS")
sys.exit(0 if bad == 0 else 1)
