"""Check scripts/sessions.py against the exchange's own timestamps (status + statistics)."""
import glob
import sys

import databento as db
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import sessions as S

st = np.concatenate([db.DBNStore.from_file(f).to_ndarray() for f in sorted(glob.glob("/srv/quant/data/raw/es-status/*/*.status.dbn.zst"))])
stat = np.concatenate([db.DBNStore.from_file(f).to_ndarray() for f in sorted(glob.glob("/srv/quant/data/raw/es-fut-statistics/*/*.statistics.dbn.zst"))])
dfn = np.concatenate([db.DBNStore.from_file(f).to_ndarray() for f in sorted(glob.glob("/srv/quant/data/raw/es-fut-definition/*/*.definition.dbn.zst"))])
iid = {}
for r in dfn:  # outright ES futures: raw symbol like ESZ6 (4 chars, no spread dash)
    s = r["raw_symbol"].decode()
    if len(s) == 4 and s.startswith("ES"):
        iid[int(r["instrument_id"])] = s
print("outright instruments:", sorted(iid.values()))
ok = True

# 1. every published preliminary settlement at ~15:00 CT: ts_ref (a date label) == our trade date of the publication
m = (stat["stat_type"] == 3) & np.isin(stat["instrument_id"], list(iid))
s3 = stat[m]
loc = S._local(s3["ts_recv"]); mins = np.asarray(loc.hour) * 60 + np.asarray(loc.minute)
at15 = (mins >= 15 * 60) & (mins < 15 * 60 + 5) & ((s3["stat_flags"] & 1) == 0)
ref = (s3["ts_ref"] // (86400 * 10**9)).astype("datetime64[D]") if s3["ts_ref"].dtype.kind == "u" else None
ref = np.array(s3["ts_ref"].astype("int64") // (86400 * 10**9), dtype="int64").astype("datetime64[D]")
td = S.trade_date(s3["ts_recv"])
eq = (ref[at15] == td[at15])
print(f"preliminary settlements published 15:00-15:05 CT: {int(at15.sum())}; trade_date == ts_ref: {int(eq.sum())}")
ok &= bool(eq.all()) and at15.sum() > 0
bad = ~eq
if bad.any():
    idx = np.nonzero(at15)[0][bad][:5]
    for i in idx: print("  mismatch", ref[i], td[i], s3["ts_recv"][i])

# 2. scheduled opens (Trading, is_trading Y, reason Scheduled) of outrights at exactly 17:00 CT; closes at 16:00 CT except early closes
o = st[np.isin(st["instrument_id"], list(iid)) & (st["reason"] == 1)]
# Databento adds a status snapshot at 00:00:00 UTC every day (its docs filter these out): not exchange transitions
not_snap = (o["ts_recv"].astype("int64") % (86400 * 10**9)) > 5 * 10**9
open_ = o[(o["action"] == 7) & (o["is_trading"] == b"Y") & not_snap]
lo = S._local(open_["ts_recv"]); om = np.asarray(lo.hour) * 60 + np.asarray(lo.minute)
good = om == 17 * 60
print(f"scheduled Trading=Y transitions (excluding 00:00 UTC snapshots): {len(open_)}; at 17:00 CT: {int(good.sum())}; others at (CT minutes after midnight): {sorted(set(om[~good]))[:6]}")
ok &= bool(good.all())
nonstd = open_[~good]
if len(nonstd):
    print("  non-17:00 opens (CT):", sorted({str(t)[:19] for t in S._local(nonstd['ts_recv'])})[:6])
close = o[(o["action"] == 12)]
lc = S._local(close["ts_recv"]); cm = np.asarray(lc.hour) * 60 + np.asarray(lc.minute)
print(f"scheduled Close transitions: {len(close)}; at 16:00 CT: {int((cm == 16 * 60).sum())}; others: {sorted(set(cm[cm != 960]))[:6]}")
ok &= int((cm == 16 * 60).sum()) == len(close)

# 3. early closes detected from data
ec = S.early_close_dates(o[o["instrument_id"] == o["instrument_id"][0]])
allec = set()
for i in set(o["instrument_id"].tolist()):
    allec |= S.early_close_dates(o[o["instrument_id"] == i])
print("early-close trade dates detected:", sorted(str(d) for d in allec))
print("ALL OK" if ok else "CHECK FAILED")
sys.exit(0 if ok else 1)
