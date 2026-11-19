# Research notes and decisions (2026-11-17 to 2026-11-19)

These were collected in the owner's desktop planning session, which runs
alongside the server session. They record what was decided, why, and the
sources. Where a point is unverified, it says so. Read with `PLAN.md`,
`HANDOFF.md` and `CLAUDE.md`.

---

## 1. Direction change: trading by hand, with an analyst instead of an RL bot (2026-11-18)

The owner trades **by hand** (prop firm, ATAS on Windows). Short-horizon
prediction and RL execution are **paused**. A 1-second passive edge would
need low-latency automation, and some prop firms restrict automated trading.
That restriction is unverified for the owner's firm.

The new goal is a decision-support analyst in three layers.

| Layer | What it does | How it is kept honest |
|---|---|---|
| **L1 Context engine** | Computes market structure: sessions (Asia/London/NY, ETH/RTH); daily, weekly and prior-session levels; settlement; VWAP and bands; volume profile (POC, value area, LVN/HVN, naked POCs); auction-market concepts (initial balance, open type, day type, single prints, poor highs/lows, excess, value migration, overnight inventory, gap vs prior value); order flow at levels (delta, absorption, stacked/pulled liquidity, iceberg refills from MBO); gamma levels | Deterministic calculations, recomputed independently and checked |
| **L2 Statistics on context** | Event studies per level type and context: hold/break, MFE/MAE, time to resolution | Walk-forward splits, confidence intervals, random-level controls, pre-registered tests |
| **L3 Written briefing** | Pre-market brief, later live updates | Built only from L1 and L2 outputs |

Order of work:
- Gamma (Phase 4) moves up, because it is directly useful for trading by hand.
- `DEFINITIONS.md` comes first. Every concept gets sources and its variants
  are listed, and the owner picks the variants and session times.
- Pending owner input: which concepts they trade, session times (Chicago
  time), and output format (morning report, ATAS chart lines, live alerts).

Notes on the Option A analysis that was proposed before the change (kept in
case short-horizon work resumes):
- A cost of 0.24 ticks round trip assumes both legs fill passively.
  Passive entry plus market exit is 0.76 ticks (0.12 + 0.64).
- Passive fills must go through the conservative queue model with latency,
  not through the mid-price path.

## 2. Data source: Databento Standard plan (decided 2026-11-18)

### What the plan includes
Verified on https://databento.com/pricing (CME section), 2026-11-18. Price:
**$199/month, monthly subscription.**

| Level | Schemas | History included |
|---|---|---|
| L0 | OHLCV-1s/1m/1h/1d, definitions, statistics, status | 16+ years |
| L1 | trades, mbp-1, tbbo, bbo, cbbo, cmbp-1 | last 12 months (rolling) |
| L2/L3 | mbp-10, mbo | last 1 month (rolling) |
| Live | yes, "No license fees" (an earlier read said personal use, up to 2 devices) | |

History beyond these windows is pay-as-you-go.

### Pay-as-you-go cost of the same data
Free quotes from `metadata.get_cost`, 2026-11-18, range 2010-06-06 to
2026-11-17:

| Data | Size | Pay-as-you-go |
|---|---|---|
| ohlcv-1m ES.v.0, 16 years | 0.32 GB | $20.99 |
| ohlcv-1s ES.v.0, 16 years | 8.55 GB | $557.44 |
| ohlcv-1d ES.FUT, 16 years | ~0 | $0.30 |
| status ES.FUT, 16 years | 0.02 GB | $0.09 |
| statistics + definition ES.FUT, 16 years | 0.59 GB | $0.62 |
| statistics + definition ES.OPT (quarterly), 16 years | 62.63 GB | $68.73 |
| statistics + definition, 24 weekly/daily/EOM roots, 16 years | n/a | $385.99 |
| trades ES.v.0, last 12 months | 5.94 GB | $154.81 |
| **Total** | | **$1,188.96** |

### Decision
- Subscribe to Standard, since every item above is included.
- Pay-as-you-go would only be cheaper for 1-minute levels alone (about $22).

### Still to confirm
- No minimum term.
- Downloaded data can be kept after cancelling.
- Personal-use status for a prop trader.
- That included data really bills $0. Test one tiny request per schema with a
  $1 cap before bulk downloads.

### Rolling windows
L1 (12 months) and L2/L3 (1 month) move forward every day, so anything
needed from them must be downloaded while still inside the window.

Proposal: a daily job that downloads the previous day's ES MBO, checksums
it, and appends it to `data/raw/es-mbo/`. The owned MBO then grows for free
while subscribed.

### Owned MBO
- ES.v.0, 2026-08-01 to 2026-11-16.
- 92 UTC-day files (76 trade dates).
- 15.76 GB compressed (about 57.7 GB uncompressed).
- About 1.03 billion records and 34.5 million trades.
- Cost $96.68.

## 3. CME facts used by the engine (with sources)

**Crossed books.** Normal in pre-open and the daily pause: CME accepts
orders but does not trade until the opening uncross.
https://databento.com/docs/knowledge-base/datasets/glbx-mdp3

**Velocity Logic.** Reserved/pre-open state after fast moves, and CME says
"Time may vary". The server agent matched five ~5 s PreOpen periods to the
five 5 s crossed runs.
https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457218368/Velocity+Logic

**Modify priority.** A quantity increase or price change loses priority; a
quantity decrease keeps it.
- https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457218521/CME+Globex+Matching+Algorithm+Steps
- https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457605721/MDP+3.0+-+Market+by+Order+-+Book+Management

**Priority from message order.** Databento doesn't carry MDOrderPriority,
but FIFO priority comes from message order within an instrument_id. Snapshot
publish order equals priority.
- https://databento.com/docs/datasets/glbx-mdp3
- https://databento.com/docs/examples/order-book/queue-position

**Matching.** ES is FIFO (algorithm F): the server agent confirmed
`match_algorithm='F'` in our definitions.

**Sessions.**
- Trading day 17:00 CT (previous day) to 16:00 CT; settlement 15:00 CT,
  12:00 CT on shortened days.
  https://cmegroup.com/education/articles-and-reports/faq-daily-settlement-price-determination-time-change.html
- RTH 08:30–15:00 CT matches NYSE 09:30–16:00 ET. Some traders use
  15:15 CT, so this is a convention, not an official CME definition.
- NYSE early close 13:00 ET (e.g. 2026-12-24).
  https://www.nyse.com/Markets/hours-calendars

## 4. Server operations notes

**SSH hardening.**
- sshd keeps the first value it reads, so the hardening drop-in must be
  `00-hardening.conf`, not `99-...`. Otherwise Ubuntu's
  `50-cloud-init.conf` (`PasswordAuthentication yes`) wins.
- Verify with `sudo sshd -T` from inside a session, never with failed
  logins.

**fail2ban.**
- Deliberate failed-login tests from the owner's IP got it banned for about
  10 minutes. Don't do that.
- The owner's IP is not in `ignoreip`.

**apt.**
- `apt-get upgrade` without `--with-new-pkgs` upgraded libfwupd while
  holding back fwupd, and the daemon refused to start on the version
  mismatch.
- Fixed with `apt-get install fwupd`. The setup scripts now use
  `--with-new-pkgs`.

**tmux clipboard.**
- Claude Code copies via OSC 52. tmux's default `set-clipboard external`
  drops OSC 52 from programs inside tmux.
- Fixed with `set -g set-clipboard on`, saved in `~/.tmux.conf`. Windows
  Terminal supports OSC 52 writes.

**Batch jobs.** `systemd-run --user --scope -p MemoryMax=... taskset -c
4-11,16-23` works without sudo. The server agent's `scripts/batch_run.sh`
wraps it.

**Raw data.** `data/raw` is read-only (`chmod -R a-w`). Unlock only to add
data, then lock again.

## 5. Hardware decision: keep the 9900X (2026-11-18)

- Phoronix Linux suite of about 400 tests: the 9900X is 9.2% faster overall
  than the 7950X3D.
- The 9900X has about 11–13% better single-thread speed and is much stronger
  at AVX-512 numeric/ML work (+15–43%).
- The 7950X3D wins compiles and rendering by 6–18%, and cache-heavy HPC code
  by 20–42%.
- The X3D's V-Cache sits on one CCD only and needs a 6.13+ kernel for its
  preference knob. Ubuntu 24.04's default kernel is 6.8.
- Main source: https://www.phoronix.com/review/amd-ryzen-9-9950x3d-linux

## 6. Open items

1. The disk is a single 1 TB Samsung 980 (DRAM-less, PCIe 3.0) with no RAID.
   The plan listed 2 TB+ RAID1; the owner is raising it with the provider.
2. Back up `data/raw` before the free re-download window ends (around
   2026-12-18 for the 2026-11-18 orders).
3. The owner should back up their SSH key and store the `quant` password.
4. Windows VPS: for ATAS live.
5. Factory box: a separate build box (`server/factory_bootstrap.sh`). It has
   no access to this server by design, and the setup installs no AI tools.
6. rag-cpp review: it can't be the engine base (no DBN reader, no per-order
   book, Windows-only). Reusable pieces: the vectorised nanobind env pattern,
   `features.hpp` and the validation tooling. Its OF24 audit found no
   out-of-sample edge, which is why validation discipline is central to
   this plan.
