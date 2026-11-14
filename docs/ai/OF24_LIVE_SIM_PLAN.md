# OF24 plan: the OF23 agent live in Sierra Chart, simulated orders only

Status: **design, 2026-11-13.** At the user's request ("run a separate
Sierra Chart sub-instance, build out the agents in C++ and Python, let it
observe the live tick data, order-flow data (MBO) and let it trade in
sim. I want to see if this can work"). Decisions below were left to this
session ("do whatever you see fit, learn from the KB then make your
decision") after the user's four answers: the live agent stays inside
Sierra; MBO is on; this session drives Sierra; keep learning in sim.

**This is an observation, not a test of an edge.** OF23 found no edge
(A92): the agent starts from a policy that lost 2-3 ticks per trade on
unseen months. Nothing here is a claim; a claim would need its own
pre-registration on forward days. Orders are **simulated only**.

## 1. Constraints that shaped the design

- **Sierra's data license** limits its data to Sierra's own interfaces.
  The user chose "inside Sierra only": the live agent is a C++ ACSIL
  study; no Python reads live data. Python's part is offline: training
  the starting policy on the already-built OF23 cache and exporting the
  weights as a text file.
- **Sierra's DTC server refuses CME data** to external programs, so an
  outside process could not have received the feed anyway.
- **MBO is live only** (Denali, orders of 3+ lots; never recorded or
  replayed), so no historical training on MBO is possible: the agent can
  only learn what MBO is worth from its own live sim trades.
- **Forward days:** days from 2026-11-12 are OF21's and the confirm
  block's. The study reads them live inside Sierra and writes only its
  own decisions, fills and weights. Nothing from them is used to tune a
  pre-registered rule; OF21's runner is frozen. The agent's log is not
  evidence for any claim.
- **SierraChart-KB** (`Desktop\SierraChart-KB`, the user's ACSIL
  knowledge base) sets the build and safety rules followed here:
  local MSVC build with its `build.ps1`, a new DLL name per build
  (`Name_v<n>_64`, never a digit suffix), a pure C++ kernel with a thin
  ACSIL adapter, heap state in a persistent pointer, sim guard on every
  call, an Arm input defaulting to off, a dedicated Sim account, no
  per-bar logging, and "verified" meaning observed at runtime.

## 2. Parts

1. **Kernel** `include/rag/live/of24_kernel.hpp` (pure C++17, no Sierra
   headers): OF23's candidates and 37 features from 1-min bars (plan
   OF23 section 3 and 5), the policy network (2 x 64 tanh, 13 actions:
   skip + the 12 exits of OF23 6b), weight file load and save, and the
   online update (section 4). Checked against `experiments/of23_mine.py`
   on synthetic days by `python/tests/test_of24.py` through
   `examples/of24_parity.cpp`: same candidates, same features, same
   policy outputs.
2. **Weights** `experiments/of24_export.py`: trains the RL-free arm of
   OF23 (same code, same settings, seed 0) on all 228 days and writes
   the policy, the value network and the feature standardization to a
   text file outside the repository.
3. **Study** `sierra/RagAgent.cpp` (ACSIL adapter): on a 1-min ESZ26
   chart it feeds completed RTH bars to the kernel; at a candidate it
   samples an action from the policy, sends a market entry with attached
   target and stop (`Target1Offset` / `Stop1Offset`), flattens at the
   time cap, and after the trade closes applies the online update. It
   reads the depth book and MBO queues each call for the live book
   features, and logs decisions and fills to a CSV in its own folder.
4. **Sub-instance:** a second Sierra instance (File >> New Instance) in
   Trade Simulation Mode with its own Sim account, so nothing touches
   the main instance's charts, recorders or the S1 strategy.

## 3. Policy inputs

The 37 OF23 features (standardized with the training means) plus 6 live
book features, zero when the data is absent (replay, or before the book
fills in):

- depth imbalance at the best level and over the 5 and 10 best levels,
  (bid − ask) / (bid + ask), signed by the trade direction;
- MBO at the best bid and ask: share of displayed size in orders of 10+
  lots on the entry side and on the far side (signed pair);
- spread in ticks.

The book inputs' weights start at zero, so at the start the live policy
equals the trained one; only the online update can give them weight.

## 4. Online learning (inside the study)

After each closed trade: reward = net ticks after the OF23 base costs
(1.28 round trip for market/stop exits, 0.76 with a limit target) / 10;
one actor-critic step on the stored inputs and sampled action (advantage
= reward − value, the value head starting at zero because OF23's critic
predicts a rest-of-day return; policy-gradient and value steps by SGD, learning rate
1e-4, gradient norm clipped at 1, entropy bonus 0.01). Skips are updated
with reward 0 at the same time. Weights are saved to the study's folder
after each update and reloaded on start; the starting file is kept.
An input freezes learning (a fixed-policy run) and another switches
sampling to greedy.

## 5. Safety

- Sim only: `sc.SendOrdersToTradeService = 0` and
  `sc.GlobalTradeSimulationIsOn` must be 1 on every call, else nothing
  is sent and one line is logged. Arm input defaults to off.
- One position, one lot, `sc.MaximumPositionAllowed = 1`; no entry
  before 08:50 or after 14:30 CT; flat by 15:00; a daily stop (default
  −120 ticks) disarms until the next session.
- Entry only when flat with no working orders; a rejected order is
  logged and not retried on the same bar.
- The study never changes global settings; the sub-instance setup is
  done once, with the user's approval of each step that touches the
  main instance.

## 6. What is and is not comparable with OF23

Sim fills are quote-based and immediate (no latency, optimistic limit
fills), OF23's fills are from the trade records with a one-tick
trade-through for targets: live sim results will look somewhat better
than OF23's for the same decisions. The log keeps both the sim fill and
the kernel's own OF23-style accounting where it can.

## 7. Verification order

1. Parity test (kernel vs Python) in the gate.
2. Compile with the KB build script; load in the sub-instance; confirm
   the module path in the log.
3. Chart replay of a recorded day with depth (from 2026-11-04 on), in
   sim: candidates fire, orders bracket, time cap flattens, log
   reconciles with Sierra's fills; MBO features are expected to be zero.
4. Live from the Globex open (Sunday 17:00 CT) for the feed checks and
   Monday 08:30 CT for the first RTH session, armed in sim.

## 8. Replay training and the profit log (added 2026-11-14)

At the user's request ("run more replay days so it keeps learning, and
keep a log of the profit"):

- **Replay training.** One continuous chart replay (Accurate Trading
  System Back Test, Skip Empty Periods) over recorded days, learning on.
  Input 8 names the last session (default 20261111, the day before the
  forward block); after that session's close, with the account flat, the
  study flattens, writes the day line, saves the weights and stops the
  replay. Restarting the replay from inside the study for each next day
  did not take in Sierra 2980, so days are not chained one by one.
- **Weights carry forward.** Replay continues `weights_replay_c<n>.txt`;
  live continues `weights_live_c<n>.txt`, else the replay-learned file,
  else the starting file. The log header says which (`weights=`).
- **Profit log.** `Data\RagAgent\profit_trades_<mode>.csv` (each closed
  trade: entry time, side, exit rule, fills, move, exit kind, net ticks
  and dollars after OF23's base costs, day and cumulative totals, update
  count) and `profit_days_<mode>.csv` (one line per session), with
  `<mode>` replay or live. Append-only; the cumulative totals are read
  back from the last line on restart.
- These numbers stay an observation: replayed days were in the starting
  policy's training data and sim limit fills are optimistic (section 6).
