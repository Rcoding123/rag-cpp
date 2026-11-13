# Lessons, distilled

The experiment log (`EXPERIMENTS.md`, E0–E14) is long. These are its
cross-cutting lessons, each with the measurement behind it. They're about
building learning systems in general, not just this task.

## Measuring

1. **Score against an exact optimum, as a share of the optimal gain.**
   "Reward went up" can't be compared across runs; "67% of optimal" can
   (A1). When the optimum is computable (a DP here), compute it.
2. **Pair your comparisons.** Episode returns have sd ≈ 55, much larger than
   the differences between good policies; common random numbers make
   differences of a few percent measurable (A2). Pairing also made a model
   check ~10× more sample-efficient: 100 episodes instead of ~1,500 (E7c).
3. **Count what learning costs, not only what it ends with.** With every
   training episode counted as real, ε starting at 1 lost over a thousand
   episodes' worth of the optimal gain before paying off (E8).
4. **Look at the tail you care about directly.** A policy more volatile
   than TWAP had a far better worst 5% (−66 vs −117), and a variance
   penalty that halved the sd barely moved the tail (E14).
5. **Unpaired online sums are noisy.** Cumulative gains over 20,000
   episodes carry about ±500 of noise; read them to ~2.5% (E11).

## Learning

6. **Model-free difficulty is set by gain-to-noise, not by the state
   space.** Execution: optimal gain 0.27 return-sds per episode, TileQ
   plateaus near 70%. Market making: 4.07, TileQ reaches 97–99% and
   actor-critic 100% in 5,000 episodes (E12).
7. **The plateau is target noise, not capacity, but how the approximator
   averages noise matters.** The default tiles can represent a 99.5% policy
   (E2b), yet coarser tiles learned better (76–77% vs 65–69%) and a small
   neural network better still (84–89%), with the same targets (E16).
   Multi-step returns helped too (n = 3; E2a).
8. **A model that's right beats any model-free learner by orders of
   magnitude:** 99–100% after 20–50 episodes vs ~70% after 40,000 (E1);
   even on the task model-free learning finds easy, 3 episodes vs 5,000
   (E15).
9. **A model that's wrong is worse than no model,** and gets worse with
   more data, because its plan exploits its own error (25% after 300
   episodes vs 43% after 20; E7a). Shrinkage toward a simple prior makes
   flexible models safe with little data (E1c).
10. **RL fine-tuning can't hold a near-optimal policy against noisy
    targets.** TD jitter (≈ √(α/2)·σ ≈ 2) exceeded the action gaps (< 1),
    so fine-tuning wrecked a distilled 99% policy within 2,500 episodes
    (E7d).
11. **Little exploration is often enough.** Zero-initialized values were
    optimistic about the bad actions, so greedy selection explored them;
    ε 0.1 learned as well as ε 1 at a fraction of the online cost (E8). For
    the model-based agent the ongoing ε 0.05, not the initial exploration,
    was the cost (E11).

## Deploying

12. **Deploy a champion, not the learner.** Learning wobbles; the champion
    only changes when a challenger beats it on the same held-out scenarios
    (A6).
13. **Never deploy below the simple baseline.** In a weak-signal market,
    Q-learning's champions were 35–106% *worse* than TWAP; a guard that
    requires beating the baseline by 2 standard errors fixed it at no cost
    elsewhere (E6b).
14. **Check a model before trusting its plan.** A wrong model is always
    optimistic about its own plan (the optimizer's curse), so "predicted
    gain > realized gain" is the signature to test for (E7c).
15. **When the model may be wrong, combine:** distill its plan into a
    model-free learner, fine-tune on real rewards, and guard with the plan
    as the baseline. 100% when the model is right, never below the plan
    when it isn't (E7d).

## Changing worlds

16. **Suspect fast, confirm exactly.** A CUSUM test on surprise reacts in
    tens of episodes but can't tell "the world changed" from "my learning
    wobbled"; re-scoring the frozen champion on fixed scenarios can, because
    in an unchanged world it reproduces exactly (A7).
17. **Good news is a change too.** A market that got better went unnoticed
    by a detector that watched for drops; the champion's re-score catches
    changes in either direction (E13).
18. **A detector must not go blind after a false alarm.** Resetting on
    every alarm left a 1,550-episode blind spot that a real change fell
    into (E13).
19. **Remember old worlds.** A bank of past champions recognized returning
    regimes within 20–131 episodes (E3), and at 99–100% on the second task
    (E13).
20. **Few parameters adapt fast.** After a flip the model-based agent was
    back at 95–100% within 20 episodes (change detection + reset), against
    5,000 episodes for the model-free agent to reach 18% (E9).

## Offline evaluation

21. **Importance sampling fails at long horizons.** At 20 steps, almost no
    logged episode follows a different deterministic policy for long
    (effective sample size ≈ 0), and errors stayed at 45–2,968% of the
    optimal gain with 10,000 logs (E10).
22. **So offline evaluation rests on a model, and the model must be
    checked:** the direct method was accurate to 2–4% with the right model
    class and biased by up to 62% with the wrong one, regardless of data
    volume (E10).

## Engineering

23. **Bugs hid in plain sight, and measurement found them:** a planner that
    received hidden market parameters (A11, and nearly again in E7); an
    exploration phase twice as long as configured (E11); a detector blind
    after false alarms (E13). Each now has a regression test.
24. **Speed up the inner loop before scaling experiments.** A precomputed
    banded expectation made the DP ~9× faster with identical decisions, and
    the slowest tests 4–10× faster (A28).
25. **Decompose the noise before reducing it.** A textbook control variate
    for the price innovation did nothing here, because the signal path, not
    the innovation, carried ~95% of the return variance; where the
    innovation was large, the same trick worked (E17).
26. **Tune globally, validate out of sample.** A local optimizer's result
    depended on its start; a global search plus held-out seeds and
    bootstrap intervals gave a reliable improvement (37% → 59%), and a
    response surface over the trials recovered the earlier lessons (E18).
27. **Independent reviewers find what the author doesn't.** Three review
    agents found 14 bugs in code that already had ~98% line coverage and a
    differential test suite: among them a heap overflow reachable through
    a config value and through a forged index file, a bit-exactness
    violation in IVF, and an RL schedule off by one doubling (A40, A41,
    R13–R17). One bug was found independently by two of them. A fourth,
    on a header whose author had tested it, found three more within hours
    (an overflowing table size, an endless loop, subnormal bin widths;
    A46), and its fuzzer re-found one of them unaided.
28. **Rank learners under one protocol, with intervals.** Numbers gathered
    by different probes (seeds, evaluation sets, deployed vs final policy)
    don't compare. One protocol (8 seeds, deployed champion, the same
    held-out scenarios, bootstrap intervals) separated the top three
    learners cleanly and showed that a configuration tuned for 10k episodes
    has a wide spread at 40k (E21).
29. **Compile what you learned.** On a small state space, a table sampled
    from the trained network kept its gain within 0.25 points at 1/1,000 of
    the decision cost. Check the grid on a policy you know first: the exact
    optimum, compiled the same way, showed what the grid alone costs (E22).
30. **Seeds are an ensemble you already paid for.** The median of 8 seeds'
    tables (97.5%) beat the best seed (94.1%) by more than the best seed
    beat the average one (91.4%). Report every subset, and confirm on fresh
    scenarios, so the ensemble isn't selected on the test set (E23).
31. **Average the weights you deploy.** A slow moving average of the
    network (τ = 1e-4) beat the network itself on every seed, at the cost
    of one extra copy of the weights; a fast one (1e-3) didn't. It also made
    the final policy usable without champion selection (E24).

## Trading on order flow (OF1–OF10; ORDERFLOW.md)

32. **A simulator that pays every day is wrong, not good.** The scalping
    agent's annualized Sharpe of ~80 with no losing day measured the
    simulator's predictability (markouts that keep drifting for 30 s), not
    a tradable edge. Check the markout curve before believing P&L (OF8, OF9).
33. **Make naive strategies fail first.** Naive passive quoting winning
    every episode exposed a too-generous queue (uniform cancellations, no
    latency competition); naive delta-following winning exposed too much
    informed flow. Calibrate until naive rules are near breakeven (OF2, OF4).
34. **Exploration must be able to reach the reward.** Uniform random
    quoting modes cancelled and rejoined at the back of a 120-lot queue, so
    the tick agent never got filled and learned not to quote; guiding
    episodes with a rule fixed the experience, not yet the edge (OF5, OF7).
35. **Same seed is not the same path.** Once our orders change the book,
    two policies on one seed see different markets; paired differences are
    nearly as noisy as unpaired ones, so validation leads are selection
    noise until confirmed on fresh seeds (OF7 review).
36. **Summary statistics don't identify mechanisms.** A fit matched
    predictability with momentum instead of the true scarce informed flow;
    and less informed trading made the market more predictable (slower
    price discovery). Calibrate to richer targets (markout curves, queue
    dynamics) (calibration recovery test).
37. **Re-run what you commit.** A results table produced by a stale
    binary (the last sweep step) went into git; regenerate result files
    from the committed code, in the same command (OF4 correction).
38. **Beating the oracle means a leak.** Making trade flow unpredictable
    (informed makers) moved the information into the book: the agent read
    the makers' tilted order rates and outperformed an oracle that sees
    the hidden value. Check every new market against an oracle and against
    the features' correlation with hidden state (OF9).
39. **Pairing needs counter-based randomness.** One stream per process
    barely helped (state-dependent draw counts desynchronize streams);
    re-seeding every step from (seed, process, step) doubled the paired
    runs' correlation (0.33 -> 0.61).
40. **Measure in the regime you trade in.** The stylized facts were
    measured on 6.5 h windows, but every RL episode starts a fresh
    simulator whose activity sits at its floor for ~1.5 h (a running mean
    initialized too high, with an 1,800 s time constant). The agents
    trained and were scored at ~4 trades/s, not ~20, for every OF2-OF9
    result, and the OF10 v2 preset was tuned on 2 h windows that leave
    the floor halfway. Run the fact checks on the exact windows the agents
    see (OF10 warm-up finding).
41. **A price that moves without the book makes imbalance useless.** Rigid
    whole-book quote jumps removed the hidden gap's 30 s predictability
    (0.57 -> 0.11) and also every signal in the book: imbalance predicted
    the next mid change at +0.02 instead of +0.33. Information must reach
    the price through the queues, or order-flow features have nothing to
    read (OF10 v1).
42. **Rate tilts leak, depletion informs.** Makers tilting order rates
    toward a value the price takes 30 s to reach put a 30 s forecast into
    the book (OF9: L1 imbalance vs the next 30 s return 0.32). Withdrawing
    quotes that value has already passed moves the price within a tick in
    ~0.1 s, so the book is still correlated with the gap (0.45) but the gap
    is gone within seconds: imbalance predicts the next mid change at +0.25
    and the next 30 s at 0.04-0.05, tail ~0 (OF10 v2). What matters is not
    whether the book knows the value, but how long the price lags it.
43. **Check the oracle actually trades.** "No rule beats the oracle"
    passed in v1 and v2 because the oracles barely traded: the gap was
    capped at the band (v1), or rarely crossed the pre-registered threshold
    (v2: 1.4-1.6 lots per episode, net -0.5), while a 2 s clairvoyant made
    +7 ticks per episode. Report the bound's activity next to it, and
    keep a clairvoyant beside the oracle (OF10).
44. **Check a feedback loop has a stable steady state before warm-starting
    it.** The warm-up looked like a bad initial value (`move_avg_` started
    too high), but the activity loop itself was unstable (loop gain
    0.045/s, R = 11.5): 10 h runs swung between the clamps, so no starting
    value could have helped. Derive the stationary law first (activity
    model 1 has an exact one), then draw the warm start from it (OF10 v3).
45. **Warm every slow state, and check it in the regime the agent sees.**
    With activity fixed, an episode-vs-long-window comparison of every
    statistic (the AGREE table) still found the meta-order population 5%
    short after the 150 s warm-up (5-6 SE on three seed sets); a 1,050 s
    burn matched. Compare the agent's windows with long windows on every
    population, not just the one you repaired (OF10 v3).
46. **A wide gate fails by chance; gate on core rows.** With ~29 rows at
    2 SE, about one row fails per run by chance; v3's one (c) miss (2.4 SE
    on tuning-seed episodes only) passed on two fresh seed sets. Decide
    the core rows before the run, and confirm an isolated miss on fresh
    seeds instead of retuning to it (OF10 v3).
47. **An edge that vanishes when the simulator gets efficient belonged to the
    simulator.** The OF3 recipe beat every rule on the slow-discovery market
    (OF3, OF8); on v3, which discovers prices in under a second at realistic
    activity, all 24 seeds lose on sealed episodes and the ensemble stops
    trading (OF11). Fix the market's known leaks first, and only then read
    an RL profit (OF10, OF11).
48. **Calibrating to real data can remove what the agent needs; check the
    oracles still have something to find.** v4 matches real ES on 25 rows,
    but its fair value carries no edge after costs: both pre-registered
    oracles lose (O1 about −8 to −12, O2 about −18 to −20 ticks/episode).
    "RL finds no edge" on such a market is close to true by construction;
    say so, and read the run as an overfitting test, not as evidence about
    order flow (OF13).
49. **One morning of real data is one observation.** The real reference
    for G2 and G4 was one calm RTH morning; a second, busy morning fell
    outside G4 and G2 itself. Record more real sessions before tightening
    gates around one of them (OF13).
50. **A flat ensemble makes "no edge" vacuous; report the mean of seeds
    and orders per episode.** OF15 declared the 12-seed ensemble as the
    headline before training, as it should, but 5 of 8 v4 and 4 of 8 v3
    ensembles placed 0 orders on the sealed test: their NO_EDGE was true
    by construction. The mean of the seeds' per-episode P&L (≤ 0
    everywhere on fresh episodes) carried the information. Put the mean
    of seeds and orders/episode next to every ensemble number (OF15).
51. **A Sharpe threshold needs a minimum-activity guard, and a t
    calibration.** R1 (annualized Sharpe > 5) fired hard on rows trading
    0.0-0.26 orders/episode, where a few trades make any Sharpe; and
    annualizing a per-episode Sharpe (x sqrt(9828) ~ 99) makes "> 5" the
    same as t > 0.87 at 300 episodes, so it would also fire on noise from
    policies that do trade. Fix both before the sealed run: the guard was
    added afterwards (R1v, A74), which is a post-hoc rule change even
    when it only relaxes flags (OF15). Since A77, R1 needs the Sharpe's
    lower 2-SE bound above 5 (about t > 2.9 at 300 episodes).
52. **Pairing tightens less when the simulator reacts to our orders.**
    Same seeds give the same market only until the first fill: PPO v4's
    P&L on the same episodes at 100 ms vs 300 ms latency correlates 0.56,
    the delta rule's 0.19, so a "paired" latency effect of -1.51 had an SE
    of 0.99. The paired CIs stay valid; just don't expect the variance
    reduction of a fixed path (LESSONS 35, OF15).
53. **An intermittent native crash needs a log that retries can't
    overwrite.** Five 0xc0000374 worker crashes and one unexplained stop
    hit ~0.15% of evaluations; the stop's cause was lost because the
    retry overwrote the runner's stdout log, and Windows kept no WER or
    event-log entry. Write each attempt's logs under a unique name and
    record the exit code of every process before retrying; a crash that
    no longer reproduces (0 in ~1,450 evaluations) can't be fixed or
    ruled out after the fact (OF15).
54. **Know what a serve command writes before pointing it at a
    directory.** `of_train_live.py serve DIR` writes its live page to
    DIR/index.html, so serving the zoo replays (`build-avx512/zoo`) with
    it replaces their index page; serve static output with a plain static
    server (`python -m http.server`, the `of-zoo` launch entry) (OF15).
55. **Score a policy in the conditions it was trained in, down to the
    feature warm-up.** The first real-data scorer warmed the features up
    for 150 s, the replay default, while every agent trained with 1050 s;
    the slow scaling averages were 22% vs 83% settled, so features came
    out ~2x off scale. Nothing would have crashed: the agents would just
    have "found no edge on real ES". `score-real` now takes the warm-up
    from the run's config and refuses a mismatch, as `check_net` does
    for the C++ nets (OF16, A78).

56. **"Beats its simulator" must mean the real result is good, not just
    less bad.** OF16's RG flag fires when a policy's real gross per window
    exceeds its simulator gross by 2 SE. It fired 6 times, and each time
    the simulator gross was negative and the real one about 0: a policy
    that trades less, or more noisily, on real "beats" its simulator. A
    leak would show real gross above 0 on its own. RG now also requires
    real gross > 0 by 2 SE on at least 2 orders/window, else the warning
    RGv, as R1 needs a lower bound and activity (A74, A77, A80).

57. **"No candidate" is only as strong as the test's power at each
    horizon.** OF17 sampled every second, so at 10 min its 87,000
    samples per month were about 143 independent moves, and a rule needed
    ~3 ticks net per trade to pass. Report the detectable size beside a
    null, and say "none detected" where it is large. Fit on one month and
    score on the other also separates two different nulls: no signal, and
    a signal whose sign does not survive a regime change (OF17: vwap_dist
    +10 ticks in Nov, −7 in Sep).

58. **An empty log after a native crash can mean the crash came after the
    work.** Under the full gate's 24-way load, `ragrl train` workers
    (ppo once, neuralq_ref once) died with 0xC0000005 and an empty
    worker.log despite `-X faulthandler`. The crashed worker's last
    metrics record ("timing", the last statement of `train_worker`) was
    there: it died in interpreter finalization, the native libraries'
    teardown, where faulthandler is already off. 200 isolated reruns were
    clean. Workers now `os._exit(0)` after their last write, like
    multiprocessing children; a crash during the work still fails. Check
    the last record a process wrote before hunting inside its work.
59. **A "persistence" measure built on a conditional subsample can be
    negative by construction.** OF18 correlated the trailing estimate of
    a feature's top-minus-bottom move with day D's own value; the own
    value existed only on days that reach both tails, which for
    vwap_dist means days that cross the VWAP, where it is negative
    whatever the regime, and it scaled with volatility. Volatility
    clustering then produced r ≈ −0.3 with no regime at all. Check a
    descriptive statistic against a no-signal simulation with the real
    volatility before reading it.
60. **Sierra stamps the records of one exchange message 1 µs apart.**
    Grouping "records sharing a timestamp" into sweeps grouped nothing
    (45,765,398 records, 45,765,213 distinct timestamps; ~80% of gaps
    exactly 1 µs). A sweep is a chain of records each 1 µs after the
    previous one (OF19). Check what "simultaneous" means in a data
    source before modelling fills on it.
61. **A positive mean on fat-tailed trades needs a trim test.** OF20's
    exhaustion at levels netted +1.44 per trade over 119 trades, but
    the median was −9.24, and dropping its top 3 trades made it −0.96.
    With stops of a few ticks and 15-min caps, a few trend days decide
    the mean. Report the median, the result without the top few trades
    and without the best days before reading any descriptive positive.
