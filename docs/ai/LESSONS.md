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

## Trading on order flow (OF1–OF9; ORDERFLOW.md)

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
