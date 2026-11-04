# Research notes: noisy-target RL, execution and market making

A literature review written by a research agent for this project (after
E16), mapping published results onto what the experiments measured, and
ranking next experiments. **Caveats:** the environment's network policy
blocked arxiv.org, so most claims were checked against abstracts,
proceedings pages and search-indexed text. Numbers marked † come from
secondary summaries; confirm them in the paper before relying on them.
**[E]** marks an established, published result; **[I]** marks inference
about this codebase.

## 0. The number that frames everything

**[I]** The samples needed to detect an effect scale like (noise/effect)².
The two tasks differ in gain-to-noise by (4.07/0.27)² ≈ 227 (E12). If
actor-critic needs ~5k episodes on market making, a naive scaling puts
execution at ~10⁶ for the same precision. So 75–77% at 200k episodes (E2)
is roughly what you'd expect: the execution task is hard because of
statistical estimation, not representation or exploration.

**[E]** The closest analytical ancestor of the execution task is Bertsimas
& Lo's model with an AR(1) information variable [BL98]; there too the
signal-adjusted strategy saves only a sliver of cost (about 241.46M vs
241.26M in a Nikkei example†).

## 1. Why a small MLP can beat tile coding when both can represent Q*

Approximation error is ruled out (tiles fitted to Q* give 99.5%). What's
left: variance of the estimates, and bias from the learning dynamics.

* **(a) Pooling: effective samples per parameter.**
  * **[E]** Tile coding generalizes locally: each weight is updated only
    by samples in its tile [SB18 §9.5.4].
  * **[E]** Tile-coding the inputs of a neural net makes it "generalize
    more locally", with less interference [GYRS18, GRLGW20].
  * **[E]** ReLU networks are biased toward low-frequency functions
    (spectral bias) [Rah19].
  * **[I]** With roughly white target noise across states, a
    smoothness-biased learner acts as a low-pass filter. The MLP has about
    4.7k parameters, while a fine tiling gives each weight far fewer
    effective samples.
* **(b) Sharing across actions.**
  * **[E]** The dueling paper argues that action gaps are tiny relative to
    Q values, so small update noise reorders actions [Wang16].
  * **[E]** VA-learning learns V faster than A "as the former is shared
    across all actions" [Tang23].
  * **[E]** A 2026 value-sharing layer increases action gaps and reduces
    overestimation [Nag26].
  * **[I]** With per-action tile weights, Q(s,a) − Q(s,a′) is estimated
    from *disjoint* noisy samples, and ε-greedy starves non-greedy actions.
    The MLP's shared torso estimates the action-dependent part through
    common features. This is the leading hypothesis.
* **(c) Optimizer and averaging.**
  * **[E]** Adam's update factorizes into a sign and a variance-adaptation
    factor that shrinks the step as relative gradient variance grows
    [BH18, KB15].
  * **[E]** The step-size schedule governs Q-learning's convergence rate
    [EDM03].
  * **[I]** The replay result (32 updates per step worse than 4) fits a
    constant-α noise floor. The weights are an exponential moving average
    over about 1/α updates, so more updates per transition at the same α
    shrinks that window in distinct transitions, and the weights get
    noisier.
  * **[I]** Target variance scales with inventory², so one α is wrong for
    high-inventory tiles; per-parameter normalization corrects that.
  * (E16 measured: SGD instead of Adam still reached 67–83%, so this is at
    most part of the story.)
* **(d) Target networks and the deadly triad.**
  * **[E]** Multi-step returns, double Q and target networks reduce "soft
    divergence"† [vH18].
  * **[E]** A target network with Polyak updates gives convergent linear
    Q-learning [ZYW21].
  * **[I]** With a 20-step horizon and 3-step returns, divergence is
    unlikely to be the driver. (E16 measured: no effect.)
* **(e) Bias at the TD fixed point.**
  * **[E]** Oracle tests of Q-learning found function-approximation error
    matters little with expressive networks, while the sampling
    distribution matters [Fu19].
  * **[I]** Tiles fitted to Q* give the *projection* of Q*. Semi-gradient
    off-policy Q-learning converges to the *projected Bellman fixed point*
    [SB18 ch. 11], which can induce a different greedy policy. Only a
    noise-free run separates bias from variance.

Ablations that discriminate between these:

| Ablation | Points to |
|---|---|
| Tile Q-learning with exact expected targets (integrate the noise out in the simulator) | still ~75%: bias (e); close to 99%: variance |
| Tiles with per-weight Adam/RMSProp, or α = 1/n^0.7 per tile, or minibatch-averaged updates | large gain: (c) |
| Tiles with 32 replays at α/8 | ≥ 4 replays at α: (c), the replay noise floor |
| Tiles with V on fine tiles + A on coarse tiles, or the action as a 4th tiled dimension | large gain: (b) |
| DQN on tile-coded inputs; DQN with separate per-action networks | drop: (a), (b) |
| Linear Fourier basis trained with Adam | matches the MLP: (a) without nonlinearity |
| Ridge LSTD-Q or backward fitted Q iteration on the same data | the lowest-variance linear result: the ceiling for tiles at this data size [LP03] |

## 2. DQN improvements that target noisy targets and action gaps

Effect sizes are median human-normalized Atari scores from Rainbow's
Table 2 [Hes18] unless noted (separately published agents, not clean
ablations; in Rainbow's ablation, prioritized replay and multi-step mattered
most†).

| Method | Mechanism | Reported effect | Plausible here? |
|---|---|---|---|
| Double DQN [vHGS16] | decouples selection from evaluation; the max over noisy estimates is biased upward | 79% → 117% | medium-high. [I] Overestimation is largest for actions that keep more inventory (noise ∝ inventory²), predicting too-slow selling: compare learned vs optimal inventory paths |
| Dueling [Wang16] | V and A streams; robust to small gaps | 151% (with Double DQN) | high, cheap |
| Advantage learning / consistent Bellman [Bel16] | widens gaps by lowering suboptimal actions' values; optimal policies preserved | a one-line change that "significantly outperform[s]" DQN on 60 games | high; also works with linear tiles |
| Munchausen RL [VPG20] | adds τ·log π to the reward: implicit KL regularization (averages Q over iterations) and gap increase | beats DQN by a large margin and C51; M-IQN beats Rainbow | high: the error bound depends on the norm of the *averaged* errors [Vie20], which suits this noise |
| C51 / QR-DQN [BDM17, Dab18] | learn the return distribution | C51 164%; QR-DQN +33% median over C51 | low for tiles (equivalent to expected RL in the tabular/linear case [Lyle19]); moderate for the MLP |
| HL-Gauss classification loss [Far24] | cross-entropy against a histogram of the target | more robust to reward noise†; same as MSE for deterministic dynamics | moderate, cheap |
| Averaged-DQN [ABS17] | target = mean of the last K networks | lower target-error variance | moderate; similar to Polyak |
| Ensembles: Bootstrapped DQN [Osb16], Maxmin [Lan20], REDQ [Chen21] | several heads; min over a subset controls bias | REDQ makes high update-to-data ratios work | moderate; addresses the replay-ratio failure |
| Huber loss | caps gradients on outliers | Adam + MSE beat Huber + RMSProp on small tasks [OCC21] | low: the noise here is Gaussian, so MSE is efficient |
| Polyak target updates [Lil16] | smooth target | standard in DDPG | low-moderate |
| Weight averaging for acting/evaluation [Nik18] | average the iterates | stabilizes A2C/DDPG; DQN's greedy action changes in ~10% of states per update [Sch22] | high, for the checkpoint fluctuation seen in E16 |

## 3. Published RL results on execution and market making

* **Execution:**
  * **[E] [NFK06]:** 1.5 years of NASDAQ data, tabular RL, "50% or more"
    improvement over submit-and-leave†. They assume their own orders don't
    move the market and update every inventory/action pair on each
    historical path, backward in time: the all-action trick of §4.
  * **[E] [HW14]:** Q-learning that rescales the Almgren–Chriss trajectory
    improved implementation shortfall by up to 10.3% on average.
  * **[E] [NLJ21]:** Double DQN beat TWAP on most of 9 stocks.
  * **[E] [ML24]:** Double DQN on Almgren–Chriss with time-varying
    liquidity "learns the optimal policy when the analytical solution is
    available".
* **Market making:**
  * **[E] [Spo18]:** linear tile codings with eligibility traces and an
    asymmetrically dampened PnL reward beat simple benchmarks.
  * **[E] [Jer22]:** PPO solves an Avellaneda–Stoikov-type problem "to
    near optimality"; vectorized environments were crucial.
  * **[E] [GM19]:** model-based actor-critic-like networks for multi-bond
    market making.
* **Analytical optima and theory:**
  * **[E]** Benchmarks: [AC00], [AS08], [GLFT13] (closed-form
    approximations of optimal quotes), [BL98]; survey [HXY23].
  * **[E] [TR19]:** for linear-quadratic control, plug-in model-based
    estimation beats LSTD by at least a factor of the state dimension, and
    certainty-equivalence control beats policy gradient by several such
    factors.
* **[I] Synthesis:** few papers compare against an exact optimum, and
  those that do use stylized, low-noise dynamics. This project's results
  line up with the literature: market making is easy (high SNR), execution
  is hard (consistent with the small value of the signal in [BL98]), and
  model-based reaching 99–100% in 20–50 episodes is the
  certainty-equivalence advantage [TR19] predicts.

## 4. Variance reduction and structure exploitation

* **4.1 Remove the exogenous price innovation (a control variate).**
  * **[I]** With execution price S − ηa and mark-to-market reward,
    r = −ηa² + q′(μ(x) + ε′), where μ is the signal-implied drift and ε′
    the price innovation. The term q′ε′ has zero mean given (s, a), so
    subtracting it leaves Q* *unchanged* (when μ is known) and removes the
    dominant noise; what remains is the randomness of the next signal.
  * **[I]** With μ estimated, the bias is q′(μ − μ̂): one regression
    parameter, which the model-based agent learns in ~20 episodes.
  * **[E]** Removing exogenous reward components speeds up learning
    [DTC18]. Input-dependent baselines are unbiased and reduce variance
    [MVSA19]. Baselines are additive control variates [GBB04].
* **4.2 All-action updates with shared noise.**
  * **[I]** Impact is temporary, so the price path doesn't depend on the
    action. For each transition, targets for *every* action can be computed
    with the same realized ε: the noise in action differences falls from
    ~q·ε to ~(a − a′)·ε. This needs the impact function (known or learned).
  * **[E]** See [Sin23] (hindsight learning in exogenous MDPs), [PCG20]
    (counterfactual augmentation) and NFK06.
* **4.3 Afterstates.**
  * **[E]** [SB18 §6.8]; [Pow09].
  * **[I]** Q(t,q,x,a) = −ηa² + W(t, q − a, x): one function W serves all
    actions. (E2 tried a version of this and logged it as a negative
    result.)
* **4.4 Common random numbers and antithetics.**
  * **[E]** PEGASUS [NJ00]; [Gla03]. This project already uses common
    random numbers; antithetic (mirrored-noise) paths would add more.
* **4.5 Potential-based shaping.**
  * **[E]** Shaping with a potential Φ preserves optimal policies [NHR99]
    and equals Q initialization in the tabular case [Wie03].
  * **[I]** Mark-to-market reward is already shaping with Φ = q·S; shaping
    toward an analytic value makes the learner fit a residual but doesn't
    remove noise.
* **4.6 Better estimators.**
  * **[E]** Zap Q-learning [DM17]; variance-reduced Q-learning [Wai19];
    Polyak–Ruppert averaging [PJ92]; LSTD/LSPI [LP03].

## 5. Ranked next experiments (as proposed)

1. **Evaluation hygiene:** antithetic paths; acting with averaged weights;
   ≥ 10 seeds with interquartile means and bootstrap intervals [Aga21].
2. **The exogenous-innovation control variate (§4.1)** for tiles, DQN and
   actor-critic. Expected: the largest single gain; tile Q-learning above
   90% at 40k if variance is the bottleneck.
3. **An oracle / noise-dial test [Fu19]:** exact expected targets, then
   targets with k·(sampled − expected) noise, k from 0 to 1, to separate
   bias from variance.
4. **Step-size and estimator ablations for tiles:** α/R with the replay
   ratio, per-weight Adam, 1/n^0.7, Polyak–Ruppert averaging, ridge LSTD-Q.
5. **Sharing across actions:** V + A decomposition for tiles; per-action
   networks and tile-coded inputs as negative controls for DQN.
6. **Gap-increasing and averaging targets:** advantage learning,
   Munchausen, dueling, Double Q; check the learned inventory path.
7. **All-action updates and afterstates (§4.2–4.3)** as a semi-model-based
   tier, to locate the boundary between model-free and model-based here.

## 6. Compiling and ensembling learned policies

Setting: `ExecutionPolicyTable` (E22) evaluates a trained policy once at the
centre of each (t, inventory, signal-bin) cell and stores one action byte;
E23 will combine tables from networks trained with different seeds by a
per-cell median action. Both steps have published relatives, though none was
found for optimal execution specifically.

### 6.1 Distillation, extraction and tabulation

* **[E] Policy distillation [Rus16]** trains a smaller student network on the
  Q-values or action distribution of a trained DQN teacher, using states the
  teacher visits; the student matches or exceeds the teacher while being much
  smaller. It builds on knowledge distillation [Hin15]. Their best-performing
  variants match the teacher's softened outputs (KL), not only its argmax:
  the student sees how close the runner-up action was.
* **[E] DAgger [RGB11]** shows why the sampling distribution matters. A
  student fitted on the teacher's states has errors that compound once its own
  mistakes move it to states the teacher never visited; training on states
  from the student's own rollouts (labelled by the teacher) removes the
  quadratic-in-horizon penalty of plain behaviour cloning.
* **[E] VIPER / Q-DAgger [Bas18]** adds the weighting. The loss at a state is
  the oracle's V(s) − min_a Q(s, a), i.e. how much a wrong action can cost,
  and the dataset is resampled in proportion to it. This gave decision trees
  an order of magnitude smaller than plain DAgger at equal reward (Pong:
  769 vs 7,967 nodes). The weight uses the worst action; an
  expected-regret weight for a specific wrong action is a small variant [I].
* **[E] Tabulating a continuous-state policy for deployment** has one
  prominent precedent in the other direction: ACAS Xu stores a dynamic-programming
  lookup table that is too large for avionics, and [Jul19] replaces it with
  networks (about 1000x less storage). Our table is the reverse trade, with a
  small state space where a table is cheap and the network is the slow part.
  No paper found studies bin-centre evaluation of a network policy.

**What this implies for a bin-centre table [I].**
1. A uniform grid spends bytes where the policy is never queried. The signal
   is AR(1), so its distribution at step t is known in closed form, and
   bins should be equal-probability (quantile) bins, or the evaluation points
   should be drawn from that distribution, not a fixed range.
2. Centre evaluation assigns a whole bin the action at one point. If the
   decision boundary crosses a bin, a better label is the action with the
   highest *cell-averaged* Q (or the cell's regret-minimising action), which
   needs only the network's Q-values at several signals per cell.
3. Errors are not equal. Neighbouring actions have small Q-gaps near
   boundaries, so a boundary miss costs little, while a miss at a large gap
   (e.g. nearly all inventory left late in the horizon) costs a lot. Weighting
   cells by Q-gap, as in [Bas18], tells where to refine bins. Because the
   exact DP optimum is available, the table's resolution ceiling can be
   measured directly by tabulating the DP policy on the same grid.

### 6.2 Ensembles of policies and value functions

* **[E] Voting over independently trained learners.** Wiering and van
  Hasselt [WvH08] combine five RL algorithms by majority voting, rank voting,
  Boltzmann multiplication and Boltzmann addition on maze tasks; majority
  voting and Boltzmann multiplication significantly outperformed the single
  algorithms. Chourasia and Singla [CS19] recast these as social-choice rules
  over Q-learners. Both combine different *algorithms*; neither isolates seed
  diversity of one architecture.
* **[E] Value-function ensembles** are mostly used inside training:
  Bootstrapped DQN [Osb16] for exploration, Averaged-DQN [ABS17] to lower
  target variance, Maxmin Q [Lan20] and REDQ [Chen21] to control
  overestimation. They average or take minima of Q-values and deploy a
  learned-in-the-loop ensemble; this is different from combining finished
  policies after the fact.
* **When voting helps [I].** A vote reduces variance that is independent
  across members and does nothing for bias shared by all (here: the
  function-class limit and the noisy-target bias of §1). The 8-seed spread
  at 40k episodes is the room available: voting can at best approach what an
  idealised seed-free learner reaches. Policy churn [Sch22] suggests members
  disagree mostly near decision boundaries, exactly where Q-gaps are small
  and the cost of disagreement is low, so expected gains are modest and the
  ensemble mostly smooths boundaries.
* **Median of ordinal actions.** I found no RL paper that uses the median of
  ordered actions. By social-choice reasoning [I] it is the L1 consensus
  (it minimises Σ|a_i − a|) and, for single-peaked preferences over an
  ordered set, the Condorcet winner; unlike a plurality vote it never selects
  an extreme action that few members favour while the rest split on the other
  side. It also needs no Q-values, so it applies to tables.

### 6.3 Weight averaging as a cheaper alternative

Averaging weights over the training trajectory gives one deployed network.
Polyak–Ruppert averaging [PJ92] is the classical form; an exponential moving
average is its constant-step variant [I]. Stochastic weight averaging [Izm18]
averages iterates under a cyclical or constant learning rate and reports
better generalisation than ordinary SGD at almost no extra cost. For RL,
[Nik18] reports more stable A2C/DDPG. Caveats [I]: averaged weights are not
the weights of any visited network, and averaging is sound only when the
iterates lie in one basin; that holds within a run but not across seeds
from different initialisations, so weight averaging does not replace a
seed ensemble. It addresses checkpoint-to-checkpoint fluctuation, not
seed-to-seed variance.

### 6.4 Ranked experiments for this codebase

1. **Table quality ceiling and fill rule (E22 follow-up).**
   *Hypothesis:* much of any table-vs-network gap is bin placement and
   centre labelling, not the network. *Implement:* (a) tabulate the exact DP
   policy on the same grid; (b) quantile bins; (c) cell-averaged-Q labels
   from K signals per cell. *Measure:* score share for bins ∈ {16, 32, 64,
   128}, versus the network. *Cost:* small; no training, a few ns per lookup.
2. **Seed-ensemble aggregation (E23).** *Hypothesis:* the cell-wise median
   of N tables beats the mean seed and approaches the best seed, with
   diminishing returns beyond N ≈ 5. *Implement:* median, plurality, and
   mean-Q-then-argmax over the same members. *Measure:* score at N = 1, 3,
   5, 8 with intervals over resampled subsets of seeds, plus per-cell
   disagreement. *Cost:* the 8 trained networks already exist; evaluation
   only.
3. **Weight averaging vs ensemble.** *Hypothesis:* EMA or last-K-checkpoint
   averaging raises the single-network mean but not above a seed ensemble.
   *Implement:* an EMA copy of `NeuralQ` weights (one extra vector), SWA over
   the last K checkpoints. *Measure:* 8-seed score and spread against
   experiment 2. *Cost:* one extra training batch of 8 seeds.
4. **Gap-weighted refinement of a table.** *Hypothesis:* spending bins by
   Q-gap, as in VIPER, beats uniform bins at equal table size. *Implement:*
   a non-uniform signal-bin edge list per t, refined where the network's
   Q-gap times visit probability is largest. *Measure:* score versus table
   bytes. *Cost:* moderate; needs a variable-bin lookup.
5. **Cross-learner voting.** *Hypothesis:* rank voting over tile,
   actor-critic and MLP tables (the Wiering-van Hasselt setup) beats the
   best member only if errors are uncorrelated. *Implement:* rank vote
   over three tables. *Measure:* score, and error correlation against DP.
   *Cost:* low. Expect little because the members differ greatly in quality.

## References

* [ABS17] Anschel, Baram, Shimkin. Averaged-DQN. 2017. https://arxiv.org/abs/1611.01929
* [AC00] Almgren, Chriss. Optimal execution of portfolio transactions. 2000.
* [Aga21] Agarwal et al. Deep RL at the edge of the statistical precipice. 2021. https://arxiv.org/abs/2108.13264
* [AS08] Avellaneda, Stoikov. High-frequency trading in a limit order book. 2008. https://math.nyu.edu/~avellane/HighFrequencyTrading.pdf
* [Bas18] Bastani, Pu, Solar-Lezama. Verifiable RL via policy extraction. NeurIPS 2018. https://arxiv.org/abs/1805.08328
* [BDM17] Bellemare, Dabney, Munos. A distributional perspective on RL. 2017. https://arxiv.org/abs/1707.06887
* [Bel16] Bellemare et al. Increasing the action gap. 2016. https://arxiv.org/abs/1512.04860
* [BH18] Balles, Hennig. Dissecting Adam. 2018. https://proceedings.mlr.press/v80/balles18a.html
* [BL98] Bertsimas, Lo. Optimal control of execution costs. 1998.
* [Chen21] Chen, Wang, Zhou, Ross. REDQ. 2021. https://arxiv.org/abs/2101.05982
* [Dab18] Dabney et al. Distributional RL with quantile regression. 2018. https://arxiv.org/abs/1710.10044
* [CS19] Chourasia, Singla. Unifying ensemble methods for Q-learning via social choice theory. 2019. https://arxiv.org/abs/1902.10646
* [DM17] Devraj, Meyn. Zap Q-learning. 2017. https://arxiv.org/abs/1707.03770
* [DTC18] Dietterich, Trimponias, Chen. Discovering and removing exogenous state variables and rewards. 2018. https://arxiv.org/abs/1806.01584
* [EDM03] Even-Dar, Mansour. Learning rates for Q-learning. 2003. https://jmlr.org/papers/v5/evendar03a.html
* [Far24] Farebrother et al. Stop regressing. 2024. https://arxiv.org/abs/2403.03950
* [Fu19] Fu, Kumar, Soh, Levine. Diagnosing bottlenecks in deep Q-learning. 2019. https://arxiv.org/abs/1902.10250
* [GBB04] Greensmith, Bartlett, Baxter. Variance reduction for gradient estimates in RL. 2004. https://jmlr.org/papers/v5/greensmith04a.html
* [Gla03] Glasserman. Monte Carlo Methods in Financial Engineering. 2003.
* [GLFT13] Guéant, Lehalle, Fernandez-Tapia. Dealing with the inventory risk. 2013. https://arxiv.org/abs/1105.3115
* [GM19] Guéant, Manziuk. Deep RL for market making in corporate bonds. 2019. https://arxiv.org/abs/1910.13205
* [GRLGW20] Ghiassian, Rafiee, Lo, Gao, White. Improving performance in RL by breaking generalization in neural networks. 2020. https://arxiv.org/abs/2003.07417
* [GYRS18] Ghiassian, Yu, Rafiee, Sutton. Two geometric input transformation methods for fast online RL with neural nets. 2018. https://arxiv.org/abs/1805.07476
* [Hes18] Hessel et al. Rainbow. 2018. https://arxiv.org/abs/1710.02298
* [Hin15] Hinton, Vinyals, Dean. Distilling the knowledge in a neural network. 2015. https://arxiv.org/abs/1503.02531
* [HW14] Hendricks, Wilcox. An RL extension to the Almgren–Chriss model. 2014. https://arxiv.org/abs/1403.2229
* [HXY23] Hambly, Xu, Yang. Recent advances in RL in finance. 2023. https://arxiv.org/abs/2112.04553
* [Izm18] Izmailov, Podoprikhin, Garipov, Vetrov, Wilson. Averaging weights leads to wider optima and better generalization. UAI 2018. https://arxiv.org/abs/1803.05407
* [Jer22] Jerome, Sánchez-Betancourt, Savani, Herdegen. Model-based gym environments for LOB trading. 2022. https://arxiv.org/abs/2209.07823
* [Jul19] Julian, Kochenderfer, Owen. Deep neural network compression for aircraft collision avoidance systems. J. Guidance, Control, and Dynamics, 2019. https://arxiv.org/abs/1810.04240
* [KB15] Kingma, Ba. Adam. 2015. https://arxiv.org/abs/1412.6980
* [Lan20] Lan, Pan, Fyshe, White. Maxmin Q-learning. 2020. https://arxiv.org/abs/2002.06487
* [Lil16] Lillicrap et al. DDPG. 2016. https://arxiv.org/abs/1509.02971
* [LP03] Lagoudakis, Parr. Least-squares policy iteration. 2003. https://jmlr.org/papers/v4/lagoudakis03a.html
* [Lyle19] Lyle, Bellemare, Castro. Expected vs distributional RL. 2019. https://arxiv.org/abs/1901.11084
* [ML24] Macrì, Lillo. RL for optimal execution when liquidity is time-varying. 2024. https://arxiv.org/abs/2402.12049
* [MVSA19] Mao et al. Variance reduction for RL in input-driven environments. 2019. https://arxiv.org/abs/1807.02264
* [Nag26] Nagarajan, Daley, White, Machado. Value-sharing across actions. 2026. https://arxiv.org/abs/2606.29806
* [NFK06] Nevmyvaka, Feng, Kearns. RL for optimized trade execution. 2006. https://www.cis.upenn.edu/~mkearns/papers/rlexec.pdf
* [NHR99] Ng, Harada, Russell. Policy invariance under reward transformations. 1999.
* [Nik18] Nikishin et al. Improving stability in deep RL with weight averaging. 2018.
* [NJ00] Ng, Jordan. PEGASUS. 2000. https://arxiv.org/abs/1301.3878
* [NLJ21] Ning, Lin, Jaimungal. Double deep Q-learning for optimal execution. 2021. https://arxiv.org/abs/1812.06600
* [OCC21] Obando-Ceron, Castro. Revisiting Rainbow. 2021. https://arxiv.org/abs/2011.14826
* [Osb16] Osband et al. Bootstrapped DQN. 2016. https://arxiv.org/abs/1602.04621
* [PCG20] Pitis, Creager, Garg. Counterfactual data augmentation. 2020. https://arxiv.org/abs/2007.02863
* [PJ92] Polyak, Juditsky. Acceleration of stochastic approximation by averaging. 1992.
* [Pow09] Powell. Approximate Dynamic Programming (2007); slides 2009.
* [Rah19] Rahaman et al. On the spectral bias of neural networks. 2019. https://arxiv.org/abs/1806.08734
* [SB18] Sutton, Barto. Reinforcement Learning: An Introduction, 2nd ed. 2018. http://incompleteideas.net/book/the-book-2nd.html
* [Sch22] Schaul et al. The phenomenon of policy churn. 2022. https://arxiv.org/abs/2206.00730
* [RGB11] Ross, Gordon, Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning (DAgger). AISTATS 2011. https://arxiv.org/abs/1011.0686
* [Rus16] Rusu et al. Policy distillation. ICLR 2016. https://arxiv.org/abs/1511.06295
* [Sin23] Sinclair et al. Hindsight learning for MDPs with exogenous inputs. 2023. https://arxiv.org/abs/2207.06272
* [Spo18] Spooner et al. Market making via RL. 2018. https://arxiv.org/abs/1804.04216
* [Tang23] Tang, Munos, Rowland, Valko. VA-learning. 2023. https://arxiv.org/abs/2305.18161
* [TR19] Tu, Recht. The gap between model-based and model-free methods on the LQR. 2019. https://arxiv.org/abs/1812.03565
* [vH18] van Hasselt et al. Deep RL and the deadly triad. 2018. https://arxiv.org/abs/1812.02648
* [vHGS16] van Hasselt, Guez, Silver. Deep RL with double Q-learning. 2016. https://arxiv.org/abs/1509.06461
* [Vie20] Vieillard et al. Leverage the average. 2020. https://arxiv.org/abs/2003.14089
* [VPG20] Vieillard, Pietquin, Geist. Munchausen RL. 2020. https://arxiv.org/abs/2007.14430
* [Wai19] Wainwright. Variance-reduced Q-learning is minimax optimal. 2019. https://arxiv.org/abs/1906.04697
* [Wang16] Wang et al. Dueling network architectures. 2016. https://arxiv.org/abs/1511.06581
* [Wie03] Wiewiora. Potential-based shaping and Q-value initialization are equivalent. 2003.
* [WvH08] Wiering, van Hasselt. Ensemble algorithms in reinforcement learning. IEEE Trans. SMC-B 38(4), 2008. https://doi.org/10.1109/TSMCB.2008.920231
* [ZYW21] Zhang, Yao, Whiteson. Breaking the deadly triad with a target network. 2021. https://arxiv.org/abs/2101.08862
