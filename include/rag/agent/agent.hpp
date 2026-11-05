// rag/agent/agent.hpp — the decision-making core of a self-learning agent.
//
//   observation ─► encoder ─► learner's Q-values ─► policy (ε-greedy) ─► action
//        ▲                         ▲
//        │                         └── learning: every step (TD + replay) or
//        │                             every episode (episodic memory)
//        └──────── reward, next observation ◄── environment
//
// The agent never sees the environment's internals: only observations, the
// action count, rewards, and the environment's feature ranges. The same
// agent learns ChainEnv and ExecutionEnv unchanged.
//
// Exploration is the price of learning: a fraction ε of decisions are random.
// ε decays as the agent gains experience, and reexplore() raises it again
// temporarily (the Trainer calls it when it detects that the world changed).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "rag/agent/env.hpp"
#include "rag/agent/features.hpp"
#include "rag/agent/learners.hpp"
#include "rag/agent/mlp.hpp"
#include "rag/agent/rng.hpp"

namespace rag::agent {

enum class LearnerKind : std::uint8_t {
  TileQ,     // parametric: tile coding + Q-learning + experience replay
  Episodic,  // instance-based: k-NN over remembered situations (episodic control)
  // Policy gradient: a softmax policy over tile-coded logits (the actor) and a
  // tile-coded state value (the critic); the critic's TD error is the
  // actor's advantage. On-policy: exploration comes from sampling the policy
  // (epsilon only applies during reexplore()), no replay. Measured best
  // model-free learner on ExecutionEnv (EXPERIMENTS.md E5).
  ActorCritic,
  // DQN-style: an MLP (ReLU, Adam) over the features, n-step targets from a
  // target network, uniform replay of n-step transitions, one minibatch per
  // step taken (applied at episode end). Measured on ExecutionEnv: 84-89% of
  // the optimal gain at 40k episodes vs 65-69% for TileQ (EXPERIMENTS.md
  // E16); ~75x the training compute per episode, 1.7 us per decision.
  NeuralQ,
};

template <std::size_t Dims>
struct AgentConfig {
  LearnerKind learner = LearnerKind::TileQ;

  // Exploration: ε falls linearly from eps_start to eps_end over
  // eps_decay_episodes, then stays at eps_end (never 0: a world that changes
  // can only be noticed by an agent that still occasionally tries things).
  double eps_start = 1.0;
  double eps_end = 0.02;
  std::uint32_t eps_decay_episodes = 1500;

  // Parametric learner.
  std::array<std::uint32_t, Dims> bins{};  // grid cells per dimension; 0 => 8
  std::uint32_t tilings = 8;
  // Step size: alpha / (1 + episodes / alpha_decay_episodes), never below
  // alpha_min. Shrinking it averages out reward noise as experience grows;
  // the floor keeps the agent able to learn if the world changes.
  double alpha = 0.1;
  double alpha_min = 0.003;
  std::uint32_t alpha_decay_episodes = 3000;  // 0 = constant step size
  double gamma = 1.0;  // discount; 1 for fixed-horizon tasks
  double init_q = 0.0;
  // Double Q-learning (van Hasselt, 2010): two estimators, each updated
  // toward the other's value of its own best next action. Removes the upward
  // bias that max() over noisy estimates puts into every Q-learning target.
  bool double_q = false;  // measured: on ExecutionEnv, halving each estimator's updates costs more than the bias it removes
  // Advantage learning (Baird 1999; Bellemare et al. 2016): subtract k times
  // the action's advantage deficit from its target, widening action gaps
  // while keeping the optimal policy. 0 = plain Q-learning. TileQ (E19) and
  // NeuralQ (E20; there double_q means Double DQN: the online network picks
  // the next action, the target network values it).
  double advantage_learning = 0.0;
  std::uint32_t replay_capacity = 100'000;
  std::uint32_t replay_per_step = 4;  // extra updates from replayed experience per real step
  // n-step returns: targets sum n real rewards before bootstrapping from
  // Q(s_{t+n}). n = 1 is plain Q-learning; n >= the horizon is Monte Carlo.
  // (No off-policy correction: exploratory actions inside the n steps bias
  // targets toward the behaviour policy.)
  std::uint32_t n_step = 3;  // measured best of 1,2,3,5,10,20 (EXPERIMENTS.md E2)

  // Actor-critic learner (annealed by the same schedule as alpha).
  double actor_alpha = 0.03;
  double critic_alpha = 0.1;
  double temperature = 1.0;  // softmax temperature of the policy
  // Actor-critic step sizes decay as 1 / (1 + episodes / ac_decay_episodes)
  // (floor alpha_min / alpha). Measured: 3,000 (the TileQ default) shrinks the
  // actor's steps before the policy forms (-235% to 7% at 10k episodes on
  // some seeds); 10,000 gives 75-78% at 10k and 80-86% at 40k.
  std::uint32_t ac_decay_episodes = 10000;
  // How ActorCritic re-explores after a drift: 0 = epsilon-uniform actions
  // (off-policy for the actor's update), > 0 = sample the policy itself at a
  // temperature raised by this factor, decaying with the boost (stays
  // on-policy). E13.
  double ac_reexplore_temperature = 0.0;

  // Episodic learner.
  EpisodicParams memory{};
  std::array<double, Dims> kernel_width{};  // similarity length scale per dimension; 0 => 10% of the range

  // NeuralQ learner.
  std::uint32_t nn_hidden = 64;          // units in each of two hidden layers
  double nn_lr = 1e-3;                   // Adam step size
  std::uint32_t nn_batch = 32;           // minibatch per update
  std::uint32_t nn_target_sync = 2000;   // updates between target-network copies (0 = no target network)
  std::uint32_t nn_warmup = 2000;        // transitions in replay before updates start
  // Weight averaging (E24): > 0 keeps an exponential moving average of the
  // network's weights, avg += nn_polyak * (net - avg) after every update.
  // Exploring actions still use the online network (training is unchanged);
  // greedy actions, snapshots and so the deployed policy use the average.
  double nn_polyak = 0.0;

  std::uint64_t seed = 1;
};

template <Environment Env>
class Agent {
 public:
  using Obs = typename Env::Observation;
  static constexpr std::size_t A = Env::kNumActions;
  static constexpr std::size_t Dims = Env::kStateDims;
  static constexpr std::size_t kEmbed = 64;  // episodic embedding size
  using Config = AgentConfig<Dims>;
  using Memory = EpisodicMemory<kEmbed, A>;

  explicit Agent(const Env& env, Config cfg = {})
      : env_(env), cfg_(defaults(cfg)),
        coder_(Env::kLo, Env::kHi, cfg_.bins, cfg_.tilings),
        embed_(cfg_.kernel_width, cfg_.seed ^ 0xE5BEDull),
        // NeuralQ uses neither tile table: allocating them anyway costs
        // (bins + 1)^Dims cells each (2^21 x 8 tilings x 5 actions = 670 MB at
        // 21 inputs) and every snapshot copies them.
        q_(cfg_.learner == LearnerKind::NeuralQ ? TileQ{} : TileQ(coder_.size(), cfg_.tilings, static_cast<std::uint32_t>(A), cfg_.alpha, cfg_.init_q)),
        q2_(cfg_.double_q && cfg_.learner != LearnerKind::NeuralQ ? q_ : TileQ{}),
        critic_(cfg_.learner == LearnerKind::ActorCritic ? TileQ(coder_.size(), cfg_.tilings, 1, cfg_.critic_alpha) : TileQ{}),
        replay_(cfg_.learner == LearnerKind::TileQ ? cfg_.replay_capacity : 0),
        mem_(cfg_.memory),
        rng_(cfg_.seed) {
    episode_.reserve(1024);
    pending_.reserve(cfg_.n_step + 1);
    if (cfg_.learner == LearnerKind::NeuralQ) {
      Rng init(cfg_.seed ^ 0x4E4E4E4Eull);
      net_ = Mlp({static_cast<std::uint32_t>(Dims), cfg_.nn_hidden, cfg_.nn_hidden, static_cast<std::uint32_t>(A)}, init);
      target_ = net_;
      avg_ = net_;
      nscratch_.resize(net_.scratch_size());
      nreplay_.reserve(std::min<std::size_t>(cfg_.replay_capacity, 1u << 16));
    }
    apply_step_sizes();  // actor-critic step sizes from the first episode on
  }

  // ------------------------------------------------------------- decide
  // explore = true: ε-greedy with random tie-breaking (training).
  // explore = false: purely greedy, deterministic (evaluation / deployment).
  // Allocation-free.
  Action act(const Obs& o, bool explore) noexcept {
    if (cfg_.learner == LearnerKind::ActorCritic) return act_policy(o, explore);
    if (explore && rng_.uniform() < epsilon()) return rng_.below(static_cast<std::uint32_t>(A));
    std::array<double, A> q;
    std::array<bool, A> known;
    if (!explore && averaging())
      estimate_with(q_, q2_, mem_, avg_, o, q.data(), known.data());  // greedy: the averaged weights
    else
      estimate(o, q.data(), known.data());
    // Unknown actions (episodic memory only): optimistic while exploring, so
    // untried actions get tried; avoided when acting for real.
    for (std::size_t a = 0; a < A; ++a)
      if (!known[a]) q[a] = explore ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
    Action best = 0;
    std::uint32_t ties = 1;
    for (Action a = 1; a < A; ++a) {
      if (q[a] > q[best]) { best = a; ties = 1; }
      else if (explore && q[a] == q[best] && rng_.below(++ties) == 0) best = a;  // uniform among ties
    }
    return best;
  }

  // The learner's current value estimates (unknown actions read as 0).
  void q_values(const Obs& o, double* out) const noexcept {
    std::array<bool, A> known;
    estimate(o, out, known.data());
  }

  // Predicted return from o under the greedy policy (the critic, for ActorCritic).
  [[nodiscard]] double value(const Obs& o) const noexcept {
    if (cfg_.learner == LearnerKind::ActorCritic) {
      std::array<std::uint32_t, kMaxTilings> idx;
      coder_.encode(env_.features(o), idx.data());
      return critic_.q(idx.data(), 0);
    }
    std::array<double, A> q;
    std::array<bool, A> known;
    estimate(o, q.data(), known.data());
    double v = -std::numeric_limits<double>::infinity();
    bool any = false;
    for (std::size_t a = 0; a < A; ++a)
      if (known[a]) { v = std::max(v, q[a]); any = true; }
    return any ? v : 0.0;
  }

  // -------------------------------------------------------------- learn
  // Called after every real step.
  void observe(const Obs& o, Action a, double r, const Obs& next, bool done) {
    if (cfg_.learner == LearnerKind::ActorCritic) {
      std::array<std::uint32_t, kMaxTilings> s, s2;
      coder_.encode(env_.features(o), s.data());
      coder_.encode(env_.features(next), s2.data());
      const double target = r + (done ? 0.0 : cfg_.gamma * critic_.q(s2.data(), 0));
      const double delta = target - critic_.q(s.data(), 0);  // TD error = advantage estimate
      critic_.update(s.data(), 0, target);
      std::array<double, A> pi;
      const double temp = sampling_temperature();
      policy(s.data(), q_, pi.data(), temp);
      // grad log pi(a|s) wrt the logit of b = (1[b == a] - pi(b)) / temperature
      for (Action b = 0; b < A; ++b) q_.update(s.data(), b, q_.q(s.data(), b) + delta * ((b == a ? 1.0 : 0.0) - pi[b]) / temp);
      return;
    }
    if (cfg_.learner == LearnerKind::Episodic) {
      episode_.push_back({env_.features(o), a, r});
      return;
    }
    if (cfg_.learner == LearnerKind::NeuralQ) {
      episode_.push_back({env_.features(o), a, r});
      if (done) neural_learn_episode();
      (void)next;
      return;
    }
    if (cfg_.n_step <= 1) {
      Transition t;
      coder_.encode(env_.features(o), t.s.data());
      coder_.encode(env_.features(next), t.next.data());
      t.reward = static_cast<float>(r);
      t.action = static_cast<std::uint16_t>(a);
      t.done = done;
      learn_from(t);
      return;
    }
    // n-step: hold the last n steps; once n are pending (or the episode
    // ends), the oldest becomes a transition to the state n steps later.
    PendingStep ps;
    coder_.encode(env_.features(o), ps.s.data());
    ps.action = a;
    ps.reward = r;
    pending_.push_back(ps);
    if (done) {
      while (!pending_.empty()) emit_pending(next, true);
    } else if (pending_.size() >= cfg_.n_step) {
      emit_pending(next, false);
    }
  }

  // Called once at the end of every training episode.
  void end_episode() {
    if (cfg_.learner == LearnerKind::NeuralQ && !episode_.empty()) neural_learn_episode();
    if (cfg_.learner == LearnerKind::Episodic) {
      double g = 0;  // Monte Carlo return-to-go, computed backwards
      alignas(kCacheLine) std::array<float, kEmbed> key;
      for (std::size_t i = episode_.size(); i-- > 0;) {
        g = episode_[i].reward + cfg_.gamma * g;
        embed_.encode(episode_[i].features, key.data());
        mem_.write(key.data(), episode_[i].action, g);
      }
      episode_.clear();
    }
    ++episodes_;
    ++plastic_episodes_;
    if (boost_left_ > 0) --boost_left_;
    apply_step_sizes();
  }

  // Current TD step size: annealed, and raised during re-exploration so the
  // agent can unlearn what no longer holds.
  [[nodiscard]] double step_size() const noexcept {
    double a = cfg_.alpha;
    if (cfg_.alpha_decay_episodes)
      a = std::max(cfg_.alpha_min, cfg_.alpha / (1.0 + static_cast<double>(plastic_episodes_) / cfg_.alpha_decay_episodes));
    return a;
  }

  // ------------------------------------------------------- exploration
  [[nodiscard]] double epsilon() const noexcept {
    const double f = std::min(1.0, static_cast<double>(episodes_) / std::max(1u, cfg_.eps_decay_episodes));
    double eps = cfg_.eps_start + (cfg_.eps_end - cfg_.eps_start) * f;
    if (boost_left_ > 0) eps = std::max(eps, boost_eps_ * boost_left_ / boost_total_);
    return eps;
  }

  // Raise ε to `eps`, decaying linearly back over `episodes` episodes.
  // Also restarts step-size annealing: re-learning needs plasticity.
  void reexplore(double eps, std::uint32_t episodes) noexcept {
    boost_eps_ = eps;
    boost_total_ = boost_left_ = std::max(1u, episodes);
    plastic_episodes_ = 0;
    apply_step_sizes();
  }

  // ---------------------------------------------------------- snapshots
  // The learned state only (no replay buffer, no RNG): what a champion
  // policy needs. Restoring one makes the agent act exactly like it.
  struct Snapshot {
    TileQ q, q2;  // q: Q-values, or the actor's logits for ActorCritic
    Memory mem;
    TileQ critic;
    Mlp net;  // NeuralQ
  };
  [[nodiscard]] Snapshot snapshot() const { return {q_, q2_, mem_, critic_, averaging() ? avg_ : net_}; }

  // Act greedily as the snapshot would, without touching the learner: the
  // deployed (champion) policy and the still-learning one coexist.
  [[nodiscard]] Action act_as(const Snapshot& s, const Obs& o) const noexcept {
    std::array<double, A> q;
    std::array<bool, A> known;
    estimate_with(s.q, s.q2, s.mem, s.net, o, q.data(), known.data());
    Action best = 0;
    for (Action a = 1; a < A; ++a)
      if ((known[a] && !known[best]) || (known[a] == known[best] && q[a] > q[best])) best = a;
    return best;
  }
  void restore(const Snapshot& s) {
    q_ = s.q;
    q2_ = s.q2;
    mem_ = s.mem;
    critic_ = s.critic;
    net_ = s.net;  // the target network is training state, not policy: left alone
    if (averaging()) avg_ = s.net;
  }

  // Supervised warm start (TileQ only): regress the Q-function onto a
  // teacher's action values, e.g. the plan of a learned model, on states
  // visited in `sim` (a simulator, e.g. the model's own market, so no real
  // experience is spent) by the teacher's greedy policy, with random actions
  // at rate `eps` for coverage. Episode count, schedules and replay are
  // untouched: RL fine-tuning then starts from the teacher's knowledge, and
  // real rewards correct whatever the teacher got wrong (E7). Returns false
  // (and does nothing) for other learners.
  template <class TeacherQ>
  bool distill(const Env& sim, TeacherQ&& teacher_q, std::uint32_t episodes, std::uint64_t seed, double eps = 0.2) {
    if (cfg_.learner != LearnerKind::TileQ) return false;
    Env env = sim;
    Rng rng(seed), pick(seed ^ 0x9E3779B97F4A7C15ull);
    const double saved = q_.alpha();
    std::array<std::uint32_t, kMaxTilings> idx;
    std::array<double, A> tq;
    for (std::uint32_t ep = 0; ep < episodes; ++ep) {
      q_.set_alpha(std::max(0.005, cfg_.alpha / (1.0 + static_cast<double>(ep) / 3000.0)));
      auto o = env.reset(rng);
      for (;;) {
        coder_.encode(env_.features(o), idx.data());
        Action best = 0;
        for (Action a = 0; a < A; ++a) {
          tq[a] = teacher_q(o, a);
          q_.update(idx.data(), a, tq[a]);
          if (tq[a] > tq[best]) best = a;
        }
        const Action act = pick.uniform() < eps ? static_cast<Action>(pick.below(static_cast<std::uint32_t>(A))) : best;
        const auto st = env.step(act, rng);
        if (st.done) break;
        o = st.obs;
      }
    }
    q_.set_alpha(saved);
    if (cfg_.double_q) q2_ = q_;
    return true;
  }

  // Drop replayed experience. After the world changes, old transitions
  // describe dynamics that no longer hold, and replaying them keeps
  // teaching the agent the past.
  void forget_experience() noexcept {
    replay_.clear();
    nreplay_.clear();
    nhead_ = 0;
  }

  [[nodiscard]] std::uint64_t episodes() const noexcept { return episodes_; }
  [[nodiscard]] const Config& config() const noexcept { return cfg_; }
  [[nodiscard]] const TileQ& q_learner() const noexcept { return q_; }
  [[nodiscard]] const Memory& memory() const noexcept { return mem_; }
  [[nodiscard]] std::size_t replay_size() const noexcept { return replay_.size(); }

 private:
  struct EpisodeStep {
    std::array<double, Dims> features;
    Action action;
    double reward;
  };
  struct PendingStep {
    std::array<std::uint32_t, kMaxTilings> s;
    Action action;
    double reward;
  };

  // Softmax policy of the logits in `actor` at tiles idx (numerically stable).
  void policy(const std::uint32_t* idx, const TileQ& actor, double* pi, double temperature = 0) const noexcept {
    if (temperature <= 0) temperature = cfg_.temperature;
    std::array<double, A> h;
    actor.q_values(idx, h.data());
    const double m = *std::max_element(h.begin(), h.end());
    double z = 0;
    for (std::size_t a = 0; a < A; ++a) z += (pi[a] = std::exp((h[a] - m) / temperature));
    for (std::size_t a = 0; a < A; ++a) pi[a] /= z;
  }

  // ActorCritic's sampling temperature: raised during a re-exploration boost
  // when ac_reexplore_temperature > 0.
  [[nodiscard]] double sampling_temperature() const noexcept {
    if (cfg_.ac_reexplore_temperature <= 0 || boost_left_ == 0) return cfg_.temperature;
    return cfg_.temperature * (1.0 + cfg_.ac_reexplore_temperature * boost_left_ / boost_total_);
  }

  // ActorCritic acting: sample the policy while exploring, its mode otherwise.
  Action act_policy(const Obs& o, bool explore) noexcept {
    std::array<std::uint32_t, kMaxTilings> idx;
    coder_.encode(env_.features(o), idx.data());
    std::array<double, A> pi;
    policy(idx.data(), q_, pi.data(), explore ? sampling_temperature() : 0.0);
    if (!explore) return static_cast<Action>(std::max_element(pi.begin(), pi.end()) - pi.begin());
    if (cfg_.ac_reexplore_temperature <= 0 && boost_left_ > 0 && rng_.uniform() < epsilon())
      return rng_.below(static_cast<std::uint32_t>(A));
    double u = rng_.uniform();
    for (Action a = 0; a + 1 < A; ++a)
      if ((u -= pi[a]) <= 0) return a;
    return static_cast<Action>(A - 1);
  }

  void apply_step_sizes() noexcept {
    const double a = step_size();
    if (cfg_.learner == LearnerKind::ActorCritic) {
      const double f = std::max(cfg_.alpha_min / cfg_.alpha,
                                1.0 / (1.0 + static_cast<double>(plastic_episodes_) / std::max(1u, cfg_.ac_decay_episodes)));
      q_.set_alpha(cfg_.actor_alpha * f);
      critic_.set_alpha(cfg_.critic_alpha * f);
      return;
    }
    q_.set_alpha(a);
    if (cfg_.double_q) q2_.set_alpha(a);
  }

  void learn_from(const Transition& t) {
    td_update(t);
    replay_.push(t);
    for (std::uint32_t i = 0; i < cfg_.replay_per_step && replay_.size() > 1; ++i) td_update(replay_.sample(rng_));
  }

  // Turn the oldest pending step into an n-step transition ending at `next`.
  void emit_pending(const Obs& next, bool done) {
    Transition t;
    const PendingStep& first = pending_.front();
    t.s = first.s;
    t.action = static_cast<std::uint16_t>(first.action);
    double g = 0, disc = 1;
    for (const PendingStep& p : pending_) {
      g += disc * p.reward;
      disc *= cfg_.gamma;
    }
    t.reward = static_cast<float>(g);
    t.steps = static_cast<std::uint8_t>(std::min<std::size_t>(pending_.size(), 255));
    t.done = done;
    coder_.encode(env_.features(next), t.next.data());
    pending_.erase(pending_.begin());
    learn_from(t);
  }

  static Config defaults(Config c) {
    for (std::size_t d = 0; d < Dims; ++d) {
      if (c.bins[d] == 0) c.bins[d] = 8;
      if (c.kernel_width[d] <= 0) c.kernel_width[d] = 0.1 * (Env::kHi[d] - Env::kLo[d]);
    }
    if (c.tilings == 0 || c.tilings > kMaxTilings) throw std::invalid_argument("AgentConfig: tilings in [1, 32]");
    if (!(c.alpha > 0) || !(c.gamma > 0) || c.gamma > 1) throw std::invalid_argument("AgentConfig: need alpha > 0, 0 < gamma <= 1");
    if (c.eps_start < c.eps_end || c.eps_end < 0 || c.eps_start > 1) throw std::invalid_argument("AgentConfig: need 0 <= eps_end <= eps_start <= 1");
    if (!(c.nn_polyak >= 0 && c.nn_polyak <= 1)) throw std::invalid_argument("AgentConfig: nn_polyak in [0, 1]");
    return c;
  }

  void estimate(const Obs& o, double* out, bool* known) const noexcept { estimate_with(q_, q2_, mem_, net_, o, out, known); }

  void estimate_with(const TileQ& q1, const TileQ& q2, const Memory& mem, const Mlp& net, const Obs& o, double* out,
                     bool* known) const noexcept {
    if (cfg_.learner == LearnerKind::NeuralQ) {
      const auto f = env_.features(o);
      net.forward(f.data(), out, nscratch_.data());
      for (std::size_t a = 0; a < A; ++a) known[a] = true;
      return;
    }
    if (cfg_.learner != LearnerKind::Episodic) {  // TileQ values, or ActorCritic logits
      std::array<std::uint32_t, kMaxTilings> idx;
      coder_.encode(env_.features(o), idx.data());
      q1.q_values(idx.data(), out);
      if (cfg_.double_q) {  // act on the average of both estimators
        std::array<double, A> o2;
        q2.q_values(idx.data(), o2.data());
        for (std::size_t a = 0; a < A; ++a) out[a] = 0.5 * (out[a] + o2[a]);
      }
      for (std::size_t a = 0; a < A; ++a) known[a] = true;
      return;
    }
    alignas(kCacheLine) std::array<float, kEmbed> key;
    embed_.encode(env_.features(o), key.data());
    mem.estimate(key.data(), out, known);
  }

  // One Q-learning step: target = r + γ max_a' Q(s', a'). With double_q,
  // a coin picks which estimator learns; it chooses a' and the other one
  // values it.
  void td_update(const Transition& t) noexcept {
    TileQ* learn = &q_;
    const TileQ* judge = &q_;
    if (cfg_.double_q) {
      const bool flip = rng_.next() & 1;
      learn = flip ? &q2_ : &q_;
      judge = flip ? &q_ : &q2_;
    }
    double target = t.reward;
    if (!t.done) {
      std::array<double, A> qn;
      learn->q_values(t.next.data(), qn.data());
      const auto best = static_cast<Action>(std::max_element(qn.begin(), qn.end()) - qn.begin());
      const double boot = cfg_.gamma == 1.0 ? 1.0 : std::pow(cfg_.gamma, t.steps);
      target += boot * judge->q(t.next.data(), best);
    }
    if (cfg_.advantage_learning > 0) {
      std::array<double, A> qs;
      learn->q_values(t.s.data(), qs.data());
      target -= cfg_.advantage_learning * (*std::max_element(qs.begin(), qs.end()) - qs[t.action]);
    }
    learn->update(t.s.data(), t.action, target);
  }

  Env env_;  // a copy, used only for features()
  Config cfg_;
  TileCoder<Dims> coder_;
  RandomFourierEncoder<Dims, kEmbed> embed_;
  TileQ q_, q2_;  // q2_ is used only with double_q; q_ holds the actor's logits for ActorCritic
  TileQ critic_;  // ActorCritic only
  ReplayBuffer replay_;
  Memory mem_;
  std::vector<EpisodeStep> episode_;
  std::vector<PendingStep> pending_;  // n-step window (n_step > 1)
  Rng rng_;
  // NeuralQ: online and target networks, replay of n-step transitions in feature space.
  struct NeuralTransition {
    std::array<double, Dims> s, next;
    Action action;
    double ret;         // discounted sum of up to n rewards
    double bootstrap;   // gamma^steps, or 0 if the episode ended within n steps
  };
  Mlp net_, target_, avg_;  // avg_: weight average (nn_polyak > 0)
  [[nodiscard]] bool averaging() const noexcept { return cfg_.learner == LearnerKind::NeuralQ && cfg_.nn_polyak > 0; }
  std::vector<NeuralTransition> nreplay_;
  std::size_t nhead_ = 0;
  std::uint64_t nn_updates_ = 0;
  mutable std::vector<double> nscratch_;

  // Turn the finished episode into n-step transitions, then run one
  // minibatch update per step it took.
  void neural_learn_episode() {
    const std::size_t T = episode_.size();
    const std::uint32_t n = std::max(1u, cfg_.n_step);
    const std::size_t cap = std::max<std::size_t>(1, cfg_.replay_capacity);
    for (std::size_t t = 0; t < T; ++t) {
      NeuralTransition tr{};
      tr.s = episode_[t].features;
      tr.action = episode_[t].action;
      double disc = 1.0;
      std::size_t k = t;
      for (; k < std::min(T, t + n); ++k) {
        tr.ret += disc * episode_[k].reward;
        disc *= cfg_.gamma;
      }
      if (k < T) {
        tr.next = episode_[k].features;
        tr.bootstrap = disc;
      }
      if (nreplay_.size() < cap) {
        nreplay_.push_back(tr);
      } else {
        nreplay_[nhead_] = tr;
        nhead_ = (nhead_ + 1) % cap;
      }
    }
    episode_.clear();
    if (nreplay_.size() < cfg_.nn_warmup) return;
    std::array<double, A> qn;
    for (std::size_t u = 0; u < T; ++u) {
      for (std::uint32_t b = 0; b < cfg_.nn_batch; ++b) {
        const NeuralTransition& tr = nreplay_[rng_.below(static_cast<std::uint32_t>(nreplay_.size()))];
        double y = tr.ret;
        const Mlp& judge = cfg_.nn_target_sync ? target_ : net_;
        if (tr.bootstrap > 0) {
          judge.forward(tr.next.data(), qn.data(), nscratch_.data());
          if (cfg_.double_q) {  // Double DQN: the online network picks, the target network values
            std::array<double, A> qo;
            net_.forward(tr.next.data(), qo.data(), nscratch_.data());
            y += tr.bootstrap * qn[static_cast<std::size_t>(std::max_element(qo.begin(), qo.end()) - qo.begin())];
          } else {
            y += tr.bootstrap * *std::max_element(qn.begin(), qn.end());
          }
        }
        if (cfg_.advantage_learning > 0) {  // gap-increasing target (E19), with the online network
          std::array<double, A> qs;
          net_.forward(tr.s.data(), qs.data(), nscratch_.data());
          y -= cfg_.advantage_learning * (*std::max_element(qs.begin(), qs.end()) - qs[tr.action]);
        }
        net_.accumulate(tr.s.data(), tr.action, y);
      }
      net_.step(cfg_.nn_lr);
      if (averaging()) avg_.blend_toward(net_, cfg_.nn_polyak);
      if (cfg_.nn_target_sync && ++nn_updates_ % cfg_.nn_target_sync == 0) target_ = net_;
    }
  }

  std::uint64_t episodes_ = 0;
  std::uint64_t plastic_episodes_ = 0;  // episodes since the last reexplore() (step-size annealing clock)
  double boost_eps_ = 0;
  std::uint32_t boost_left_ = 0, boost_total_ = 1;
};

}  // namespace rag::agent
