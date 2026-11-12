"""Imitation of a privileged teacher: behaviour cloning, then DAgger.

Teacher: the fair-value oracle (BuiltinPolicy::oracle) on gap = fair value - mid,
threshold chosen on val_ckpt with the C++ run_builtin (nothing hidden leaves
C++ on validation). Labels on the train stream come from the privileged gap
column (train block only). Student: an MLP on observations only (16 -> hidden
-> 3), cross-entropy. DAgger iteration i executes the teacher's action with
probability beta0^i (per env and step, seeded) and the student's otherwise, and
labels every visited state with the teacher's action; the dataset aggregates.

On discovery_v4 both oracles lose after costs, so the student is expected to
lose too: its value is the learnability probe (the share of the teacher's gross
and net the student keeps on val_ckpt)."""
import numpy as np

from .. import native
from ..envs import oracle_actions
from ..policies import MLPStudent, freeze_module


def _fit(student, X, Y, p, gen):
    import torch as th
    opt = th.optim.Adam(student.parameters(), lr=p["learning_rate"])
    Xt = th.as_tensor(X, dtype=th.float32)
    Yt = th.as_tensor(Y, dtype=th.long)
    n = len(Xt)
    student.train()
    for _ in range(p["epochs"]):
        perm = th.randperm(n, generator=gen)
        for i in range(0, n, p["batch_size"]):
            idx = perm[i: i + p["batch_size"]]
            loss = th.nn.functional.cross_entropy(student(Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    student.eval()


def train_bc_dagger(ctx):
    import torch as th
    R = native.load()
    p = ctx.p
    vsrc = ctx.ckpt.source()
    teacher_val = {}
    for th_t in p["teacher_thresholds"]:
        teacher_val[th_t] = R.run_builtin(ctx.params, vsrc, ("oracle", float(th_t)), threads=ctx.eval_threads)
    means = {k: float(np.mean(v["net_pnl_ticks"])) for k, v in teacher_val.items()}
    th_t = max(means, key=lambda k: (means[k], -p["teacher_thresholds"].index(k)))
    ctx.extra.update(teacher_threshold=th_t, teacher_val_ckpt=means)

    th.manual_seed(ctx.seed)
    gen = th.Generator().manual_seed(ctx.seed)
    rng = np.random.default_rng(ctx.seed)
    student = MLPStudent.build(p["hidden"])
    env = ctx.train_env(privileged=True)
    n, T = env.n_envs, ctx.steps_per_episode
    rounds_total = ctx.episodes // n
    bc_rounds = max(1, int(round(p["bc_fraction"] * rounds_total)))
    iters = int(p["dagger_iterations"])
    rest = rounds_total - bc_rounds
    plan = [(bc_rounds, 1.0)] + [(rest // iters + (1 if i < rest % iters else 0), p["beta0"] ** (i + 1)) for i in range(iters)]
    plan = [x for x in plan if x[0] > 0]
    obs = env.reset()
    Xs, Ys = [], []
    steps = episodes = 0
    if ctx.live:
        ctx.live.ckpt(MLPStudent(freeze_module(student), p["hidden"]), 0, [], [], 1.0)
    for k, (rounds, beta) in enumerate(plan):
        returns = []
        for _ in range(rounds):
            for _t in range(T):
                teach = oracle_actions(obs, env.priv, th_t)
                if beta < 1.0:
                    with th.no_grad():
                        stud = th.argmax(student(th.as_tensor(obs.astype(np.float32))), dim=1).numpy().astype(np.int64)
                    a = np.where(rng.random(n) < beta, teach, stud)
                else:
                    a = teach
                Xs.append(obs.astype(np.float32))
                Ys.append(teach)
                obs, _, done = env.step(a)
                steps += n
            assert done[0]
            episodes += n
            returns += env.episode_stats["net_pnl_ticks"].tolist()
        _fit(student, np.concatenate(Xs), np.concatenate(Ys), p, gen)
        frozen = MLPStudent(freeze_module(student), p["hidden"])
        ctx.ckpt.offer(frozen, steps, episodes, {"phase": "bc" if k == 0 else f"dagger{k}", "beta": beta})
        if ctx.live:
            ctx.live.ckpt(frozen, episodes, returns, ctx.ckpt.pop_evals(), beta)
    if ctx.live:
        ctx.live.done(ctx.ckpt.champion_score, "")
    # learnability probe on val_ckpt: the student's share of the teacher's net and gross
    tv = teacher_val[th_t]
    t_net = float(np.mean(tv["net_pnl_ticks"]))
    t_gross = float(np.mean(tv["net_pnl_ticks"] + tv["fees_ticks"]))
    from ..evaluate import run_policy
    sv = run_policy(ctx.params, vsrc, ctx.ckpt.champion, ctx.eval_threads)
    s_net = float(np.mean(sv["net_pnl_ticks"]))
    s_gross = float(np.mean(sv["net_pnl_ticks"] + sv["fees_ticks"]))
    ctx.extra.update(dataset_rows=int(sum(len(x) for x in Xs)), teacher_net=t_net, teacher_gross=t_gross, student_net=s_net,
                     student_gross=s_gross, gross_share=s_gross / t_gross if t_gross else None,
                     net_share=s_net / t_net if t_net else None)
