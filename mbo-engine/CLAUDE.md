# Working rules for mbo-engine

These apply to any AI session working in this folder. They come from the
owner's explicit instructions. The rest of rag-cpp is Windows-only; **this
folder targets the Linux server** (Ubuntu 24.04, gcc/clang, Python 3.12).

1. **Research before answering.**
   - Facts about products, prices, hardware, vendor APIs, licensing or
     benchmarks: look them up and cite sources before answering.
   - Never answer from memory.
   - Say plainly when something could not be verified.
2. **Test every script before handing it over.**
   - Run every path that can be run safely: dry runs, quote-only modes,
     simulated failures, re-runs, tiny or already-paid inputs.
   - Check library APIs against the installed source, not memory.
   - State exactly what remains untested and why.
3. **Never spend money without an explicit go-ahead.**
   - Databento orders bill at submission and are final.
   - `scripts/order_data.py` is quote-only unless run with `--buy` and the
     user types `BUY`.
4. **No risky actions on the server.**
   - No deliberate failed logins: fail2ban bans the IP.
   - Before touching SSH, firewall, fail2ban, sudo, reboots or anything that
     could cut access, say what will happen and wait for approval.
   - Verify hardening from inside an authenticated session (`sudo sshd -T`).
5. **Engine work (PLAN.md Phase 1+) waits for the owner's approval.**
6. **Never install AI coding tools** (or anything not asked for) from
   scripts. The owner installs those themselves.
7. **This repo is public.** Never commit `.env`, API keys, SSH keys, server
   IPs, `data/`, or `orders.json`.
8. **Runtime copies of scripts live in `/srv/quant`.** Edit and test here,
   then copy over.
9. **Heavy jobs go on the batch cores with the batch memory cap.** The live
   cores (0-3) are reserved. Use `scripts/batch_run.sh <cmd>` (no sudo needed;
   `quant-batch.slice` itself needs sudo).
10. **`/srv/quant/data/raw` is read-only** (`chmod -R a-w`, owner request
    2026-11-18). Lift only the directory (`chmod u+w`) to add data, re-lock
    right after, and tell the owner.
11. **Don't trust replay profits.** Follow the validation principles in
    PLAN.md: score on unseen later days, compare with random baselines, and
    report confidence intervals.
