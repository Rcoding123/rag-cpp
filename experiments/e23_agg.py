import csv, statistics as st, sys, random
for bins in ('64', '256'):
    rows = list(csv.reader(open(f"{sys.argv[1]}/e23_{bins}.csv")))
    print(f"\n{bins} bins")
    print("| Members | Ensembles | Median vote: mean (min–max) | Plurality vote: mean (min–max) | Median vote beats the members' mean |")
    print("|---|---|---|---|---|")
    single = {r[2]: float(r[3]) for r in rows if r[1] == '1'}
    for size in ('1','2','3','4','5','6','7','8'):
        sel = [r for r in rows if r[1] == size]
        med = [float(r[3]) for r in sel]; plu = [float(r[4]) for r in sel]
        beats = sum(1 for r in sel if float(r[3]) > st.mean(single[c] for c in r[2]))
        print(f"| {size} | {len(sel)} | {100*st.mean(med):.1f}% ({100*min(med):.1f}–{100*max(med):.1f}) | {100*st.mean(plu):.1f}% ({100*min(plu):.1f}–{100*max(plu):.1f}) | {beats}/{len(sel)} |")
    print("best single seed:", max(single, key=single.get), f"{100*max(single.values()):.1f}%")
