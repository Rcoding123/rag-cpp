#!/usr/bin/env python3
"""of24_live.py - live view of the OF24 Sierra agent learning (docs/ai/OF24_LIVE_SIM_PLAN.md section 8).

    python tools/of24_live.py serve [--dir C:/SierraChart/SierraChartInstance_2/Data/RagAgent] [--mode replay]
                                    [--days 228] [--port 8775]

Serves a page on http://127.0.0.1:PORT/ (local only) that polls /api/state every 4 s. The state is built from
the study's own output only: the profit ledgers (profit_days_<mode>.csv, profit_trades_<mode>.csv), the newest
decision log (<mode>-c1-*.csv: one D line per candidate with the sampled action, its probability, the critic
value and the book inputs) and the weight file (weights_<mode>_c1.txt: how much weight the 6 book inputs have
gained; they start at zero), plus the verdict lines of the last experiments/of24_audit.py report (--audit).
No market data is read. Standard library only.
"""
import argparse
import datetime as dt
import functools
import glob
import http.server
import json
import math
import os
import sys
import threading

MENU = [(8, 4, 30, 0), (8, 4, 30, 1), (12, 6, 60, 0), (12, 6, 60, 1), (16, 8, 60, 0), (16, 8, 60, 1),
        (24, 8, 120, 0), (24, 8, 120, 1), (32, 12, 0, 0), (32, 12, 0, 1), (48, 16, 0, 0), (48, 16, 0, 1)]
GROUPS = ["8/4", "12/6", "16/8", "24/8", "32/12", "48/16"]  # action a (1..12) -> group (a - 1) // 2
EPOCH = dt.date(1899, 12, 30)  # SCDateTime day 0


def sc_date(n):
    return (EPOCH + dt.timedelta(days=int(n))).isoformat()


def read_csv(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    if not lines:
        return []
    head = lines[0].split(",")
    return [dict(zip(head, ln.split(","))) for ln in lines[1:] if ln]


class State:
    def __init__(self, d, mode, days):
        self.dir, self.mode, self.days = d, mode, days
        self.lock = threading.Lock()
        self.log, self.off, self.part = None, 0, ""
        self.per_day = {}   # date -> aggregates of D lines
        self.last_d = None
        self.events = {"T": 0, "X": 0, "C": []}
        self.w_mtime, self.w_hist = 0.0, []
        self.audit = ""     # path of the of24_audit.py report (its verdict is shown)

    def _scan_log(self):
        logs = [p for p in glob.glob(os.path.join(self.dir, f"{self.mode}-c1-*.csv"))]
        if not logs:
            return
        newest = max(logs, key=os.path.getmtime)
        if newest != self.log:
            self.log, self.off, self.part, self.per_day, self.last_d = newest, 0, "", {}, None
            self.events = {"T": 0, "X": 0, "C": []}
        with open(self.log, "rb") as f:
            f.seek(self.off)
            chunk = f.read()
        self.off += len(chunk)
        text = self.part + chunk.decode("utf-8", "replace")
        lines = text.split("\n")
        self.part = lines.pop()
        for ln in lines:
            p = ln.rstrip("\r").split(",")
            k = p[0]
            if k == "D" and len(p) >= 18:
                try:
                    date, a, prob, val = sc_date(p[1]), int(p[4]), float(p[5]), float(p[6])
                    book = [float(x) for x in p[8:14]]
                    levels = min(int(p[14]), 10)  # the study reads at most 10; the log has what Sierra reports
                except ValueError:
                    continue
                g = self.per_day.setdefault(date, {"n": 0, "skip": 0, "act": [0] * 13, "val": 0.0, "lev": 0, "lev_max": 0,
                                                   "imb": 0.0, "imb_n": 0})
                g["n"] += 1
                g["skip"] += a == 0
                g["act"][a] += 1
                g["val"] += val
                g["lev"] += levels
                g["lev_max"] = max(g["lev_max"], levels)
                if levels >= 5:
                    g["imb"] += abs(book[1])
                    g["imb_n"] += 1
                self.last_d = {"date": date, "min": int(p[2]), "dir": int(p[3]), "action": a, "p": prob, "value": val,
                               "book": book, "levels": levels, "mbo": int(p[15]) if p[15].lstrip("-").isdigit() else 0}
            elif k in ("T", "X"):
                self.events[k] += 1
            elif k == "C":
                self.events["C"].append(ln.strip())

    def _scan_weights(self):
        path = os.path.join(self.dir, f"weights_{self.mode}_c1.txt")
        try:
            m = os.path.getmtime(path)
        except OSError:
            return
        if m == self.w_mtime:
            return
        try:
            with open(path, encoding="ascii") as f:
                lines = f.read().split("\n")
        except OSError:
            return
        hdr = lines[1].split() if len(lines) > 1 else []
        try:
            n_in, hidden, updates = int(hdr[1]), int(hdr[3]), int(hdr[7])
            i = next(j for j, ln in enumerate(lines) if ln.startswith("pol_W1 "))
            w = [float(x) for x in lines[i + 1].split()]
        except (ValueError, IndexError, StopIteration):
            return  # a save in progress: try again on the next poll
        if len(w) != n_in * hidden:
            return
        self.w_mtime = m
        col = [math.sqrt(sum(w[r * n_in + c] ** 2 for r in range(hidden))) for c in range(n_in)]
        feat = sorted(col[:n_in - 6])
        rec = {"updates": updates, "depth": math.sqrt(sum(col[c] ** 2 for c in range(n_in - 6, n_in - 3))),
               "mbo": math.sqrt(col[n_in - 3] ** 2 + col[n_in - 2] ** 2), "spread": col[n_in - 1],
               "feat_median": feat[len(feat) // 2]}
        if not self.w_hist or self.w_hist[-1]["updates"] != updates:
            self.w_hist.append(rec)
            if len(self.w_hist) > 4000:
                self.w_hist = self.w_hist[::2]

    def _audit(self):
        """The verdict lines of the last experiments/of24_audit.py run, and when it ran."""
        try:
            txt = open(self.audit, encoding="utf-8").read()
            when = dt.datetime.fromtimestamp(os.path.getmtime(self.audit)).isoformat(timespec="minutes")
        except OSError:
            return None
        lines = [x.strip() for x in txt.split("VERDICT", 1)[-1].splitlines() if x.strip()] if "VERDICT" in txt else []
        return {"when": when, "lines": lines}

    def snapshot(self):
        with self.lock:
            self._scan_log()
            self._scan_weights()
            days = read_csv(os.path.join(self.dir, f"profit_days_{self.mode}.csv"))
            trades = read_csv(os.path.join(self.dir, f"profit_trades_{self.mode}.csv"))
            nets = [float(t["net_ticks"]) for t in trades if t.get("net_ticks")]
            roll, w, s = [], 100, 0.0
            for i, x in enumerate(nets):
                s += x - (nets[i - w] if i >= w else 0.0)
                if i + 1 >= 20 and (i % 5 == 4 or i == len(nets) - 1):
                    roll.append([i + 1, s / min(i + 1, w)])
            mix = []
            for t in trades:
                r = t.get("menu_exit") or t.get("exit_rule", "")  # the menu exit (a scaled bracket keeps its group)
                tgt = r.split("/")[0].lstrip("T") + "/" + r.split("/")[1].lstrip("S") if r.count("/") >= 1 else "?"
                mix.append((t["date"], tgt))
            by_day_mix = {}
            for d, g in mix:
                by_day_mix.setdefault(d, [0] * len(GROUPS))
                if g in GROUPS:
                    by_day_mix[d][GROUPS.index(g)] += 1
            per_day = []
            for d in sorted(self.per_day):
                g = self.per_day[d]
                per_day.append({"date": d, "n": g["n"], "trade_rate": 1 - g["skip"] / g["n"], "value": g["val"] / g["n"],
                                "levels": g["lev"] / g["n"], "lev_max": g["lev_max"],
                                "imb": g["imb"] / g["imb_n"] if g["imb_n"] else None, "mix": by_day_mix.get(d, [0] * 6)})
            return {
                "mode": self.mode, "total_days": self.days, "log": os.path.basename(self.log or ""),
                "days": [{k: d[k] for k in ("date", "trades", "net_ticks", "net_usd", "cum_ticks", "cum_usd", "updates")}
                         for d in days],
                "n_trades": len(nets), "net_ticks": sum(nets), "roll": roll, "groups": GROUPS,
                "recent": trades[-12:][::-1], "per_day": per_day, "last": self.last_d, "events": self.events,
                "weights": self.w_hist, "now": dt.datetime.now().isoformat(timespec="seconds"),
                "audit": self._audit(),
            }


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OF24 agent learning</title>
<style>
:root{color-scheme:light;--bg:#fcfcfb;--card:#ffffff;--fg:#0b0b0b;--fg2:#52514e;--mut:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
 --ring:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--neg:#e34948}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--bg:#1a1a19;--card:#222221;--fg:#f0efec;
 --fg2:#c3c2b7;--mut:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;
 --s4:#c98500;--s5:#d55181;--s6:#008300;--neg:#e66767}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#1a1a19;--card:#222221;--fg:#f0efec;--fg2:#c3c2b7;--mut:#898781;--grid:#2c2c2a;
 --axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--neg:#e66767}
*{box-sizing:border-box}body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;padding:16px}
h1{font-size:19px;margin:0}.sub{color:var(--fg2);margin:2px 0 14px}.mut{color:var(--mut)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:14px}
.tile{background:var(--card);border:1px solid var(--ring);border-radius:10px;padding:10px 12px}
.tile .k{color:var(--fg2);font-size:12px}.tile .v{font-size:22px;font-variant-numeric:tabular-nums;font-weight:600}
.tile .n{color:var(--mut);font-size:12px}.bar{height:6px;background:var(--grid);border-radius:3px;margin-top:6px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--s1);border-radius:3px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--ring);border-radius:10px;padding:12px;min-width:0;position:relative}
.card h2{font-size:14px;margin:0}.card .d{color:var(--fg2);font-size:12px;margin:2px 0 6px}
svg{width:100%;height:210px;display:block;overflow:visible}svg text{fill:var(--mut);font-size:11px}
.leg{display:flex;flex-wrap:wrap;gap:4px 12px;font-size:12px;color:var(--fg2);margin-bottom:4px}
.leg span{display:inline-flex;align-items:center;gap:5px}.leg i{width:10px;height:10px;border-radius:2px;display:inline-block}
.tip{position:absolute;pointer-events:none;background:var(--card);border:1px solid var(--ring);border-radius:8px;padding:6px 8px;
 font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.15);display:none;white-space:nowrap;z-index:5;font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:12px}
td,th{padding:4px 6px;text-align:right;border-bottom:1px solid var(--grid)}th{color:var(--fg2);font-weight:500}td:first-child,th:first-child{text-align:left}
.pos{color:var(--fg)}.wrap{overflow-x:auto}
</style></head><body>
<h1>OF24 agent: learning in a Sierra replay</h1>
<div class="sub">Starts from the PPO-trained policy; one actor-critic step after every closed sim trade. Net is after OF23's base costs, one ES lot.
 Sim fills and in-sample days: this shows behaviour, not an edge. <span class="mut" id="st"></span></div>
<div class="tiles" id="tiles"></div>
<div class="grid">
 <div class="card"><h2>Cumulative net</h2><div class="d">Dollars, one lot, by replayed session</div><svg id="cum"></svg></div>
 <div class="card"><h2>Net per session</h2><div class="d">Ticks after costs</div><svg id="daily"></svg></div>
 <div class="card"><h2>Is it getting better?</h2><div class="d">Mean net ticks per trade over the last 100 trades</div><svg id="roll"></svg></div>
 <div class="card"><h2>Exit choices</h2><div class="d">Share of each session's trades by target/stop (ticks; plain and trailing together)</div><div class="leg" id="mixleg"></div><svg id="mix"></svg></div>
 <div class="card"><h2>How often it trades</h2><div class="d">Share of candidate entries taken (the rest are skipped)</div><svg id="rate"></svg></div>
 <div class="card"><h2>Critic's expected reward</h2><div class="d">Mean value at decisions (reward units: net ticks / 10); it starts at zero</div><svg id="val"></svg></div>
 <div class="card"><h2>Book inputs: weight learned</h2><div class="d" id="wd">Norm of the policy's first-layer weights on the book inputs (they start at zero); recorded while this page's server runs</div><div class="leg" id="wleg"></div><svg id="w"></svg></div>
 <div class="card"><h2>Depth seen at decisions</h2><div class="d">Mean book levels read per decision (1 = best bid/ask only; 10 = recorded full depth)</div><svg id="lev"></svg></div>
</div>
<div class="card" style="margin-top:12px"><h2>Skill or luck?</h2><div class="d" id="auw">The last audit (python experiments/of24_audit.py)</div><ul id="au" style="margin:6px 0 0 18px;padding:0;font-size:13px;line-height:1.5"></ul></div>
<div class="card" style="margin-top:12px"><h2>Latest trades</h2><div class="wrap"><table id="tb"></table></div></div>
<div class="tip" id="tip"></div>
<script>
const NS="http://www.w3.org/2000/svg",css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const tip=document.getElementById("tip");let S=null;
const EXITS=[[8,4,30,0],[8,4,30,1],[12,6,60,0],[12,6,60,1],[16,8,60,0],[16,8,60,1],[24,8,120,0],[24,8,120,1],[32,12,0,0],[32,12,0,1],[48,16,0,0],[48,16,0,1]]
 .map(([t,s,h,tr])=>`${t}/${s}`+(h?` ${h}m`:" to close")+(tr?" trail":""));
function el(t,a,p){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);p&&p.appendChild(e);return e}
const f0=x=>x.toLocaleString(undefined,{maximumFractionDigits:0}),f1=x=>x.toLocaleString(undefined,{minimumFractionDigits:1,maximumFractionDigits:1}),
 f2=x=>x.toFixed(2),usd=x=>(x<0?"−$":"$")+f0(Math.abs(x)),sg=x=>(x>0?"+":x<0?"−":"")+Math.abs(x).toFixed(1);
function nice(lo,hi,n=4){if(lo===hi){lo-=1;hi+=1}const s=Math.pow(10,Math.floor(Math.log10((hi-lo)/n))),m=[1,2,2.5,5,10].find(m=>(hi-lo)/(m*s)<=n)*s;
 const a=Math.floor(lo/m)*m,b=Math.ceil(hi/m)*m,t=[];for(let v=a;v<=b+m/2;v+=m)t.push(+v.toFixed(10));return t}
// frame: x is a category index 0..n-1 (sessions or trades); returns scale helpers
function frame(svg,n,ys,o={}){svg.innerHTML="";const W=svg.clientWidth||600,H=svg.clientHeight||210,L=o.L||52,R=8,T=6,B=22;
 let lo=Math.min(...ys,o.zero?0:Infinity),hi=Math.max(...ys,o.zero?0:-Infinity);if(o.lo!=null)lo=o.lo;if(o.hi!=null)hi=o.hi;
 const ticks=nice(lo,hi);lo=ticks[0];hi=ticks[ticks.length-1];
 const X=i=>L+(n<=1?0.5:i/(n-1))*(W-L-R),Y=v=>T+(1-(v-lo)/(hi-lo||1))*(H-T-B);
 for(const t of ticks){el("line",{x1:L,x2:W-R,y1:Y(t),y2:Y(t),stroke:t===0&&o.zero?css("--axis"):css("--grid"),"stroke-width":1},svg);
  el("text",{x:L-6,y:Y(t)+4,"text-anchor":"end"},svg).textContent=(o.yf||f0)(t)}
 const lab=o.labels||[];if(lab.length){const k=Math.max(1,Math.ceil(lab.length/6));for(let i=0;i<lab.length;i+=k)
  el("text",{x:X(i),y:H-6,"text-anchor":i===0?"start":"middle"},svg).textContent=lab[i]}
 return{W,H,L,R,T,B,X,Y,lo,hi}}
function hover(svg,F,n,html,bw){svg.onmousemove=e=>{const r=svg.getBoundingClientRect(),x=e.clientX-r.left;
  let i=n<=1?0:Math.round((x-F.L)/(F.W-F.L-F.R)*(n-1));i=Math.max(0,Math.min(n-1,i));let g=svg.querySelector(".hv");g&&g.remove();
  g=el("g",{class:"hv"},svg);el("line",{x1:F.X(i),x2:F.X(i),y1:F.T,y2:F.H-F.B,stroke:css("--axis"),"stroke-width":1},g);
  tip.innerHTML=html(i);tip.style.display="block";const c=svg.closest(".card").getBoundingClientRect();
  let tx=r.left+F.X(i)+12+window.scrollX,ty=r.top+window.scrollY+8;if(tx+tip.offsetWidth>window.scrollX+document.documentElement.clientWidth-8)tx-=tip.offsetWidth+24;
  tip.style.left=tx+"px";tip.style.top=ty+"px"};svg.onmouseleave=()=>{tip.style.display="none";const g=svg.querySelector(".hv");g&&g.remove()}}
function line(svg,F,ys,col,w=2){if(!ys.length)return;let d="";ys.forEach((y,i)=>{if(y==null)return;d+=(d?"L":"M")+F.X(i).toFixed(1)+","+F.Y(y).toFixed(1)});
 el("path",{d,fill:"none",stroke:col,"stroke-width":w,"stroke-linejoin":"round","stroke-linecap":"round"},svg)}
function bars(svg,F,ys,colf,n){const bw=Math.max(1,Math.min(14,(F.W-F.L-F.R)/Math.max(n,1)-2));ys.forEach((y,i)=>{if(y==null)return;
 const y0=F.Y(Math.max(0,F.lo)),y1=F.Y(y),h=Math.abs(y1-y0);el("rect",{x:F.X(i)-bw/2,y:Math.min(y0,y1),width:bw,height:Math.max(h,0.5),rx:Math.min(2,bw/2),fill:colf(y,i)},svg)})}
function leg(id,items){document.getElementById(id).innerHTML=items.map(([n,c])=>`<span><i style="background:${c}"></i>${n}</span>`).join("")}
function render(){if(!S)return;const D=S.days,P=S.per_day,lab=D.map(d=>d.date.slice(2)),last=D[D.length-1];
 const done=D.length,pct=Math.min(1,done/S.total_days),cur=S.last?S.last.date:"–";
 const trades=S.n_trades,net=S.net_ticks,upd=S.weights.length?S.weights[S.weights.length-1].updates:(last?+last.updates:0);
 const tiles=[["Replayed session",cur,`${done} of ${S.total_days} closed`,pct],["Trades",f0(trades),trades?`${f2(net/trades)} ticks per trade`:""],
  ["Net",usd(net*12.5),`${sg(net)} ticks after costs`],["Learning steps",f0(upd),"actor-critic updates so far"],
  ["Last decision",S.last?(S.last.action?(S.last.dir>0?"Long ":"Short ")+EXITS[S.last.action-1]:"Skip"):"–",S.last?`p=${f2(S.last.p)} · value ${f2(S.last.value)} · ${S.last.levels} level${S.last.levels===1?"":"s"}`:""]];
 document.getElementById("tiles").innerHTML=tiles.map(([k,v,n,p])=>`<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="n">${n}</div>${p!=null?`<div class="bar"><i style="width:${(p*100).toFixed(1)}%"></i></div>`:""}</div>`).join("");
 document.getElementById("st").textContent=`Updated ${S.now.slice(11)} · log ${S.log}`+(S.events.C.length?" · "+S.events.C[S.events.C.length-1]:"");
 // cumulative $
 let s=document.getElementById("cum"),ys=D.map(d=>+d.cum_usd),F=frame(s,D.length,ys,{zero:1,labels:lab,yf:usd});line(s,F,ys,css("--s1"));
 hover(s,F,D.length,i=>`<b>${D[i].date}</b><br>Cumulative ${usd(+D[i].cum_usd)}<br>Session ${sg(+D[i].net_ticks)} ticks, ${D[i].trades} trades`);
 s=document.getElementById("daily");ys=D.map(d=>+d.net_ticks);F=frame(s,D.length,ys,{zero:1,labels:lab});bars(s,F,ys,y=>y>=0?css("--s1"):css("--neg"),D.length);
 hover(s,F,D.length,i=>`<b>${D[i].date}</b><br>${sg(+D[i].net_ticks)} ticks (${usd(+D[i].net_usd)})<br>${D[i].trades} trades`);
 s=document.getElementById("roll");const R=S.roll;ys=R.map(r=>r[1]);F=frame(s,R.length,ys,{zero:1,labels:R.map(r=>"#"+r[0]),yf:f1});line(s,F,ys,css("--s1"));
 hover(s,F,R.length,i=>`Trades to #${R[i][0]}<br>${f2(R[i][1])} ticks per trade`);
 const cols=["--s1","--s2","--s3","--s4","--s5","--s6"].map(css);leg("mixleg",S.groups.map((g,i)=>[g,cols[i]]));
 s=document.getElementById("mix");const plab=P.map(p=>p.date.slice(2));F=frame(s,P.length,[0,1],{lo:0,hi:1,labels:plab,yf:v=>Math.round(v*100)+"%"});
 const bw=Math.max(1,Math.min(14,(F.W-F.L-F.R)/Math.max(P.length,1)-2));P.forEach((p,i)=>{const tot=p.mix.reduce((a,b)=>a+b,0);if(!tot)return;let acc=0;
  p.mix.forEach((m,k)=>{if(!m)return;const y0=F.Y(acc/tot),y1=F.Y((acc+m)/tot);acc+=m;el("rect",{x:F.X(i)-bw/2,y:y1+(acc<tot?1:0),width:bw,height:Math.max(0.5,y0-y1-(acc<tot?1:0)),fill:cols[k]},s)})});
 hover(s,F,P.length,i=>{const t=P[i].mix.reduce((a,b)=>a+b,0);return `<b>${P[i].date}</b> · ${t} trades<br>`+S.groups.map((g,k)=>P[i].mix[k]?`${g}: ${P[i].mix[k]}`:"").filter(Boolean).join("<br>")});
 s=document.getElementById("rate");ys=P.map(p=>p.trade_rate);F=frame(s,P.length,ys,{lo:0,labels:plab,yf:v=>Math.round(v*100)+"%"});line(s,F,ys,css("--s1"));
 hover(s,F,P.length,i=>`<b>${P[i].date}</b><br>Took ${Math.round(P[i].trade_rate*100)}% of ${P[i].n} candidates`);
 s=document.getElementById("val");ys=P.map(p=>p.value);F=frame(s,P.length,ys,{zero:1,labels:plab,yf:f2});line(s,F,ys,css("--s1"));
 hover(s,F,P.length,i=>`<b>${P[i].date}</b><br>Mean value ${P[i].value.toFixed(3)}`);
 const Wh=S.weights,wc=[["Depth imbalance (3)",css("--s1"),"depth"],["MBO large orders (2)",css("--s2"),"mbo"],["Spread",css("--s3"),"spread"]];
 leg("wleg",wc.map(w=>[w[0],w[1]]));
 if(Wh.length)document.getElementById("wd").textContent=`Norm of the policy's first-layer weights on the book inputs (they start at zero). For scale, a bar feature's median is ${Wh[Wh.length-1].feat_median.toFixed(3)}. Recorded while this page's server runs.`;
 s=document.getElementById("w");if(Wh.length){const all=Wh.flatMap(h=>[h.depth,h.mbo,h.spread]);const fx=v=>v===0?"0":v<0.001?v.toExponential(1):v.toFixed(3);
  F=frame(s,Wh.length,all,{lo:0,labels:Wh.map(h=>"#"+h.updates),yf:fx,L:58});for(const w of wc)line(s,F,Wh.map(h=>h[w[2]]),w[1]);
  hover(s,F,Wh.length,i=>`After update #${Wh[i].updates}<br>`+wc.map(w=>`${w[0]}: ${Wh[i][w[2]].toFixed(4)}`).join("<br>")+`<br>Bar feature median: ${Wh[i].feat_median.toFixed(4)}`)}
 else{s.innerHTML="";el("text",{x:10,y:20},s).textContent="Waiting for the first weight save"}
 s=document.getElementById("lev");ys=P.map(p=>p.levels);F=frame(s,P.length,ys,{lo:0,hi:10,labels:plab,yf:f0});bars(s,F,ys,()=>css("--s1"),P.length);
 hover(s,F,P.length,i=>`<b>${P[i].date}</b><br>Mean ${f1(P[i].levels)} levels (max ${P[i].lev_max})`+(P[i].imb!=null?`<br>Mean |5-level imbalance| ${f2(P[i].imb)}`:""));
 const H=["date","entry_ct","side","exit_rule","entry_px","exit_px","exit","net_ticks","net_usd","cum_usd"],N=["Date","Entry (CT)","Side","Exit rule","Entry","Exit","Closed by","Net ticks","Net $","Cumulative $"];
 document.getElementById("tb").innerHTML="<tr>"+N.map(n=>`<th>${n}</th>`).join("")+"</tr>"+S.recent.map(t=>"<tr>"+H.map(h=>{let v=t[h]??"";
  if(h==="net_ticks")v=sg(+v);else if(h==="net_usd"||h==="cum_usd")v=usd(+v);return `<td>${v}</td>`}).join("")+"</tr>").join("");
 const A=S.audit;document.getElementById("auw").textContent=A?`The last audit, run ${A.when.replace("T"," ")} (python experiments/of24_audit.py)`:"No audit yet (python experiments/of24_audit.py)";
 document.getElementById("au").innerHTML=A?A.lines.map(l=>{const i=l.indexOf(":");return `<li><b>${esc(l.slice(0,i))}</b>${esc(l.slice(i))}</li>`}).join(""):""}
function esc(t){return t.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}
async function poll(){try{const r=await fetch("api/state",{cache:"no-store"});S=await r.json();render()}catch(e){document.getElementById("st").textContent="server not reachable"}}
poll();setInterval(poll,4000);addEventListener("resize",render);
</script></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *a, state=None, **k):
        self.state = state
        super().__init__(*a, **k)

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        elif path == "/api/state":
            body, ctype = json.dumps(self.state.snapshot()).encode(), "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--dir", default=r"C:\SierraChart\SierraChartInstance_2\Data\RagAgent")
    s.add_argument("--mode", choices=("replay", "live"), default="replay")
    s.add_argument("--days", type=int, default=228, help="sessions in the replay, for the progress bar")
    s.add_argument("--port", type=int, default=8775)
    s.add_argument("--audit", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build-avx512", "of24", "audit.txt"),
                   help="the report of experiments/of24_audit.py; its verdict is shown on the page")
    a = ap.parse_args(argv)
    st = State(a.dir, a.mode, a.days)
    st.audit = a.audit
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), functools.partial(Handler, state=st))
    print(f"OF24 live view: http://127.0.0.1:{a.port}/ ({a.mode}, {a.dir})", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
