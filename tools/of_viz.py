#!/usr/bin/env python3
"""of_viz.py - turn an of_trace CSV into ONE self-contained HTML viewer.

    python tools/of_viz.py trace.csv [out.html]

Standard library only; the page has inline JS and SVG, no CDN, no network.
Per episode: price (bid/ask band, mid, hidden fair value, the agent's fills),
position strips (agent, 30 s delta rule, fair-value oracle), the two signals
of the policy map, Q-values, cumulative net P&L; a shared crosshair shows
every column of the hovered step. Over all episodes: the policy map (action
by L1 imbalance x 30 s delta, optionally given the position held), action
shares by each signal, holding times and P&L per round trip.

Round trips: a run of steps with the same non-zero position. Its P&L is the
change in cumulative net P&L from the step before the entry to the step that
closes or flips it (a flip's new entry cost lands on the closing trip), so
the trips of an episode sum to its P&L exactly.
"""
import csv
import json
import os
import sys

FEATURES = ["f_delta5", "f_delta30", "f_delta120", "f_ofi5", "f_ofi30", "f_relvol", "f_imb_l1",
            "f_imb_l5", "f_ret5", "f_ret30", "f_ret120", "f_vwap", "f_absorb"]
STRATS = [("agent", "pos", "pnl"), ("rule", "rule_pos", "rule_pnl"), ("oracle", "oracle_pos", "oracle_pnl")]
IMB_BINS = 10    # raw L1 imbalance (bid-ask)/(bid+ask) over [-1, 1]
DELTA_BINS = 8   # normalized 30 s delta feature over [-4, 4] (the feature clip)
SCORE_LABEL = {"q": "Q-values", "prob": "log-probabilities", "score": "model scores", "vote": "votes"}


def num(s):
    if s == "":
        return None
    x = float(s)
    return int(x) if x.is_integer() and abs(x) < 1e15 else x


def read(path):
    meta, body = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for line in f:
            (meta if line.startswith("#") else body).append(line)
    rows = list(csv.DictReader(body))
    if not rows:
        sys.exit(f"{path}: no rows")
    cols = list(rows[0].keys())
    episodes = {}
    for r in rows:
        ep = episodes.setdefault(int(r["episode"]), {c: [] for c in cols})
        for c in cols:
            ep[c].append(num(r[c]))
    return " ".join(m.lstrip("# ").strip() for m in meta), cols, [episodes[k] for k in sorted(episodes)]


def runs(pos):
    """(first, last) row of each run of one non-zero position."""
    out, start = [], None
    for i, p in enumerate(pos + [0]):
        if start is not None and (i == len(pos) or p != pos[start]):
            out.append((start, i - 1))
            start = None
        if start is None and i < len(pos) and p != 0:
            start = i
    return out


def trips(pos, pnl):
    res = []
    for a, b in runs(pos):
        end = min(b + 1, len(pnl) - 1)  # the step that closes (or flips) the position
        # After a flip, step a (the flip) already belongs to the closing trip.
        before = 0.0 if a == 0 else pnl[a] if pos[a - 1] != 0 else pnl[a - 1]
        res.append({"hold": b - a + 1, "pnl": round(pnl[end] - before, 3), "side": pos[a]})
    return res


def bin_of(x, lo, hi, n):
    if x is None:
        return None
    k = int((x - lo) / (hi - lo) * n)
    return max(0, min(n - 1, k))


def aggregate(episodes):
    # policy[position held before the decision + 1][imbalance bin][delta bin] = [short, flat, long]
    policy = [[[[0, 0, 0] for _ in range(DELTA_BINS)] for _ in range(IMB_BINS)] for _ in range(3)]
    trip = {s: [] for s, _, _ in STRATS}
    for ep in episodes:
        for i, a in enumerate(ep["action"]):
            held = ep["pos"][i - 1] if i > 0 else 0
            bi = bin_of(ep["imb_l1"][i], -1.0, 1.0, IMB_BINS)
            bd = bin_of(ep["f_delta30"][i], -4.0, 4.0, DELTA_BINS)
            if bi is not None and bd is not None and a is not None:
                policy[max(-1, min(1, held)) + 1][bi][bd][a] += 1
        for s, pc, pl in STRATS:
            for t in trips(ep[pc], ep[pl]):
                t["ep"] = ep["episode"][0]
                trip[s].append(t)
    for s, pc, pl in STRATS:  # integrity: trips partition each episode's P&L
        total = sum(t["pnl"] for t in trip[s])
        final = sum(ep[pl][-1] for ep in episodes)
        if abs(total - final) > 0.05 * len(episodes) + 1e-6:
            print(f"warning: {s} round trips sum to {total:.2f}, episodes to {final:.2f}", file=sys.stderr)
    return {"policy": policy, "trips": trip, "imbBins": IMB_BINS, "deltaBins": DELTA_BINS}


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 2
    src = argv[1]
    dst = argv[2] if len(argv) > 2 else os.path.splitext(src)[0] + ".html"
    meta, cols, episodes = read(src)
    missing = [c for c in ["episode", "step", "bid", "ask", "mid", "fair_value", "imb_l1", "q_short", "q_flat", "q_long",
                           "action", "pos", "fill_px", "pnl", "rule_pos", "rule_pnl", "oracle_pos", "oracle_pnl"] + FEATURES
               if c not in cols]
    if missing:
        sys.exit(f"{src}: missing columns {missing}")
    kv = dict(t.split("=", 1) for t in meta.split() if "=" in t)  # of_zoo_trace: agent=, policy=, score_kind=
    agent = f'{kv["agent"]} ({kv.get("policy", "")})' if "agent" in kv else "RL agent"
    data = {"meta": meta, "source": os.path.basename(src), "features": FEATURES, "episodes": episodes, "agent": agent,
            "scores": SCORE_LABEL.get(kv.get("score_kind", "q"), "scores"), **aggregate(episodes)}
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = PAGE.replace("/*DATA*/null", payload)
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(page)
    n = sum(len(e["step"]) for e in episodes)
    print(f"wrote {dst}: {len(episodes)} episodes, {n} steps, {os.path.getsize(dst) / 1e6:.1f} MB")
    return 0


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RL scalping episode viewer</title>
<style>
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--agent:#2a78d6;--rule:#eb6834;--oracle:#1baf7a;--fv:#c98500;--long:#2a78d6;--short:#e34948;--neutral:#f0efec;--band:#e9e8e3;
--longbg:rgba(42,120,214,.10);--shortbg:rgba(227,73,72,.10)}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--axis:#383835;--agent:#3987e5;--rule:#d95926;--oracle:#199e70;--fv:#eda100;--long:#3987e5;--short:#e66767;--neutral:#383835;--band:#2c2c2a;
--longbg:rgba(57,135,229,.16);--shortbg:rgba(230,103,103,.16)}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:13px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:14px 20px 4px}h1{font-size:18px;margin:0 0 2px}h2{font-size:15px;margin:22px 0 6px}h3{font-size:13px;margin:10px 0 2px;font-weight:600}
.sub{color:var(--ink2)}.muted{color:var(--muted)}main{padding:0 20px 30px}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:8px 0}
select,button{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--axis);border-radius:6px;padding:3px 8px}
button{cursor:pointer}.wrap{display:flex;gap:16px;align-items:flex-start}.charts{flex:1;min-width:0}
.card{background:var(--surface);border:1px solid var(--grid);border-radius:8px;padding:8px 10px;margin-bottom:10px}
.side{width:270px;flex:none;position:sticky;top:8px;font-size:12px}.side table{width:100%;border-collapse:collapse}
.side td{padding:1px 4px;border-bottom:1px solid var(--grid)}.side td:last-child{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
svg{display:block;width:100%;overflow:visible}text{fill:var(--muted);font-size:11px}.lab{fill:var(--ink2)}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--ink2);font-size:12px;margin:2px 0 4px}
.key{display:inline-block;width:14px;height:0;border-top:2px solid;vertical-align:middle;margin-right:5px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:-1px;margin-right:5px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:10px}
table.sum{border-collapse:collapse;font-variant-numeric:tabular-nums;width:100%}
table.sum th,table.sum td{padding:3px 8px;border-bottom:1px solid var(--grid);text-align:right}table.sum th{color:var(--ink2);font-weight:600}
table.sum td:first-child,table.sum th:first-child{text-align:left}table.sum tr.ep{cursor:pointer}table.sum tr.ep:hover,table.sum tr.cur{background:var(--band)}
#tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--axis);border-radius:6px;padding:6px 8px;font-size:12px;display:none;z-index:9;box-shadow:0 2px 8px rgba(0,0,0,.15)}
#tip b{font-variant-numeric:tabular-nums}.note{color:var(--ink2);font-size:12px;max-width:900px}
</style></head><body>
<header><h1>RL scalping episode viewer</h1><div class="sub" id="meta"></div></header>
<main>
<div class="bar"><button id="prev" title="previous episode (left arrow)">&larr;</button><select id="pick" aria-label="episode"></select>
<button id="next" title="next episode (right arrow)">&rarr;</button><span class="muted" id="epinfo"></span></div>
<div class="wrap"><div class="charts" id="charts"></div>
<aside class="side card"><h3 id="sidehead">Hover a chart</h3><table id="side"></table></aside></div>
<h2>All episodes</h2>
<div class="grid2">
<div class="card"><h3>Policy map: agent's action by L1 imbalance and 30 s delta</h3>
<div class="bar">Position held before the decision: <select id="held"><option value="all">any</option><option value="1">flat</option><option value="2">long</option><option value="0">short</option></select></div>
<div class="legend"><span><span class="sw" style="background:var(--short)"></span>mostly short</span><span><span class="sw" style="background:var(--neutral);border:1px solid var(--axis)"></span>mostly flat / mixed</span><span><span class="sw" style="background:var(--long)"></span>mostly long</span><span class="muted">colour = share long &minus; share short; faint = under 10 decisions</span></div>
<div id="pmap"></div></div>
<div class="card"><h3>Action share by signal bucket</h3><div class="legend"><span><span class="sw" style="background:var(--short)"></span>short</span><span><span class="sw" style="background:var(--neutral);border:1px solid var(--axis)"></span>flat</span><span><span class="sw" style="background:var(--long)"></span>long</span></div>
<div id="share1"></div><div id="share2"></div></div>
<div class="card"><h3>Holding time per round trip</h3><div class="legend" id="lg1"></div><div id="hold"></div></div>
<div class="card"><h3>P&amp;L per round trip (ticks, net of fees)</h3><div class="legend" id="lg2"></div><div id="trip"></div></div>
</div>
<div class="card"><h3>Round trips</h3><table class="sum" id="tripsum"></table></div>
<div class="card"><h3>Episodes (click a row)</h3><table class="sum" id="epsum"></table></div>
<p class="note">Market columns are the agent's own path; the rule and the oracle trade on the same seeds in their own copies of the market (their orders move it slightly). The fair value is hidden from the agent and shown for reference only; the oracle trades on it (gap &gt; 2 ticks). Visualization seeds (960M+): never select or score anything on them.</p>
</main><div id="tip"></div>
<script>
"use strict";
const D=/*DATA*/null;
const NS="http://www.w3.org/2000/svg",STR=[["agent","pos","pnl"],["rule","rule_pos","rule_pnl"],["oracle","oracle_pos","oracle_pnl"]];
const NAMES={agent:D.agent||"RL agent",rule:"30 s delta rule",oracle:"fair-value oracle"},ACT=["short","flat","long"];
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function el(tag,attrs,parent){const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);if(parent)parent.appendChild(e);return e}
function h(tag,parent,text,cls){const e=document.createElement(tag);if(text!=null)e.textContent=text;if(cls)e.className=cls;if(parent)parent.appendChild(e);return e}
function lin(d0,d1,r0,r1){const k=(r1-r0)/((d1-d0)||1);return v=>r0+(v-d0)*k}
function nice(lo,hi,n){if(!(hi>lo)){lo-=1;hi+=1}const raw=(hi-lo)/n,p=Math.pow(10,Math.floor(Math.log10(raw))),m=raw/p,st=(m<1.5?1:m<3?2:m<7?5:10)*p;
 const out=[];for(let v=Math.ceil(lo/st)*st;v<=hi+1e-9;v+=st)out.push(+v.toFixed(10));return out}
const f=(x,d=2)=>x==null?"–":(+x).toFixed(d),sg=(x,d=2)=>x==null?"–":(x>0?"+":"")+(+x).toFixed(d);
function clock(s){if(s==null)return"–";s=Math.round(s)+9.5*3600;const hh=Math.floor(s/3600),mm=Math.floor(s/60)%60,ss=s%60;return`${hh}:${String(mm).padStart(2,"0")}:${String(ss).padStart(2,"0")}`}
const tip=document.getElementById("tip");
function showTip(ev,rows){tip.replaceChildren();for(const [k,v,c] of rows){const d=h("div",tip);if(c){const s=h("span",d,null,"key");s.style.borderColor=c}h("b",d,v);d.append(" "+k)}
 tip.style.display="block";const x=Math.min(ev.clientX+14,innerWidth-tip.offsetWidth-8);tip.style.left=x+"px";tip.style.top=(ev.clientY+14)+"px"}
function hideTip(){tip.style.display="none"}

// ---------- per-episode charts ----------
const M={l:52,r:12,t:6,b:18};let cur=0,cross=[],hoverStep=null,lastPanel=null;
function panel(host,title,height,lo,hi,legend){
 const card=h("div",host,null,"card");h("h3",card,title);if(legend){const lg=h("div",card,null,"legend");for(const [n,c,dash] of legend){const s=h("span",lg);const k=h("span",s,null,"key");k.style.borderColor=c;if(dash)k.style.borderTopStyle="dashed";s.append(n)}}
 const W=Math.max(360,card.clientWidth-20),svg=el("svg",{viewBox:`0 0 ${W} ${height}`,height},card);
 const ep=D.episodes[cur],n=ep.step.length,x=lin(0,n-1,M.l,W-M.r),y=lin(lo,hi,height-M.b,M.t);
 for(const v of nice(lo,hi,4)){el("line",{x1:M.l,x2:W-M.r,y1:y(v),y2:y(v),stroke:css("--grid")},svg);const t=el("text",{x:M.l-6,y:y(v)+4,"text-anchor":"end"},svg);t.textContent=+v.toFixed(3)}
 el("line",{x1:M.l,x2:W-M.r,y1:height-M.b,y2:height-M.b,stroke:css("--axis")},svg);
 for(const s of nice(0,n-1,6)){if(s>n-1)continue;const t=el("text",{x:x(s),y:height-4,"text-anchor":"middle"},svg);t.textContent=clock(ep.tod_s[s])}
 const g=el("g",{},svg),cl=el("line",{y1:M.t,y2:height-M.b,stroke:css("--ink2"),"stroke-width":1,visibility:"hidden"},svg);cross.push({cl,x});
 const hit=el("rect",{x:M.l,y:0,width:W-M.l-M.r,height,fill:"transparent"},svg);
 hit.addEventListener("pointermove",ev=>{const r=svg.getBoundingClientRect(),px=(ev.clientX-r.left)*W/r.width;const i=Math.max(0,Math.min(n-1,Math.round((px-M.l)/(W-M.l-M.r)*(n-1))));setHover(i,ev)});
 hit.addEventListener("pointerleave",()=>{hideTip()});
 lastPanel={svg:g,x,y,W,H:height,n};return lastPanel}
function path(p,ys,attrs){let d="",pen=false;for(let i=0;i<ys.length;i++){const v=ys[i];if(v==null){pen=false;continue}d+=(pen?"L":"M")+p.x(i).toFixed(1)+","+p.y(v).toFixed(1);pen=true}
 return el("path",Object.assign({d,fill:"none","stroke-width":2,"stroke-linejoin":"round"},attrs),p.svg)}
function ext(arrs){let lo=Infinity,hi=-Infinity;for(const a of arrs)for(const v of a)if(v!=null){if(v<lo)lo=v;if(v>hi)hi=v}return[lo,hi]}
function posBands(p,pos,y0,y1){let s=0;for(let i=1;i<=pos.length;i++){if(i==pos.length||pos[i]!=pos[s]){if(pos[s]!=0){const xa=p.x(s),xb=i<pos.length?p.x(i):p.x(i-1);
 el("rect",{x:xa,y:Math.min(y0,y1),width:Math.max(1,xb-xa),height:Math.abs(y1-y0),fill:pos[s]>0?css("--longbg"):css("--shortbg")},p.svg)}s=i}}}
function render(){
 const host=document.getElementById("charts");host.replaceChildren();cross=[];const ep=D.episodes[cur],n=ep.step.length,m0=ep.mid[0];
 const rel=a=>a.map(v=>v==null?null:v-m0);
 // price
 const [plo,phi]=ext([rel(ep.bid),rel(ep.ask),rel(ep.fair_value)]);
 const pp=panel(host,"Price (ticks from the episode's first mid) with the agent's position shaded",230,plo-0.5,phi+0.5,[["mid",css("--ink")],...(ep.bid[0]!=null?[["bid / ask",css("--axis")]]:[]),["hidden fair value",css("--fv"),1],["buy ▲ / sell ▼ fills",css("--ink2")]]);
 posBands(pp,ep.pos,pp.y(phi+0.5),pp.y(plo-0.5));
 if(ep.bid[0]!=null){let d="";for(let i=0;i<n;i++)d+=(i?"L":"M")+pp.x(i).toFixed(1)+","+pp.y(ep.ask[i]-m0).toFixed(1);for(let i=n-1;i>=0;i--)d+="L"+pp.x(i).toFixed(1)+","+pp.y(ep.bid[i]-m0).toFixed(1);
 el("path",{d:d+"Z",fill:css("--band"),stroke:"none"},pp.svg)}  // of_zoo_trace has no quotes
 path(pp,rel(ep.mid),{stroke:css("--ink"),"stroke-width":1.5});path(pp,rel(ep.fair_value),{stroke:css("--fv"),"stroke-dasharray":"5 3"});
 for(let i=0;i<n;i++){const dp=ep.pos[i]-(i?ep.pos[i-1]:0);if(!dp&&!(i==n-1&&ep.pos[i-1]))continue;
  const buy=dp>0||(dp==0&&ep.pos[i-1]<0),px=ep.fill_px[i]!=null?ep.fill_px[i]:ep.mid[i],X=pp.x(i),Y=pp.y(px-m0);
  el("path",{d:buy?`M${X-5},${Y+7}L${X+5},${Y+7}L${X},${Y-1}Z`:`M${X-5},${Y-7}L${X+5},${Y-7}L${X},${Y+1}Z`,fill:buy?css("--long"):css("--short"),stroke:css("--surface"),"stroke-width":1.5},pp.svg)}
 // positions
 const sp=panel(host,"Position (after each decision): long ▮ blue, short ▮ red",84,0,3,null);
 sp.svg.parentNode.querySelectorAll("text").forEach((t,i)=>{if(t.getAttribute("text-anchor")=="end")t.remove()});
 STR.forEach(([s,pc],k)=>{const y0=sp.y(3-k)+3,y1=sp.y(2-k)-3,a=ep[pc];let st=0;
  for(let i=1;i<=n;i++){if(i==n||a[i]!=a[st]){if(a[st]!=0)el("rect",{x:sp.x(st),y:y0,width:Math.max(1,(i<n?sp.x(i):sp.x(n-1)+1)-sp.x(st)),height:y1-y0,fill:a[st]>0?css("--long"):css("--short"),rx:1},sp.svg);st=i}}
  const t=el("text",{x:M.l-6,y:(y0+y1)/2+4,"text-anchor":"end",class:"lab"},sp.svg);t.textContent=s});
 // signals
 const gp=panel(host,"Signals the policy map uses",120,-1,1,[["L1 imbalance (raw, −1..1)",css("--agent")],["30 s delta (normalized ÷ 4)",css("--rule")]]);
 el("line",{x1:M.l,x2:gp.W-M.r,y1:gp.y(0),y2:gp.y(0),stroke:css("--axis")},gp.svg);
 path(gp,ep.imb_l1,{stroke:css("--agent"),"stroke-width":1.2});path(gp,ep.f_delta30.map(v=>v==null?null:v/4),{stroke:css("--rule"),"stroke-width":1.5});
 // Q
 const relq=document.getElementById("relq")?document.getElementById("relq").checked:true;
 const qs=[ep.q_short,ep.q_flat,ep.q_long].map(a=>relq?a.map((v,i)=>v-ep.q_flat[i]):a);const [qlo,qhi]=ext(qs);
 const qp=panel(host,relq?`${D.scores||"Q-values"} relative to flat (advantage of short / long over flat)`:(D.scores||"Q-values"),150,qlo,qhi,[["short",css("--short")],["flat",css("--muted")],["long",css("--long")]]);
 const qc=[css("--short"),css("--muted"),css("--long")];qs.forEach((a,k)=>path(qp,a,{stroke:qc[k],"stroke-width":k==1?1:1.5}));
 const cb=h("label",qp.svg.closest(".card").querySelector("h3"));cb.style.cssText="font-weight:400;margin-left:12px;color:var(--ink2)";
 const inp=h("input",cb);inp.type="checkbox";inp.id="relq";inp.checked=relq;cb.append(" relative to flat");inp.onchange=render;
 // P&L
 const [lo,hi]=ext([ep.pnl,ep.rule_pnl,ep.oracle_pnl,[0]]);
 const lp=panel(host,"Cumulative net P&L (ticks)",170,lo,hi,STR.map(([s])=>[NAMES[s],css("--"+s)]));
 el("line",{x1:M.l,x2:lp.W-M.r,y1:lp.y(0),y2:lp.y(0),stroke:css("--axis")},lp.svg);
 STR.forEach(([s,,pl])=>path(lp,ep[pl],{stroke:css("--"+s)}));
 STR.forEach(([s,,pl])=>{const v=ep[pl][n-1],t=el("text",{x:lp.W-M.r+2,y:lp.y(v)+4,class:"lab"},lp.svg);t.textContent=sg(v,1)});
 if(hoverStep!=null)setHover(Math.min(hoverStep,n-1));
 document.querySelectorAll("#epsum tr.ep").forEach(r=>r.classList.toggle("cur",+r.dataset.i==cur));
 const e=D.episodes[cur];document.getElementById("epinfo").textContent=`seed ${e.seed[0]} · ${clock(e.tod_s[0])}–${clock(e.tod_s[n-1])} · agent ${sg(e.pnl[n-1])}, rule ${sg(e.rule_pnl[n-1])}, oracle ${sg(e.oracle_pnl[n-1])} ticks · ${e.orders[n-1]} orders`}
function setHover(i,ev){hoverStep=i;const ep=D.episodes[cur];for(const c of cross){const X=c.x(i);c.cl.setAttribute("x1",X);c.cl.setAttribute("x2",X);c.cl.setAttribute("visibility","visible")}
 document.getElementById("sidehead").textContent=`Step ${i} · ${clock(ep.tod_s[i])}`;const t=document.getElementById("side");t.replaceChildren();
 const row=(k,v)=>{const r=h("tr",t);h("td",r,k);h("td",r,v)};const m0=ep.mid[0];
 row("bid / ask",ep.bid[i]==null?"–":`${ep.bid[i]-m0} / ${ep.ask[i]-m0}`);row("fair value − mid",sg(ep.fair_value[i]-ep.mid[i]));row("L1 sizes bid / ask",ep.bid_sz1[i]==null?"–":`${ep.bid_sz1[i]} / ${ep.ask_sz1[i]}`);row("L1 imbalance",sg(ep.imb_l1[i],3));
 for(const k of D.features)row(k.replace("f_","feat "),sg(ep[k][i],2));row("open P&L",sg(ep.open_pnl[i]));
 row("Q short / flat / long",`${f(ep.q_short[i])} / ${f(ep.q_flat[i])} / ${f(ep.q_long[i])}`);row("action",ACT[ep.action[i]]);row("position after",ep.pos[i]);
 row("fill price",ep.fill_px[i]==null?"–":sg(ep.fill_px[i]-m0));row("orders / fees",`${ep.orders[i]} / ${f(ep.fees[i])}`);
 row("P&L agent / rule / oracle",`${sg(ep.pnl[i],1)} / ${sg(ep.rule_pnl[i],1)} / ${sg(ep.oracle_pnl[i],1)}`);row("rule / oracle pos",`${ep.rule_pos[i]} / ${ep.oracle_pos[i]}`);
 if(ev)showTip(ev,[["action",ACT[ep.action[i]]],["Q long − flat",sg(ep.q_long[i]-ep.q_flat[i])],["Q short − flat",sg(ep.q_short[i]-ep.q_flat[i])],["agent P&L",sg(ep.pnl[i],1),css("--agent")]])}

// ---------- aggregates ----------
function mix(a,b,t){const p=s=>[1,3,5].map(k=>parseInt(s.slice(k,k+2),16)),A=p(a),B=p(b);return"rgb("+A.map((v,k)=>Math.round(v+(B[k]-v)*t)).join(",")+")"}
function divColor(v){const mid=css("--neutral");return v>=0?mix(mid,css("--long"),Math.min(1,v)):mix(mid,css("--short"),Math.min(1,-v))}
function policyCounts(held){const P=D.policy,out=[];for(let i=0;i<D.imbBins;i++){out.push([]);for(let j=0;j<D.deltaBins;j++){const c=[0,0,0];
 for(let k=0;k<3;k++)if(held=="all"||+held==k)for(let a=0;a<3;a++)c[a]+=P[k][i][j][a];out[i].push(c)}}return out}
const imbLabel=i=>`${(-1+2*i/D.imbBins).toFixed(1)}…${(-1+2*(i+1)/D.imbBins).toFixed(1)}`,dLabel=j=>`${-4+8*j/D.deltaBins}…${-4+8*(j+1)/D.deltaBins}`;
function renderPolicy(){const held=document.getElementById("held").value,C=policyCounts(held),host=document.getElementById("pmap");host.replaceChildren();
 const W=Math.max(360,host.clientWidth),H=300,ml=70,mb=40,cw=(W-ml-10)/D.deltaBins,ch=(H-mb-6)/D.imbBins,svg=el("svg",{viewBox:`0 0 ${W} ${H}`,height:H},host);
 for(let i=0;i<D.imbBins;i++)for(let j=0;j<D.deltaBins;j++){const c=C[i][j],n=c[0]+c[1]+c[2],x=ml+j*cw,y=6+(D.imbBins-1-i)*ch;
  const r=el("rect",{x:x+1,y:y+1,width:cw-2,height:ch-2,rx:3,fill:n?divColor((c[2]-c[0])/n):css("--surface"),stroke:n?"none":css("--grid"),opacity:n&&n<10?0.35:1},svg);
  r.addEventListener("pointermove",ev=>{r.setAttribute("stroke",css("--ink"));showTip(ev,[["decisions",String(n)],["short",n?Math.round(100*c[0]/n)+"%":"–",css("--short")],["flat",n?Math.round(100*c[1]/n)+"%":"–",css("--muted")],["long",n?Math.round(100*c[2]/n)+"%":"–",css("--long")],["imbalance",imbLabel(i)],["30 s delta (norm.)",dLabel(j)]])});
  r.addEventListener("pointerleave",()=>{r.setAttribute("stroke",n?"none":css("--grid"));hideTip()})}
 for(let i=0;i<=D.imbBins;i+=2){const t=el("text",{x:ml-6,y:6+(D.imbBins-i)*ch+4,"text-anchor":"end"},svg);t.textContent=(-1+2*i/D.imbBins).toFixed(1)}
 for(let j=0;j<=D.deltaBins;j++){const t=el("text",{x:ml+j*cw,y:H-mb+14,"text-anchor":"middle"},svg);t.textContent=-4+8*j/D.deltaBins}
 let t=el("text",{x:ml+(W-ml)/2,y:H-6,"text-anchor":"middle",class:"lab"},svg);t.textContent="30 s delta (normalized feature) → buyers";
 t=el("text",{x:12,y:6+(H-mb)/2,"text-anchor":"middle",class:"lab",transform:`rotate(-90 12 ${6+(H-mb)/2})`},svg);t.textContent="L1 imbalance (bid heavy ↑)";
 renderShare("share1","by L1 imbalance",D.imbBins,i=>C[i].reduce((s,c)=>s.map((v,k)=>v+c[k]),[0,0,0]),imbLabel);
 renderShare("share2","by 30 s delta (normalized)",D.deltaBins,j=>C.reduce((s,row)=>s.map((v,k)=>v+row[j][k]),[0,0,0]),dLabel)}
function renderShare(id,title,nb,get,label){const host=document.getElementById(id);host.replaceChildren();h("div",host,title,"muted");
 const W=Math.max(360,host.clientWidth),H=120,ml=34,mb=18,bw=(W-ml-6)/nb,svg=el("svg",{viewBox:`0 0 ${W} ${H}`,height:H},host),y=lin(0,1,H-mb,4);
 for(const v of[0,.5,1]){el("line",{x1:ml,x2:W-6,y1:y(v),y2:y(v),stroke:css("--grid")},svg);const t=el("text",{x:ml-4,y:y(v)+4,"text-anchor":"end"},svg);t.textContent=v*100+"%"}
 const col=[css("--short"),css("--neutral"),css("--long")];
 for(let b=0;b<nb;b++){const c=get(b),n=c[0]+c[1]+c[2];let acc=0;const x=ml+b*bw;
  if(n)for(const a of[0,1,2]){const s=c[a]/n;if(s>0)el("rect",{x:x+1,y:y(acc+s)+1,width:bw-2,height:Math.max(0,y(acc)-y(acc+s)-2),fill:col[a],rx:2},svg);acc+=s}
  const hit=el("rect",{x,y:0,width:bw,height:H-mb,fill:"transparent"},svg);
  hit.addEventListener("pointermove",ev=>showTip(ev,[["bucket",label(b)],["decisions",String(n)],...ACT.map((a,k)=>[a,n?Math.round(100*c[k]/n)+"%":"–",k==1?css("--muted"):col[k]])]));hit.addEventListener("pointerleave",hideTip);
  if(nb<=10){const t=el("text",{x:x+bw/2,y:H-4,"text-anchor":"middle"},svg);t.textContent=label(b).split("…")[0]}}}
function grouped(id,bins,valueOf,fmtBin){const host=document.getElementById(id);host.replaceChildren();
 const counts=STR.map(([s])=>{const c=new Array(bins.length).fill(0);for(const t of D.trips[s]){const b=valueOf(t);c[b]++}return c});
 const W=Math.max(360,host.clientWidth),H=190,ml=40,mb=30,gw=(W-ml-6)/bins.length,bw=Math.max(2,(gw-6)/3),mx=Math.max(1,...counts.flat()),y=lin(0,mx,H-mb,6);
 const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,height:H},host);
 for(const v of nice(0,mx,4)){el("line",{x1:ml,x2:W-6,y1:y(v),y2:y(v),stroke:css("--grid")},svg);const t=el("text",{x:ml-4,y:y(v)+4,"text-anchor":"end"},svg);t.textContent=v}
 bins.forEach((b,i)=>{const x=ml+i*gw+3;STR.forEach(([s],k)=>{const v=counts[k][i];if(v)el("rect",{x:x+k*bw+1,y:y(v),width:bw-2,height:y(0)-y(v),fill:css("--"+s),rx:2},svg)});
  const hit=el("rect",{x:ml+i*gw,y:0,width:gw,height:H-mb,fill:"transparent"},svg);
  hit.addEventListener("pointermove",ev=>showTip(ev,[["bin",fmtBin(b)],...STR.map(([s],k)=>[NAMES[s],String(counts[k][i]),css("--"+s)])]));hit.addEventListener("pointerleave",hideTip);
  const t=el("text",{x:ml+(i+.5)*gw,y:H-mb+14,"text-anchor":"middle"},svg);t.textContent=fmtBin(b)});
 el("line",{x1:ml,x2:W-6,y1:y(0),y2:y(0),stroke:css("--axis")},svg)}
function legend(id){const lg=document.getElementById(id);lg.replaceChildren();for(const [s] of STR){const sp=h("span",lg);const k=h("span",sp,null,"sw");k.style.background=css("--"+s);sp.append(NAMES[s])}}
function renderAgg(){renderPolicy();legend("lg1");legend("lg2");
 const hb=[[1,1],[2,2],[3,4],[5,8],[9,16],[17,32],[33,64],[65,128],[129,1e9]];
 grouped("hold",hb,t=>hb.findIndex(([a,b])=>t.hold>=a&&t.hold<=b),([a,b])=>b>1e8?`${a}s+`:a==b?`${a}s`:`${a}-${b}s`);
 const pb=[];for(let v=-8;v<=8;v++)pb.push(v);
 grouped("trip",pb,t=>Math.max(0,Math.min(pb.length-1,Math.round(t.pnl)+8)),v=>v==-8?"≤−8":v==8?"≥8":String(v));
 const tb=document.getElementById("tripsum");tb.replaceChildren();let r=h("tr",tb);for(const c of["strategy","round trips","per episode","mean P&L/trip","win %","mean hold (s)","long share","P&L/episode"])h("th",r,c);
 const ne=D.episodes.length;for(const [s] of STR){const T=D.trips[s],n=T.length,sum=T.reduce((a,t)=>a+t.pnl,0);r=h("tr",tb);
  h("td",r,NAMES[s]);h("td",r,String(n));h("td",r,f(n/ne,1));h("td",r,n?sg(sum/n):"–");h("td",r,n?f(100*T.filter(t=>t.pnl>0).length/n,0):"–");
  h("td",r,n?f(T.reduce((a,t)=>a+t.hold,0)/n,1):"–");h("td",r,n?f(100*T.filter(t=>t.side>0).length/n,0)+"%":"–");h("td",r,sg(sum/ne))}}
function renderEpTable(){const tb=document.getElementById("epsum");tb.replaceChildren();let r=h("tr",tb);
 for(const c of["episode","seed","start","agent","rule","oracle","orders","agent time long / short"])h("th",r,c);
 D.episodes.forEach((e,i)=>{const n=e.step.length;r=h("tr",tb,null,"ep");r.dataset.i=i;r.onclick=()=>pick(i);
  const lg=e.pos.filter(v=>v>0).length,sh=e.pos.filter(v=>v<0).length;
  for(const v of[String(e.episode[0]),String(e.seed[0]),clock(e.tod_s[0]),sg(e.pnl[n-1]),sg(e.rule_pnl[n-1]),sg(e.oracle_pnl[n-1]),String(e.orders[n-1]),`${Math.round(100*lg/n)}% / ${Math.round(100*sh/n)}%`])h("td",r,v)})}
function pick(i){cur=(i+D.episodes.length)%D.episodes.length;document.getElementById("pick").value=cur;render()}
// ---------- boot ----------
document.getElementById("meta").textContent=`${D.source} · ${D.meta} · ${D.episodes.length} episodes · prices in ticks (1 tick = 0.25 ES point = $12.50); times are session clock from the 09:30 open`;
const sel=document.getElementById("pick");D.episodes.forEach((e,i)=>{const o=h("option",sel,`episode ${e.episode[0]}  (agent ${sg(e.pnl[e.pnl.length-1],1)})`);o.value=i});
sel.onchange=()=>pick(+sel.value);document.getElementById("prev").onclick=()=>pick(cur-1);document.getElementById("next").onclick=()=>pick(cur+1);
document.getElementById("held").onchange=renderPolicy;
addEventListener("keydown",e=>{if(e.target.tagName=="SELECT"||e.target.tagName=="INPUT")return;if(e.key=="ArrowLeft")pick(cur-1);if(e.key=="ArrowRight")pick(cur+1)});
let rt;addEventListener("resize",()=>{clearTimeout(rt);rt=setTimeout(()=>{render();renderAgg()},150)});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",()=>{render();renderAgg()});
renderEpTable();render();renderAgg();
</script></body></html>
"""

if __name__ == "__main__":
    sys.exit(main(sys.argv))
