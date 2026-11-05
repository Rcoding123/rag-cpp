#!/usr/bin/env python3
"""of_train_live.py - live training animation for of_train_live checkpoints.

    python tools/of_train_live.py serve DIR [--port 8770] [--file live.jsonl]
        writes DIR/index.html and serves DIR on http://127.0.0.1:PORT/ (local only);
        the page polls DIR/live.jsonl every 2 s, so start of_train_live with
        --out DIR/live.jsonl before or after, and watch training as it happens.
    python tools/of_train_live.py bake live.jsonl out.html
        one self-contained HTML file (data inline, no network) that replays a
        finished (or partial) run as an animation.
    python tools/of_train_live.py page out.html
        just the polling page (for any static server next to live.jsonl).

Standard library only. The page shows: the learning curve (training returns,
the Trainer's validation scores) and exploration; then, at the selected
checkpoint, the learner's greedy replay of a fixed probe episode (price,
hidden fair value, fills, position vs the 30 s delta rule, P&L with a trail of
earlier checkpoints, Q-values) and its policy map on fixed states given
flat / long / short; and probe results over training. Play animates through
the checkpoints; "follow live" jumps to each new one while training runs.
Probe episodes come from the visualization seed block (960M+): look, never select.
"""
import functools
import http.server
import json
import os
import sys


def page(baked_text=None):
    payload = "null" if baked_text is None else json.dumps(baked_text).replace("</", "<\\/")
    return PAGE.replace("/*BAKED*/null", payload)


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def bake(src, dst):
    with open(src, encoding="utf-8") as f:
        text = f.read()
    lines = [l for l in text.splitlines() if l.strip()]
    good = []
    for i, l in enumerate(lines):
        try:
            json.loads(l)
            good.append(l)
        except json.JSONDecodeError:
            if i != len(lines) - 1:  # only a half-written last line is expected
                sys.exit(f"{src}: line {i + 1} is not JSON")
    if not good or json.loads(good[0]).get("type") != "header":
        sys.exit(f"{src}: no of_train_live header")
    write(dst, page("\n".join(good)))
    n = sum(1 for l in good if json.loads(l).get("type") == "ckpt")
    print(f"wrote {dst}: {n} checkpoints, {os.path.getsize(dst) / 1e6:.1f} MB")


class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):  # quiet: the page polls every 2 s
        pass


def serve(directory, port, file):
    os.makedirs(directory, exist_ok=True)
    write(os.path.join(directory, "index.html"), page())
    handler = functools.partial(NoCache, directory=directory)
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"serving {os.path.abspath(directory)}: open http://127.0.0.1:{port}/index.html?src={file}  (Ctrl+C to stop)", flush=True)
        httpd.serve_forever()


def main(argv):
    if len(argv) < 3 or argv[1] not in ("serve", "bake", "page"):
        print(__doc__)
        return 2
    if argv[1] == "bake":
        if len(argv) < 4:
            print(__doc__)
            return 2
        bake(argv[2], argv[3])
    elif argv[1] == "page":
        write(argv[2], page())
        print(f"wrote {argv[2]}")
    else:
        opts = dict(zip(argv[3::2], argv[4::2]))
        serve(argv[2], int(opts.get("--port", 8770)), opts.get("--file", "live.jsonl"))
    return 0


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RL scalping: live training</title>
<style>
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--agent:#2a78d6;--rule:#eb6834;--champ:#1baf7a;--fv:#c98500;--long:#2a78d6;--short:#e34948;--neutral:#f0efec;--band:#e9e8e3;
--longbg:rgba(42,120,214,.10);--shortbg:rgba(227,73,72,.10);--good:#0ca30c}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--axis:#383835;--agent:#3987e5;--rule:#d95926;--champ:#199e70;--fv:#eda100;--long:#3987e5;--short:#e66767;--neutral:#383835;--band:#2c2c2a;
--longbg:rgba(57,135,229,.16);--shortbg:rgba(230,103,103,.16)}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:13px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:14px 20px 4px;display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}h1{font-size:18px;margin:0}
h3{font-size:13px;margin:0 0 2px;font-weight:600}.sub{color:var(--ink2)}.muted{color:var(--muted)}main{padding:0 20px 30px}
.pill{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--axis);border-radius:99px;padding:1px 10px;font-size:12px;color:var(--ink2)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--muted)}.live .dot{background:var(--good);animation:pulse 1.2s infinite}
@keyframes pulse{50%{opacity:.25}}
.controls{position:sticky;top:0;z-index:5;background:var(--page);padding:8px 0;display:flex;gap:10px;align-items:center;flex-wrap:wrap;border-bottom:1px solid var(--grid);margin-bottom:10px}
select,button{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--axis);border-radius:6px;padding:3px 10px}button{cursor:pointer;min-width:74px}
input[type=range]{flex:1;min-width:200px}.prog{height:4px;background:var(--grid);border-radius:2px;width:180px;overflow:hidden}.prog>div{height:100%;background:var(--agent)}
.card{background:var(--surface);border:1px solid var(--grid);border-radius:8px;padding:8px 10px;margin-bottom:10px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:10px}.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
svg{display:block;width:100%;overflow:visible}text{fill:var(--muted);font-size:11px}.lab{fill:var(--ink2)}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--ink2);font-size:12px;margin:2px 0 4px}
.key{display:inline-block;width:14px;height:0;border-top:2px solid;vertical-align:middle;margin-right:5px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:-1px;margin-right:5px}
.stat{font-variant-numeric:tabular-nums}.stat b{font-size:15px;color:var(--ink)}
#tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--axis);border-radius:6px;padding:6px 8px;font-size:12px;display:none;z-index:9;box-shadow:0 2px 8px rgba(0,0,0,.15)}
#tip b{font-variant-numeric:tabular-nums}.note{color:var(--ink2);font-size:12px;max-width:960px}#msg{padding:40px 20px;color:var(--ink2)}
</style></head><body>
<header><h1>RL scalping: live training</h1><span class="pill" id="status"><span class="dot"></span><span id="statustext">waiting for data</span></span>
<span class="sub" id="meta"></span></header>
<main>
<div class="controls"><button id="play">▶ Play</button>
<input type="range" id="slider" min="0" max="0" value="0" aria-label="checkpoint">
<span class="stat" id="where"></span>
<label class="muted">speed <select id="speed"><option value="900">0.5×</option><option value="450" selected>1×</option><option value="220">2×</option><option value="100">4×</option></select></label>
<label class="muted"><input type="checkbox" id="follow" checked> follow live</label>
<div class="prog" title="training progress"><div id="progbar" style="width:0"></div></div><span class="muted stat" id="eta"></span></div>
<div id="msg">Waiting for <code id="srcname"></code>… start <code>of_train_live --out &lt;this folder&gt;/live.jsonl</code>.</div>
<div id="app" style="display:none">
<div class="grid2">
 <div class="card"><h3>Learning curve</h3><div class="legend" id="lg-learn"></div><div id="learn"></div><div id="eps"></div></div>
 <div class="card"><h3>Probe episodes over training (greedy learner, visualization seeds)</h3><div class="legend" id="lg-probe"></div><div id="probe"></div><div id="mkt"></div></div>
</div>
<div class="card"><h3 id="replay-title">Probe episode replay</h3><div class="legend" id="lg-price"></div><div id="price"></div><div id="strip"></div>
 <div class="legend" id="lg-pnl"></div><div id="pnl"></div><div class="legend" id="lg-q"></div><div id="q"></div></div>
<div class="card"><h3>Policy map at this checkpoint: the learner's action on fixed states, by L1 imbalance × 30 s delta</h3>
 <div class="legend"><span><span class="sw" style="background:var(--short)"></span>short</span><span><span class="sw" style="background:var(--neutral);border:1px solid var(--axis)"></span>flat / mixed</span><span><span class="sw" style="background:var(--long)"></span>long</span><span class="muted">colour = share long − share short among the cell's states; faint = under 10 states</span></div>
 <div class="grid3"><div><div class="muted">when flat (enter?)</div><div id="pm1"></div></div><div><div class="muted">when long (hold or exit?)</div><div id="pm2"></div></div><div><div class="muted">when short (hold or exit?)</div><div id="pm0"></div></div></div></div>
<p class="note">The learner shown is the current greedy policy (averaged weights), not the champion; the validation curve is the Trainer's (seeds 700M+), the probes are fixed visualization episodes (960M+): our orders move the market a little, so each checkpoint's probe path can differ slightly. The fair value is hidden from the agent.</p>
</div><div id="tip"></div>
<script>
"use strict";
const BAKED=/*BAKED*/null;
const NS="http://www.w3.org/2000/svg",ACT=["short","flat","long"];
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function el(t,a,p){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function h(t,p,txt,cls){const e=document.createElement(t);if(txt!=null)e.textContent=txt;if(cls)e.className=cls;if(p)p.appendChild(e);return e}
function lin(d0,d1,r0,r1){const k=(r1-r0)/((d1-d0)||1);return v=>r0+(v-d0)*k}
function nice(lo,hi,n){if(!(hi>lo)){lo-=1;hi+=1}const raw=(hi-lo)/n,p=Math.pow(10,Math.floor(Math.log10(raw))),m=raw/p,st=(m<1.5?1:m<3?2:m<7?5:10)*p;
 const o=[];for(let v=Math.ceil(lo/st)*st;v<=hi+1e-9;v+=st)o.push(+v.toFixed(10));return o}
const f=(x,d=2)=>x==null||!isFinite(x)?"–":(+x).toFixed(d),sg=(x,d=2)=>x==null||!isFinite(x)?"–":(x>0?"+":"")+(+x).toFixed(d);
const mean=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:NaN;
function clock(s){if(s==null)return"–";s=Math.round(s)+9.5*3600;return`${Math.floor(s/3600)}:${String(Math.floor(s/60)%60).padStart(2,"0")}:${String(s%60).padStart(2,"0")}`}
function ext(arrs){let lo=Infinity,hi=-Infinity;for(const a of arrs)for(const v of a)if(v!=null&&isFinite(v)){if(v<lo)lo=v;if(v>hi)hi=v}return lo<=hi?[lo,hi]:[0,1]}
const tip=document.getElementById("tip");
function showTip(ev,rows){tip.replaceChildren();for(const [k,v,c] of rows){const d=h("div",tip);if(c){const s=h("span",d,null,"key");s.style.borderColor=c}h("b",d,v);d.append(" "+k)}
 tip.style.display="block";tip.style.left=Math.min(ev.clientX+14,innerWidth-tip.offsetWidth-8)+"px";tip.style.top=(ev.clientY+14)+"px"}
const hideTip=()=>{tip.style.display="none"};
function legend(id,items){const lg=document.getElementById(id);lg.replaceChildren();for(const [n,c,kind] of items){const s=h("span",lg);const k=h("span",s,null,kind=="sw"?"sw":"key");
 if(kind=="sw")k.style.background=c;else{k.style.borderColor=c;if(kind=="dash")k.style.borderTopStyle="dashed";if(kind=="faint")k.style.opacity=.35}s.append(n)}}
// A chart frame: x domain [x0,x1], y domain [lo,hi]; xfmt labels the x axis; onHover(xValue, event) -> tooltip rows.
function frame(id,H,x0,x1,lo,hi,xfmt,onHover,yfmt){const host=document.getElementById(id);host.replaceChildren();
 const W=Math.max(320,host.clientWidth),M={l:48,r:40,t:6,b:18},svg=el("svg",{viewBox:`0 0 ${W} ${H}`,height:H},host);
 const x=lin(x0,x1,M.l,W-M.r),y=lin(lo,hi,H-M.b,M.t);
 for(const v of nice(lo,hi,4)){el("line",{x1:M.l,x2:W-M.r,y1:y(v),y2:y(v),stroke:css("--grid")},svg);const t=el("text",{x:M.l-6,y:y(v)+4,"text-anchor":"end"},svg);t.textContent=yfmt?yfmt(v):+v.toFixed(3)}
 el("line",{x1:M.l,x2:W-M.r,y1:H-M.b,y2:H-M.b,stroke:css("--axis")},svg);
 for(const v of nice(x0,x1,6)){if(v<x0||v>x1)continue;const t=el("text",{x:x(v),y:H-4,"text-anchor":"middle"},svg);t.textContent=xfmt(v)}
 const g=el("g",{},svg),cl=el("line",{y1:M.t,y2:H-M.b,stroke:css("--ink2"),visibility:"hidden"},svg);
 if(onHover){const hit=el("rect",{x:M.l,y:0,width:W-M.l-M.r,height:H,fill:"transparent"},svg);
  hit.addEventListener("pointermove",ev=>{const r=svg.getBoundingClientRect(),px=(ev.clientX-r.left)*W/r.width,xv=x0+(px-M.l)/(W-M.l-M.r)*(x1-x0);
   const res=onHover(xv);if(!res)return;cl.setAttribute("x1",x(res[0]));cl.setAttribute("x2",x(res[0]));cl.setAttribute("visibility","visible");showTip(ev,res[1])});
  hit.addEventListener("pointerleave",()=>{cl.setAttribute("visibility","hidden");hideTip()})}
 return {g,x,y,W,H,M,svg}}
function line(fr,xs,ys,attrs){let d="",pen=false;for(let i=0;i<ys.length;i++){const v=ys[i];if(v==null||!isFinite(v)){pen=false;continue}d+=(pen?"L":"M")+fr.x(xs[i]).toFixed(1)+","+fr.y(v).toFixed(1);pen=true}
 return el("path",Object.assign({d,fill:"none","stroke-width":2,"stroke-linejoin":"round"},attrs),fr.g)}
function vmark(fr,xv){el("line",{x1:fr.x(xv),x2:fr.x(xv),y1:fr.M.t,y2:fr.H-fr.M.b,stroke:css("--ink"),"stroke-width":1,"stroke-dasharray":"3 3"},fr.g)}
function endLabel(fr,yv,txt){if(yv==null||!isFinite(yv))return;const t=el("text",{x:fr.W-fr.M.r+3,y:fr.y(yv)+4,class:"lab"},fr.g);t.textContent=txt}

// ---------- data ----------
let HDR=null,CK=[],DONE=null,cur=0,playing=false,timer=null,lastData=0,src="live.jsonl";
function ingest(text){const lines=text.split("\n").filter(l=>l.trim());const ck=[];let hdr=null,done=null;
 for(const l of lines){let o;try{o=JSON.parse(l)}catch(e){continue}if(o.type=="header")hdr=o;else if(o.type=="ckpt")ck.push(o);else if(o.type=="done")done=o}
 if(!hdr)return false;const grew=ck.length!=CK.length||(!!done)!=(!!DONE);HDR=hdr;CK=ck;DONE=done;return grew}
async function poll(){try{const r=await fetch(src+(src.includes("?")?"&":"?")+"t="+Date.now(),{cache:"no-store"});if(r.ok){const grew=ingest(await r.text());lastData=Date.now();
  if(grew){if(document.getElementById("follow").checked&&!playing)cur=CK.length-1;update()}}}catch(e){}
 status();if(!DONE)setTimeout(poll,2000)}
function status(){const s=document.getElementById("status"),t=document.getElementById("statustext");
 const txt=BAKED?(DONE?"replay of a finished run":"replay of a partial run"):DONE?"finished":HDR?(Date.now()-lastData<6000?"live: training":"live: no update"):"waiting for data";
 t.textContent=txt;s.classList.toggle("live",!BAKED&&!DONE&&!!HDR)}

// ---------- rendering ----------
function update(){if(!HDR||!CK.length)return;document.getElementById("msg").style.display="none";document.getElementById("app").style.display="";
 const sl=document.getElementById("slider");sl.max=CK.length-1;cur=Math.max(0,Math.min(cur,CK.length-1));sl.value=cur;
 document.getElementById("meta").textContent=`market ${HDR.market} · warm-up ${HDR.warmup} s · seed ${HDR.seed} · ${HDR.episodes} episodes · checkpoint every ${HDR.every} · ${HDR.probe} probe episodes from seed ${HDR.probe_seed} · hidden ${HDR.hidden}, γ ${HDR.gamma}, n ${HDR.nstep}, lr ${HDR.lr}`;
 const last=CK[CK.length-1],c=CK[cur];document.getElementById("progbar").style.width=(100*last.episode/HDR.episodes)+"%";
 const rate=last.episode?last.elapsed/last.episode:0;document.getElementById("eta").textContent=DONE?`done in ${f(last.elapsed,0)} s`:`${last.episode}/${HDR.episodes} · ${f(last.elapsed,0)} s` + (rate?` · ~${f(rate*(HDR.episodes-last.episode),0)} s left`:"");
 document.getElementById("where").textContent=`checkpoint ${cur+1}/${CK.length} · episode ${c.episode} · ε ${f(c.epsilon,3)}`;
 renderLearn();renderProbe();renderReplay();renderPolicy()}
function allEvals(){const e=[];for(const c of CK)for(const v of c.evals)e.push(v);return e}
function renderLearn(){const R=[];let ep=0;for(const c of CK)for(const r of c.returns)R.push([++ep,r]);
 const roll=[],k=Math.max(5,Math.round(HDR.episodes/50));let s=0;for(let i=0;i<R.length;i++){s+=R[i][1];if(i>=k)s-=R[i-k][1];roll.push(i>=k-1?s/k:null)}
 const E=allEvals(),c=CK[cur];let [lo,hi]=ext([R.map(r=>r[1]),E.map(e=>e.challenger),E.map(e=>e.champion)]);
 // clip the y range to the 2nd-98th percentile of returns so the first random episodes don't flatten everything
 const sorted=R.map(r=>r[1]).sort((a,b)=>a-b);if(sorted.length>20){lo=Math.min(sorted[Math.floor(.02*sorted.length)],...E.map(e=>e.champion));hi=Math.max(sorted[Math.floor(.98*sorted.length)],...E.map(e=>e.champion),0)}
 legend("lg-learn",[["training return (one dot per episode, with exploration)",css("--muted"),"sw"],[`rolling mean of ${k}`,css("--ink")],["validation: learner",css("--agent")],["validation: champion",css("--champ")]]);
 const fr=frame("learn",210,0,HDR.episodes,lo,hi,v=>String(v),xv=>{const i=Math.max(0,Math.min(R.length-1,Math.round(xv)-1));if(!R.length)return null;
  const ev=E.reduce((b,e)=>Math.abs(e.episode-xv)<Math.abs((b?b.episode:1e18)-xv)?e:b,null);
  return[R[i][0],[["episode",String(R[i][0])],["return",sg(R[i][1])],["rolling mean",sg(roll[i]),css("--ink")]].concat(ev?[[`validation @${ev.episode}: learner`,sg(ev.challenger),css("--agent")],["champion",sg(ev.champion),css("--champ")]]:[])]});
 el("line",{x1:fr.M.l,x2:fr.W-fr.M.r,y1:fr.y(0),y2:fr.y(0),stroke:css("--axis")},fr.g);
 const cl=v=>Math.max(lo,Math.min(hi,v));for(const [e,r] of R)el("circle",{cx:fr.x(e),cy:fr.y(cl(r)),r:1.6,fill:css("--muted"),opacity:.45},fr.g);
 line(fr,R.map(r=>r[0]),roll,{stroke:css("--ink"),"stroke-width":1.5});
 line(fr,E.map(e=>e.episode),E.map(e=>e.challenger),{stroke:css("--agent")});line(fr,E.map(e=>e.episode),E.map(e=>e.champion),{stroke:css("--champ")});
 for(const e of E)if(e.promoted)el("circle",{cx:fr.x(e.episode),cy:fr.y(e.champion),r:3,fill:css("--champ"),stroke:css("--surface"),"stroke-width":1.5},fr.g);
 if(E.length)endLabel(fr,E[E.length-1].champion,sg(E[E.length-1].champion,1));vmark(fr,c.episode);
 const fe=frame("eps",80,0,HDR.episodes,0,1,v=>String(v),null,v=>v==0||v==1?String(v):"");const t=el("text",{x:fe.M.l+4,y:fe.M.t+10,class:"lab"},fe.g);t.textContent="exploration ε";
 line(fe,[0].concat(CK.map(c=>c.episode)),[1].concat(CK.map(c=>c.epsilon)),{stroke:css("--ink2"),"stroke-width":1.5});vmark(fe,c.episode)}
function renderProbe(){const xs=CK.map(c=>c.episode),pm=CK.map(c=>mean(c.probe_pnl)),rule=mean(HDR.rule_pnl),c=CK[cur];
 const [lo,hi]=ext([pm,[rule,0]]);legend("lg-probe",[["learner, mean P&L per probe episode (ticks)",css("--agent")],["30 s delta rule on the same probes",css("--rule"),"dash"]]);
 const fr=frame("probe",170,0,HDR.episodes,lo,hi,v=>String(v),xv=>{const i=nearest(xv);return[CK[i].episode,[["episode",String(CK[i].episode)],["learner",sg(pm[i]),css("--agent")],["rule",sg(rule),css("--rule")],["orders/episode",f(CK[i].orders,1)]]]});
 el("line",{x1:fr.M.l,x2:fr.W-fr.M.r,y1:fr.y(0),y2:fr.y(0),stroke:css("--axis")},fr.g);
 el("line",{x1:fr.M.l,x2:fr.W-fr.M.r,y1:fr.y(rule),y2:fr.y(rule),stroke:css("--rule"),"stroke-dasharray":"5 3","stroke-width":1.5},fr.g);
 line(fr,xs,pm,{stroke:css("--agent")});for(let i=0;i<xs.length;i++)el("circle",{cx:fr.x(xs[i]),cy:fr.y(pm[i]),r:i==cur?5:2.5,fill:css("--agent"),stroke:css("--surface"),"stroke-width":1.5},fr.g);
 endLabel(fr,pm[pm.length-1],sg(pm[pm.length-1],1));vmark(fr,c.episode);
 // time in market: stacked long (bottom) and short (top) shares
 const fm=frame("mkt",120,0,HDR.episodes,0,1,v=>String(v),xv=>{const i=nearest(xv);return[CK[i].episode,[["episode",String(CK[i].episode)],["long",f(100*CK[i].long_share,0)+"%",css("--long")],["short",f(100*CK[i].short_share,0)+"%",css("--short")],["flat",f(100*(1-CK[i].long_share-CK[i].short_share),0)+"%"]]]},v=>Math.round(v*100)+"%");
 const area=(a,b,col)=>{let d="";xs.forEach((x,i)=>d+=(i?"L":"M")+fm.x(x)+","+fm.y(b[i]));for(let i=xs.length-1;i>=0;i--)d+="L"+fm.x(xs[i])+","+fm.y(a[i]);el("path",{d:d+"Z",fill:col,opacity:.85},fm.g)};
 const L=CK.map(c=>c.long_share),S=CK.map((c,i)=>L[i]+c.short_share);area(CK.map(()=>0),L,css("--long"));area(L,S,css("--short"));
 const t=el("text",{x:fm.M.l+4,y:fm.M.t+10,class:"lab"},fm.g);t.textContent="time in market on the probes: long (blue) + short (red)";vmark(fm,c.episode)}
function nearest(xv){let b=0;CK.forEach((c,i)=>{if(Math.abs(c.episode-xv)<Math.abs(CK[b].episode-xv))b=i});return b}
function bands(fr,pos,y0,y1){let s=0;for(let i=1;i<=pos.length;i++){if(i==pos.length||pos[i]!=pos[s]){if(pos[s]){const xa=fr.x(s),xb=fr.x(Math.min(i,pos.length-1));
 el("rect",{x:xa,y:Math.min(y0,y1),width:Math.max(1,xb-xa),height:Math.abs(y1-y0),fill:pos[s]>0?css("--longbg"):css("--shortbg")},fr.g)}s=i}}}
function renderReplay(){const c=CK[cur],T=c.track,R=HDR.rule,n=T.mid.length,m0=T.mid[0],xs=[...Array(n).keys()],tf=v=>clock(T.tod[Math.max(0,Math.min(n-1,Math.round(v)))]);
 document.getElementById("replay-title").textContent=`Probe episode ${HDR.probe_seed} at checkpoint ${cur+1} (episode ${c.episode}): learner ${sg(T.pnl[n-1],1)} ticks, rule ${sg(R.pnl[R.pnl.length-1],1)}`;
 const rel=a=>a.map(v=>v==null?null:v-m0),hov=xv=>{const i=Math.max(0,Math.min(n-1,Math.round(xv)));return[i,[["time",clock(T.tod[i])],["mid",sg(T.mid[i]-m0,1)],["fair value − mid",sg(T.fv[i]-T.mid[i]),css("--fv")],
  ["learner position",String(T.pos[i])],["rule position",String(R.pos[i])],["Q long − flat",sg(T.ql[i]-T.qf[i]),css("--long")],["Q short − flat",sg(T.qs[i]-T.qf[i]),css("--short")],["learner P&L",sg(T.pnl[i],1),css("--agent")],["rule P&L",sg(R.pnl[i],1),css("--rule")]]]};
 legend("lg-price",[["mid",css("--ink")],["hidden fair value",css("--fv"),"dash"],["buy ▲ / sell ▼ (learner's fills); shading = learner's position",css("--ink2")]]);
 const [plo,phi]=ext([rel(T.mid),rel(T.fv)]),fp=frame("price",200,0,n-1,plo-1,phi+1,tf,hov);bands(fp,T.pos,fp.y(phi+1),fp.y(plo-1));
 line(fp,xs,rel(T.mid),{stroke:css("--ink"),"stroke-width":1.5});line(fp,xs,rel(T.fv),{stroke:css("--fv"),"stroke-dasharray":"5 3"});
 for(let i=0;i<n;i++){const prev=i?T.pos[i-1]:0,dp=T.pos[i]-prev;if(!dp)continue;const buy=dp>0,px=T.fill[i]!=null?T.fill[i]:T.mid[i],X=fp.x(i),Y=fp.y(px-m0);
  el("path",{d:buy?`M${X-5},${Y+7}L${X+5},${Y+7}L${X},${Y-1}Z`:`M${X-5},${Y-7}L${X+5},${Y-7}L${X},${Y+1}Z`,fill:buy?css("--long"):css("--short"),stroke:css("--surface"),"stroke-width":1.5},fp.g)}
 const fs=frame("strip",56,0,n-1,0,2,tf,hov,()=>"");[["learner",T.pos],["rule",R.pos]].forEach(([nm,a],k)=>{const y0=fs.y(2-k)+3,y1=fs.y(1-k)-3;let s=0;
  for(let i=1;i<=n;i++){if(i==n||a[i]!=a[s]){if(a[s])el("rect",{x:fs.x(s),y:y0,width:Math.max(1,fs.x(Math.min(i,n-1))-fs.x(s)),height:y1-y0,rx:1,fill:a[s]>0?css("--long"):css("--short")},fs.g);s=i}}
  const t=el("text",{x:fs.M.l-6,y:(y0+y1)/2+4,"text-anchor":"end",class:"lab"},fs.g);t.textContent=nm});
 const trail=CK.slice(Math.max(0,cur-5),cur);const [lo,hi]=ext([T.pnl,R.pnl,[0]].concat(trail.map(t=>t.track.pnl)));
 legend("lg-pnl",[["learner at this checkpoint (net P&L, ticks)",css("--agent")],["earlier checkpoints",css("--agent"),"faint"],["30 s delta rule",css("--rule")]]);
 const fl=frame("pnl",170,0,n-1,lo,hi,tf,hov);el("line",{x1:fl.M.l,x2:fl.W-fl.M.r,y1:fl.y(0),y2:fl.y(0),stroke:css("--axis")},fl.g);
 trail.forEach((t,k)=>line(fl,xs,t.track.pnl,{stroke:css("--agent"),"stroke-width":1.2,opacity:(.1+.12*(k+1)).toFixed(2)}));
 line(fl,xs,R.pnl,{stroke:css("--rule")});line(fl,xs,T.pnl,{stroke:css("--agent"),"stroke-width":2.5});endLabel(fl,T.pnl[n-1],sg(T.pnl[n-1],1));
 const al=T.ql.map((v,i)=>v-T.qf[i]),as=T.qs.map((v,i)=>v-T.qf[i]),[qlo,qhi]=ext([al,as,[0]]);
 legend("lg-q",[["Q long − Q flat",css("--long")],["Q short − Q flat",css("--short")],["above 0: preferred to flat",css("--muted"),"sw"]]);
 const fq=frame("q",130,0,n-1,qlo,qhi,tf,hov);el("line",{x1:fq.M.l,x2:fq.W-fq.M.r,y1:fq.y(0),y2:fq.y(0),stroke:css("--axis")},fq.g);
 line(fq,xs,as,{stroke:css("--short"),"stroke-width":1.3});line(fq,xs,al,{stroke:css("--long"),"stroke-width":1.3})}
function mix(a,b,t){const p=s=>[1,3,5].map(k=>parseInt(s.slice(k,k+2),16)),A=p(a),B=p(b);return"rgb("+A.map((v,k)=>Math.round(v+(B[k]-v)*t)).join(",")+")"}
const divColor=v=>v>=0?mix(css("--neutral"),css("--long"),Math.min(1,v)):mix(css("--neutral"),css("--short"),Math.min(1,-v));
function renderPolicy(){const P=CK[cur].policy,IB=HDR.imb_bins,DB=HDR.delta_bins;
 for(const held of[0,1,2]){const host=document.getElementById("pm"+held);host.replaceChildren();const W=Math.max(220,host.clientWidth),H=230,ml=34,mb=30,cw=(W-ml-4)/DB,ch=(H-mb-4)/IB;
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,height:H},host);
  for(let i=0;i<IB;i++)for(let j=0;j<DB;j++){const o=((held*IB+i)*DB+j)*3,c=[P[o],P[o+1],P[o+2]],n=c[0]+c[1]+c[2],x=ml+j*cw,y=4+(IB-1-i)*ch;
   const r=el("rect",{x:x+1,y:y+1,width:cw-2,height:ch-2,rx:2,fill:n?divColor((c[2]-c[0])/n):css("--surface"),stroke:n?"none":css("--grid"),opacity:n&&n<10?.35:1},svg);
   r.addEventListener("pointermove",ev=>{r.setAttribute("stroke",css("--ink"));showTip(ev,[["states",String(n)],...ACT.map((a,k)=>[a,n?Math.round(100*c[k]/n)+"%":"–",k==1?css("--muted"):k==0?css("--short"):css("--long")]),
    ["L1 imbalance",`${(-1+2*i/IB).toFixed(1)}…${(-1+2*(i+1)/IB).toFixed(1)}`],["30 s delta (norm.)",`${-4+8*j/DB}…${-4+8*(j+1)/DB}`]])});
   r.addEventListener("pointerleave",()=>{r.setAttribute("stroke",n?"none":css("--grid"));hideTip()})}
  for(let i=0;i<=IB;i+=5){const t=el("text",{x:ml-4,y:4+(IB-i)*ch+4,"text-anchor":"end"},svg);t.textContent=(-1+2*i/IB).toFixed(0)}
  for(let j=0;j<=DB;j+=2){const t=el("text",{x:ml+j*cw,y:H-mb+13,"text-anchor":"middle"},svg);t.textContent=-4+8*j/DB}
  const t=el("text",{x:ml+(W-ml)/2,y:H-4,"text-anchor":"middle",class:"lab"},svg);t.textContent="30 s delta → (imbalance ↑)"}}
// ---------- controls ----------
function setPlaying(p){playing=p;document.getElementById("play").textContent=p?"⏸ Pause":"▶ Play";clearTimeout(timer);if(p)tick()}
function tick(){if(!playing)return;if(cur>=CK.length-1){if(DONE||BAKED){setPlaying(false);return}}else{cur++;update()}timer=setTimeout(tick,+document.getElementById("speed").value)}
document.getElementById("play").onclick=()=>{if(!playing&&cur>=CK.length-1)cur=0;setPlaying(!playing)};
document.getElementById("slider").oninput=e=>{cur=+e.target.value;document.getElementById("follow").checked=false;update()};
addEventListener("keydown",e=>{if(e.target.tagName=="SELECT"||e.target.tagName=="INPUT"&&e.target.type!="range")return;if(e.key==" "){e.preventDefault();document.getElementById("play").click()}
 if(e.key=="ArrowLeft"){cur=Math.max(0,cur-1);update()}if(e.key=="ArrowRight"){cur=Math.min(CK.length-1,cur+1);update()}});
let rt;addEventListener("resize",()=>{clearTimeout(rt);rt=setTimeout(update,150)});matchMedia("(prefers-color-scheme: dark)").addEventListener("change",update);
src=new URLSearchParams(location.search).get("src")||"live.jsonl";document.getElementById("srcname").textContent=src;
if(BAKED){ingest(BAKED);document.getElementById("follow").parentNode.style.display="none";cur=0;update();status();setPlaying(true)}else poll();
setInterval(status,2000);
</script></body></html>
"""

if __name__ == "__main__":
    sys.exit(main(sys.argv))
