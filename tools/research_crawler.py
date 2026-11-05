#!/usr/bin/env python3
"""research_crawler.py - collect reinforcement-learning trading research.

Searches public research APIs for papers and articles on RL for trading
(order flow, limit order books, market making, tick scalping, execution,
futures), ranks them for relevance, and writes a corpus the RAG engine can
index:

  OUT/papers.jsonl        one JSON record per paper/article (merged, deduplicated)
  OUT/digest.md           ranked list: title, year, venue, citations, link, abstract
  OUT/docs/<id>.md        one text document per item (title, metadata, abstract,
                          article text): `rag_pdf index corpus.ragidx OUT/docs`
  OUT/pdfs/<id>.pdf       open-access PDFs (with --pdfs)
  OUT/cache/              raw responses, so re-runs don't re-download

Sources (no accounts or API keys):
  arxiv       export.arxiv.org API (q-fin and cs categories)
  openalex    api.openalex.org (journals, conferences, SSRN and more)
  s2          api.semanticscholar.org (with citation counts)
  hn          hn.algolia.com: Hacker News stories (blog posts, articles);
              --fetch-articles also downloads the linked pages' text

Anonymous by design: the requests carry a generic user agent
("research-crawler/1.0"), no email address, no API key, no cookies and no
referrer, so nothing identifies who runs it. It is also polite: it checks
robots.txt before fetching any web page or PDF, waits between requests to the
same host (arXiv asks for 3 s), backs off on 429/5xx, and caches everything.
It does not rotate IPs or disguise itself to get past blocks: a source that
refuses it is skipped.

Usage:
  python3 tools/research_crawler.py                      # all sources, default queries
  python3 tools/research_crawler.py --since 2018 --max-per-query 50 --pdfs
  python3 tools/research_crawler.py --sources arxiv,s2 --query "deep reinforcement learning limit order book"
  python3 tools/research_crawler.py --queries my_queries.txt --fetch-articles

Standard library only (Python 3.9+); runs on Linux, macOS and Windows.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import html.parser
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

USER_AGENT = "research-crawler/1.0"

DEFAULT_QUERIES = [
    "reinforcement learning order flow trading",
    "reinforcement learning limit order book",
    "deep reinforcement learning market making",
    "reinforcement learning high-frequency trading",
    "reinforcement learning optimal execution",
    "reinforcement learning futures trading",
    "deep reinforcement learning intraday trading",
    "queue position limit order book",
    "order flow imbalance price prediction",
    "deep learning limit order book prediction",
    "market microstructure reinforcement learning",
    "agent-based market simulator reinforcement learning",
    "offline reinforcement learning trading",
    "reinforcement learning trading backtest overfitting",
]

# Relevance lexicon: (pattern, weight). Title hits count 3x.
RL_TERMS = [
    (r"reinforcement learning", 3), (r"\bq-learning\b", 2), (r"\bdqn\b|deep q-network", 2), (r"\bppo\b|proximal policy", 2),
    (r"actor-critic|\bsac\b|soft actor", 2), (r"policy gradient", 2), (r"offline rl|offline reinforcement", 2),
    (r"\bmdp\b|markov decision", 1), (r"\bagent\b", 0.5),
]
MARKET_TERMS = [
    (r"order flow", 3), (r"limit order book|\blob\b|order book", 3), (r"market making|market maker", 3), (r"queue", 2),
    (r"imbalance", 2), (r"high[- ]frequency|\bhft\b", 2), (r"optimal execution|trade execution|execution", 1.5),
    (r"microstructure", 2), (r"\btick\b|tick-level|tick data", 2), (r"scalp", 2), (r"spread", 1), (r"liquidity", 1),
    (r"futures", 1.5), (r"e-mini|s&p 500|\bcme\b", 2), (r"intraday", 1.5), (r"trading", 1),
]
VALIDATION_TERMS = [
    (r"backtest", 1), (r"overfit", 1.5), (r"walk[- ]forward", 1.5), (r"out-of-sample", 1), (r"transaction cost", 1),
    (r"simulator|simulation", 0.5),
]


@dataclass
class Item:
    key: str                     # dedup key: doi:, arxiv:, or title:
    source: str                  # first source that found it
    kind: str                    # paper | article
    title: str
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    venue: str = ""
    abstract: str = ""
    url: str = ""
    pdf_url: str = ""
    doi: str = ""
    arxiv_id: str = ""
    citations: int | None = None
    sources: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    text: str = ""               # fetched article text (hn with --fetch-articles)
    score: float = 0.0


# ------------------------------------------------------------------ helpers
def normalize_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def make_key(doi: str = "", arxiv_id: str = "", title: str = "") -> str:
    if doi:
        return "doi:" + doi.lower().removeprefix("https://doi.org/")
    if arxiv_id:
        return "arxiv:" + re.sub(r"v\d+$", "", arxiv_id)
    return "title:" + normalize_title(title)


def slug(key: str) -> str:
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def abstract_from_inverted_index(inv: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}."""
    if not inv:
        return ""
    pos = {}
    for word, places in inv.items():
        for p in places:
            pos[p] = word
    return " ".join(pos[i] for i in sorted(pos))


def lexicon_score(text: str, terms) -> float:
    return sum(w for pat, w in terms if re.search(pat, text))


def relevance(item: Item, now_year: int) -> float:
    """Keyword relevance (title 3x), boosted by citations and recency; items
    without both an RL term and a market term are pushed down."""
    title, body = item.title.lower(), (item.abstract + " " + item.text[:5000]).lower()
    rl = 3 * lexicon_score(title, RL_TERMS) + lexicon_score(body, RL_TERMS)
    mk = 3 * lexicon_score(title, MARKET_TERMS) + lexicon_score(body, MARKET_TERMS)
    va = lexicon_score(title + " " + body, VALIDATION_TERMS)
    s = rl + mk + va
    if rl == 0 or mk == 0:
        s *= 0.3
    if item.citations:
        s += 1.5 * math.log1p(item.citations)
    if item.year:
        s += max(0.0, 3.0 - 0.3 * max(0, now_year - item.year))
    return round(s, 2)


class TextExtractor(html.parser.HTMLParser):
    """Visible text of an HTML page (scripts, styles and navigation dropped)."""
    SKIP = {"script", "style", "noscript", "nav", "header", "footer", "svg", "form"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.depth = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.depth += 1
        if tag == "title":
            self._in_title = True
        if tag in ("p", "br", "li", "h1", "h2", "h3", "h4", "tr", "div"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.depth:
            self.depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.depth:
            self.parts.append(data)

    def text(self) -> str:
        t = "".join(self.parts)
        t = re.sub(r"[ \t\r\f\v]+", " ", t)
        return re.sub(r"\n\s*\n+", "\n\n", t).strip()


def html_to_text(page: str) -> tuple[str, str]:
    p = TextExtractor()
    p.feed(page)
    return p.title.strip(), p.text()


# ------------------------------------------------------------------ HTTP
class Fetcher:
    """Anonymous, polite HTTP: generic user agent, no cookies/referrer/keys;
    per-host delay, robots.txt for pages, retries with backoff, disk cache."""

    API_HOSTS = {"export.arxiv.org", "api.openalex.org", "api.semanticscholar.org", "hn.algolia.com"}

    def __init__(self, cache: Path, delays: dict[str, float] | None = None, default_delay: float = 1.0, timeout: float = 30):
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.delays = delays or {"export.arxiv.org": 3.1, "arxiv.org": 3.1, "api.semanticscholar.org": 1.2}
        self.default_delay = default_delay
        self.timeout = timeout
        self.last: dict[str, float] = {}
        self.robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        # No cookie processor is installed: urllib sends no cookies.
        self.opener = urllib.request.build_opener()
        self.opener.addheaders = [("User-Agent", USER_AGENT), ("Accept-Encoding", "gzip")]

    def _wait(self, host: str):
        delay = self.delays.get(host, self.default_delay)
        dt = time.monotonic() - self.last.get(host, -1e9)
        if dt < delay:
            time.sleep(delay - dt)
        self.last[host] = time.monotonic()

    def allowed(self, url: str) -> bool:
        parts = urllib.parse.urlsplit(url)
        if parts.hostname in self.API_HOSTS:
            return True  # documented public APIs for automated access
        base = f"{parts.scheme}://{parts.netloc}"
        if base not in self.robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                raw = self.get(base + "/robots.txt", check_robots=False, cache=True, quiet=True)
                rp.parse(raw.decode("utf-8", "replace").splitlines() if raw is not None else [])
                self.robots[base] = rp
            except Exception:
                self.robots[base] = None  # unreachable robots.txt: treat as allowed
        rp = self.robots[base]
        return True if rp is None else rp.can_fetch(USER_AGENT, url)

    def get(self, url: str, check_robots: bool = True, cache: bool = True, quiet: bool = False, retries: int = 4) -> bytes | None:
        path = self.cache / (hashlib.sha1(url.encode()).hexdigest() + ".bin")
        if cache and path.exists():
            return path.read_bytes()
        if check_robots and not self.allowed(url):
            if not quiet:
                print(f"  robots.txt disallows {url}; skipped", file=sys.stderr)
            return None
        host = urllib.parse.urlsplit(url).hostname or ""
        for attempt in range(retries + 1):
            self._wait(host)
            try:
                with self.opener.open(urllib.request.Request(url), timeout=self.timeout) as r:
                    data = r.read()
                    if r.headers.get("Content-Encoding") == "gzip":
                        data = gzip.decompress(data)
                if cache:
                    path.write_bytes(data)
                return data
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                    wait = float(e.headers.get("Retry-After") or 0) or 5 * 2**attempt
                    time.sleep(min(wait, 120))
                    continue
                if not quiet:
                    print(f"  HTTP {e.code} for {url}", file=sys.stderr)
                return None
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt < retries:
                    time.sleep(3 * 2**attempt)
                    continue
                if not quiet:
                    print(f"  failed {url}: {e}", file=sys.stderr)
                return None
        return None


# ------------------------------------------------------------------ sources
def parse_arxiv(atom: bytes, query: str) -> list[Item]:
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    root = ET.fromstring(atom)
    out = []
    for e in root.findall("a:entry", ns):
        aid = (e.findtext("a:id", "", ns) or "").rsplit("/abs/", 1)[-1]
        title = " ".join((e.findtext("a:title", "", ns) or "").split())
        if not title:
            continue
        pdf = next((l.get("href", "") for l in e.findall("a:link", ns) if l.get("title") == "pdf"), "")
        doi = e.findtext("arxiv:doi", "", ns) or ""
        year = (e.findtext("a:published", "", ns) or "")[:4]
        out.append(Item(
            key=make_key(doi=doi, arxiv_id=aid, title=title), source="arxiv", kind="paper", title=title,
            year=int(year) if year.isdigit() else None,
            authors=[a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)],
            venue=e.findtext("arxiv:journal_ref", "", ns) or "arXiv",
            abstract=" ".join((e.findtext("a:summary", "", ns) or "").split()),
            url=f"https://arxiv.org/abs/{aid}", pdf_url=pdf.replace("http://", "https://"), doi=doi, arxiv_id=aid,
            sources=["arxiv"], queries=[query]))
    return out


def parse_openalex(data: bytes, query: str) -> list[Item]:
    out = []
    for w in json.loads(data).get("results", []):
        title = w.get("title") or w.get("display_name") or ""
        if not title:
            continue
        doi = (w.get("doi") or "").removeprefix("https://doi.org/")
        loc = w.get("primary_location") or {}
        src = loc.get("source") or {}
        oa = w.get("open_access") or {}
        out.append(Item(
            key=make_key(doi=doi, title=title), source="openalex", kind="paper", title=title,
            year=w.get("publication_year"),
            authors=[(a.get("author") or {}).get("display_name", "") for a in w.get("authorships", [])][:20],
            venue=src.get("display_name") or "", abstract=abstract_from_inverted_index(w.get("abstract_inverted_index")),
            url=loc.get("landing_page_url") or (f"https://doi.org/{doi}" if doi else w.get("id", "")),
            pdf_url=oa.get("oa_url") or "", doi=doi, citations=w.get("cited_by_count"),
            sources=["openalex"], queries=[query]))
    return out


def parse_s2(data: bytes, query: str) -> list[Item]:
    out = []
    for p in json.loads(data).get("data", []) or []:
        title = p.get("title") or ""
        if not title:
            continue
        ext = p.get("externalIds") or {}
        doi, aid = ext.get("DOI") or "", ext.get("ArXiv") or ""
        out.append(Item(
            key=make_key(doi=doi, arxiv_id=aid, title=title), source="s2", kind="paper", title=title, year=p.get("year"),
            authors=[a.get("name", "") for a in p.get("authors") or []][:20], venue=p.get("venue") or "",
            abstract=p.get("abstract") or "", url=p.get("url") or "", pdf_url=(p.get("openAccessPdf") or {}).get("url") or "",
            doi=doi, arxiv_id=aid, citations=p.get("citationCount"), sources=["s2"], queries=[query]))
    return out


def parse_hn(data: bytes, query: str) -> list[Item]:
    out = []
    for h in json.loads(data).get("hits", []):
        url, title = h.get("url") or "", h.get("title") or ""
        if not url or not title:
            continue
        out.append(Item(
            key="url:" + url.split("#")[0].rstrip("/"), source="hn", kind="article", title=title,
            year=int((h.get("created_at") or "0000")[:4]) or None, venue=urllib.parse.urlsplit(url).hostname or "",
            url=url, citations=h.get("points"), sources=["hn"], queries=[query]))
    return out


def search(fetcher: Fetcher, source: str, query: str, n: int, since: int) -> list[Item]:
    q = urllib.parse.quote
    if source == "arxiv":
        terms = " AND ".join(f'all:"{w}"' if " " in w else f"all:{w}" for w in split_query(query))
        cats = "(cat:q-fin.* OR cat:cs.LG OR cat:cs.AI OR cat:stat.ML)"
        url = (f"https://export.arxiv.org/api/query?search_query={q(f'({terms}) AND {cats}')}"
               f"&start=0&max_results={n}&sortBy=relevance")
        data = fetcher.get(url)
        items = parse_arxiv(data, query) if data else []
    elif source == "openalex":
        url = (f"https://api.openalex.org/works?search={q(query)}&per-page={min(n, 200)}"
               f"&filter=from_publication_date:{since}-01-01")
        data = fetcher.get(url)
        items = parse_openalex(data, query) if data else []
    elif source == "s2":
        fields = "title,abstract,year,venue,citationCount,externalIds,openAccessPdf,url,authors"
        url = (f"https://api.semanticscholar.org/graph/v1/paper/search?query={q(query)}"
               f"&limit={min(n, 100)}&fields={fields}&year={since}-")
        data = fetcher.get(url)
        items = parse_s2(data, query) if data else []
    elif source == "hn":
        url = f"https://hn.algolia.com/api/v1/search?query={q(query)}&tags=story&hitsPerPage={min(n, 100)}"
        data = fetcher.get(url)
        items = parse_hn(data, query) if data else []
    else:
        raise ValueError(f"unknown source {source}")
    return [i for i in items if not i.year or i.year >= since]


def split_query(query: str) -> list[str]:
    """Keep known multi-word phrases together for arXiv's field search."""
    phrases = ["reinforcement learning", "deep reinforcement learning", "order flow", "limit order book", "order book",
               "market making", "high-frequency trading", "optimal execution", "queue position", "market microstructure",
               "deep learning", "price prediction", "agent-based"]
    rest, found = query.lower(), []
    for ph in sorted(phrases, key=len, reverse=True):
        if ph in rest:
            found.append(ph)
            rest = rest.replace(ph, " ")
    return found + [w for w in rest.split() if len(w) > 2]


def merge(into: dict[str, Item], items: list[Item]) -> None:
    for it in items:
        cur = into.get(it.key)
        if cur is None:
            # Also merge a title match against an existing doi/arxiv record.
            tkey = "title:" + normalize_title(it.title)
            cur = next((v for v in into.values() if normalize_title(v.title) == tkey[6:]), None) if it.kind == "paper" else None
            if cur is None:
                into[it.key] = it
                continue
        for f in ("abstract", "pdf_url", "doi", "arxiv_id", "venue", "url"):
            if not getattr(cur, f) and getattr(it, f):
                setattr(cur, f, getattr(it, f))
        if it.citations is not None and (cur.citations is None or it.citations > cur.citations):
            cur.citations = it.citations
        cur.year = cur.year or it.year
        cur.authors = cur.authors or it.authors
        cur.sources = sorted(set(cur.sources) | set(it.sources))
        cur.queries = sorted(set(cur.queries) | set(it.queries))


# ------------------------------------------------------------------ output
def write_outputs(out: Path, items: list[Item]) -> None:
    (out / "docs").mkdir(parents=True, exist_ok=True)
    with open(out / "papers.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
    lines = [f"# RL trading research digest ({len(items)} items)", "",
             "Ranked by keyword relevance (RL x market-microstructure terms), citations and recency.", ""]
    for n, it in enumerate(items, 1):
        meta = ", ".join(x for x in [str(it.year or ""), it.venue, f"{it.citations} citations" if it.kind == "paper" and it.citations is not None
                                     else (f"{it.citations} points" if it.citations is not None else "")] if x)
        lines.append(f"{n}. **{it.title}** ({meta}) score {it.score}  ")
        lines.append(f"   {it.url}" + (f" · [pdf]({it.pdf_url})" if it.pdf_url else ""))
        if it.abstract:
            lines.append(f"   > {it.abstract[:400]}{'...' if len(it.abstract) > 400 else ''}")
        lines.append("")
    (out / "digest.md").write_text("\n".join(lines), encoding="utf-8")
    for it in items:
        body = [f"# {it.title}", "", f"- kind: {it.kind}", f"- year: {it.year or ''}", f"- authors: {', '.join(it.authors[:12])}",
                f"- venue: {it.venue}", f"- link: {it.url}", f"- citations: {it.citations if it.citations is not None else ''}",
                f"- relevance: {it.score}", ""]
        if it.abstract:
            body += ["## Abstract", "", it.abstract, ""]
        if it.text:
            body += ["## Text", "", it.text, ""]
        (out / "docs" / f"{slug(it.key)}.md").write_text("\n".join(body), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="research_corpus", help="output directory")
    ap.add_argument("--sources", default="arxiv,openalex,s2,hn", help="comma-separated: arxiv,openalex,s2,hn")
    ap.add_argument("--query", action="append", help="a search query (repeatable); default: built-in list")
    ap.add_argument("--queries", help="file with one query per line")
    ap.add_argument("--max-per-query", type=int, default=40)
    ap.add_argument("--since", type=int, default=2015, help="earliest publication year")
    ap.add_argument("--min-score", type=float, default=4.0, help="drop items below this relevance")
    ap.add_argument("--fetch-articles", action="store_true", help="download the text of linked articles (hn)")
    ap.add_argument("--pdfs", action="store_true", help="download open-access PDFs of the top items")
    ap.add_argument("--max-pdfs", type=int, default=100)
    args = ap.parse_args(argv)

    out = Path(args.out)
    fetcher = Fetcher(out / "cache")
    queries = args.query or []
    if args.queries:
        queries += [l.strip() for l in Path(args.queries).read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    queries = queries or DEFAULT_QUERIES
    found: dict[str, Item] = {}
    for source in [s.strip() for s in args.sources.split(",") if s.strip()]:
        for query in queries:
            items = search(fetcher, source, query, args.max_per_query, args.since)
            print(f"{source:9s} {len(items):4d}  {query}", file=sys.stderr)
            merge(found, items)
    now_year = time.gmtime().tm_year
    if args.fetch_articles:
        for it in found.values():
            if it.kind == "article":
                raw = fetcher.get(it.url)
                if raw:
                    _, it.text = html_to_text(raw.decode("utf-8", "replace"))
                    it.text = it.text[:50_000]
    for it in found.values():
        it.score = relevance(it, now_year)
    ranked = sorted((i for i in found.values() if i.score >= args.min_score), key=lambda i: -i.score)
    write_outputs(out, ranked)
    if args.pdfs:
        (out / "pdfs").mkdir(exist_ok=True)
        got = 0
        for it in ranked:
            if got >= args.max_pdfs:
                break
            if it.pdf_url:
                url = it.pdf_url.replace("://arxiv.org/", "://export.arxiv.org/")  # arXiv asks bulk clients to use export
                data = fetcher.get(url, cache=False)
                if data and data[:4] == b"%PDF":
                    (out / "pdfs" / f"{slug(it.key)}.pdf").write_bytes(data)
                    got += 1
        print(f"downloaded {got} PDFs", file=sys.stderr)
    print(f"{len(found)} items found, {len(ranked)} kept (score >= {args.min_score}); wrote {out / 'digest.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
