"""Offline tests for research_crawler.py: parsers on small hand-written
responses in each API's documented shape, merging, ranking, robots.txt and
the anonymity of requests. No network access is needed or used.

    python3 -m unittest discover -s tools -p "test_*.py"
"""
import json
import sys
import tempfile
import unittest
import urllib.robotparser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_crawler as rc  # noqa: E402

ARXIV = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2101.00001v2</id>
    <published>2021-01-04T00:00:00Z</published>
    <title>Deep Reinforcement Learning for
      Market Making in a Limit Order Book</title>
    <summary>We train a PPO agent on order flow imbalance and queue position features.</summary>
    <author><name>A. Author</name></author><author><name>B. Author</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2101.00001v2"/>
    <arxiv:doi>10.1000/example.1</arxiv:doi>
  </entry>
</feed>"""

OPENALEX = json.dumps({"results": [{
    "id": "https://openalex.org/W1", "doi": "https://doi.org/10.1000/EXAMPLE.1",
    "title": "Deep reinforcement learning for market making in a limit order book",
    "publication_year": 2021, "cited_by_count": 42,
    "authorships": [{"author": {"display_name": "A. Author"}}],
    "primary_location": {"landing_page_url": "https://example.org/paper", "source": {"display_name": "Quant Finance"}},
    "abstract_inverted_index": {"We": [0], "train": [1], "agents": [2]},
    "open_access": {"oa_url": ""}}]}).encode()

S2 = json.dumps({"data": [{
    "title": "Order flow imbalance and short-term price prediction", "year": 2019, "venue": "J. Fin. Econometrics",
    "abstract": "Order flow imbalance at the best quotes predicts price changes.", "citationCount": 10,
    "externalIds": {"ArXiv": "1907.00002"}, "openAccessPdf": {"url": "https://arxiv.org/pdf/1907.00002"},
    "url": "https://www.semanticscholar.org/paper/x", "authors": [{"name": "C. Author"}]}]}).encode()

HN = json.dumps({"hits": [
    {"title": "Scalping ES futures with reinforcement learning", "url": "https://blog.example.com/rl-es#top",
     "points": 120, "created_at": "2023-05-01T00:00:00Z"},
    {"title": "Ask HN: no link", "url": None, "points": 5, "created_at": "2023-01-01T00:00:00Z"}]}).encode()


class Parsers(unittest.TestCase):
    def test_arxiv(self):
        [it] = rc.parse_arxiv(ARXIV, "q")
        self.assertEqual(it.title, "Deep Reinforcement Learning for Market Making in a Limit Order Book")
        self.assertEqual(it.arxiv_id, "2101.00001v2")
        self.assertEqual(it.key, "doi:10.1000/example.1")
        self.assertEqual(it.year, 2021)
        self.assertEqual(it.authors, ["A. Author", "B. Author"])
        self.assertEqual(it.pdf_url, "https://arxiv.org/pdf/2101.00001v2")

    def test_openalex_rebuilds_the_abstract(self):
        [it] = rc.parse_openalex(OPENALEX, "q")
        self.assertEqual(it.abstract, "We train agents")
        self.assertEqual(it.doi, "10.1000/EXAMPLE.1")
        self.assertEqual(it.key, "doi:10.1000/example.1")  # keys are case-insensitive
        self.assertEqual(it.citations, 42)
        self.assertEqual(it.venue, "Quant Finance")

    def test_s2_and_hn(self):
        [p] = rc.parse_s2(S2, "q")
        self.assertEqual(p.key, "arxiv:1907.00002")
        self.assertEqual(p.citations, 10)
        hits = rc.parse_hn(HN, "q")
        self.assertEqual(len(hits), 1)  # stories without a link are dropped
        self.assertEqual(hits[0].kind, "article")
        self.assertEqual(hits[0].key, "url:https://blog.example.com/rl-es")
        self.assertEqual(hits[0].venue, "blog.example.com")


class MergeAndRank(unittest.TestCase):
    def test_same_paper_from_two_sources_merges(self):
        found = {}
        rc.merge(found, rc.parse_arxiv(ARXIV, "q1"))
        rc.merge(found, rc.parse_openalex(OPENALEX, "q2"))
        self.assertEqual(len(found), 1)
        [it] = found.values()
        self.assertEqual(it.sources, ["arxiv", "openalex"])
        self.assertEqual(it.queries, ["q1", "q2"])
        self.assertEqual(it.citations, 42)  # the larger count wins
        self.assertTrue(it.pdf_url)

    def test_relevance_prefers_rl_on_microstructure(self):
        core = rc.Item(key="a", source="s", kind="paper", title="Reinforcement learning for market making with order flow",
                       abstract="queue position and order book imbalance", year=2024)
        generic = rc.Item(key="b", source="s", kind="paper", title="Reinforcement learning for robot control",
                          abstract="locomotion", year=2024)
        finance_only = rc.Item(key="c", source="s", kind="paper", title="Order book imbalance and spreads",
                               abstract="microstructure of futures markets", year=2024)
        s = {i.key: rc.relevance(i, 2025) for i in (core, generic, finance_only)}
        self.assertGreater(s["a"], s["c"])
        self.assertGreater(s["a"], s["b"])
        self.assertGreater(s["c"], 0)

    def test_outputs(self):
        found = {}
        rc.merge(found, rc.parse_arxiv(ARXIV, "q"))
        rc.merge(found, rc.parse_hn(HN, "q"))
        items = list(found.values())
        for it in items:
            it.score = rc.relevance(it, 2025)
        with tempfile.TemporaryDirectory() as d:
            rc.write_outputs(Path(d), items)
            lines = (Path(d) / "papers.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("Market Making", (Path(d) / "digest.md").read_text(encoding="utf-8"))
            self.assertEqual(len(list((Path(d) / "docs").glob("*.md"))), 2)


class Helpers(unittest.TestCase):
    def test_html_to_text_drops_scripts_and_navigation(self):
        title, text = rc.html_to_text("<html><head><title>T</title><script>var x=1;</script></head>"
                                      "<body><nav>menu</nav><p>Order flow &amp; RL.</p><p>Second.</p></body></html>")
        self.assertEqual(title, "T")
        self.assertIn("Order flow & RL.", text)
        self.assertNotIn("var x", text)
        self.assertNotIn("menu", text)

    def test_split_query_keeps_phrases(self):
        self.assertEqual(rc.split_query("reinforcement learning limit order book"), ["reinforcement learning", "limit order book"])

    def test_requests_are_anonymous(self):
        f = rc.Fetcher(Path(tempfile.mkdtemp()))
        headers = dict(f.opener.addheaders)
        self.assertEqual(headers["User-Agent"], "research-crawler/1.0")
        self.assertNotIn("From", headers)
        self.assertNotIn("Cookie", headers)
        self.assertNotIn("Referer", headers)
        self.assertFalse(any("@" in v for v in headers.values()))

    def test_robots_txt_is_respected_for_web_pages(self):
        f = rc.Fetcher(Path(tempfile.mkdtemp()))
        rp = urllib.robotparser.RobotFileParser()
        rp.parse(["User-agent: *", "Disallow: /private/"])
        f.robots["https://blog.example.com"] = rp
        self.assertTrue(f.allowed("https://blog.example.com/post"))
        self.assertFalse(f.allowed("https://blog.example.com/private/x"))
        self.assertIsNone(f.get("https://blog.example.com/private/x", quiet=True))  # skipped without a request
        self.assertTrue(f.allowed("https://export.arxiv.org/api/query?x"))  # documented public API


if __name__ == "__main__":
    unittest.main()
