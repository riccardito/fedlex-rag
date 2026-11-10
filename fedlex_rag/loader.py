"""Fetch the consolidated German text of each source from fedlex and split it into articles.

fedlex publishes every consolidated version as Akoma Ntoso XML. The SPARQL endpoint names the
file of the newest version already in force; the parser keeps one record per article, drops the
footnotes, and skips repealed articles.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from .sources import BY_ABBR, SOURCES, Source

SPARQL = "https://fedlex.data.admin.ch/sparqlendpoint"
AKN = "{http://docs.oasis-open.org/legaldocml/ns/akn/3.0}"
DATA = Path(__file__).resolve().parent.parent / "data"
ARTICLES = DATA / "articles.jsonl"
MANIFEST = DATA / "sources.json"

_QUERY = """PREFIX jolux: <http://data.legilux.public.lu/resource/ontology/jolux#>
SELECT ?date ?file WHERE {
  ?cons jolux:isMemberOf <%s> ; jolux:dateApplicability ?date ; jolux:isRealizedBy ?expr .
  ?expr jolux:language <http://publications.europa.eu/resource/authority/language/DEU> ;
        jolux:isEmbodiedBy ?manif .
  ?manif jolux:isExemplifiedBy ?file .
  FILTER(CONTAINS(STR(?file), "/xml/") && ?date <= "%s"^^<http://www.w3.org/2001/XMLSchema#date>)
} ORDER BY DESC(?date) LIMIT 1"""

_REPEALED = re.compile(r"^(aufgehoben|\.\.\.|…)\W*$", re.I)


@dataclass(frozen=True)
class Article:
    id: str        # "DSG Art. 5"
    act: str
    number: str    # "5", "321c"
    heading: str   # the article's own heading, else the nearest enclosing one
    context: str   # the enclosing headings, outermost first: "Beendigung ... > Kündigungsfristen > ..."
    text: str
    url: str

    def passage(self) -> str:
        """What gets embedded and shown: the citation, where it sits, the text."""
        return f"{self.id} ({self.context})\n{self.text}" if self.context else f"{self.id}\n{self.text}"


def latest_xml(source: Source, today: date | None = None, fetch=None) -> tuple[str, str]:
    """(applicability date, XML file URL) of the newest German version in force."""
    query = _QUERY % (source.eli, (today or date.today()).isoformat())
    url = SPARQL + "?" + urllib.parse.urlencode({"query": query})
    body = (fetch or _get)(url, "application/sparql-results+json")
    rows = json.loads(body)["results"]["bindings"]
    if not rows:
        raise LookupError(f"fedlex lists no German XML version of SR {source.sr} in force")
    return rows[0]["date"]["value"], rows[0]["file"]["value"]


def _get(url: str, accept: str) -> str:
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "fedlex-rag/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")


_BLOCKS = {AKN + t for t in ("p", "item", "listIntroduction", "blockList", "paragraph", "content", "br")}


def _text(el: ET.Element) -> str:
    """The visible text of an element, without footnotes; block boundaries become spaces."""
    parts = [el.text or ""]
    for child in el:
        if child.tag == AKN + "br" and "".join(parts).endswith("-"):
            parts.append("\0")              # "Adoptions-<br/>urlaub": the break splits one word
        elif child.tag != AKN + "authorialNote":
            sep = " " if child.tag in _BLOCKS else ""
            parts.append(sep + _text(child) + sep)
        parts.append(child.tail or "")
    return "".join(parts).replace("-\0", "")


# a paragraph or list item that a repeal left behind: "3 …", "a. …"
_GAP = re.compile(r"^(\d+\w*\.?|[a-z]\.)?\s*(…|\.\.\.)$")


def _drop_gaps(el: ET.Element) -> None:
    for parent in el.iter():
        for child in list(parent):
            if child.tag in (AKN + "paragraph", AKN + "item") and _GAP.match(_clean(_text(child))):
                parent.remove(child)


def _own_heading(el: ET.Element) -> str:
    h = el.find(AKN + "heading")
    return _clean(_text(h)) if h is not None else ""


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def parse_articles(xml: str, source: Source) -> list[Article]:
    root = ET.fromstring(xml)
    parent = {child: el for el in root.iter() for child in el}
    out: list[Article] = []
    for art in root.iter(AKN + "article"):
        num_el = art.find(AKN + "num")
        number = _clean(_text(num_el)).removeprefix("Art.").replace(" ", "") if num_el is not None else ""
        match = re.match(r"(\d+)", number)
        if not match or not source.keeps(int(match.group(1))):
            continue
        chain, up = [], parent.get(art)
        while up is not None and up.tag != AKN + "body":
            if h := _own_heading(up):
                chain.insert(0, h)
            up = parent.get(up)
        heading = _own_heading(art) or (chain[-1] if chain else "")
        _drop_gaps(art)
        # paragraphs keep their number ("2 Es ist nicht anwendbar auf: a. ...") -- the citation needs it
        body = [_clean(_text(child)) for child in art if child.tag not in (AKN + "num", AKN + "heading")]
        text = " ".join(b for b in body if b)
        if not text or _REPEALED.match(text):
            continue
        context = " > ".join(chain + ([_own_heading(art)] if _own_heading(art) else []))
        out.append(Article(f"{source.abbr} Art. {number}", source.abbr, number, heading, context, text,
                           source.article_url(art.get("eId", ""))))
    return out


def fetch_all(today: date | None = None, fetch=None) -> tuple[list[Article], list[dict]]:
    articles, manifest = [], []
    for source in SOURCES:
        applies, xml_url = latest_xml(source, today, fetch)
        parsed = parse_articles((fetch or _get)(xml_url, "application/xml"), source)
        articles += parsed
        manifest.append({"abbr": source.abbr, "title": source.title, "sr": source.sr, "eli": source.eli,
                         "applicable_from": applies, "file": xml_url, "articles": len(parsed),
                         "retrieved": (today or date.today()).isoformat()})
    return articles, manifest


def write_fixture(articles: list[Article], manifest: list[dict], data: Path = DATA) -> None:
    data.mkdir(parents=True, exist_ok=True)
    with open(data / ARTICLES.name, "w", encoding="utf-8", newline="\n") as fh:
        for a in articles:
            fh.write(json.dumps(asdict(a), ensure_ascii=False) + "\n")
    note = ("Official enactments of the Swiss Confederation; not protected by copyright "
            "(Art. 5 para. 1 let. a URG, SR 231.1). Regenerate with `make fetch`.")
    (data / MANIFEST.name).write_text(json.dumps({"note": note, "sources": manifest}, ensure_ascii=False,
                                                 indent=2) + "\n", encoding="utf-8", newline="\n")


def load_articles(path: Path = ARTICLES) -> list[Article]:
    with open(path, encoding="utf-8") as fh:
        rows = [Article(**json.loads(line)) for line in fh if line.strip()]
    unknown = {r.act for r in rows} - set(BY_ABBR)
    if unknown:
        raise ValueError(f"articles from unknown acts: {sorted(unknown)}")
    return rows
