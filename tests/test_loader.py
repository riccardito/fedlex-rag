import json
from urllib.parse import unquote_plus

import pytest

from fedlex_rag.loader import Article, fetch_all, latest_xml, load_articles, parse_articles, write_fixture
from fedlex_rag.sources import BY_ABBR, Source

NS = 'xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0"'
XML = f"""<akomaNtoso {NS}><act><body>
<level eId="lvl_A"><num>A. </num><heading>Beendigung</heading>
  <level eId="lvl_A/lvl_1"><heading>Fristen</heading>
    <article eId="art_10"><num><b>Art. 10</b><authorialNote><p>Eingefügt durch AS 1988</p></authorialNote></num>
      <paragraph eId="art_10/para_1"><num>1</num><content><p>Die Frist beträgt<authorialNote><p>Fussnote</p>
      </authorialNote> einen Monat.</p></content></paragraph>
      <paragraph eId="art_10/para_2"><num>2</num><content><blockList>
      <listIntroduction>Sie gilt für:</listIntroduction>
      <item><num>a. </num><p>Arbeitnehmer;</p></item><item><num>b. </num><p>Arbeitgeber.</p></item>
      <item><num>c. </num><p>…</p></item></blockList>
      </content></paragraph>
      <paragraph eId="art_10/para_3"><num>3</num><content><p> …<authorialNote><p>Aufgehoben</p></authorialNote></p>
      </content></paragraph></article>
    <article eId="art_10_a"><num><b>Art. 10</b><i>a</i></num><heading>Probe-<br/>zeit</heading>
      <paragraph><content><p>Sieben Tage.</p></content></paragraph></article>
    <article eId="art_11"><num><b>Art. 11</b></num>
      <paragraph><content><p>Aufgehoben</p></content></paragraph></article>
    <article eId="art_99"><num><b>Art. 99</b></num><paragraph><content><p>Ausserhalb.</p></content></paragraph>
    </article>
  </level></level></body></act></akomaNtoso>"""
SRC = Source("OR", "Test", "220", "https://fedlex.data.admin.ch/eli/cc/27/317_321_377", first=10, last=20)


def test_parse_keeps_range_drops_footnotes_and_repealed():
    arts = parse_articles(XML, SRC)
    assert [a.id for a in arts] == ["OR Art. 10", "OR Art. 10a"]
    first = arts[0]
    assert "Fussnote" not in first.text and "AS 1988" not in first.text
    assert first.text == "1 Die Frist beträgt einen Monat. 2 Sie gilt für: a. Arbeitnehmer; b. Arbeitgeber."


def test_heading_chain_and_own_heading():
    a10, a10a = parse_articles(XML, SRC)
    assert a10.heading == "Fristen" and a10.context == "Beendigung > Fristen"
    assert a10a.heading == "Probezeit" and a10a.context == "Beendigung > Fristen > Probezeit"
    assert a10a.url == "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de#art_10_a"
    assert a10a.passage().startswith("OR Art. 10a (Beendigung > Fristen > Probezeit)\n")


def test_passage_without_context():
    a = Article("DSG Art. 1", "DSG", "1", "", "", "Text.", "u")
    assert a.passage() == "DSG Art. 1\nText."


def _sparql(file):
    return json.dumps({"results": {"bindings": [{"date": {"value": "2026-01-01"}, "file": {"value": file}}]}})


def test_latest_xml_reads_the_newest_row():
    seen = []

    def fetch(url, accept):
        seen.append(url)
        return _sparql("https://x/file.xml")

    assert latest_xml(SRC, fetch=fetch) == ("2026-01-01", "https://x/file.xml")
    assert "dateApplicability" in unquote_plus(seen[0])


def test_latest_xml_without_a_version_raises():
    with pytest.raises(LookupError):
        latest_xml(SRC, fetch=lambda url, accept: json.dumps({"results": {"bindings": []}}))


def test_fetch_write_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("fedlex_rag.loader.SOURCES", (SRC,))

    def fetch(url, accept):
        return _sparql("https://x/or.xml") if "sparql" in url else XML

    arts, manifest = fetch_all(fetch=fetch)
    assert manifest[0]["articles"] == 2 and manifest[0]["applicable_from"] == "2026-01-01"
    write_fixture(arts, manifest, tmp_path)
    assert load_articles(tmp_path / "articles.jsonl") == arts
    assert "Art. 5 para. 1 let. a URG" in json.loads((tmp_path / "sources.json").read_text("utf-8"))["note"]


def test_load_refuses_unknown_acts(tmp_path):
    row = {"id": "XYZ Art. 1", "act": "XYZ", "number": "1", "heading": "", "context": "", "text": "t", "url": "u"}
    (tmp_path / "a.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_articles(tmp_path / "a.jsonl")


def test_source_range():
    assert BY_ABBR["OR"].keeps(319) and BY_ABBR["OR"].keeps(362)
    assert not BY_ABBR["OR"].keeps(318) and not BY_ABBR["OR"].keeps(363)
    assert BY_ABBR["DSG"].keeps(1)
