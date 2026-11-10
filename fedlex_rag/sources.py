"""The enactments this project reads: three Swiss federal acts, selected by SR number."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    abbr: str          # the citation prefix: "DSG Art. 5"
    title: str
    sr: str            # Systematische Rechtssammlung number
    eli: str           # the consolidated act on fedlex
    first: int | None = None   # keep only articles numbered first..last (None = all)
    last: int | None = None

    def keeps(self, number: int) -> bool:
        return (self.first is None or number >= self.first) and (self.last is None or number <= self.last)

    def article_url(self, eid: str) -> str:
        return f"https://www.fedlex.admin.ch/{self.eli.split('fedlex.data.admin.ch/', 1)[1]}/de#{eid}"


SOURCES: tuple[Source, ...] = (
    Source("DSG", "Bundesgesetz über den Datenschutz", "235.1",
           "https://fedlex.data.admin.ch/eli/cc/2022/491"),
    Source("OR", "Obligationenrecht -- der Arbeitsvertrag (Art. 319-362)", "220",
           "https://fedlex.data.admin.ch/eli/cc/27/317_321_377", first=319, last=362),
    Source("KVG", "Bundesgesetz über die Krankenversicherung", "832.10",
           "https://fedlex.data.admin.ch/eli/cc/1995/1328_1328_1328"),
)

BY_ABBR = {s.abbr: s for s in SOURCES}
