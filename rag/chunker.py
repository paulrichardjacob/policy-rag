"""Split documents into chunks.

Markdown files are split on headings first, so a chunk never mixes two
sections (a waiting-period rule should not share a chunk with an exclusion).
Long sections are then cut into overlapping word windows.
"""

import re
from dataclasses import asdict, dataclass

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    id: int
    source: str
    section: str
    text: str

    @property
    def indexed_text(self) -> str:
        """Text used for retrieval. The section title is included because it
        often carries the key term (e.g. "Grace period")."""
        return f"{self.section}. {self.text}" if self.section else self.text

    def to_dict(self) -> dict:
        return asdict(self)


def split_sections(text: str) -> list:
    """Return [(section_title, body)] using markdown headings as boundaries."""
    sections, title, lines = [], "", []
    for line in text.splitlines():
        match = HEADING.match(line.strip())
        if match:
            if any(l.strip() for l in lines):
                sections.append((title, "\n".join(lines)))
            title, lines = match.group(2).strip(), []
        else:
            lines.append(line)
    if any(l.strip() for l in lines):
        sections.append((title, "\n".join(lines)))
    return sections


def window(words: list, size: int, overlap: int) -> list:
    """Cut a word list into windows of `size` words that share `overlap` words."""
    if len(words) <= size:
        return [words]
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        out.append(words[start : start + size])
        if start + size >= len(words):
            break
    return out


def chunk_document(text: str, source: str, max_words: int = 120, overlap: int = 30, start_id: int = 0) -> list:
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")
    chunks = []
    for section, body in split_sections(text):
        words = body.split()
        for piece in window(words, max_words, overlap):
            chunks.append(Chunk(start_id + len(chunks), source, section, " ".join(piece)))
    return chunks
