"""Load lecture passages from the course-notes HTML files."""
from pathlib import Path

from bs4 import BeautifulSoup


def load_notes(folder):
    """Return a list of passages, each a dict with week, section and text.

    Only the lecture content is kept: each <h3> section heading and its paragraph.
    Boilerplate such as "Check your understanding" prompts, discussion
    instructions and assignment details is skipped, because questions about
    those would not test the course material.
    """
    passages = []
    for path in sorted(Path(folder).glob("*.html")):
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        week = soup.find("h1").get_text(strip=True)
        for h3 in soup.find_all("h3"):
            paragraphs = []
            for sibling in h3.find_next_siblings():
                if sibling.name in ("h2", "h3"):
                    break
                text = sibling.get_text(" ", strip=True)
                if sibling.name == "p" and not text.startswith("Check your understanding"):
                    paragraphs.append(text)
            if paragraphs:
                passages.append({
                    "week": week,
                    "section": h3.get_text(strip=True),
                    "text": " ".join(paragraphs),
                })
    return passages