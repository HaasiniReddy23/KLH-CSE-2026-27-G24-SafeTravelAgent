from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    source: str  # the file name; shown to the user as the citation
    text: str


def clean_text(text: str) -> str:
    """Tidy spaces and blank lines, but keep paragraph breaks."""
    lines = [" ".join(line.split()) for line in text.splitlines()]
    cleaned = "\n".join(lines)
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip()


def load_documents(folder: Path) -> list[Document]:
    """Load all .txt files from a folder (UTF-8, so Hindi and Telugu work)."""
    documents = []
    for path in sorted(folder.glob("*.txt")):
        text = clean_text(path.read_text(encoding="utf-8"))
        if text:
            documents.append(Document(source=path.name, text=text))
    return documents
