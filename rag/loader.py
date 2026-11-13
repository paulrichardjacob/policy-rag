"""Read .md, .txt and .pdf files into plain text."""

from pathlib import Path

SUPPORTED = {".md", ".txt", ".pdf"}


def load_file(path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError(f"Unsupported file type: {path.name}")


def load_folder(folder) -> dict:
    """Return {filename: text} for every supported file in the folder."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Document folder not found: {folder}")
    docs = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            text = load_file(path)
            if text.strip():
                docs[path.name] = text
    if not docs:
        raise ValueError(f"No .md, .txt or .pdf files with text found in {folder}")
    return docs
