from pathlib import Path

from bs4 import BeautifulSoup

from src.models import Document


def _detect_format(path: Path) -> str:
    p = path
    return p.suffix[1:]


def _read_normalized(path: Path) -> str:
    """Read a text file and normalize Windows line endings to \n."""
    raw = path.read_text()
    return raw.replace("\r\n", "\n")


def _html_to_clean_text(html: str) -> str:
    """Strip script/style, convert h1-h6 to markdown headings, extract text."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    for level in range(1, 7):
        for heading in soup.find_all(f"h{level}"):
            heading.replace_with(f"\n{'#' * level} {heading.get_text(strip=True)}\n")

    return soup.get_text(separator=" ").strip()


def load_file(path: Path, base_dir: Path) -> Document:
    """Load a single file and return a normalized Document."""
    fmt = _detect_format(path)
    source_name = path.relative_to(base_dir).as_posix()

    if fmt in ("md", "txt"):
        clean_text = _read_normalized(path)
        raw_text = path.read_text()
        metadata = {}
    elif fmt == "html":
        raw_text = path.read_text()
        clean_text = _html_to_clean_text(raw_text)
        metadata = {}
    else:
        raise ValueError(f"unsupported format: {fmt}")

    return Document(
        doc_id=source_name,
        source_path=str(path),
        source_name=source_name,
        fmt=fmt,
        raw_text=raw_text,
        clean_text=clean_text,
        metadata=metadata,
    )
