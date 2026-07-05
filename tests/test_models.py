from src.models import Document, Chunk

def test_document_construction():
    doc = Document(
        doc_id="d1", source_path="corpus/foo.md", source_name="foo.md",
        fmt="md", raw_text="# Foo\nbar", clean_text="# Foo\nbar", metadata={},
    )
    assert doc.fmt == "md"

def test_chunk_to_metadata_excludes_text():
    chunk = Chunk(
        chunk_id="c1", doc_id="d1", source_name="foo.md", chunk_index=0,
        text="bar", section_heading="Foo", chunking_strategy="fixed",
        char_count=3, token_count=1,end_char=199,start_char=1
    )
    meta = chunk.to_metadata()
    assert "text" not in meta
    assert meta["chunk_index"] == 0
    assert meta["section_heading"] == "Foo"