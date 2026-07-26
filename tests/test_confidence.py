from src.confidence import completeness


def test_completeness_trailing_citation_gets_full_credit():
    assert completeness("ERR_2043 means the upload was dropped. [1]") == 1.0


def test_completeness_all_sentences_cited():
    assert completeness("The default is 3 [1]. The endpoint is /v1/health [2].") == 1.0


def test_completeness_partial_citations():
    assert completeness("This has no citation. This one does [1].") == 0.5


def test_completeness_no_citations_at_all():
    assert completeness("No citations here at all.") == 0.0


def test_completeness_empty_answer_returns_zero():
    assert completeness("") == 0.0
