from src.llm import extract_claims


def test_extract_claims_trailing_citation_merges_into_sentence():
    claims = extract_claims("ERR_2043 means the upload was dropped. [1]")
    assert len(claims) == 1
    claim, citation_numbers = claims[0]
    assert claim == "ERR_2043 means the upload was dropped. [1]"
    assert citation_numbers == [1]


def test_extract_claims_inline_citation_stays_attached():
    claims = extract_claims("The default is 3 [1]. The endpoint is /v1/health [2].")
    assert len(claims) == 2
    assert claims[0] == ("The default is 3 [1].", [1])
    assert claims[1] == ("The endpoint is /v1/health [2].", [2])


def test_extract_claims_sentence_without_citation_is_dropped():
    claims = extract_claims("This has no citation. This one does [1].")
    assert len(claims) == 1
    assert claims[0] == ("This one does [1].", [1])


def test_extract_claims_no_citations_returns_empty():
    assert extract_claims("No citations here at all.") == []
