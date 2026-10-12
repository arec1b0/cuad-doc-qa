"""ground and check_citations: tiers, offsets, normalization and rejection."""

import pytest

from cuad_doc_qa.citations import check_citations, ground
from cuad_doc_qa.domain import Answer, Chunk, Citation

TEXT = (
    "THIS  DISTRIBUTOR  AGREEMENT is made by Electric City Corp. (“Company”)\n"
    "and the Distributor. This Agreement shall be governed by the laws of the State of "
    "Delaware — without regard to conflict of laws. The Distributor shall pay on time."
)
CHUNK = Chunk(chunk_id="D::0003", doc_id="D", start=1000, end=1000 + len(TEXT), text=TEXT)
OTHER = Chunk(chunk_id="D::0004", doc_id="D", start=5000, end=5013, text="Term: 5 years")


def _ground(quote: str, chunk: Chunk = CHUNK):
    return ground(Citation(chunk.chunk_id, quote), chunk)


def _local(substring: str) -> tuple[int, int]:
    start = TEXT.index(substring)
    return CHUNK.start + start, CHUNK.start + start + len(substring)


def test_verbatim_quote_gets_document_offsets() -> None:
    grounded = _ground("governed by the laws of the State of Delaware")

    assert grounded is not None
    assert (grounded.start, grounded.end) == _local("governed by the laws of the State of Delaware")
    assert grounded.text == "governed by the laws of the State of Delaware"
    assert grounded.doc_id == "D"
    assert (grounded.fuzzy, grounded.score) == (False, 100.0)


def test_whitespace_quotes_and_dashes_are_normalized_but_text_stays_original() -> None:
    grounded = _ground('This distributor agreement is made by Electric City Corp. ("Company")')

    assert grounded is not None
    assert (
        grounded.text == "THIS  DISTRIBUTOR  AGREEMENT is made by Electric City Corp. (“Company”)"
    )
    assert not grounded.fuzzy

    dashed = _ground("State of Delaware - without regard")
    assert dashed is not None and dashed.text == "State of Delaware — without regard"


def test_case_sensitive_occurrence_is_preferred_over_an_earlier_one() -> None:
    grounded = _ground("Distributor")

    assert grounded is not None
    assert (grounded.start, grounded.end) == _local("Distributor")
    assert grounded.text == "Distributor"


def test_quote_with_a_typo_is_grounded_fuzzily_on_the_original_text() -> None:
    grounded = _ground("governed by the laws of the State of Delawre")

    assert grounded is not None
    assert grounded.fuzzy
    assert 90.0 <= grounded.score < 100.0
    start, end = _local("governed by the laws of the State of Delaware")
    assert grounded.start < end and start < grounded.end
    assert grounded.text == TEXT[grounded.start - CHUNK.start : grounded.end - CHUNK.start]


@pytest.mark.parametrize(
    "quote",
    [
        "Delawre law",
        "The agreement is subject to New York law and arbitration in Manhattan.",
        "   \n ",
        "",
    ],
    ids=["short-typo", "paraphrase", "whitespace", "empty"],
)
def test_unlocatable_quotes_are_rejected(quote: str) -> None:
    assert _ground(quote) is None


def test_quote_must_be_in_the_chunk_it_cites() -> None:
    assert _ground("Term: 5 years") is None
    assert _ground("Term: 5 years", OTHER) is not None


def test_non_ascii_lowercasing_keeps_offsets_aligned() -> None:
    text = "Seat of arbitration: İstanbul, Turkey."
    chunk = Chunk(chunk_id="T::0000", doc_id="T", start=0, end=len(text), text=text)

    grounded = ground(Citation("T::0000", "İSTANBUL, turkey"), chunk)

    assert grounded is not None
    assert grounded.text == "İstanbul, Turkey"


def test_check_partitions_citations_in_claim_order() -> None:
    answer = Answer(
        found=True,
        text="Delaware law governs.",
        citations=(
            Citation("D::0003", "laws of the State of Delaware"),
            Citation("D::0003", "governed by English law"),
            Citation("D::0099", "laws of the State of Delaware"),
            Citation("D::0004", "Term: 5 years"),
        ),
    )

    check = check_citations(answer, [CHUNK, OTHER])

    assert [g.citation for g in check.grounded] == [answer.citations[0], answer.citations[3]]
    assert check.rejected == (answer.citations[1], answer.citations[2])


def test_answer_without_citations_has_nothing_to_check() -> None:
    check = check_citations(Answer(found=False, text="", citations=()), [CHUNK])

    assert check.grounded == () and check.rejected == ()
