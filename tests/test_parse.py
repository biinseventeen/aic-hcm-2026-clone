"""P7 — query understanding.

The property under test is that the *entity* channel is only fed real names. Its input is a
fuzzy character n-gram match over OCR strings, so a phrase that is not a name does not fail
loudly: it returns plausible-looking hits for the wrong reason, and it also flips
``has_named_entity`` in the channel weighting. Both failures are silent, which is why they are
locked by tests rather than left to inspection.
"""

from __future__ import annotations

import pytest

from aic.query.parse import RuleParser, _vietnamese_uppercase, parse_query


@pytest.fixture
def parser() -> RuleParser:
    return RuleParser()


def test_uppercase_class_holds_no_lowercase_letters():
    # The range À-Ỹ would: it spans blocks where the two cases interleave.
    upper = _vietnamese_uppercase()
    assert not [character for character in upper if character.islower()]
    assert {"Đ", "Ỹ", "Ê", "Ơ"} <= set(upper)


@pytest.mark.parametrize(
    "query",
    [
        "người đàn ông mặc áo đỏ đang nấu ăn",
        "đầu bếp cho nguyên liệu vào chảo dầu nóng",
        "cảnh quay từ trên cao xuống một con đường",
    ],
)
def test_all_lowercase_queries_yield_no_entities(parser: RuleParser, query: str):
    """A query with no capital letter contains no proper noun, whatever its diacritics."""
    assert parser._entities(query) == []


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Nguyễn Văn A phát biểu tại Hà Nội", ["Nguyễn Văn A", "Hà Nội"]),
        # "phố" is lowercase, so the capitalised run breaks; the "Thành" fragment is dropped
        # rather than handed to the fuzzy channel as a name.
        ("phóng viên đưa tin ở Thành phố Hồ Chí Minh", ["Hồ Chí Minh"]),
        ("biển số xe 51F-123.45 trên phố", ["51F-123.45"]),
        ("Hồ Chí Minh và Hà Nội", ["Hồ Chí Minh", "Hà Nội"]),
    ],
)
def test_real_names_and_figures_are_still_extracted(
    parser: RuleParser, query: str, expected: list[str]
):
    assert parser._entities(query) == expected


def test_entity_flag_follows_the_extraction(parser: RuleParser):
    """The channel weighting reads ``entities``; an over-eager extractor silently reweights it."""
    assert parse_query("người đàn ông mặc áo đỏ").entities == []
    assert parse_query("ông Nguyễn Văn A mặc áo đỏ").entities == ["Nguyễn Văn A"]


def test_leading_capital_of_a_sentence_is_not_an_entity(parser: RuleParser):
    assert "Tìm" not in parser._entities("Tìm cảnh một người đang chạy")


def test_a_generic_head_noun_alone_is_not_an_entity(parser: RuleParser):
    """Single-word only: the filter must not swallow a name that begins with such a word."""
    assert parser._entities("cảnh quay bên Hồ Tây") == ["Hồ Tây"]
    assert parser._entities("cảnh quay bên Hồ và bờ kè") == []
