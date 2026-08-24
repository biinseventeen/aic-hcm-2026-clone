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


# --------------------------------------------------------------------------
# TRAKE moment labels — the form the organisers publish
# --------------------------------------------------------------------------


def test_e_labelled_moments_route_to_trake(parser: RuleParser):
    """The published set writes moments as "E1: ...", with no numbered-list markers at all."""
    query = parser.parse(
        "E1: Khoảnh khắc đầu tiên bột được bỏ vào tô.\n"
        "E2: Khoảnh khắc miếng đầu tiên tiếp xúc với dầu.\n"
        "E3: Khoảnh khắc miếng đầu tiên rời khỏi chảo.\n"
        "E4: Khoảnh khắc miếng cuối cùng nằm trên dĩa."
    )
    assert query.task == "trake"
    assert query.n_moments == 4
    assert query.moments[0].startswith("Khoảnh khắc đầu tiên bột")


def test_the_lead_in_sentence_is_not_counted_as_a_moment(parser: RuleParser):
    """N is fixed by the query: counting the lead-in would submit one frame too many."""
    query = parser.parse(
        "Đoạn video múa lân, tìm các sự kiện sau:\n"
        "E1: Lân bắt đầu xoay vòng trên cột.\n"
        "E2: Khoảnh khắc 4 chân chạm đất.\n"
        "E3: Hai người biểu diễn chào ban giám khảo."
    )
    assert query.task == "trake"
    assert query.n_moments == 3


def test_duplicate_label_values_do_not_change_the_moment_count(parser: RuleParser):
    """The published set numbers one query E1, E2, E2, E4 — it still has four moments."""
    query = parser.parse(
        "Trong đoạn video nấu ăn, gồm các khoảnh khắc sơ chế:\n"
        "E1: Khoảnh khắc đầu tiên thấy cắt nấm.\n"
        "E2: Khoảnh khắc đầu tiên cắt củ năng.\n"
        "E2: Khoảnh khắc đầu tiên cắt đậu hủ.\n"
        "E4: Khoảnh khắc chảo đặt lên bếp."
    )
    assert query.n_moments == 4
    assert "đậu hủ" in query.moments[2]


# --------------------------------------------------------------------------
# Query-side term selection for the sparse channel
# --------------------------------------------------------------------------


def test_informative_terms_drops_instruction_words_and_keeps_the_subject():
    """Corpus IDF rates "đoạn clip" as rare and therefore informative. It is neither."""
    from aic.index.text import TextDoc, TextIndex

    docs = [
        TextDoc(0, "L26_V004", "PANNA COTTA KEM MUỐI MÓN NGON MỖI NGÀY", source="media-info"),
        TextDoc(1, "L26_V005", "CANH CHUA CÁ BÔNG LAU MÓN NGON MỖI NGÀY", source="media-info"),
        TextDoc(2, "L27_V013", "Việt Nam đi là ghiền khám phá món ăn cảnh đẹp mọi miền"),
    ]
    index = TextIndex(docs=docs).build(with_fuzzy=False)
    terms = index.informative_terms(
        "Đoạn clip cần tìm là cảnh 3 ly panna cotta trên đĩa", max_terms=4
    )
    assert "panna" in terms and "cotta" in terms
    assert not {"đoạn", "clip", "cảnh", "3"} & set(terms)


def test_titles_are_indexed_without_the_description():
    from aic.index.text import docs_from_titles

    media = {
        "L30_V072": {
            "title": "12 năm mang yêu thương trao nơi vùng xa",
            "description": "Đăng ký kênh https://example.com #hashtag TÒA SOẠN Địa chỉ ...",
            "keywords": ["tin tức", "thời sự"],
        },
        "L30_V073": {"title": "", "description": "no title, so no document"},
    }
    docs = docs_from_titles(media)
    assert [doc.video_id for doc in docs] == ["L30_V072"]
    assert docs[0].text == "12 năm mang yêu thương trao nơi vùng xa"


# --------------------------------------------------------------------------
# A video-level text hit must not become the video's title sequence
# --------------------------------------------------------------------------


def test_a_video_level_hit_does_not_collapse_onto_the_first_shots():
    """Measured defect: 37 % of submitted rows landed in the first five seconds of their video.

    Every episode of a series shares its title sequence, so expanding a BM25 hit into the first
    twelve shots put interchangeable credits into the most valuable slots — and the one query whose
    correct video was found had its frame inside the intro, which scores zero.
    """
    import numpy as np

    from aic.index.shots import Shot, ShotTable
    from aic.query.retrieve import Retriever

    n_shots = 40
    shots = [
        Shot(shot_id=i, video_id="L26_V004", start=i * 100, end=i * 100 + 99, keyframes=(i + 1,))
        for i in range(n_shots)
    ]
    table = ShotTable(shots={"L26_V004": shots})

    class FakeDense:
        dim = 4

        def __init__(self):
            # One row per shot; only the row of shot 30 matches the query direction.
            self.vectors = np.zeros((n_shots, 4), dtype=np.float32)
            self.vectors[:, 1] = 1.0
            self.vectors[30] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        def video_rows(self, video_id):  # noqa: ARG002 — signature must match DenseIndex
            return np.arange(n_shots)

        def decode(self, row):
            return ("L26_V004", row + 1, row * 100 + 50)

    retriever = Retriever(dense=FakeDense(), shots=table, max_shots_per_video=5)
    query_vector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    chosen = retriever._shots_for_video_level_hit("L26_V004", query_vector)
    assert chosen[0] == 30, "the shot matching the query must come first, not shot 0"

    # With no encoder the fallback must still not pile onto the opening shots.
    fallback = retriever._shots_for_video_level_hit("L26_V004", None)
    assert fallback[:5] != [0, 1, 2, 3, 4], "the fallback must spread over the video"
    assert max(fallback) > n_shots // 2, "the fallback must reach the second half of the video"
