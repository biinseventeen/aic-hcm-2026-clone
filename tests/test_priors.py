"""The domain prior: labels must stay consistent between query understanding and the L-group table.

If the two label sets diverge, a domain hint inferred from a query matches no video and the prior
silently does nothing — a failure that never raises, because everything still runs and still returns
results.
"""

import pytest

from aic.index.priors import GROUP_DOMAIN, TRAKE_FRIENDLY, DomainPrior, group_of
from aic.query.parse import _DOMAIN_KEYWORDS, parse_query


def test_group_of():
    assert group_of("L26_V444") == "L26"
    assert group_of("L21_V001") == "L21"


def test_all_ten_batch_1_groups_are_present():
    assert sorted(GROUP_DOMAIN) == [f"L{index}" for index in range(21, 31)]


def test_every_domain_has_a_trake_score():
    """A domain missing from TRAKE_FRIENDLY would silently fall back to the default."""
    assert {domain for domain, _ in GROUP_DOMAIN.values()} <= set(TRAKE_FRIENDLY)


def test_the_label_sets_match_between_parse_and_priors():
    """The most fragile property whenever a domain is added."""
    assert {domain for domain, _ in GROUP_DOMAIN.values()} == set(_DOMAIN_KEYWORDS)


def test_the_prior_is_neutral_without_a_hint():
    """A query that says nothing about the domain gives no licence to guess."""
    assert DomainPrior().for_hints("L26_V001", []) == 1.0


def test_the_prior_never_excludes_a_group_outright():
    """The cost of guessing wrong is the whole query, so the multiplier needs a positive floor."""
    prior = DomainPrior()
    for group in GROUP_DOMAIN:
        assert prior.for_hints(f"{group}_V001", ["cooking"]) >= prior.floor > 0.0
        assert prior.for_trake(f"{group}_V001") >= prior.floor > 0.0


def test_the_prior_can_be_disabled_entirely():
    prior = DomainPrior(strength=0.0)
    for group in GROUP_DOMAIN:
        assert prior.for_hints(f"{group}_V001", ["cooking"]) == 1.0
        assert prior.for_trake(f"{group}_V001") == 1.0


def test_trake_tilts_towards_the_sports_groups():
    """L23 and L24 are the only batch 1 groups with a definable event sequence."""
    prior = DomainPrior()
    sports = min(prior.for_trake(f"{group}_V001") for group in ("L23", "L24"))
    for group in ("L21", "L22", "L25", "L26", "L27", "L28", "L29", "L30"):
        assert prior.for_trake(f"{group}_V001") < sports, group


def test_apply_returns_a_distribution():
    prior = DomainPrior()
    result = prior.apply({"L26_V001": 0.5, "L23_V001": 0.5}, task="trake")
    assert sum(result.values()) == pytest.approx(1.0)
    # TRAKE: the sports group must be lifted above cooking despite equal raw scores.
    assert result["L23_V001"] > result["L26_V001"]


def test_apply_never_invents_a_candidate():
    result = DomainPrior().apply({"L23_V001": 1.0}, task="kis", domain_hints=["cooking"])
    assert set(result) == {"L23_V001"}


def test_apply_handles_an_all_zero_input():
    assert DomainPrior().apply({"L21_V001": 0.0}) == {"L21_V001": 0.0}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("đua xe đạp về đích chặng 1", "sport_cycling"),
        ("đoàn lân múa mai hoa thung", "sport_liondance"),
        ("đầu bếp nêm nguyên liệu vào chảo", "cooking"),
        ("phóng viên tại buổi họp báo", "news"),
        ("chuyên đề ôn thi THPT môn toán", "education"),
        ("khách mời trò chuyện trong toạ đàm", "talkshow"),
        ("ký sự làng nghề ven sông Mê Kông", "documentary"),
        ("khám phá miệt vườn và đặc sản", "travel"),
    ],
)
def test_domain_hints_are_inferred_from_the_query(text, expected):
    """The query strings stay Vietnamese: they are the language the parser reads."""
    hints = parse_query(text).domain_hints
    assert expected in hints, hints


def test_report_lists_every_group():
    report = DomainPrior().report()
    for group in GROUP_DOMAIN:
        assert group in report
