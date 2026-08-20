"""Content-domain prior, derived from the batch 1 corpus audit.

Basis
-----
The corpus is seven YouTube channels across six genres. Each L group corresponds to exactly
one channel, and each channel publishes exactly one kind of content. The genre is therefore
derivable from the ``video_id`` prefix with no inference over the content at all, which is
what makes this prior nearly free. Full measurements are in ``docs/DATA_AUDIT.md`` section
2.1; the table below is the part this module needs.

======  ==========================  ======  ======  ==============================
group   channel                     videos  hours   genre
======  ==========================  ======  ======  ==============================
L21     60 Giây Official (HTV)          29     8.9  television news bulletin
L22     60 Giây Official (HTV)          31    10.2  television news bulletin
L23     HTV Sports                      25     2.7  road cycling race
L24     HTV Sports                      43     6.1  lion dance competition
L25     Báo Thanh Niên                  88    36.2  exam revision lessons (88/88)
L26     ViVU TV                        498    43.8  cooking show
L27     HTV Giải Trí                    16     2.6  travel programme
L28     HTV Entertainment               24     7.6  documentary (Tản Mạn Mê Kông)
L29     HTV Entertainment               23     6.8  documentary (Đôi Mắt Mê Kông)
L30     Báo Tuổi Trẻ                    96     5.7  talk show (95/96)
======  ==========================  ======  ======  ==============================

Three consequences that affect the score:

1. **News bulletins are only 6.9 % of videos and 14.6 % of hours** (L21, L22). The argument
   made in P4 and P5 of ``DESIGN.md`` — on-screen text carries names and job titles, the
   voice-over names identifiable entities — holds only over that fraction. Across 33.5 % of
   hours (cooking) and 27.7 % (exam revision), the text and speech characteristics are
   entirely different. N4 is not a future risk but the present state.

2. **Weighting by video and weighting by hour give different pictures.** L26 is 498 short
   videos (5.3 minutes on average); L25 is 88 long ones (24.7 minutes). Since ``pi_v`` and
   ``max_shots_per_video`` are per video, video count is the right unit for the slot
   allocation layer.

3. **TRAKE queries will almost certainly come from L23 and L24.** TRAKE needs a structured
   event sequence with definable semantic moments — the rules' own example is a high jump:
   approach, take-off, clearance, landing. In batch 1 only cycling and lion dance have that
   structure; cooking has a sequence of steps but much blurrier moment boundaries. Because a
   wrong video multiplies the TRAKE score by zero, shifting probability mass towards the 68
   videos of L23 and L24 (7.8 % of the corpus) is the highest-leverage intervention available
   to that branch.

The prior tilts, it does not filter: no group is ever excluded outright. The cost of guessing
the domain wrong is losing the whole query, while the benefit of guessing right is only a
better rank. That asymmetry forces the prior to stay soft, and is the reason for ``floor``
and ``ceiling``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["GROUP_DOMAIN", "TRAKE_FRIENDLY", "DomainPrior", "group_of"]

#: L group -> (domain label, channel)
GROUP_DOMAIN: dict[str, tuple[str, str]] = {
    "L21": ("news", "60 Giây Official"),
    "L22": ("news", "60 Giây Official"),
    "L23": ("sport_cycling", "HTV Sports"),
    "L24": ("sport_liondance", "HTV Sports"),
    "L25": ("education", "Báo Thanh Niên"),
    "L26": ("cooking", "ViVU TV"),
    "L27": ("travel", "HTV Giải Trí"),
    "L28": ("documentary", "HTV Entertainment"),
    "L29": ("documentary", "HTV Entertainment"),
    "L30": ("talkshow", "Báo Tuổi Trẻ"),
}

#: How clearly a domain carries an event sequence usable for TRAKE. These values are inferred
#: from genre and are **not** calibrated against real labels — see R6 in docs/CONSTRAINTS.md.
TRAKE_FRIENDLY: dict[str, float] = {
    "sport_cycling": 1.0,
    "sport_liondance": 1.0,
    "cooking": 0.35,  # has a sequence of steps, but blurry moment boundaries
    "travel": 0.25,
    "documentary": 0.25,
    "talkshow": 0.15,
    "news": 0.15,  # reportage sometimes has a sequence, but rarely
    "education": 0.10,  # lectures: almost no visual event sequence at all
}

#: Fallback score for a domain missing from TRAKE_FRIENDLY. Deliberately mid-range so an
#: unrecognised domain is neither promoted nor suppressed.
UNKNOWN_TRAKE_FRIENDLINESS = 0.5


def group_of(video_id: str) -> str:
    """``"L26_V444"`` -> ``"L26"``."""
    return video_id.split("_", 1)[0]


@dataclass
class DomainPrior:
    """Per-video prior multiplier, used to tilt ``pi_v`` before slot allocation.

    The prior *multiplies* the retrieval score and then renormalises, so it can never create
    a new candidate — it only reorders the candidates the retrieval layer already found. A
    video that retrieval missed cannot be rescued by the prior.
    """

    #: tilt strength; 0 disables the prior entirely, 1 applies it in full.
    strength: float = 1.0
    #: floor and ceiling of the multiplier, so no group is ever fully excluded.
    floor: float = 0.25
    ceiling: float = 4.0

    def _clamp(self, multiplier: float) -> float:
        if self.strength <= 0.0:
            return 1.0
        scaled = 1.0 + self.strength * (multiplier - 1.0)
        return max(self.floor, min(self.ceiling, scaled))

    def for_trake(self, video_id: str) -> float:
        """Multiplier for the TRAKE branch — tilts hard towards structured-sequence domains."""
        domain, _ = GROUP_DOMAIN.get(group_of(video_id), ("unknown", "?"))
        friendliness = TRAKE_FRIENDLY.get(domain, UNKNOWN_TRAKE_FRIENDLINESS)
        # Rescale around 1.0: friendly domains reach ~2.5x, distant ones ~0.4x.
        return self._clamp(0.4 + 2.1 * friendliness)

    def for_hints(self, video_id: str, domain_hints: list[str]) -> float:
        """Multiplier once query understanding (P7) has inferred a domain from the query.

        No hints returns 1.0 (neutral). That is both the common case and the correct one: when
        the query says nothing about the domain, there is no licence to guess.
        """
        if not domain_hints:
            return 1.0
        domain, _ = GROUP_DOMAIN.get(group_of(video_id), ("unknown", "?"))
        return self._clamp(2.2 if domain in domain_hints else 0.55)

    def apply(
        self,
        scores: dict[str, float],
        *,
        task: str = "kis",
        domain_hints: list[str] | None = None,
    ) -> dict[str, float]:
        """Apply the prior to video-level scores, then renormalise to a distribution."""
        hints = domain_hints or []
        weighted: dict[str, float] = {}
        for video_id, score in scores.items():
            multiplier = self.for_hints(video_id, hints)
            if task == "trake":
                multiplier *= self.for_trake(video_id)
            weighted[video_id] = score * multiplier
        total = sum(weighted.values())
        return {v: s / total for v, s in weighted.items()} if total > 0 else scores

    def report(self) -> str:
        lines = [f"Prior multipliers (strength={self.strength:.2f})", ""]
        lines.append(f"  {'group':<6} {'domain':<18} {'TRAKE':>7}  {'on hint match':>15}")
        for group in sorted(GROUP_DOMAIN):
            domain, _ = GROUP_DOMAIN[group]
            video_id = f"{group}_V001"
            lines.append(
                f"  {group:<6} {domain:<18} {self.for_trake(video_id):>7.2f}  "
                f"{self.for_hints(video_id, [domain]):>15.2f}"
            )
        return "\n".join(lines)
