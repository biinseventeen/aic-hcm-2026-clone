"""The service layer: the contract a backend relies on.

Two properties are locked here:

1. :meth:`SolveResult.to_dict` returns **plain JSON** — no numpy, no dataclasses. If that breaks,
   the backend dies while serialising the response, i.e. after the work is already done.
2. The engine fails **clearly and early** when the index is absent, rather than with a vague error
   part-way through.
"""

import json

import pytest

from aic.config import Config
from aic.core.allocator import FrameCandidate, allocate_kis
from aic.core.coverage import CoverageModel, Locus, VideoBelief
from aic.query.parse import parse_query
from aic.service import Engine, EngineStatus, SolveResult
from aic.submit.writer import Answer, QuerySubmission


def make_result(task="kis", n=4) -> SolveResult:
    """Build a real SolveResult through the allocator — no mocks, so to_dict sees real data."""
    model = CoverageModel.from_beliefs(
        [
            VideoBelief("L21_V001", 0.6, [Locus(0, 199)]),
            VideoBelief("L21_V002", 0.4, [Locus(0, 199)]),
        ],
        answer_len=25,
    )
    candidates = [
        FrameCandidate(video_id, frame, score=0.5, source=f"shot#{frame}")
        for video_id in ("L21_V001", "L21_V002")
        for frame in (24, 49, 74, 99)
    ]
    trace = allocate_kis(model, candidates, budget=n)
    answers = []
    for allocation in trace.allocations:
        if task == "trake":
            answers.append(
                Answer(allocation.video_id, frames=(allocation.frame_id, allocation.frame_id + 50))
            )
        elif task == "qa":
            answers.append(Answer(allocation.video_id, frame=allocation.frame_id, answer="5"))
        else:
            answers.append(Answer(allocation.video_id, frame=allocation.frame_id))
    return SolveResult(
        query_id="q1",
        task=task,
        submission=QuerySubmission("q1", task, answers),
        trace=trace,
        query=parse_query("a speaker in a red shirt"),
        n_candidates=len(candidates),
        channel_sizes={"dense_original": 2000},
        elapsed_s=1.234,
        degraded=True,
        notes=["test note"],
    )


# --------------------------------------------------------------------------
# SolveResult
# --------------------------------------------------------------------------


def test_to_dict_is_json_serialisable():
    for task in ("kis", "qa", "trake"):
        payload = make_result(task).to_dict()
        # A round trip through json is the real test: a numpy scalar would make it raise.
        assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload


def test_to_dict_carries_every_key_a_backend_needs():
    payload = make_result().to_dict()
    assert set(payload) >= {
        "query_id",
        "task",
        "degraded",
        "n_answers",
        "elapsed_s",
        "query",
        "retrieval",
        "allocation",
        "answers",
        "notes",
    }
    assert set(payload["allocation"]) >= {
        "expected_final",
        "coverage_at_k",
        "n_distinct_videos",
        "exhausted_at",
    }
    assert set(payload["query"]) >= {"raw", "task_confidence", "parser", "keywords"}


def test_to_dict_preserves_order_and_rank():
    """Order is the score — if the response reorders, the client submits the wrong thing."""
    payload = make_result(n=4).to_dict()
    assert [answer["rank"] for answer in payload["answers"]] == [1, 2, 3, 4]


def test_to_dict_shape_per_task():
    kis = make_result("kis").to_dict()["answers"][0]
    assert "frame_id" in kis
    assert "frames" not in kis
    assert "answer" not in kis

    qa = make_result("qa").to_dict()["answers"][0]
    assert qa["answer"] == "5"
    assert "frame_id" in qa

    trake = make_result("trake").to_dict()["answers"][0]
    assert isinstance(trake["frame_ids"], list)
    assert len(trake["frame_ids"]) == 2


def test_top_limits_the_answers_returned():
    result = make_result(n=4)
    assert len(result.to_dict(top=2)["answers"]) == 2
    # ``top`` only truncates the response; it does not change what was computed.
    assert result.to_dict(top=2)["n_answers"] == 4
    assert len(result.to_dict()["answers"]) == 4


def test_notes_combine_the_trace_and_the_service():
    result = make_result()
    result.trace.notes.append("from the trace")
    notes = result.to_dict()["notes"]
    assert "from the trace" in notes
    assert "test note" in notes


def test_rows_match_the_submission_format():
    assert make_result("kis").rows()[0][0].startswith("L21_V")
    assert len(make_result("kis").rows()[0]) == 2
    assert len(make_result("qa").rows()[0]) == 3
    assert len(make_result("trake").rows()[0]) == 3  # video + 2 frames


def test_report_is_empty_without_a_solution_object():
    assert make_result().report() == ""


def test_expected_final_and_n_answers():
    result = make_result(n=4)
    assert result.n_answers == 4
    assert 0.0 <= result.expected_final <= 1.0


# --------------------------------------------------------------------------
# EngineStatus
# --------------------------------------------------------------------------


def test_status_is_not_ready_when_the_encoder_is_a_stub():
    status = EngineStatus(
        ready=False,
        data_root="d",
        index_dir="i",
        n_videos=873,
        n_keyframes=177321,
        dense_dim=512,
        encoder="stub",
        encoder_is_stub=True,
        has_text_index=True,
        n_shots=1000,
        warnings=["stub"],
    )
    payload = status.to_dict()
    assert payload["ready"] is False
    assert payload["encoder_is_stub"] is True
    assert json.loads(json.dumps(payload)) == payload


# --------------------------------------------------------------------------
# Engine.load
# --------------------------------------------------------------------------


def test_load_names_the_missing_index_clearly(tmp_path):
    cfg = Config().resolve(tmp_path)
    with pytest.raises(FileNotFoundError, match="build-index"):
        Engine.load(cfg)


def test_load_fails_when_there_is_no_data(tmp_path):
    (tmp_path / "data" / "processed" / "index").mkdir(parents=True)
    cfg = Config().resolve(tmp_path)
    with pytest.raises(FileNotFoundError):
        Engine.load(cfg)
