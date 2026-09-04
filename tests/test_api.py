"""The HTTP layer: input validation at the boundary, and no web framework in the core package.

The validation tests run **without** fastapi — that is the point: the rules live in
standard-library dataclasses so they apply to HTTP callers and in-process callers alike. The
application tests skip themselves when ``.[backend]`` is not installed.
"""

import importlib.util
import json
import zipfile
from typing import ClassVar

import pytest

from aic.api.preview import PreviewLookupError, enrich_for_ui, require_video_id
from aic.api.schemas import (
    MAX_BATCH,
    MAX_QUERY_CHARS,
    BatchSolveRequest,
    ErrorResponse,
    HealthResponse,
    PackageRequest,
    ReviewRequest,
    SolveRequest,
    SolveResponse,
    SubmitRequest,
)
from aic.submit.writer import MAX_ANSWER_CHARS, package_submission

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None
HAS_TEST_CLIENT = HAS_FASTAPI and importlib.util.find_spec("httpx") is not None


# --------------------------------------------------------------------------
# The core package must stay independent of any web framework
# --------------------------------------------------------------------------


def test_importing_aic_does_not_pull_in_fastapi():
    """``import aic.service`` must work in an environment with no web packages at all."""
    import os
    import subprocess
    import sys

    from aic.config import find_project_root

    code = (
        "import sys, aic.service, aic.api, aic.config, aic.submit.writer;"
        "assert 'fastapi' not in sys.modules, sorted(m for m in sys.modules if 'fast' in m)"
    )
    # A subprocess does not inherit pytest's pythonpath, so pass it explicitly.
    env = {**os.environ, "PYTHONPATH": str(find_project_root() / "src")}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
    )
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def test_a_valid_request_is_normalised():
    request = SolveRequest(text="  a speaker in a red shirt  ", query_id=" ").validated()
    assert request.text == "a speaker in a red shirt"
    assert request.query_id == "1"  # blank falls back to the default
    assert request.task is None
    assert request.top is None
    assert request.hedge_answers is True
    assert request.answers == []
    assert request.pins == []


def test_empty_text_is_rejected():
    for bad in ("", "   ", None):
        with pytest.raises(ValueError, match="text"):
            SolveRequest(text=bad).validated()  # type: ignore[arg-type]


def test_overlong_text_is_rejected():
    SolveRequest(text="a" * MAX_QUERY_CHARS).validated()
    with pytest.raises(ValueError, match="over the limit"):
        SolveRequest(text="a" * (MAX_QUERY_CHARS + 1)).validated()


def test_an_unknown_task_is_rejected():
    with pytest.raises(ValueError, match="task"):
        SolveRequest(text="x", task="ocr").validated()  # type: ignore[arg-type]
    for task in ("kis", "qa", "trake"):
        assert SolveRequest(text="x", task=task).validated().task == task


def test_qna_is_an_alias_of_qa():
    """Frontend pages send ``qna``; submission filenames and the engine use ``qa``."""
    assert SolveRequest(text="x", task="qna").validated().task == "qa"
    assert ReviewRequest(text="x", task="qna").validated().task == "qa"


def test_answers_and_pins_are_accepted():
    request = SolveRequest(
        text="x",
        answers=[{"text": "Giang Ly", "prob": 0.5}],
        pins=[{"video_id": "L30_V072", "frame": 676}],
    ).validated()
    assert request.engine_answers() == [("Giang Ly", 0.5)]
    assert request.engine_pins() == [("L30_V072", 676)]


def test_an_overlong_answer_is_rejected():
    with pytest.raises(ValueError, match="100"):
        SolveRequest(text="x", answers=[{"text": "a" * (MAX_ANSWER_CHARS + 1)}]).validated()


def test_a_malformed_or_duplicate_pin_is_rejected():
    with pytest.raises(ValueError, match="video_id"):
        SolveRequest(text="x", pins=[{"video_id": "../etc/passwd", "frame": 1}]).validated()
    with pytest.raises(ValueError, match="video_id"):
        SolveRequest(text="x", pins=[{"video_id": "L21_V001/../../x", "frame": 1}]).validated()
    with pytest.raises(ValueError, match="frame"):
        SolveRequest(text="x", pins=[{"video_id": "L21_V001", "frame": -1}]).validated()
    with pytest.raises(ValueError, match="duplicate"):
        SolveRequest(
            text="x",
            pins=[
                {"video_id": "L21_V001", "frame": 1},
                {"video_id": "L21_V001", "frame": 1},
            ],
        ).validated()


def test_package_request_rejects_empty_or_unsafe_ids():
    with pytest.raises(ValueError, match="empty"):
        PackageRequest([]).validated()
    with pytest.raises(ValueError, match="query_id"):
        PackageRequest(["../etc/passwd"]).validated()
    with pytest.raises(ValueError, match="set_name"):
        PackageRequest(["p1-1"], set_name="..").validated()
    ok = PackageRequest(["p1-1"], zip_name="team_round1").validated()
    assert ok.zip_name == "team_round1.zip"


def test_submit_request_keeps_strict_default():
    request = SubmitRequest(text="x").validated()
    assert request.strict is True
    assert SubmitRequest(text="x", strict=False).validated().strict is False


def test_require_video_id_rejects_traversal():
    with pytest.raises(PreviewLookupError) as caught:
        require_video_id("../etc/passwd")
    assert caught.value.status == 404
    with pytest.raises(PreviewLookupError):
        require_video_id("L21_V001/../../secret")
    with pytest.raises(PreviewLookupError):
        require_video_id("L21_V001", known={"L22_V001"})
    assert require_video_id("L21_V001") == "L21_V001"
    assert require_video_id("L21_V001", known={"L21_V001"}) == "L21_V001"


class _DummyEngine:
    tables: ClassVar[dict] = {}


def test_enrich_for_ui_adds_trake_frame_id_and_milestones_without_inventing_names():
    payload = {
        "task": "trake",
        "query": {"moments": ["take-off"]},
        "answers": [{"rank": 1, "video_id": "L23_V001", "frame_ids": [101, 156]}],
    }
    enrich_for_ui(payload, _DummyEngine(), base_url="http://example")
    answer = payload["answers"][0]
    assert answer["frame_id"] == 101
    assert [item["frameId"] for item in answer["milestones"]] == [101, 156]
    assert answer["milestones"][0]["stepName"] == "take-off"
    assert answer["milestones"][1]["stepName"] == ""  # no invented name


def test_enrich_for_ui_builds_qna_envelope_from_rank_one():
    payload = {
        "task": "qa",
        "answers": [
            {
                "rank": 1,
                "video_id": "L30_V072",
                "frame_id": 676,
                "answer": "Giang Ly",
                "gain": 0.5,
            }
        ],
    }
    enrich_for_ui(payload, _DummyEngine(), base_url="http://example")
    envelope = payload["qna_answer"]
    assert envelope["answer_text"] == "Giang Ly"
    assert envelope["source_segment"] == "L30_V072"
    assert envelope["evidence_frames"][0]["frame_id"] == 676
    assert payload["answers"][0]["rank"] == 1  # ranking contract unchanged


def test_top_outside_its_range_is_rejected():
    for bad in (0, -1, 101):
        with pytest.raises(ValueError, match="top"):
            SolveRequest(text="x", top=bad).validated()
    assert SolveRequest(text="x", top=100).validated().top == 100


def test_a_query_id_may_not_contain_path_characters():
    """``query_id`` becomes a submission filename — block directory escapes at the boundary."""
    for bad in ("../../etc/passwd", "a/b", "a\\b", "c:x", "..", ".", "a*b", "a?b", 'a"b'):
        with pytest.raises(ValueError, match="query_id"):
            SolveRequest(text="x", query_id=bad).validated()
    assert SolveRequest(text="x", query_id="query-12").validated().query_id == "query-12"


def test_an_empty_or_oversized_batch_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        BatchSolveRequest([]).validated()
    ok = [SolveRequest(text="x", query_id=str(i)) for i in range(MAX_BATCH)]
    assert len(BatchSolveRequest(ok).validated().queries) == MAX_BATCH
    with pytest.raises(ValueError, match="over the limit"):
        BatchSolveRequest([*ok, SolveRequest(text="x", query_id="extra")]).validated()


def test_a_duplicate_query_id_in_a_batch_is_rejected():
    """Duplicate ids mean two queries write to the same submission file."""
    duplicated = [SolveRequest(text="a", query_id="1"), SolveRequest(text="b", query_id="1")]
    with pytest.raises(ValueError, match="duplicate"):
        BatchSolveRequest(duplicated).validated()


def test_a_batch_propagates_per_query_errors():
    with pytest.raises(ValueError, match="text"):
        BatchSolveRequest([SolveRequest(text="ok"), SolveRequest(text="")]).validated()


# --------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------


def test_solve_response_lifts_degraded_to_the_top_level():
    from test_service import make_result  # reuse the real result builder

    response = SolveResponse.of(make_result(), top=2).to_dict()
    assert response["degraded"] is True
    assert len(response["answers"]) == 2
    assert json.loads(json.dumps(response, ensure_ascii=False)) == response


def test_health_response_drops_empty_fields():
    assert HealthResponse("loading").to_dict() == {"status": "loading"}
    assert HealthResponse("ready", engine={"ready": True}).to_dict() == {
        "status": "ready",
        "engine": {"ready": True},
    }


def test_error_response_drops_empty_fields():
    assert ErrorResponse("x").to_dict() == {"error": "x"}
    assert ErrorResponse("x", "because").to_dict() == {"error": "x", "detail": "because"}


# --------------------------------------------------------------------------
# The application — skipped when .[backend] is not installed
# --------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TEST_CLIENT, reason="needs .[backend] and httpx")
def test_cors_allows_vite_when_it_hops_off_5173():
    from fastapi.testclient import TestClient

    from aic.api.app import create_app, reset_engine

    reset_engine()
    origin = "http://localhost:5174"
    with TestClient(create_app(eager=False)) as client:
        health = client.get("/health", headers={"Origin": origin})
        assert health.status_code == 200
        assert health.headers.get("access-control-allow-origin") == origin

        preflight = client.options(
            "/solve",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert preflight.status_code in (200, 204)
        assert preflight.headers.get("access-control-allow-origin") == origin


@pytest.mark.skipif(not HAS_TEST_CLIENT, reason="needs .[backend] and httpx")
def test_health_answers_before_the_engine_is_loaded():
    from fastapi.testclient import TestClient

    from aic.api.app import create_app, reset_engine

    reset_engine()
    with TestClient(create_app(eager=False)) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] in ("loading", "ready", "degraded", "error")


@pytest.mark.skipif(not HAS_TEST_CLIENT, reason="needs .[backend] and httpx")
def test_solve_returns_422_on_invalid_input():
    from fastapi.testclient import TestClient

    from aic.api.app import create_app, reset_engine

    reset_engine()
    with TestClient(create_app(eager=False)) as client:
        assert client.post("/solve", json={"text": "   "}).status_code == 422
        assert client.post("/solve", json={"text": "x", "top": 999}).status_code == 422


@pytest.mark.skipif(not HAS_TEST_CLIENT, reason="needs .[backend] and httpx")
def test_review_and_submit_reject_empty_text_without_loading_the_engine():
    from fastapi.testclient import TestClient

    from aic.api.app import create_app, reset_engine

    reset_engine()
    with TestClient(create_app(eager=False)) as client:
        assert client.post("/review", json={"text": ""}).status_code == 422
        assert client.post("/submit", json={"text": ""}).status_code == 422
        assert client.post("/submit/package", json={"query_ids": []}).status_code == 422


def test_package_submission_puts_csvs_inside_a_submission_directory(tmp_path):
    first = tmp_path / "query-p1-1-kis.csv"
    second = tmp_path / "query-p1-2-qa.csv"
    first.write_text("L21_V001,10\r\n", encoding="utf-8", newline="")
    second.write_text("L21_V002,20,hello\r\n", encoding="utf-8", newline="")
    archive = package_submission([first, second], tmp_path / "submission.zip")
    with zipfile.ZipFile(archive) as zipped:
        names = sorted(zipped.namelist())
    assert names == ["submission/query-p1-1-kis.csv", "submission/query-p1-2-qa.csv"]
