"""The HTTP layer: input validation at the boundary, and no web framework in the core package.

The validation tests run **without** fastapi — that is the point: the rules live in
standard-library dataclasses so they apply to HTTP callers and in-process callers alike. The
application tests skip themselves when ``.[backend]`` is not installed.
"""

import importlib.util
import json

import pytest

from aic.api.schemas import (
    MAX_BATCH,
    MAX_QUERY_CHARS,
    BatchSolveRequest,
    ErrorResponse,
    HealthResponse,
    SolveRequest,
    SolveResponse,
)

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
    from tests.test_service import make_result  # reuse the real result builder

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
