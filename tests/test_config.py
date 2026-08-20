"""Paths must be absolute against the **project root**, never against the CWD.

That is the condition for one configuration to work both from the CLI (CWD = project root) and
inside a backend process (CWD arbitrary). If it breaks, the backend reads the index from a place
that does not exist and only reports the error on the first request.
"""

import json
from pathlib import Path

import pytest

from aic.config import (
    Config,
    PathConfig,
    default_config_path,
    find_project_root,
    load_config,
)


def test_the_defaults_are_relative():
    """A shared configuration file must not contain one machine's absolute paths."""
    defaults = PathConfig()
    for name in PathConfig.FIELDS:
        assert not Path(getattr(defaults, name)).is_absolute(), name


def test_the_default_data_root_points_at_the_in_repo_link():
    assert PathConfig().data_root == "data/raw"


def test_generated_output_lives_under_data_processed():
    """Everything derived from the raw data is reproducible and must stay out of git."""
    defaults = PathConfig()
    for name in ("index_dir", "submission_dir", "report_dir", "video_cache_dir"):
        assert getattr(defaults, name).startswith("data/processed/"), name


def test_the_devset_lives_outside_data():
    """The evaluation set is hand-annotated ground truth and must be version-controlled."""
    assert not PathConfig().devset_dir.startswith("data/")


def test_resolved_makes_every_path_absolute(tmp_path):
    resolved = PathConfig().resolved(tmp_path)
    for name in PathConfig.FIELDS:
        path = Path(getattr(resolved, name))
        assert path.is_absolute(), name
        assert str(path).startswith(str(tmp_path.resolve())), name


def test_resolved_leaves_already_absolute_paths_alone(tmp_path):
    absolute_root = (tmp_path / "corpus").resolve()
    resolved = PathConfig(data_root=str(absolute_root)).resolved(tmp_path / "elsewhere")
    assert Path(resolved.data_root) == absolute_root


def test_resolved_does_not_mutate_the_original(tmp_path):
    paths = PathConfig()
    paths.resolved(tmp_path)
    assert paths.data_root == "data/raw"


def test_config_resolve_returns_a_copy(tmp_path):
    config = Config()
    resolved = config.resolve(tmp_path)
    assert resolved is not config
    assert Path(resolved.paths.index_dir).is_absolute()
    assert not Path(config.paths.index_dir).is_absolute()
    # Resolving paths must not touch the hyperparameters.
    assert resolved.answer_span == config.answer_span
    assert resolved.retrieval == config.retrieval


def test_find_project_root_finds_pyproject():
    root = find_project_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "src" / "aic").is_dir()


def test_find_project_root_is_independent_of_the_cwd(tmp_path, monkeypatch):
    """Changing directory must not change the project root — that is the point of the function."""
    before = find_project_root()
    monkeypatch.chdir(tmp_path)
    assert find_project_root() == before


def test_aic_project_root_overrides_the_search(tmp_path, monkeypatch):
    monkeypatch.setenv("AIC_PROJECT_ROOT", str(tmp_path))
    find_project_root.cache_clear()
    try:
        assert find_project_root() == tmp_path.resolve()
    finally:
        monkeypatch.delenv("AIC_PROJECT_ROOT", raising=False)
        find_project_root.cache_clear()


def test_the_default_config_path_exists():
    assert default_config_path().exists()


def test_load_config_reads_a_file(tmp_path, monkeypatch):
    monkeypatch.delenv("AIC_DATA_ROOT", raising=False)
    path = tmp_path / "cfg.json"
    path.write_text(
        json.dumps(
            {
                "paths": {"data_root": "corpus", "index_dir": "index"},
                "answer_span": {"kis": 7},
                "retrieval": {"rrf_eta": 13},
                "seed": 99,
            }
        ),
        encoding="utf-8",
    )
    config = load_config(path, base=tmp_path)
    assert config.answer_span.kis == 7
    # A field absent from the file keeps its default.
    assert config.answer_span.qa == 25
    assert config.retrieval.rrf_eta == 13
    assert config.seed == 99
    assert Path(config.paths.data_root) == (tmp_path / "corpus").resolve()
    assert Path(config.paths.index_dir) == (tmp_path / "index").resolve()


def test_load_config_raises_for_a_named_file_that_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.json")


def test_load_config_works_with_no_arguments():
    assert load_config() is not None


def test_aic_data_root_beats_the_config_file(tmp_path, monkeypatch):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"paths": {"data_root": "from-file"}}), encoding="utf-8")
    monkeypatch.setenv("AIC_DATA_ROOT", str(tmp_path / "from-env"))
    config = load_config(path, base=tmp_path)
    assert Path(config.paths.data_root) == (tmp_path / "from-env").resolve()


def test_resolve_false_is_for_writing_a_file_back(tmp_path, monkeypatch):
    """Baking one machine's absolute paths into a shared file is wrong, so it must be avoidable."""
    monkeypatch.delenv("AIC_DATA_ROOT", raising=False)
    config = load_config(resolve=False)
    assert config.paths.data_root == "data/raw"
    out = config.save(tmp_path / "written.json")
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["paths"]["data_root"] == "data/raw"


def test_the_config_file_in_the_repo_matches_the_dataclass(monkeypatch):
    """``configs/default.json`` must be an image of the dataclass, never drift from it."""
    monkeypatch.delenv("AIC_DATA_ROOT", raising=False)
    on_disk = json.loads(default_config_path().read_text(encoding="utf-8"))
    assert json.loads(Config().to_json()) == on_disk


def test_answer_span_rejects_an_unknown_task():
    with pytest.raises(ValueError):
        Config().answer_span.for_task("nonexistent")


def test_fields_matches_the_real_dataclass_fields():
    from dataclasses import fields

    assert set(PathConfig.FIELDS) == {field.name for field in fields(PathConfig)}
