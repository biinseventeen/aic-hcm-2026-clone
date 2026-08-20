"""The two sources (zip / extracted directory) must yield the *same* keys and *same* content.

If the two branches diverge, extracting the data would silently change the behaviour of the whole
pipeline — exactly the kind of failure that never raises an exception.
"""

import json
import zipfile

import pytest

from aic.data.layout import ArchiveSet, DataRoot, DirSet, parse_video_id


def test_parse_video_id():
    assert parse_video_id("L26_V444") == ("L26", 444)
    assert parse_video_id("L21_V001") == ("L21", 1)
    for bad in ("L21-V001", "L21_V001.mp4", "V001", "l21_v001", ""):
        with pytest.raises(ValueError):
            parse_video_id(bad)


VIDEOS = ["L21_V001", "L21_V002"]

_ZIP_NAMES = {
    "map-keyframes": "map-keyframes-aic25-b1.zip",
    "media-info": "media-info-aic25-b1.zip",
    "clip-features": "clip-features-32-aic25-b1.zip",
    "objects": "objects-aic25-b1.zip",
    "keyframes": "Keyframes_L21.zip",
}


def _families() -> dict[str, list[tuple[str, str]]]:
    return {
        "map-keyframes": [
            (f"map-keyframes/{video}.csv", "n,pts_time,fps,frame_idx\n1,0.0,25.0,0\n")
            for video in VIDEOS
        ],
        "media-info": [
            (f"media-info/{video}.json", json.dumps({"title": video})) for video in VIDEOS
        ],
        "clip-features": [(f"clip-features-32/{video}.npy", "not-a-real-npy") for video in VIDEOS],
        "objects": [
            (f"objects/{video}/001.json", json.dumps({"detection_class_entities": ["Car"]}))
            for video in VIDEOS
        ],
        "keyframes": [(f"keyframes/{video}/001.jpg", f"fake-jpeg-{video}") for video in VIDEOS],
    }


def _make_root(tmp_path, *, extracted: bool):
    """Build a tiny data root shaped like the organiser's real one."""
    root = tmp_path / "root"
    root.mkdir()
    for family, entries in _families().items():
        with zipfile.ZipFile(root / _ZIP_NAMES[family], "w") as archive:
            for name, body in entries:
                archive.writestr(name, body)
        if extracted:
            for name, body in entries:
                # extract_data.py drops the leading directory component.
                out = root / "extracted" / family / name.split("/", 1)[1]
                out.parent.mkdir(parents=True, exist_ok=True)
                # write_bytes, not write_text: on Windows write_text turns \n into \r\n and the
                # two sources would differ byte-wise despite identical logical content.
                out.write_bytes(body.encode("utf-8"))
    return root


def test_the_extracted_directory_is_preferred_when_present(tmp_path):
    root = DataRoot(_make_root(tmp_path, extracted=True))
    assert isinstance(root.map_keyframes, DirSet)
    assert isinstance(root.keyframes, DirSet)


def test_it_falls_back_to_archives_without_a_directory(tmp_path):
    root = DataRoot(_make_root(tmp_path, extracted=False))
    assert isinstance(root.map_keyframes, ArchiveSet)
    assert isinstance(root.keyframes, ArchiveSet)


def test_prefer_extracted_false_forces_reading_archives(tmp_path):
    root = DataRoot(_make_root(tmp_path, extracted=True), prefer_extracted=False)
    assert isinstance(root.map_keyframes, ArchiveSet)


def test_both_sources_give_the_same_keys_and_the_same_bytes(tmp_path):
    path = _make_root(tmp_path, extracted=True)
    from_zip = DataRoot(path, prefer_extracted=False)
    from_dir = DataRoot(path, prefer_extracted=True)
    for family in ("map_keyframes", "media_info", "clip_features", "objects", "keyframes"):
        zip_source, dir_source = getattr(from_zip, family), getattr(from_dir, family)
        zip_keys, dir_keys = sorted(zip_source.keys()), sorted(dir_source.keys())
        assert zip_keys == dir_keys, family
        assert len(zip_source) == len(dir_source) == len(zip_keys), family
        for key in zip_keys:
            assert zip_source.read(key) == dir_source.read(key), (family, key)
            assert key in zip_source
            assert key in dir_source


def test_iterating_a_source_yields_its_keys(tmp_path):
    path = _make_root(tmp_path, extracted=True)
    for prefer in (True, False):
        source = DataRoot(path, prefer_extracted=prefer).map_keyframes
        assert sorted(source) == sorted(source.keys())


def test_keyframe_keys_keep_their_prefix(tmp_path):
    """map-keyframes refers to keyframes as ``keyframes/<video>/<nnn>.jpg``."""
    path = _make_root(tmp_path, extracted=True)
    for prefer in (True, False):
        source = DataRoot(path, prefer_extracted=prefer).keyframes
        assert sorted(source.keys()) == [
            "keyframes/L21_V001/001.jpg",
            "keyframes/L21_V002/001.jpg",
        ]


def test_other_families_drop_their_prefix(tmp_path):
    path = _make_root(tmp_path, extracted=True)
    for prefer in (True, False):
        root = DataRoot(path, prefer_extracted=prefer)
        assert sorted(root.map_keyframes.keys()) == ["L21_V001.csv", "L21_V002.csv"]
        assert sorted(root.objects.keys()) == ["L21_V001/001.json", "L21_V002/001.json"]


def test_a_missing_key_raises_key_error(tmp_path):
    path = _make_root(tmp_path, extracted=True)
    for prefer in (True, False):
        with pytest.raises(KeyError):
            DataRoot(path, prefer_extracted=prefer).media_info.read("L99_V999.json")


def test_local_path_exists_only_for_a_directory_source(tmp_path):
    path = _make_root(tmp_path, extracted=True)
    key = "keyframes/L21_V001/001.jpg"
    assert DataRoot(path).keyframes.local_path(key) is not None
    assert DataRoot(path, prefer_extracted=False).keyframes.local_path(key) is None
    assert DataRoot(path).keyframes.local_path("keyframes/none/missing.jpg") is None


def test_read_text_strips_a_bom(tmp_path):
    path = _make_root(tmp_path, extracted=True)
    csv_path = path / "extracted" / "map-keyframes" / "L21_V001.csv"
    csv_path.write_bytes(b"\xef\xbb\xbfn,pts_time,fps,frame_idx\n")
    assert DataRoot(path).map_keyframes.read_text("L21_V001.csv").startswith("n,pts_time")


def test_video_ids_come_from_map_keyframes(tmp_path):
    assert DataRoot(_make_root(tmp_path, extracted=True)).video_ids() == VIDEOS


def test_missing_data_raises_a_clear_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="map-keyframes"):
        _ = DataRoot(empty).map_keyframes


def test_a_nonexistent_root_raises_immediately(tmp_path):
    with pytest.raises(FileNotFoundError):
        DataRoot(tmp_path / "no-such-directory")


def test_inventory_survives_a_missing_family(tmp_path):
    root = _make_root(tmp_path, extracted=False)
    (root / _ZIP_NAMES["objects"]).unlink()
    text = DataRoot(root).inventory()
    assert "MISSING" in text
    assert "map-keyframes" in text


def test_inventory_count_all_counts_the_large_families(tmp_path):
    text = DataRoot(_make_root(tmp_path, extracted=True)).inventory(count_all=True)
    keyframe_line = next(line for line in text.splitlines() if "keyframes" in line)
    assert "—" not in keyframe_line


def test_locate_and_extract_video(tmp_path):
    root = _make_root(tmp_path, extracted=False)
    with zipfile.ZipFile(root / "Videos_L21_a.zip", "w") as archive:
        archive.writestr("video/L21_V001.mp4", "not-a-real-mp4")
    data_root = DataRoot(root)
    assert data_root.locate_video("L21_V001") is not None
    assert data_root.locate_video("L21_V999") is None
    out = data_root.extract_video("L21_V001", tmp_path / "cache")
    assert out.read_text(encoding="utf-8") == "not-a-real-mp4"
    # Calling again does not re-extract.
    assert data_root.extract_video("L21_V001", tmp_path / "cache") == out
    with pytest.raises(FileNotFoundError):
        data_root.extract_video("L21_V999", tmp_path / "cache")


def test_locate_video_matches_the_basename_not_a_suffix(tmp_path):
    """A suffix test would also accept an entry merely ending with the target name."""
    root = _make_root(tmp_path, extracted=False)
    with zipfile.ZipFile(root / "Videos_L21_a.zip", "w") as archive:
        archive.writestr("video/XL21_V001.mp4", "decoy")
    assert DataRoot(root).locate_video("L21_V001") is None
