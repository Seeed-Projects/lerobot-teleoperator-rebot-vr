"""Real LeRobot v3 round trips using tiny synthetic videos; no cameras or CAN."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from lerobot_teleoperator_rebot_vr.recording.config import CameraConfig, RecordingConfig
from lerobot_teleoperator_rebot_vr.recording.dataset import LocalDataset


def make_frame(config, i, offset=0):
    return {"observation.state": np.arange(7, dtype=np.float32)+i,
            "action": np.arange(7, dtype=np.float32)+offset,
            "observation.host_time": np.array([i/config.fps], dtype=np.float64),
            "action.host_time": np.array([i/config.fps+.01], dtype=np.float64),
            "observation.camera_time": np.array([i/config.fps-.001]*2, dtype=np.float64),
            "observation.images.main": np.full((48, 64, 3), [240, 10, 10], np.uint8),
            "observation.images.wrist": np.full((48, 64, 3), [10, 10, 240], np.uint8),
            "task": config.task}


@pytest.fixture
def native(tmp_path, monkeypatch):
    pytest.importorskip("av", reason="install .[record] for real video round-trip tests")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import datasets.config
    import huggingface_hub
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path/'cache')
    def no_network(*args, **kwargs):
        raise AssertionError("local recording must not use the Hub")
    monkeypatch.setattr(huggingface_hub.HfApi, "repo_info", no_network)
    monkeypatch.setattr(huggingface_hub.HfApi, "upload_folder", no_network)
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    monkeypatch.setattr(LeRobotDatasetMetadata, "_pull_from_repo", no_network)
    config = RecordingConfig(tmp_path, {"main": CameraConfig(0, 64, 48), "wrist": CameraConfig(2, 64, 48)},
                             task="抓取红色积木")
    writer = LocalDataset(config, tmp_path/'RS_抓取红色积木_20261010_120000',
                          SimpleNamespace(robot_model='b601_rs', robot_id='rebot_b601_rs_left'), 100)
    yield config, writer, LeRobotDataset
    writer.close()


def test_native_saved_episodes_are_readable_after_each_save_and_discard_is_not_training_data(native):
    config, writer, LeRobotDataset = native
    for episode, save in enumerate((True, False, True)):
        writer.begin({"started_at": "2026-10-10T12:00:00+08:00", "robot_family": "RS", "task": config.task})
        for i in range(4):
            writer.add_frame(make_frame(config, i, offset=episode))
        writer.end(save=save, metadata={"ended_at": "2026-10-10T12:00:01+08:00", "frames": 4})
        if save:
            # Before close(), SAVED must already mean complete Parquet and videos.
            reader = LeRobotDataset(writer.repo_id, root=writer.root, video_backend='pyav')
            assert reader.num_episodes == (1 if episode == 0 else 2)
            assert len(reader) == (4 if episode == 0 else 8)
            row = reader[len(reader)-4]
            np.testing.assert_allclose(row['action'], np.arange(7)+episode)
            np.testing.assert_allclose(row['observation.state'], np.arange(7))
            assert row['task'] == config.task
            assert row['timestamp'].item() == 0
            assert row['observation.images.main'].shape == (3, 48, 64)
            # RGB order survives both camera streams' native H.264 round trip.
            assert row['observation.images.main'][0].mean() > .8
            assert row['observation.images.main'][2].mean() < .1
            assert row['observation.images.wrist'][2].mean() > .8
            assert row['observation.images.wrist'][0].mean() < .1
    info = json.loads((writer.root/'meta/info.json').read_text())
    assert info['codebase_version'] == 'v3.0'
    assert info['total_frames'] == 8 and info['fps'] == 25
    assert info['features']['action']['names'][-1] == 'gripper'
    manifest = json.loads((writer.root/'meta/recording.json').read_text())
    assert manifest['robot_family'] == 'RS'
    assert manifest['task'] == '抓取红色积木'
    assert manifest['created_at']
    assert manifest['state_units'] == manifest['action_units'] == 'degrees'
    assert 'active_episode' not in manifest
    assert [e['status'] for e in manifest['episodes']] == ['saved', 'discarded', 'saved']
    assert [e['episode_index'] for e in manifest['episodes'] if e['status'] == 'saved'] == [0, 1]


def test_native_empty_or_aborted_run_can_close_and_first_success_keeps_index_zero(native):
    config, writer, _ = native
    writer.begin({'started_at': 'test'})
    writer.end(save=False, metadata={'frames': 0})
    writer.begin({'started_at': 'test'})
    writer.add_frame(make_frame(config, 0))
    writer.end(save=False, metadata={'frames': 1})
    writer.begin({'started_at': 'test'})
    writer.add_frame(make_frame(config, 0))
    writer.end(save=True, metadata={'frames': 1})
    assert writer.saved == 1
    writer.close()


def test_native_no_episode_close_is_idempotent(native):
    _, writer, _ = native
    writer.close()
    writer.close()
    assert json.loads((writer.root/'meta/info.json').read_text())['total_episodes'] == 0


def test_native_encoder_drop_never_becomes_a_saved_episode(native, monkeypatch):
    config, writer, _ = native
    writer.begin({'started_at': 'test'})
    encoder = writer._dataset.writer._streaming_encoder
    def drop(video_key, image):
        encoder._dropped_frames[video_key] = 1
    monkeypatch.setattr(encoder, 'feed_frame', drop)
    with pytest.raises(RuntimeError, match='refusing to save'):
        writer.add_frame(make_frame(config, 0))
    with pytest.raises(RuntimeError, match='refusing to save'):
        writer.end(save=True, metadata={})
    writer.fail('encoder dropped frames')
    writer.close()
    assert writer.saved == 0
    assert json.loads((writer.root/'meta/info.json').read_text())['total_episodes'] == 0
    assert json.loads((writer.root/'meta/recording.json').read_text())['active_episode']['status'] == 'error'
