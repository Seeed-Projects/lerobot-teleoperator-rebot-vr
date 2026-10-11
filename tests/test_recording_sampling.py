"""Timing regressions with a 30 Hz camera and variable control work duration."""
import numpy as np
import pytest

from lerobot_teleoperator_rebot_vr.recording.sampling import CycleSampler


def test_send_latency_jitter_does_not_switch_to_adjacent_control_cycles():
    sampler = CycleSampler(50, 25)
    selected = []
    old_selected = []
    deadline = 0
    for cycle in range(1000):
        # Feedback finishes at 10 ms; varying IK/send work straddles deadlines.
        now = cycle * 20_000_000 + (16_000_000 if cycle % 4 == 0 else 12_000_000)
        if sampler.due(now):
            selected.append(cycle)
        if now >= deadline:
            old_selected.append(cycle)
            deadline = (now if not deadline else deadline) + 40_000_000
    assert len(selected) == 500
    assert set(np.diff(selected)) == {2}
    assert {1, 3} <= set(np.diff(old_selected))
    # At 30 Hz, adjacent 50 Hz cycles can reuse images. Every second cycle
    # with the same feedback latency selects a fresh image from both cameras.
    for phase_ns in (0, 17_000_000):
        camera_ids = [(i*20_000_000 + 10_000_000 + phase_ns) // 33_333_333 for i in selected]
        assert np.all(np.diff(camera_ids) > 0)


@pytest.mark.parametrize('ratio', [1, 2, 3, 4])
def test_sampling_ratio_and_episode_reset(ratio):
    sampler = CycleSampler(25*ratio, 25)
    interval = round(1e9/(25*ratio))
    assert [i for i in range(20) if sampler.due(i*interval)] == list(range(0, 20, ratio))
    sampler.reset()
    assert sampler.due(0)


def test_sampling_rejects_long_gaps_even_on_skipped_cycles():
    sampler = CycleSampler(50, 25)
    assert sampler.due(1_000_000_000)
    with pytest.raises(RuntimeError, match='deadline missed'):
        sampler.due(1_080_000_000)


@pytest.mark.parametrize('fps', [20, 40, float('nan')])
def test_sampling_requires_integer_cycle_ratio(fps):
    with pytest.raises(ValueError, match='integer multiple'):
        CycleSampler(fps, 25)
