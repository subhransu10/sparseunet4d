"""CPU-only regression tests for real-sensor deployment adaptations."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mos_inference import MOSInference
from sparseunet4d.datasets.residual_features import residual_channels


def _point(pitch_deg, distance):
    pitch = np.deg2rad(pitch_deg)
    return np.asarray([[distance * np.cos(pitch), 0.0,
                        distance * np.sin(pitch)]], np.float32)


def test_sensor_fov_prevents_upper_beam_aliasing():
    now = _point(10.0, 10.0)
    # Same beam at 9 m plus a different upper beam at 2 m. With the old
    # SemanticKITTI +3-degree ceiling, both past returns collapse onto row 0.
    past = np.concatenate([_point(10.0, 9.0), _point(14.0, 2.0)])
    adapted = residual_channels(
        now, [past], H=16, W=2048, fov_down_deg=-15.0, fov_up_deg=15.0,
        normalize=False)
    np.testing.assert_allclose(adapted, [[1.0]], atol=1e-5)


def test_explicit_history_uses_input_scan_offsets():
    mos = MOSInference.__new__(MOSInference)
    mos.offsets = [1, 2, 4, 8]
    mos._infer = lambda stack: stack

    history = []
    for offset in range(9):
        xyz = np.asarray([[float(offset), 0.0, 0.0]], np.float32)
        remission = np.asarray([[offset / 255.0]], np.float32)
        pose = np.eye(4)
        pose[0, 3] = -float(offset)
        history.append((xyz, remission, pose))

    stack = mos.infer_history(history)
    assert [item[3] for item in stack] == [0, 1, 2, 4, 8]
    np.testing.assert_allclose(
        [item[2][0, 3] for item in stack], [0.0, -1.0, -2.0, -4.0, -8.0])


if __name__ == "__main__":
    test_sensor_fov_prevents_upper_beam_aliasing()
    test_explicit_history_uses_input_scan_offsets()
    print("Deployment adaptation regression tests passed")
