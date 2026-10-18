"""Exercise the actual collector methods without importing vendor .NET DLLs."""
import ast
import __future__
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def collector():
    path = Path(__file__).parents[1] / "raman_control" / "princeton.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "SpectraCollector")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    instance = object.__new__(namespace["SpectraCollector"])
    instance._experiment = MagicMock()
    instance._daq_controller = MagicMock()
    instance.set_rm_exposure = MagicMock()
    instance._set_full_vertical_binning = MagicMock()
    instance._convert_capture = MagicMock(return_value=np.ones((1, 1340)))
    return instance


@pytest.mark.parametrize("count", [1, 25, 100])
def test_batch_scans_points_during_one_frame(collector, count):
    points = np.zeros((count, 2))
    result = collector.collect_spectra_pts_batch(points, 250)
    args, kwargs = collector._daq_controller.prepare_for_collection.call_args
    np.testing.assert_array_equal(args[0], points.T)
    assert args[0].flags.c_contiguous
    assert kwargs == {"batch": True, "exposure": .25}
    collector.set_rm_exposure.assert_called_once_with(250)
    collector._experiment.Capture.assert_called_once_with(1)
    collector._daq_controller.galvo.stop.assert_called_once()
    assert result.shape == (1, 1340)


def test_failed_batch_stops_galvo(collector):
    collector._experiment.Capture.side_effect = RuntimeError("detector failed")
    with pytest.raises(RuntimeError, match="detector failed"):
        collector.collect_spectra_pts_batch(np.zeros((25, 2)), 100)
    collector._daq_controller.galvo.stop.assert_called_once()


@pytest.mark.parametrize("points,exposure", [([], 100), ([[np.nan, 0]], 100), ([[11, 0]], 100),
                                           ([[0, 0]], 0), ([[0, 0]], np.inf)])
def test_invalid_batch_rejected_before_hardware(collector, points, exposure):
    with pytest.raises(ValueError):
        collector.collect_spectra_pts_batch(points, exposure)
    collector._experiment.Capture.assert_not_called()
    collector._daq_controller.prepare_for_collection.assert_not_called()
