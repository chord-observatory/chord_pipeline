"""Tests for chord_pipeline.analysis.inputs (SelectInputs, RemoveRFIMonitors)."""

import numpy as np
import pytest
from chord_util import andata

from chord_pipeline.analysis.inputs import RemoveRFIMonitors, SelectInputs
from chord_pipeline.core.container import CHORDTimeStream


def _run(task_cls, data, **params):
    task = task_cls()
    task.read_config(params)
    return task.process(data)


@pytest.fixture
def ts(acq):
    _, files, _ = acq
    return CHORDTimeStream.from_xengine_files(files)


def test_remove_rfi_monitors_matches_dish_read(acq, ts):
    """Removing the RFI monitors gives the same as reading only the dishes."""
    _, files, truth = acq
    dish = _run(RemoveRFIMonitors, ts)
    isel = np.flatnonzero(truth["input_type"] == andata.INPUT_TYPE_DISH)
    ref = CHORDTimeStream.from_xengine_files(files, input_sel=isel)

    assert dish.input.tolist() == ref.input.tolist()
    assert dish.prod.tolist() == ref.prod.tolist()
    for name in andata._walk_datasets(ref._data):
        if name == "config_json":
            continue
        np.testing.assert_array_equal(dish[name][:], ref[name][:], err_msg=name)
    assert dish.attrs["num_elements"] == len(isel)
    assert dish.attrs["num_prod"] == len(ref.prod)


def test_select_rfi_and_exclude_labels(ts):
    rfi = _run(SelectInputs, ts, input_type="rfi")
    assert rfi.is_rfi_monitor.all()

    some = _run(SelectInputs, ts, input_type="all", exclude_labels=["B3p1", "B3p2"])
    assert "B3p1" not in some.labels and "B3p2" not in some.labels
    assert len(some.input) == len(ts.input) - 2

    # Products are re-indexed to the new inputs and point at the same data
    labels = ts.labels.tolist()
    index = {(labels[a], labels[b]): i for i, (a, b) in enumerate(ts.prod)}
    sub_labels = some.labels.tolist()
    pmap = [index[(sub_labels[a], sub_labels[b])] for a, b in some.prod]
    np.testing.assert_array_equal(some.vis[:], ts.vis[:][:, pmap])
    np.testing.assert_array_equal(some.reverse_map["stack"]["stack"], np.arange(len(pmap)))


def test_unchanged_datasets(ts):
    """Datasets without an input/prod axis are copied unchanged."""
    dish = _run(RemoveRFIMonitors, ts)
    for name in ["flags/frac_lost", "bin_ERA_deg", "dish_positions_in_grid_coords"]:
        np.testing.assert_array_equal(dish[name][:], ts[name][:])
    np.testing.assert_array_equal(dish.time, ts.time)


def test_no_inputs_selected(ts):
    with pytest.raises(ValueError):
        _run(SelectInputs, ts, input_type="dish", exclude_labels=list(ts.labels))
