"""Tests for chord_pipeline.core.container (CHORDTimeStream and RawContainer)."""

import numpy as np
import pytest
from caput.containers import empty_like
from chord_util import andata
from draco.analysis import sensitivity, transform

from chord_pipeline.core.container import CHORDTimeStream, RawContainer

from conftest import StandInTelescope


def _gather(dset):
    arr = dset[:]
    return arr.allgather() if hasattr(arr, "allgather") else np.asarray(arr)


@pytest.fixture
def ts(acq):
    """The acquisition, distributed over frequency as in the pipeline."""
    _, files, _ = acq
    return CHORDTimeStream.from_xengine_files(files, distributed=True)


def test_from_corrdata_layout(acq, ts):
    """The CorrData datasets are moved to where a draco TimeStream expects them."""
    _, files, truth = acq
    cd = andata.CorrData.from_acq_h5(files)

    assert isinstance(ts, CHORDTimeStream)
    assert isinstance(ts, RawContainer)
    assert list(ts.vis.attrs["axis"]) == ["freq", "stack", "time"]
    assert list(ts.weight.attrs["axis"]) == ["freq", "stack", "time"]
    np.testing.assert_array_equal(_gather(ts.vis), cd.vis[:])
    np.testing.assert_array_equal(_gather(ts.weight), cd.weight[:])
    np.testing.assert_array_equal(ts.input_flags[:], truth["input_flags"])
    assert "flags/vis_weight" not in ts and "flags/inputs" not in ts
    np.testing.assert_array_equal(_gather(ts.frac_lost), cd.frac_lost[:])


def test_only_spec_datasets(ts):
    """Everything in the container is in the dataset spec."""
    names = andata._walk_datasets(ts._data)
    assert set(names) <= set(ts.dataset_spec)
    assert "config_json" in names and "input_info/type" in names


def test_properties(acq, ts):
    _, _, truth = acq
    assert ts.labels.tolist() == truth["labels"]
    np.testing.assert_array_equal(ts.input_type, truth["input_type"])
    np.testing.assert_array_equal(ts.is_dish, truth["input_type"] == 0)
    np.testing.assert_array_equal(ts.is_rfi_monitor, truth["input_type"] == 1)
    assert not ts.is_stacked


def test_save_load_roundtrip(ts, mpi_tmp_path):
    fname = str(mpi_tmp_path / "ts.h5")
    ts.save(fname)
    back = CHORDTimeStream.from_file(fname)
    assert type(back) is CHORDTimeStream
    for name in andata._walk_datasets(ts._data):
        if name == "config_json":
            assert len(back[name]) == len(ts[name])
            continue
        np.testing.assert_array_equal(_gather(back[name]), _gather(ts[name]), err_msg=name)
    for name, imap in ts.index_map.items():
        assert back.index_map[name].tolist() == imap.tolist(), name


def test_raw_container_from_acq_h5(ts, mpi_tmp_path):
    fname = str(mpi_tmp_path / "ts.h5")
    ts.save(fname)
    part = CHORDTimeStream.from_acq_h5(fname, start=2, stop=7)
    np.testing.assert_array_equal(_gather(part.vis), _gather(ts.vis)[..., 2:7])
    np.testing.assert_array_equal(part.time, ts.time[2:7])
    with pytest.raises(ValueError):
        CHORDTimeStream.from_acq_h5(fname, start=2)


def test_optional_axes_copied(ts):
    """Containers made from this one keep the CHORD axes, so all datasets can be made."""
    new = empty_like(ts)
    assert type(new) is CHORDTimeStream
    for axis in ["xyz", "grid_xy", "dish", "threshold", "config", "file", "ev"]:
        assert axis in new.index_map, axis
    new.add_dataset("input_info/feed_positions_m")
    new.add_dataset("evec")

    # Not needed when creating a container from scratch
    prod = np.array(
        [(a, b) for a in range(3) for b in range(a, 3)],
        dtype=[("input_a", "<u2"), ("input_b", "<u2")],
    )
    bare = CHORDTimeStream(freq=4, input=3, prod=prod, time=5)
    assert "xyz" not in bare.index_map
    assert bare.vis.shape == (4, 6, 5)


def test_draco_select_freq(ts):
    """draco's SelectFreq keeps the CHORD datasets, including those in groups."""
    task = transform.SelectFreq()
    task.read_config({"channel_range": [2, 9]})
    sub = task.process(ts)

    assert type(sub) is CHORDTimeStream
    np.testing.assert_array_equal(_gather(sub.vis), _gather(ts.vis)[2:9])
    np.testing.assert_array_equal(_gather(sub["flags/frac_lost"]), _gather(ts["flags/frac_lost"])[2:9])
    np.testing.assert_array_equal(_gather(sub["digital_gain"]), _gather(ts["digital_gain"])[2:9])
    np.testing.assert_array_equal(sub["input_info/type"][:], ts["input_info/type"][:])
    assert "config_json" in sub


def test_draco_system_sensitivity(ts):
    """draco's ComputeSystemSensitivity finds everything it needs."""
    task = sensitivity.ComputeSystemSensitivity()
    task.read_config({"exclude_intracyl": False})
    task.setup(StandInTelescope(ts))
    out = task.process(ts)
    assert list(out.pol) == ["XX", "XY", "YY"]
    np.testing.assert_array_equal(_gather(out.frac_lost), _gather(ts.frac_lost))
    assert np.all(np.isfinite(_gather(out.measured)))
