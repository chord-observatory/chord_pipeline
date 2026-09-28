"""Tests for chord_pipeline.core.io (QueryAcquisitionFiles, LoadCorrDataFiles).

These work under MPI too, e.g. ``mpirun -np 3 python -m pytest tests``.
"""

import os

import numpy as np
import pytest
from caput import pipeline
from caput.pipeline import exceptions
from chord_util import andata

from chord_pipeline.core.container import CHORDTimeStream
from chord_pipeline.core.io import LoadCorrDataFiles, QueryAcquisitionFiles


def _gather(dset):
    arr = dset[:]
    return arr.allgather() if hasattr(arr, "allgather") else np.asarray(arr)


def _load(files, **params):
    """Run LoadCorrDataFiles until it stops, returning all its outputs."""
    task = LoadCorrDataFiles()
    task.read_config(params)
    task.setup(files)
    out = []
    while True:
        try:
            out.append(task.process())
        except exceptions.PipelineStopIteration:
            return out


@pytest.fixture(scope="module")
def serial(acq):
    """The whole acquisition read serially, for comparison."""
    _, files, _ = acq
    return CHORDTimeStream.from_xengine_files(files)


# QueryAcquisitionFiles
# ---------------------


def _query(**params):
    task = QueryAcquisitionFiles()
    task.read_config(params)
    return task.setup()


def test_query_by_name(acq):
    root, files, _ = acq
    name = os.path.basename(os.path.dirname(files[0]))
    assert _query(acquisition=name, data_root=str(root)) == files
    assert _query(acquisition=os.path.dirname(files[0])) == files
    assert _query(acquisition="acq_2026*", data_root=str(root)) == files
    assert _query(acquisition=name, data_root=str(root), file_range=[1, 3]) == files[1:3]
    with pytest.raises(RuntimeError):
        _query(acquisition="acq_missing", data_root=str(root))


# LoadCorrDataFiles
# -----------------


def test_single_timestream(acq, serial):
    """By default all files are concatenated into one distributed timestream."""
    _, files, truth = acq
    (ts,) = _load(files)
    assert isinstance(ts, CHORDTimeStream)
    assert ts.vis.distributed
    assert ts.attrs["tag"] == os.path.basename(os.path.dirname(files[0]))
    np.testing.assert_array_equal(_gather(ts.vis), serial.vis[:])
    np.testing.assert_array_equal(_gather(ts.weight), serial.weight[:])
    np.testing.assert_array_equal(ts.input_flags[:], truth["input_flags"])
    np.testing.assert_allclose(ts.time, truth["time"], atol=1e-6)


def test_files_per_container(acq, serial):
    _, files, _ = acq
    out = _load(files, files_per_container=2)
    assert len(out) == 2
    assert out[0].attrs["tag"].endswith("_4224075-4224076")
    assert out[1].attrs["tag"].endswith("_4224077-4224077")
    vis = np.concatenate([_gather(ts.vis) for ts in out], axis=-1)
    np.testing.assert_array_equal(vis, serial.vis[:])


@pytest.mark.parametrize(
    "params, index",
    [
        ({"channel_range": [3, 11]}, np.arange(3, 11)),
        ({"channel_range": [2, 14, 4]}, np.arange(2, 14, 4)),
        ({"channel_index": [0, 5, 6]}, np.array([0, 5, 6])),
    ],
)
def test_freq_selection(acq, serial, params, index):
    _, files, _ = acq
    (ts,) = _load(files, **params)
    np.testing.assert_array_equal(_gather(ts.vis), serial.vis[:][index])
    np.testing.assert_array_equal(ts.freq, serial.freq[index])


def test_freq_physical(acq, serial):
    _, files, _ = acq
    (ts,) = _load(files, freq_physical=[float(serial.freq[7]) + 0.01, float(serial.freq[2])])
    np.testing.assert_array_equal(ts.freq, serial.freq[[2, 7]])


def test_input_type(acq, serial):
    _, files, truth = acq
    (dish,) = _load(files, input_type="dish")
    assert not dish.is_rfi_monitor.any()
    assert dish.labels.tolist() == [
        lbl for lbl, t in zip(truth["labels"], truth["input_type"]) if t == 0
    ]
    (rfi,) = _load(files, input_type="rfi")
    assert rfi.is_rfi_monitor.all()


def test_only_autos(acq, serial):
    _, files, _ = acq
    (autos,) = _load(files, only_autos=True, input_type="dish")
    assert np.all(autos.prod["input_a"] == autos.prod["input_b"])
    assert autos.vis.global_shape[1] == int(serial.is_dish.sum())


def test_datasets(acq):
    _, files, _ = acq
    (ts,) = _load(files)
    assert "eval" in ts and "gain" not in ts
    (ts,) = _load(files, exclude_datasets=["eval", "evec", "erms"])
    assert "eval" not in ts and "evec" not in ts and "vis" in ts
    (ts,) = _load(files, datasets=["vis", "flags/vis_weight", "flags/inputs"])
    assert "flags/frac_lost" not in ts and "vis_weight" in ts and "input_flags" in ts


def test_corrdata_output_and_no_gain(acq, serial):
    _, files, truth = acq
    (cd,) = _load(files, use_draco_container=False, apply_gain=False)
    assert isinstance(cd, andata.CorrData)
    assert not cd.attrs["digital_gains_applied"]
    g = truth["gain"][0]
    gg = g[:, truth["prod"]["input_a"]] * g[:, truth["prod"]["input_b"]].conj()
    np.testing.assert_allclose(_gather(cd.vis), gg[..., None] * truth["vis"], rtol=1e-5)


def test_pipeline_config(acq, mpi_tmp_path):
    """The tasks run in a pipeline, as in the production config."""
    root, files, _ = acq
    name = os.path.basename(os.path.dirname(files[0]))
    config = f"""
pipeline:
  tasks:
    - type: chord_pipeline.core.io.QueryAcquisitionFiles
      out: filelist
      params:
        acquisition: {name}
        data_root: {root}

    - type: chord_pipeline.core.io.LoadCorrDataFiles
      requires: filelist
      out: tstream
      params:
        exclude_datasets: [eval, evec, erms]
        save: true
        output_name: "{mpi_tmp_path}/tstream_{{tag}}.h5"

    - type: chord_pipeline.analysis.inputs.RemoveRFIMonitors
      in: tstream
      params:
        save: true
        output_name: "{mpi_tmp_path}/tstream_dish_{{tag}}.h5"
"""
    pipeline.Manager.from_yaml_str(config).run()

    full = CHORDTimeStream.from_file(str(mpi_tmp_path / f"tstream_{name}.h5"))
    dish = CHORDTimeStream.from_file(str(mpi_tmp_path / f"tstream_dish_{name}.h5"))
    assert type(full) is CHORDTimeStream and type(dish) is CHORDTimeStream
    assert "eval" not in full
    assert len(dish.input) == int(full.is_dish.sum())
    assert not dish.is_rfi_monitor.any()
