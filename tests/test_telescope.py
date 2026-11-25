"""Tests for the pathfinder telescope."""

import h5py
import numpy as np
import pytest

from chord_pipeline.core.telescope import CHORDPathfinder, LoadPathfinderTelescope


@pytest.fixture
def layout_file(tmp_path):
    path = tmp_path / "layout.txt"
    path.write_text(
        "# label chan_id ew_m ns_m pol\n"
        "A01X 0 0.0 0.0 X\n"
        "A02X 1 6.300156854906823 0.0 X\n"
        "A01Y 2 0.0 0.0 Y\n"
        "A02Y 3 6.300156854906823 0.0 Y\n"
    )
    return str(path)


def test_telescope_from_layout(layout_file):
    tel = CHORDPathfinder.from_config(
        {"layout_file": layout_file, "bad_inputs": ["A02Y"]}
    )
    assert tel.nfeed == 4
    assert tel.feedpositions[1, 0] == 6.300156854906823
    assert list(tel.polarisation) == ["X", "X", "Y", "Y"]
    assert list(tel.input_index["correlator_input"]) == ["A01X", "A02X", "A01Y", "A02Y"]
    assert not hasattr(tel, "cylinder_width")
    assert (
        tel.feedmask[3, 3] == 0 and tel.feedmask[0, 0] == 1 and tel.feedmask[0, 3] == 0
    )


def test_explicit_frequencies(layout_file):
    tel = CHORDPathfinder.from_config(
        {"layout_file": layout_file, "freq_list": [400.0, 410.5]}
    )
    assert np.array_equal(tel.frequencies, [400.0, 410.5])


def test_load_task_requires_pointing(layout_file):
    task = LoadPathfinderTelescope.from_config({"layout_file": layout_file})
    with pytest.raises(ValueError):
        task.setup()


def test_load_task_explicit(layout_file):
    conf = {"layout_file": layout_file, "dec_pointing": 22.0, "bad_inputs": ["A01X"]}
    tel = LoadPathfinderTelescope.from_config(conf).setup()
    assert tel.elevation_pointing_offset == pytest.approx(22.0 - tel.latitude)
    assert tel.feedmask[0, 0] == 0


@pytest.mark.parametrize("which", ["xengine", "timestream"])
def test_load_task_from_data_file(acq, tmp_path, which):
    from chord_util import layout as chord_layout

    from chord_pipeline.core.io import LoadCorrDataFiles

    _, files, truth = acq
    if which == "xengine":
        data_file = files[0]
    else:
        task = LoadCorrDataFiles()
        task.read_config({})
        task.setup(files[:1])
        data_file = str(tmp_path / "tstream.h5")
        task.process().save(data_file)

    with h5py.File(files[0], "r") as fh:
        dec = chord_layout.pointing_dec(fh.attrs)
        t_ns = fh["time_center_t_inst_ns"][:]
    time = t_ns[t_ns != 0][0] * 1e-9

    feedmask = tmp_path / "feedmask.yaml"
    feedmask.write_text(
        "- start: '2000-01-01T00:00:00Z'\n"
        "  bad_inputs: {A01X: old}\n"
        f"- start: {time - 1.0}\n"
        "  bad_inputs: {A02Y: dead, B01X: dead}\n"
        f"- start: {time + 3600.0}\n"
        "  bad_inputs: {A03X: later}\n"
    )
    conf = {"data_file": data_file, "feedmask_file": str(feedmask)}
    tel = LoadPathfinderTelescope.from_config(conf).setup()
    assert tel.elevation_pointing_offset == pytest.approx(dec - tel.latitude)
    assert tel.bad_inputs == ["A02Y", "B01X"]
