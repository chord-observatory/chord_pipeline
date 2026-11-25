"""Tests for the pathfinder telescope."""

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
