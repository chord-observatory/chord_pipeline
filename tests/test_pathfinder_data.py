"""Checks against real CHORD pathfinder X-engine data.

These run only where the pathfinder data is readable (on the cluster, with
access to the rrg-kmsmith project), and are skipped elsewhere. They read ~50
frequency channels, so they need a few GB of memory and take about a minute.

The data used can be changed with environment variables:

- ``CHORD_XENGINE_DATA``: directory holding the acquisitions (default: the
  pathfinder subset stream on the cluster).
- ``CHORD_XENGINE_ACQ``: the acquisition (default:
  ``acq_20260911_232919_986789057``). Checks specific to the default
  acquisition are skipped for others.

The reference used everywhere is an independent read of the raw files with
``h5py``, with the digital gains divided out by code in this file (not by
`chord_util.andata`).

Run with e.g.::

    python -m pytest tests/test_pathfinder_data.py -v
    mpirun -np 4 python -m pytest tests/test_pathfinder_data.py
"""

import glob
import json
import os

import h5py
import numpy as np
import pytest
from chord_util import andata
from draco.analysis import sensitivity, transform

from chord_pipeline.analysis.inputs import RemoveRFIMonitors
from chord_pipeline.core.container import CHORDTimeStream
from chord_pipeline.core.io import LoadCorrDataFiles

from conftest import StandInTelescope

DEFAULT_ACQ = "acq_20260911_232919_986789057"
DATA_ROOT = os.environ.get(
    "CHORD_XENGINE_DATA", "/project/rrg-kmsmith/chord-data/kotekan_vis_files/subset"
)
ACQ = os.environ.get("CHORD_XENGINE_ACQ", DEFAULT_ACQ)
ALL_FILES = sorted(glob.glob(os.path.join(DATA_ROOT, ACQ, "vis_*.h5")))

pytestmark = pytest.mark.skipif(
    len(ALL_FILES) < 4 or not os.access(ALL_FILES[0], os.R_OK),
    reason=f"CHORD pathfinder data not available in {os.path.join(DATA_ROOT, ACQ)}",
)

# The first file (which has unfilled time bins), two consecutive files from the
# middle, and the last file (unfilled bins at the end)
if len(ALL_FILES) >= 4:
    MID = len(ALL_FILES) // 2
    FILES = [ALL_FILES[0], ALL_FILES[MID], ALL_FILES[MID + 1], ALL_FILES[-1]]
else:
    FILES = []

# Channels 2990-3040 (~884-894 MHz): includes heavily RFI-excised samples
CHANNELS = slice(2990, 3040)

# The pathfinder wiring from kotekan config/chord_pathfinder.j2
PATHFINDER_REMAP = [0, 2, 4, 6, 8, 10, 12, 15, 1, 3, 5, 7, 9, 11, 13, 14]

default_acq_only = pytest.mark.skipif(
    ACQ != DEFAULT_ACQ, reason=f"Only checked for {DEFAULT_ACQ}"
)


def _gather(dset):
    arr = dset[:]
    return arr.allgather() if hasattr(arr, "allgather") else np.asarray(arr)


def _remap_from_config(fh):
    """The CRS board remap recorded in the receiver config snapshot."""
    for snapshot in fh["config_json"][:]:
        config = json.loads(snapshot).get("config", {})
        if "crs_board_remap" in config.get("dpdk", {}):
            return list(config["dpdk"]["crs_board_remap"])
    return PATHFINDER_REMAP


@pytest.fixture(scope="module")
def reference():
    """An independent read of the raw files, with the gains divided out."""
    vis, weight, frac_lost, frames_added, eflags, t_ns, gains = [], [], [], [], [], [], []
    for fname in FILES:
        with h5py.File(fname, "r") as fh:
            filled = fh["time_center_t_inst_ns"][:] != 0
            prod = fh["index_map/prod"][:]
            labels = [x.decode() if isinstance(x, bytes) else x for x in fh["index_map/label"][:]]
            input_type = fh["index_map/type"][:]
            input_list = fh.attrs["input_list"][:]
            remap = _remap_from_config(fh)

            # Gain of element e: lane e % 8 of the board feeding slot e // 8.
            # gain_exp is ignored (fixed to -1 by the CHORD F-engine).
            fengine_input = np.array([8 * remap[e // 8] + e % 8 for e in input_list])
            dg = fh["digital_gains"]
            upd = int(dg.attrs.get("selected_update_idx", 0))
            gain_freq = dg["index_map/freq"]["centre"][:]
            freq = fh["index_map/freq"]["centre"][CHANNELS]
            first = int(np.argmin(np.abs(gain_freq - freq[0])))
            coeff = dg["gain_coeff"][upd, first : first + len(freq)][:, fengine_input]
            gains.append(coeff.astype(np.complex128))

            vis.append(fh["vis"][CHANNELS][..., filled].astype(np.complex128))
            weight.append(fh["vis_weight"][CHANNELS][..., filled].astype(np.float64))
            frac_lost.append(fh["frac_lost"][CHANNELS][:, filled])
            frames_added.append(fh["frames_added"][CHANNELS][:, filled])
            eflags.append(fh["flags"][CHANNELS][..., filled])
            t_ns.append(fh["time_center_t_inst_ns"][:][filled])

    return {
        "vis": vis,
        "weight": weight,
        "frac_lost": np.concatenate(frac_lost, axis=-1),
        "frames_added": np.concatenate(frames_added, axis=-1),
        "element_flags": np.concatenate(eflags, axis=-1),
        "time": np.concatenate(t_ns) * 1e-9,
        "gains": gains,
        "prod": prod,
        "labels": labels,
        "input_type": input_type,
        "remap": remap,
        "freq": freq,
    }


def _corrected(ref, norm):
    """The reference vis and weights with the (normalised) gains divided out."""
    vis, weight = [], []
    for v, w, g in zip(ref["vis"], ref["weight"], ref["gains"]):
        g = g / norm
        gg = g[:, ref["prod"]["input_a"]] * g[:, ref["prod"]["input_b"]].conj()
        vis.append(v / gg[..., None])
        weight.append(w * np.abs(gg[..., None]) ** 2)
    vis, weight = np.concatenate(vis, -1), np.concatenate(weight, -1)
    missing = (ref["frames_added"] == 0) | (ref["frac_lost"] >= 1)
    weight[np.broadcast_to(missing[:, None, :], weight.shape)] = 0
    return vis, weight


@pytest.fixture(scope="module")
def data():
    return andata.CorrData.from_acq_h5(FILES, freq_sel=CHANNELS)


# Reading
# -------


def test_remap_read_from_files(data, reference):
    assert data.attrs["crs_board_remap_source"] == "config_json"
    assert list(data.attrs["crs_board_remap"]) == reference["remap"]


@default_acq_only
def test_remap_matches_wiring(reference):
    assert reference["remap"] == PATHFINDER_REMAP


def test_matches_raw_files(data, reference):
    """vis and weights equal the independent gain-corrected read."""
    vis, weight = _corrected(reference, data.attrs["digital_gain_norm"])
    np.testing.assert_allclose(data.vis[:], vis, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(data.weight[:], weight, rtol=2e-5)
    np.testing.assert_allclose(data.freq, reference["freq"])
    assert data.prod.tolist() == reference["prod"].tolist()
    assert data.labels.tolist() == reference["labels"]


def test_flags_and_times(data, reference):
    np.testing.assert_array_equal(data.frac_lost[:], reference["frac_lost"])
    np.testing.assert_array_equal(data.flags["element_flags"][:], reference["element_flags"])
    np.testing.assert_array_equal(data.input_flags[:], (reference["element_flags"] > 0).any(axis=0))
    np.testing.assert_allclose(data.time, reference["time"], rtol=0, atol=1e-6)
    assert np.all(np.diff(data.time) > 0)


def test_gain_normalisation(data, reference):
    """The normalisation is the median dish |gain_coeff| over the whole band."""
    with h5py.File(FILES[0], "r") as fh:
        dg = fh["digital_gains"]
        coeff = dg["gain_coeff"][int(dg.attrs.get("selected_update_idx", 0))]
        freq = fh["index_map/freq"]["centre"][:]
        first = int(np.argmin(np.abs(dg["index_map/freq"]["centre"][:] - freq[0])))
        input_list = fh.attrs["input_list"][:]
    remap = reference["remap"]
    fin = np.array([8 * remap[e // 8] + e % 8 for e in input_list])
    g = np.abs(coeff[first : first + len(freq)][:, fin[reference["input_type"] == 0]])
    np.testing.assert_allclose(data.attrs["digital_gain_norm"], np.median(g[g > 0]))


@default_acq_only
def test_saturated_gains_on_dead_inputs(reference):
    """The saturated F-engine gains are on the inputs that carry no signal.

    This is what confirmed the CRS board remap: with it, the ~1e10 gains land on
    A4, A3, A1 (both polarisations) and RFIA4p1, which show no correlated signal.
    """
    g = np.median(np.abs(reference["gains"][0]), axis=0)
    saturated = {lbl for lbl, gi in zip(reference["labels"], g) if gi > 20 * np.median(g)}
    assert saturated == {"A4p1", "A3p1", "A1p1", "A4p2", "A3p2", "RFIA4p1"}


@default_acq_only
def test_whole_acquisition_time_axis():
    """All files of the acquisition give one continuous timestream."""
    reader = andata.CorrReader(ALL_FILES)
    assert len(reader.files) == len(ALL_FILES) == 233
    assert len(reader.time) == 4645
    assert np.all(np.diff(reader.time) > 0)
    reader.freq_sel = slice(3000, 3002)
    reader.dataset_sel = ["vis"]
    part = reader.read()
    assert part.vis.shape == (2, 1176, 4645)
    assert np.all(np.diff(part["bin_abs_index"][:].astype(np.int64)) == 1)


def test_distributed_matches_serial(data):
    dist = andata.CorrData.from_acq_h5(FILES, freq_sel=CHANNELS, distributed=True)
    np.testing.assert_array_equal(_gather(dist.vis), data.vis[:])
    np.testing.assert_array_equal(_gather(dist.weight), data.weight[:])
    np.testing.assert_array_equal(dist.input_flags[:], data.input_flags[:])


# Pipeline tasks and draco
# ------------------------


@pytest.fixture(scope="module")
def tstream():
    task = LoadCorrDataFiles()
    task.read_config({"channel_range": [CHANNELS.start, CHANNELS.stop]})
    task.setup(FILES)
    return task.process()


def test_load_corr_data_files(tstream, data):
    assert isinstance(tstream, CHORDTimeStream)
    np.testing.assert_array_equal(_gather(tstream.vis), data.vis[:])
    np.testing.assert_array_equal(_gather(tstream.weight), data.weight[:])


def test_remove_rfi_monitors(tstream, reference):
    task = RemoveRFIMonitors()
    task.read_config({})
    dish = task.process(tstream)
    assert len(dish.input) == int((reference["input_type"] == 0).sum())
    assert not dish.is_rfi_monitor.any()

    labels = tstream.labels.tolist()
    index = {(labels[a], labels[b]): i for i, (a, b) in enumerate(tstream.prod)}
    dl = dish.labels.tolist()
    pmap = [index[(dl[a], dl[b])] for a, b in dish.prod]
    np.testing.assert_array_equal(_gather(dish.vis), _gather(tstream.vis)[:, pmap])


def test_draco_tasks(tstream):
    sel = transform.SelectFreq()
    sel.read_config({"channel_range": [0, 10]})
    sub = sel.process(tstream)
    assert isinstance(sub, CHORDTimeStream)
    np.testing.assert_array_equal(_gather(sub.vis), _gather(tstream.vis)[:10])
    assert "input_info/type" in sub and "flags/frac_lost" in sub

    task = sensitivity.ComputeSystemSensitivity()
    task.read_config({"exclude_intracyl": False})
    task.setup(StandInTelescope(tstream))
    out = task.process(tstream)
    assert list(out.pol) == ["XX", "XY", "YY"]


def test_save_and_reload(tstream, mpi_tmp_path):
    fname = str(mpi_tmp_path / "tstream.h5")
    tstream.save(fname)
    back = CHORDTimeStream.from_file(fname)
    assert type(back) is CHORDTimeStream
    np.testing.assert_array_equal(back.vis[:], _gather(tstream.vis))
    np.testing.assert_array_equal(back.weight[:], _gather(tstream.weight))
