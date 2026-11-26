"""Tests for chord_pipeline.analysis.calibration."""

import numpy as np
import pytest
from draco.core import containers

from chord_pipeline.analysis.calibration import (
    FlagAmplitude,
    InterpolateGainOverFrequency,
    PerformEigenDecomp,
)
from chord_pipeline.core.container import CHORDTimeStream, TimeEigenDecomp
from chord_pipeline.core.telescope import CHORDPathfinder

NDISH = 8
T0 = 1791355000.0


@pytest.fixture
def tel(tmp_path):
    """Dishes 6.3 m apart east-west, the X and Y feeds of a dish at one position."""
    lines = ["# label chan_id ew_m ns_m pol"]
    for p, pol in enumerate("XY"):
        for d in range(NDISH):
            lines.append(f"A0{d + 1}{pol} {d + 64 * p} {6.3 * d} 0.0 {pol}")
    fname = tmp_path / "layout.txt"
    fname.write_text("\n".join(lines) + "\n")
    return CHORDPathfinder.from_config({"layout_file": str(fname)})


def _set(dset, full):
    """Set a (possibly distributed) dataset from the full array."""
    arr = dset[:]
    if hasattr(arr, "local_offset"):
        sl = [slice(None)] * full.ndim
        sl[arr.axis] = slice(
            arr.local_offset[arr.axis],
            arr.local_offset[arr.axis] + arr.local_shape[arr.axis],
        )
        arr.local_array[:] = full[tuple(sl)]
    else:
        dset[:] = full


def _full(dset):
    arr = dset[:]
    return arr.allgather() if hasattr(arr, "allgather") else np.asarray(arr)


def _labels():
    return [f"A0{d + 1}{p}" for p in "XY" for d in range(NDISH)]


def _gains(seed=0):
    """Complex gains of the X inputs (first half) and the Y inputs."""
    rng = np.random.default_rng(seed)
    ninput = 2 * NDISH
    return rng.uniform(0.5, 1.5, ninput) * np.exp(2j * np.pi * rng.uniform(size=ninput))


def make_point_source_ts(nfreq=4, ntime=6, tsys=50.0, seed=0):
    """Visibilities of a point source, with receiver noise in the autos.

    ``V_ab = g_a g_b^* S`` for two inputs of the same polarisation, zero between
    polarisations, plus ``tsys`` on the autocorrelations: the noise power that a
    rank-one model of the sky does not describe.
    """
    labels = _labels()
    ninput = len(labels)
    inp = np.zeros(ninput, dtype=[("chan_id", "<u2"), ("correlator_input", "U16")])
    inp["chan_id"] = [d + 64 * p for p in range(2) for d in range(NDISH)]
    inp["correlator_input"] = labels
    ia, ib = np.triu_indices(ninput)
    prod = np.zeros(len(ia), dtype=[("input_a", "<u2"), ("input_b", "<u2")])
    prod["input_a"], prod["input_b"] = ia, ib
    rstack = np.zeros(len(ia), dtype=[("stack", "<u4"), ("conjugate", "u1")])
    rstack["stack"] = np.arange(len(ia))
    freq = np.zeros(nfreq, dtype=[("centre", "<f8"), ("width", "<f8")])
    freq["width"] = 3200.0 / 16384
    freq["centre"] = 600.0 + np.arange(nfreq) * freq["width"]
    ts = CHORDTimeStream(
        freq=freq,
        input=inp,
        prod=prod,
        time=T0 + 10.0 * np.arange(ntime),
        reverse_map_stack=rstack,
    )
    ts.input_flags[:] = 1

    g = _gains(seed)
    pol = np.repeat([0, 1], NDISH)
    flux = np.array([1.0, 0.6])  # X and Y differ so the eigenvalues are distinct
    vis = g[ia] * g[ib].conj() * (pol[ia] == pol[ib]) * flux[pol[ia]]
    vis = vis + tsys * (ia == ib)
    vis = np.broadcast_to(vis[None, :, None], (nfreq, len(ia), ntime)).copy()
    _set(ts.vis, vis.astype(np.complex64))
    _set(ts.weight, np.ones(vis.shape, dtype=np.float32))
    return ts, g


def test_eigendecomp_recovers_the_gains(tel):
    ts, g = make_point_source_ts()
    task = PerformEigenDecomp()
    # Mask the products within a dish (autos and X-Y of one dish), as for CHORD
    task.read_config(
        {"mask_type": "baseline", "mask_threshold": 0.5, "niter": 20, "rank": 2}
    )
    task.setup(tel)
    eigen = task.process(ts)

    assert isinstance(eigen, TimeEigenDecomp)
    evec = _full(eigen.evec)
    evalue = _full(eigen.eval)

    # The two largest eigenvalues: X (flux 1) then Y (flux 0.6)
    gx = np.where(np.arange(2 * NDISH) < NDISH, g, 0)
    gy = np.where(np.arange(2 * NDISH) >= NDISH, g, 0)
    for k, gp in enumerate([gx, gy]):
        v = evec[:, k]  # (freq, input, time)
        overlap = np.abs(np.einsum("fit,i->ft", v.conj(), gp)) / np.linalg.norm(gp)
        np.testing.assert_allclose(overlap, 1.0, atol=1e-4)
        np.testing.assert_allclose(
            evalue[:, k], np.sum(np.abs(gp) ** 2) * [1.0, 0.6][k], rtol=1e-3
        )


def test_eigendecomp_without_mask_is_biased_by_the_autos(tel):
    """Without masking, the receiver noise of the autos changes the eigenvalues."""
    ts, g = make_point_source_ts()
    task = PerformEigenDecomp()
    task.read_config({"mask_type": "baseline", "mask_threshold": 0.0, "niter": 0})
    task.setup(tel)
    evalue = _full(task.process(ts).eval)
    gx = g[:NDISH]
    assert np.all(evalue[:, 0] > np.sum(np.abs(gx) ** 2) + 10.0)


def _static_gain(nfreq=64, seed=1):
    labels = _labels()
    ninput = len(labels)
    inp = np.zeros(ninput, dtype=[("chan_id", "<u2"), ("correlator_input", "U16")])
    inp["chan_id"] = [d + 64 * p for p in range(2) for d in range(NDISH)]
    inp["correlator_input"] = labels
    freq = np.zeros(nfreq, dtype=[("centre", "<f8"), ("width", "<f8")])
    freq["width"] = 3200.0 / 16384
    freq["centre"] = 600.0 + np.arange(nfreq) * 1.0
    gain = containers.StaticGainData(freq=freq, input=inp)
    gain.add_dataset("weight")
    rng = np.random.default_rng(seed)
    gfull = (1.0 + 0.01 * rng.standard_normal((nfreq, ninput))) * np.exp(
        1j * freq["centre"][:, None] / 50.0
    )
    return gain, gfull


def test_flag_amplitude_flags_an_outlier(tel):
    gain, gfull = _static_gain()
    gfull[10, 3] *= 30.0  # an outlier amplitude
    _set(gain.gain, gfull.astype(np.complex64))
    _set(gain.weight, np.ones(gfull.shape, dtype=np.float32))

    task = FlagAmplitude()
    task.read_config({})
    task.setup(tel)
    out = task.process(gain)

    w = _full(out.weight)
    assert w[10, 3] == 0
    assert (w > 0).mean() > 0.95


def test_interpolate_gain_over_frequency(tel):
    gain, gfull = _static_gain()
    _set(gain.gain, gfull.astype(np.complex64))
    weight = np.full(gfull.shape, 1e4, dtype=np.float32)
    weight[30:33] = 0.0
    _set(gain.weight, weight)

    task = InterpolateGainOverFrequency()
    task.read_config({"interp_scale": 10.0})
    out = task.process(gain)

    g = _full(out.gain)
    w = _full(out.weight)
    np.testing.assert_array_equal(g[:30], gfull[:30].astype(np.complex64))
    np.testing.assert_allclose(g[30:33], gfull[30:33], atol=0.05)
    assert (w[30:33] > 0).all()
