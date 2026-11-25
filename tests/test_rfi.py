"""Tests for ComputeSystemSensitivity and RFISensitivityMask."""

import itertools

import numpy as np
import pytest
from chord_util import rfi as crfi

from chord_pipeline.analysis.rfi import ComputeSystemSensitivity, RFISensitivityMask
from chord_pipeline.core.container import CHORDTimeStream

NSAMP = 1.0e5
# Night of 2026-10-06/07 (PDT): Cyg A transits in the first hour
T0 = 1791355000.0


def _run(task_cls, data, **params):
    task = task_cls()
    task.read_config(params)
    return task.process(data)


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
    """The full array of a (possibly distributed) dataset."""
    arr = dset[:]
    return arr.allgather() if hasattr(arr, "allgather") else np.asarray(arr)


def make_ts(
    nfreq=48, ntime=60, ndish=5, freq0=440.0, noise=0.0, seed=0, edit_weight=None
):
    """A small unstacked timestream with radiometer-noise weights.

    The autocorrelations are smooth and positive, and the weights of every product
    are ``w_ab = N / (V_aa V_bb)`` (times ``1 + noise * gaussian``), so the ratio of
    the measured to the radiometric noise is 1 without noise. `edit_weight` is
    applied to the full weight array before it is stored.
    """
    rng = np.random.default_rng(seed)
    labels = [f"A0{d + 1}{p}" for p in "XY" for d in range(ndish)]
    ninput = len(labels)
    inp = np.zeros(ninput, dtype=[("chan_id", "<u2"), ("correlator_input", "U16")])
    inp["chan_id"] = [d + 64 * p for p in range(2) for d in range(ndish)]
    inp["correlator_input"] = labels
    ia, ib = np.triu_indices(ninput)
    prod = np.zeros(len(ia), dtype=[("input_a", "<u2"), ("input_b", "<u2")])
    prod["input_a"], prod["input_b"] = ia, ib
    freq = np.zeros(nfreq, dtype=[("centre", "<f8"), ("width", "<f8")])
    freq["width"] = 3200.0 / 16384
    freq["centre"] = freq0 + np.arange(nfreq) * freq["width"]
    time = T0 + 10.0 * np.arange(ntime)

    ts = CHORDTimeStream(freq=freq, input=inp, prod=prod, time=time)
    for name, val in (
        ("input_info/pol", np.repeat([0, 1], ndish)),
        ("input_info/dish_idx", np.tile(np.arange(ndish), 2)),
        ("input_info/type", np.zeros(ninput, dtype=int)),
    ):
        ts.add_dataset(name)
        ts[name][:] = val
    ts.add_dataset("flags/valid_fpga_count")
    _set(ts["flags/valid_fpga_count"], np.full((nfreq, ntime), NSAMP))
    ts.add_dataset("flags/frac_lost")
    _set(ts["flags/frac_lost"], np.zeros((nfreq, ntime)))
    ts.input_flags[:] = 1

    auto = (1.0 + np.arange(ninput))[None, :, None] * (
        1.0 + 0.2 * np.sin(np.arange(ntime) / 7.0)
    )[None, None, :]
    auto = (
        np.broadcast_to(auto, (nfreq, ninput, ntime))
        * (1.0 + 0.01 * np.arange(nfreq))[:, None, None]
    )
    vis = (
        rng.standard_normal((nfreq, len(ia), ntime))
        + 1j * rng.standard_normal((nfreq, len(ia), ntime))
    ) * 1e-3
    vis[:, ia == ib] = auto[:, ia[ia == ib]]
    _set(ts.vis, vis)
    w = NSAMP / (auto[:, ia] * auto[:, ib])
    w = w * (1.0 + noise * rng.standard_normal(w.shape))
    if edit_weight is not None:
        edit_weight(w)
    _set(ts.weight, w)
    return ts


def rho_reference(ts, pa, pb, bad=()):
    """The radiometer statistic of the RFI notebooks, written out directly."""
    labels = list(ts.labels)
    pol = ts["input_info/pol"][:]
    dish = ts["input_info/dish_idx"][:]
    prod = ts.prod.tolist()
    vis = ts.vis[:].local_array
    w = ts.weight[:].local_array.astype(np.float64)
    nsamp = ts["flags/valid_fpga_count"][:].local_array.astype(np.float64)
    num = np.zeros(nsamp.shape)
    den = np.zeros(nsamp.shape)
    for a, b in itertools.combinations(range(len(labels)), 2):
        if dish[a] == dish[b] or labels[a] in bad or labels[b] in bad:
            continue
        if sorted("XY"[pol[a]] + "XY"[pol[b]]) != sorted(pa + pb):
            continue
        p = prod.index((a, b))
        ok = w[:, p] > 0
        num[ok] += 1.0 / w[:, p][ok]
        aa = vis[:, prod.index((a, a))].real
        bb = vis[:, prod.index((b, b))].real
        den[ok] += (aa * bb / nsamp)[ok]
    return np.sqrt(num / den)


def test_ratio_is_one_for_radiometer_noise():
    ts = make_ts()
    sens = _run(ComputeSystemSensitivity, ts, pol=["XX", "YY", "XY"])
    assert list(sens.pol) == ["XX", "YY", "XY"]
    ratio = sens.measured[:].local_array / sens.radiometer[:].local_array
    np.testing.assert_allclose(ratio, 1.0, rtol=1e-5)
    # 5 dishes: 10 cross-dish co-polar baselines, 20 cross-polar ones (XY and YX)
    np.testing.assert_array_equal(np.unique(sens.weight[:].local_array[:, 0]), [10])
    np.testing.assert_array_equal(np.unique(sens.weight[:].local_array[:, 2]), [20])


def test_matches_the_notebook_statistic():
    ts = make_ts(noise=0.05, seed=3)
    bad = ["A02X", "A02Y"]
    sens = _run(ComputeSystemSensitivity, ts, exclude_labels=bad)
    ratio = sens.measured[:].local_array / sens.radiometer[:].local_array
    for k, (pa, pb) in enumerate((("X", "X"), ("Y", "Y"))):
        np.testing.assert_allclose(
            ratio[:, k], rho_reference(ts, pa, pb, bad=bad), rtol=1e-5
        )
    np.testing.assert_array_equal(np.unique(sens.weight[:].local_array), [6])


def test_autos_and_same_dish_not_used():
    ts = make_ts()
    ia, ib = ts.prod["input_a"], ts.prod["input_b"]
    dish = ts["input_info/dish_idx"][:]
    same = dish[ia] == dish[ib]
    w = ts.weight[:].local_array
    w[:, same] = 1e-6  # a huge variance: would dominate if used
    sens = _run(ComputeSystemSensitivity, ts)
    np.testing.assert_allclose(
        sens.measured[:].local_array / sens.radiometer[:].local_array, 1.0, rtol=1e-5
    )


def _burst(w):
    w[5:8, :, 20:25] /= 4.0  # variance x 4 within the integration: ratio 2


def test_rfi_raises_the_ratio_and_flags_remove_baselines():
    ts = make_ts(edit_weight=_burst)
    flags = ts.input_flags[:]
    flags[0, 40:] = 0  # A01X flagged for the second part
    sens = _run(ComputeSystemSensitivity, ts)
    ratio = _full(sens.measured)[:, 0] / _full(sens.radiometer)[:, 0]
    np.testing.assert_allclose(ratio[5:8, 20:25], 2.0, rtol=1e-5)
    np.testing.assert_allclose(np.delete(ratio, np.s_[5:8], axis=0), 1.0, rtol=1e-5)
    nbl = _full(sens.weight)[:, 0]
    assert (nbl[:, :40] == 10).all() and (nbl[:, 40:] == 6).all()

    sens2 = _run(ComputeSystemSensitivity, ts, use_input_flags=False)
    assert (_full(sens2.weight)[:, 0] == 10).all()


def test_nsamp_from_bandwidth():
    ts = make_ts()
    _set(ts["flags/frac_lost"], np.full((48, 60), 0.5))
    sens = _run(ComputeSystemSensitivity, ts, nsamp="bandwidth")
    ref = _run(ComputeSystemSensitivity, ts)
    n_bw = 3200.0 / 16384 * 1e6 * 10.0 * 0.5
    np.testing.assert_allclose(
        sens.radiometer[:].local_array,
        ref.radiometer[:].local_array * np.sqrt(NSAMP / n_bw),
        rtol=1e-5,
    )


def test_unknown_product():
    with pytest.raises(ValueError):
        _run(ComputeSystemSensitivity, make_ts(), pol=["XX", "QQ"])


@pytest.mark.parametrize("mask_type", ["mad", "sumthreshold", "combine"])
def test_mask_flags_rfi_and_static(mask_type):
    # 600-660 MHz includes static bands (cellular, 617-652 MHz)
    nfreq, ntime = 300, 240

    def rfi(w):
        w[40:43, :, 100:110] /= 3.0  # a burst
        w[150, :, :] /= 1.5  # a persistent narrow line

    ts = make_ts(
        nfreq=nfreq, ntime=ntime, freq0=600.0, noise=0.02, seed=5, edit_weight=rfi
    )
    sens = _run(ComputeSystemSensitivity, ts)
    mask = _run(
        RFISensitivityMask,
        sens,
        mask_type=mask_type,
        base_size=[9, 31],
        mad_size=[21, 11],
        win_f_1d=31,
    )

    m = mask.mask[:]
    assert m.shape == (nfreq, ntime)
    assert m[40:43, 100:110].all()
    assert m[150].mean() > 0.9
    static = crfi.frequency_mask(sens.freq, timestamp=sens.time[0])
    assert static.any() and m[static].all()
    clean = ~static
    clean[[40, 41, 42, 150]] = False
    assert m[clean].mean() < 0.05
    assert mask.attrs["mask_type"] == mask_type


def test_mask_matches_chord_util():
    def rfi(w):
        w[60:62, :, 50:60] /= 3.0

    ts = make_ts(nfreq=200, ntime=200, freq0=440.0, noise=0.02, seed=7, edit_weight=rfi)
    sens = _run(ComputeSystemSensitivity, ts)
    params = dict(base_size=(9, 31), mad_size=(21, 11), win_f_1d=31)
    mask = _run(
        RFISensitivityMask,
        sens,
        mask_type="mad",
        include_pol=["XX"],
        **{k: list(v) if isinstance(v, tuple) else v for k, v in params.items()},
    )

    sens.redistribute("freq")
    ratio = _full(sens.measured)[:, 0] / _full(sens.radiometer)[:, 0]
    static = crfi.frequency_mask(sens.freq, timestamp=sens.time[0])
    ref = crfi.mad_sumthreshold_mask(
        ratio,
        np.broadcast_to(static[:, None], ratio.shape),
        sens.freq,
        mask_type="mad",
        **params,
    )
    np.testing.assert_array_equal(mask.mask[:], ref)


def test_optional_parameters_from_config():
    """nsigma_1d, tv_fraction and frac_lost_max can be null in a config."""
    task = RFISensitivityMask()
    task.read_config({"nsigma_1d": None, "tv_fraction": None, "frac_lost_max": None})
    assert (
        task.nsigma_1d is None
        and task.tv_fraction is None
        and task.frac_lost_max is None
    )
    task.read_config({"nsigma_1d": 4, "frac_lost_max": "0.2"})
    assert task.nsigma_1d == 4.0 and task.frac_lost_max == 0.2
