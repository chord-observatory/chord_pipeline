"""Tasks for RFI flagging of CHORD data.

The CHORD versions of draco's sensitivity-based RFI flagging:

- :class:`ComputeSystemSensitivity`: the measured noise of the cross-dish
  baselines (from the visibility weights) and its radiometer estimate (from the
  autocorrelations), per polarisation product.
- :class:`RFISensitivityMask`: an RFI mask from their ratio, with the MAD and
  SumThreshold flagger of :func:`chord_util.rfi.mad_sumthreshold_mask` and the
  CHORD static mask.

Both write draco containers (:class:`draco.core.containers.SystemSensitivity`
and :class:`draco.core.containers.RFIMask`), so they can be used with the draco
tasks that read and apply masks.

Why not draco's tasks: draco's ``ComputeSystemSensitivity`` can only exclude
baselines by east-west separation and a cylinder width, which has no meaning for
a dish array. Without that, it includes the
autocorrelations and the same-dish products, whose correlated noise biases the
ratio, it uses the nominal integration time for the number of samples, and it
combines XY and YX into one product. Its mask task flags every frequency outside
the 398-800 MHz TV band through ``tv_channels_flag``.
"""

from __future__ import annotations

import numpy as np
from caput import config
from caput.pipeline import tasklib
from chord_util import rfi as crfi
from draco.core import containers

from ..core.container import INPUT_TYPE_DISH

_POLS = ("X", "Y")

# Products of draco's flagger kept here: the products are flagged separately,
# and a sample is flagged if it is flagged in any of them
_PRODUCTS = ("XX", "YY", "XY")


def _float_or_none(x):
    """Property type for an optional float (None is kept)."""
    return None if x is None else float(x)


def _pol_product(pa, pb):
    """Name of the polarisation product of two inputs, XY and YX combined."""
    return "".join(sorted(_POLS[pa] + _POLS[pb]))


class ComputeSystemSensitivity(tasklib.base.ContainerTask):
    """Measured and radiometric noise of the cross-dish baselines.

    For each polarisation product ``p`` and each (frequency, time), over the
    baselines ``ab`` between two different, good dishes:

    - measured noise: ``sqrt(sum_ab 1 / w_ab) / n``, from the visibility weights
      ``w_ab`` (the inverse variance that the X-engine measured within the
      integration);
    - radiometer estimate: ``sqrt(sum_ab V_aa V_bb / N) / n``, from the
      autocorrelations, with ``N`` independent samples per integration;

    where ``n`` is the number of baselines. Their ratio is 1 for Gaussian noise,
    and is raised by RFI that varies within the integration. It is the statistic
    ``rho`` of the X-engine RFI notebooks.

    The autocorrelations and the X-Y products of one dish are not used: their
    noise is correlated with the autocorrelations. Only dish inputs are used.

    Attributes
    ----------
    pol : list of str
        Polarisation products to compute: "XX", "YY" and/or "XY" (the cross-polar
        products of different dishes, XY and YX combined). Default ``["XX", "YY"]``.
    exclude_labels : list of str
        Labels of inputs not to use (e.g. dead or hot inputs), in addition to
        those flagged in ``input_flags``. Default none.
    use_input_flags : bool
        Exclude the inputs where ``input_flags`` (the X-engine's bad-feed mask) is
        zero, per time. Default True.
    nsamp : str
        Number of independent samples per integration: "valid_fpga_count" (the
        number of valid samples that the X-engine recorded, per frequency and
        time) or "bandwidth" (channel width times integration time times
        1 - ``frac_lost``, as in draco). Default "valid_fpga_count".
    prod_block : int
        Number of products processed at once (memory). Default 256.
    """

    pol = config.list_type(str, default=["XX", "YY"])
    exclude_labels = config.Property(proptype=list, default=[])
    use_input_flags = config.Property(proptype=bool, default=True)
    nsamp = config.enum(["valid_fpga_count", "bandwidth"], default="valid_fpga_count")
    prod_block = config.Property(proptype=int, default=256)

    def process(self, data):
        """Compute the sensitivity.

        Parameters
        ----------
        data : CHORDTimeStream
            Unstacked timestream with the autocorrelations, the weights, the
            ``input_info`` datasets and (for ``nsamp="valid_fpga_count"``)
            ``flags/valid_fpga_count``.

        Returns
        -------
        sens : draco.core.containers.SystemSensitivity
            ``measured`` and ``radiometer`` as above, ``weight`` the number of
            baselines used, ``frac_lost`` copied from the data.
        """
        for p in self.pol:
            if p not in _PRODUCTS:
                raise ValueError(
                    f"Unknown polarisation product {p!r}, use one of {_PRODUCTS}."
                )

        data.redistribute("freq")

        labels = data.labels
        pol = np.asarray(data["input_info/pol"][:]).astype(int)
        dish = np.asarray(data["input_info/dish_idx"][:])
        good = (np.asarray(data["input_info/type"][:]) == INPUT_TYPE_DISH) & ~np.isin(
            labels, self.exclude_labels
        )

        prod = data.prod
        ia, ib = prod["input_a"].astype(int), prod["input_b"].astype(int)
        auto_prod = np.full(len(labels), -1)
        auto_prod[ia[ia == ib]] = np.flatnonzero(ia == ib)
        if (auto_prod[good] < 0).any():
            raise RuntimeError(
                "The data must contain the autocorrelations of all the inputs used."
            )

        vis = data.vis[:].local_array
        weight = data.weight[:].local_array
        nfreq, _, ntime = vis.shape

        # Autocorrelations, NaN-free, and where they can be used
        auto = np.zeros((nfreq, len(labels), ntime), dtype=np.float64)
        auto_ok = np.zeros((nfreq, len(labels), ntime), dtype=bool)
        auto[:, good] = vis[:, auto_prod[good]].real
        auto_ok[:, good] = (weight[:, auto_prod[good]] > 0) & (auto[:, good] > 0)

        if self.use_input_flags:
            inp_ok = np.asarray(data.input_flags[:]).astype(bool)
        else:
            inp_ok = np.ones((len(labels), ntime), dtype=bool)

        nsamp = self._nsamp(data)
        nsamp_ok = nsamp > 0
        inv_nsamp = np.where(nsamp_ok, 1.0 / np.where(nsamp_ok, nsamp, 1.0), 0.0)

        pname = np.array([_pol_product(pa, pb) for pa, pb in zip(pol[ia], pol[ib])])
        cross = good[ia] & good[ib] & (dish[ia] != dish[ib])

        measured = np.zeros((nfreq, len(self.pol), ntime), dtype=np.float32)
        radiometer = np.zeros_like(measured)
        counter = np.zeros_like(measured)

        for pp, p in enumerate(self.pol):
            sel = np.flatnonzero(cross & (pname == p))
            num = np.zeros((nfreq, ntime), dtype=np.float64)
            den = np.zeros((nfreq, ntime), dtype=np.float64)
            cnt = np.zeros((nfreq, ntime), dtype=np.float64)
            for start in range(0, len(sel), self.prod_block):
                blk = sel[start : start + self.prod_block]
                a, b = ia[blk], ib[blk]
                w = weight[:, blk].astype(np.float64)
                ok = (
                    (w > 0)
                    & auto_ok[:, a]
                    & auto_ok[:, b]
                    & inp_ok[a][np.newaxis]
                    & inp_ok[b][np.newaxis]
                    & nsamp_ok[:, np.newaxis]
                )
                num += np.sum(np.where(ok, 1.0 / np.where(ok, w, 1.0), 0.0), axis=1)
                den += np.sum(np.where(ok, auto[:, a] * auto[:, b], 0.0), axis=1)
                cnt += ok.sum(axis=1)
            inv_cnt = np.where(cnt > 0, 1.0 / np.where(cnt > 0, cnt, 1.0), 0.0)
            measured[:, pp] = np.sqrt(num) * inv_cnt
            radiometer[:, pp] = np.sqrt(den * inv_nsamp) * inv_cnt
            counter[:, pp] = cnt
            self.log.debug(f"{p}: {len(sel)} cross-dish baselines")

        sens = containers.SystemSensitivity(
            pol=np.array(self.pol, dtype="<U2"),
            axes_from=data,
            attrs_from=data,
            comm=data.comm,
            distributed=data.distributed,
        )
        sens.redistribute("freq")
        sens.measured[:] = measured
        sens.radiometer[:] = radiometer
        sens.weight[:] = counter
        if "flags/frac_lost" in data:
            sens.frac_lost[:] = data["flags/frac_lost"][:].local_array
        else:
            sens.frac_lost[:] = 0.0
        sens.attrs["nsamp"] = self.nsamp
        return sens

    def _nsamp(self, data):
        """Independent samples per integration, (local freq, time)."""
        if self.nsamp == "valid_fpga_count":
            return data["flags/valid_fpga_count"][:].local_array.astype(np.float64)

        nfreq = data.vis.local_shape[0]
        sf = data.vis.local_offset[0]
        width = data.index_map["freq"]["width"][sf : sf + nfreq] * 1e6
        tint = np.median(np.abs(np.diff(data.time)))
        if "flags/frac_lost" in data:
            frac_lost = data["flags/frac_lost"][:].local_array
        else:
            frac_lost = np.zeros((nfreq, len(data.time)))
        return width[:, np.newaxis] * tint * (1.0 - frac_lost)


class RFISensitivityMask(tasklib.base.ContainerTask):
    """RFI mask from the ratio of the measured to the radiometric noise.

    The ratio of each polarisation product is flagged with
    :func:`chord_util.rfi.mad_sumthreshold_mask` (the algorithm of draco's
    ``RFISensitivityMask``, as used in the X-engine RFI notebooks), starting from
    the CHORD static mask (:func:`chord_util.rfi.frequency_mask`) and the samples
    with no baselines. A sample is flagged if it is flagged in any product.

    With ``mask_type="combine"``, MAD is used around the transits of bright sources
    and of the Sun (:func:`chord_util.rfi.bright_source_flag`), where SumThreshold
    flags real signal, and SumThreshold elsewhere (as in CHIME).

    Attributes
    ----------
    mask_type : str
        "mad", "sumthreshold" or "combine". Default "combine".
    include_pol : list of str, optional
        Polarisation products to flag. Default all of them.
    nsigma_1d, quantile_1d, win_f_1d
        The channel-flagging stage (`chord_util.rfi.mask_1d`); ``nsigma_1d=None``
        skips it. Defaults 5.0, 0.15, 191.
    nsigma, niter, rho
        Final threshold (sigma), number of iterations, and the factor by which
        the threshold drops at each. Defaults 5.0, 5, 1.5.
    base_size, mad_size : list of int
        (channels, times) of the moving medians for the background and the noise.
        Defaults (37, 181) and (101, 31).
    tv_fraction : float, optional
        Also flag whole 6 MHz TV channels where more than this fraction of the
        samples exceed the threshold. Default None (the static mask covers the TV
        channels seen at DRAO).
    max_m : int
        Largest SumThreshold window. Default 64.
    sir_eta : float
        Aggressiveness of SIR for "combine". Default 0.2.
    frac_lost_max : float, optional
        Samples where more than this fraction of the integration was lost
        (``frac_lost``) count as missing (cf. ch_pipeline's ``MaskHighFracLost``).
        Default 0.1, as in the X-engine RFI notebooks; None to not use it.
    unmask_bands : list of [float, float], optional
        Frequency ranges (MHz) where the static mask is not applied, e.g. to test
        bands filtered in the X-engine. The channels are then flagged by their
        statistics alone.
        Default none.
    sources : list of str
        Sources (``chord_util.rfi.bright_source_flag``) whose transits use MAD
        for "combine". Default Cas A, Cyg A, Tau A, B0329+54.
    nsigma_source, nsigma_sun : float
        Half-width of those windows, in primary-beam sigmas. Defaults 1.0 and 3.0.
    """

    mask_type = config.enum(["mad", "sumthreshold", "combine"], default="combine")
    include_pol = config.list_type(str, default=None)

    nsigma_1d = config.Property(proptype=_float_or_none, default=5.0)
    quantile_1d = config.Property(proptype=float, default=0.15)
    win_f_1d = config.Property(proptype=int, default=191)

    nsigma = config.Property(proptype=float, default=5.0)
    niter = config.Property(proptype=int, default=5)
    rho = config.Property(proptype=float, default=1.5)
    base_size = config.list_type(int, length=2, default=(37, 181))
    mad_size = config.list_type(int, length=2, default=(101, 31))
    tv_fraction = config.Property(proptype=_float_or_none, default=None)
    max_m = config.Property(proptype=int, default=64)
    sir_eta = config.Property(proptype=float, default=0.2)
    frac_lost_max = config.Property(proptype=_float_or_none, default=0.1)
    unmask_bands = config.Property(proptype=list, default=[])

    sources = config.list_type(str, default=["CAS_A", "CYG_A", "TAU_A", "B0329+54"])
    nsigma_source = config.Property(proptype=float, default=1.0)
    nsigma_sun = config.Property(proptype=float, default=3.0)

    def process(self, sens):
        """Make the mask.

        Parameters
        ----------
        sens : draco.core.containers.SystemSensitivity
            From :class:`ComputeSystemSensitivity`.

        Returns
        -------
        mask : draco.core.containers.RFIMask
            True where the data is to be masked: RFI, the static mask, and the
            missing samples (no baselines, or ``frac_lost > frac_lost_max``).
        """
        # All frequencies and times of a product are needed at once
        sens.redistribute("pol")

        freq = sens.freq
        times = sens.time
        pols = list(sens.pol)
        po = sens.measured[:].local_offset[1]
        npol_local = sens.measured[:].local_shape[1]

        measured = sens.measured[:].local_array
        radiometer = sens.radiometer[:].local_array
        nbl = sens.weight[:].local_array
        lost = np.zeros((len(freq), len(times)), dtype=bool)
        if self.frac_lost_max is not None:
            lost = _full(sens.frac_lost) > self.frac_lost_max

        static = crfi.frequency_mask(freq, timestamp=times[0])
        for lo, hi in self.unmask_bands:
            static &= ~((freq >= lo) & (freq <= hi))
        madtimes = None
        if self.mask_type == "combine":
            madtimes = crfi.bright_source_flag(
                times,
                freq,
                sources=tuple(self.sources),
                nsigma_source=self.nsigma_source,
                nsigma_sun=self.nsigma_sun,
            )

        local = np.zeros((len(freq), len(times)), dtype=bool)
        for k in range(npol_local):
            p = pols[po + k]
            if self.include_pol and p not in self.include_pol:
                continue
            missing = (nbl[:, k] == 0) | (radiometer[:, k] == 0) | lost
            ratio = np.where(
                missing, 0.0, measured[:, k] / np.where(missing, 1.0, radiometer[:, k])
            )
            m, stats = crfi.mad_sumthreshold_mask(
                ratio,
                missing | static[:, np.newaxis],
                freq,
                mask_type=self.mask_type,
                madtimes=madtimes,
                nsigma_1d=self.nsigma_1d,
                quantile_1d=self.quantile_1d,
                win_f_1d=self.win_f_1d,
                nsigma=self.nsigma,
                niter=self.niter,
                rho=self.rho,
                base_size=tuple(self.base_size),
                mad_size=tuple(self.mad_size),
                tv_fraction=self.tv_fraction,
                max_m=self.max_m,
                sir_eta=self.sir_eta,
                return_stats=True,
            )
            extra = stats[-1]["total"] - stats[0]["total"]
            self.log.info(
                f"{p}: {stats[-1]['total']:.2f} % flagged "
                f"({extra:.2f} % beyond the static mask and missing data)"
            )
            local |= m

        final = _any_over_ranks(local, sens.comm)
        self.log.info(f"{100 * final.mean():.2f} % of the samples are masked")

        mask = containers.RFIMask(axes_from=sens, attrs_from=sens)
        mask.mask[:] = final
        mask.attrs["mask_type"] = self.mask_type
        return mask


def _full(dset):
    """The full array of a (possibly distributed) dataset."""
    arr = dset[:]
    return arr.allgather() if hasattr(arr, "allgather") else np.asarray(arr)


def _any_over_ranks(mask, comm):
    """Logical or of a boolean array over all MPI ranks."""
    if comm is None or comm.size == 1:
        return mask
    from mpi4py import MPI

    buf = np.ascontiguousarray(mask, dtype=np.uint8)
    out = np.empty_like(buf)
    comm.Allreduce(buf, out, op=MPI.MAX)
    return out.astype(bool)
