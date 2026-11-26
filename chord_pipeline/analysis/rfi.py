"""Tasks for RFI flagging of CHORD data.

- :class:`ComputeSystemSensitivity`: a copy of draco's
  :class:`~draco.analysis.sensitivity.ComputeSystemSensitivity` for a dish array.
- :class:`RFISensitivityMask`: an RFI mask from the ratio of the measured to the
  radiometric noise, with the MAD and SumThreshold flagger of
  :func:`chord_util.rfi.mad_sumthreshold_mask` and the CHORD static mask.

Both write draco containers (:class:`draco.core.containers.SystemSensitivity`
and :class:`draco.core.containers.RFIMask`), so they can be used with the draco
tasks that read and apply masks.

draco's ``RFISensitivityMask`` is not used as is because it flags every
frequency outside the 398-800 MHz TV band through ``tv_channels_flag``.
"""

from __future__ import annotations

import numpy as np
from caput import config
from caput.pipeline import tasklib
from chord_util import rfi as crfi
from draco.core import containers, io
from draco.util import tools


def _float_or_none(x):
    """Property type for an optional float (None is kept)."""
    return None if x is None else float(x)


class ComputeSystemSensitivity(tasklib.base.ContainerTask):
    """Compute the sensitivity of beamformed visibilities.

    A copy of :class:`draco.analysis.sensitivity.ComputeSystemSensitivity`
    (draco commit dde3c0f) for a dish array. Only two things are changed, marked
    with "CHORD:" below:

    - The baselines left out are those within a dish (the autocorrelations and
      the X-Y products of each dish) instead of those within a cylinder. As for
      the intracylinder baselines of CHIME, the noise of these baselines is not
      independent, which the radiometer equation assumes. Two feeds are on the
      same dish if they are closer than half the minimum separation of the
      dishes of the telescope.
    - The number of independent samples is the number counted by the X-engine
      (``flags/valid_fpga_count``), when the data has it. draco calculates it
      from the median integration time, but the X-engine integrations are not
      all of one length (they differ by one frame, 0.85 %).

    Parameters
    ----------
    exclude_intradish : bool
        Exclude the intradish baselines (the autocorrelations and the X-Y
        products of each dish) in the sensitivity estimate. Default is True.
        Note that a RuntimeError will be raised if exclude_intradish is True and
        the visibilities have already been stacked over dish.
    """

    # CHORD: replaces draco's exclude_intracyl (default False). The CHIME daily
    # pipeline runs with exclude_intracyl: true.
    exclude_intradish = config.Property(proptype=bool, default=True)

    def setup(self, telescope):
        """Save the telescope model.

        Parameters
        ----------
        telescope : TransitTelescope
            Telescope object to use
        """
        self.telescope = io.get_telescope(telescope)

    def process(self, data):
        """Estimate the sensitivity of the input data.

        Parameters
        ----------
        data : TODContainer
            Must have a weight property that contains an
            estimate of the inverse variance of the noise
            in each visibility.  The visibilities can be
            stacked to any level of redundancy.

        Returns
        -------
        metrics : SystemSensitivity
            Contains the measured and radiometric estimates of
            the noise in the beamformed visibilities.
        """
        # Ensure we are distributed over frequency. Get shape of visibilities.
        data.redistribute("freq")

        nfreq, _, ntime = data.vis.local_shape

        # Extract the input flags.  If container has a gain dataset,
        # then also check for the default gain 1.0 + 0.0j as this indicates
        # that an input was masked for a particular time and frequency.
        inpflg = data.input_flags[:].view(np.ndarray).astype(bool)
        niff = 1

        if "gain" in data.datasets:
            # Derive frequency dependent flags from gains
            gainflg = data.gain[:].local_array != (1.0 + 0.0j)
            inpflg = np.swapaxes(inpflg[np.newaxis, :, :] & gainflg, 0, 1)
            # Flatten frequency and time axis so we can use numpy's unique
            inpflg = inpflg.reshape(inpflg.shape[0], -1)
            niff = nfreq

        # Find unique sets of input flags
        uniq_inpflg, index_cnt = np.unique(inpflg, return_inverse=True, axis=1)

        # Calculate redundancy for each unique set of input flags
        cnt = tools.calculate_redundancy(
            uniq_inpflg.astype(np.float32),
            data.prod,
            data.reverse_map["stack"]["stack"],
            data.stack.size,
        )

        # Determine stack axis
        stack_new, stack_flag = tools.redefine_stack_index_map(
            self.telescope, data.input, data.prod, data.stack, data.reverse_map["stack"]
        )

        if not np.all(stack_flag):
            self.log.warning(
                f"There are {np.sum(~stack_flag):0.0f} stacked baselines that are "
                "masked in the telescope instance."
            )

        ps = data.prod[stack_new["prod"]]
        conj = stack_new["conjugate"]

        prodstack = ps.copy()
        prodstack["input_a"] = np.where(conj, ps["input_b"], ps["input_a"])
        prodstack["input_b"] = np.where(conj, ps["input_a"], ps["input_b"])

        # Figure out mapping between inputs in data file and inputs in telescope
        tel_index = tools.find_inputs(
            self.telescope.input_index, data.input, require_match=False
        )

        # Use the mapping to extract polarisation and position of each input
        input_pol = np.array(
            [
                self.telescope.polarisation[ti] if ti is not None else "N"
                for ti in tel_index
            ]
        )

        # CHORD: the full (EW, NS) position, where draco uses the EW position only
        position = np.array(
            [
                self.telescope.feedpositions[ti, :2]
                if ti is not None
                else (np.nan, np.nan)
                for ti in tel_index
            ],
            dtype=np.float64,
        ).reshape(-1, 2)

        # Next we determine indices into the stack axis for each polarisation product
        # The next three lines result in XY and YX being
        # combined into a single polarisation product
        pa, pb = input_pol[prodstack["input_a"]], input_pol[prodstack["input_b"]]
        pol_a = np.where(pa <= pb, pa, pb)
        pol_b = np.where(pa <= pb, pb, pa)

        # CHORD: np.char.add is the same function as draco's
        # np.core.defchararray.add, without the numpy deprecation warning
        baseline_pol = np.char.add(pol_a, pol_b)

        # CHORD: two feeds are on the same dish if they are closer than half the
        # minimum separation of the dishes. This replaces draco's test
        # |EW separation| < cylinder_width / 2. Feeds not in the telescope have
        # a NaN position and are never intradish.
        intra_dish_sep = 0.5 * min(self.telescope.u_width, self.telescope.v_width)

        def _intra_dish(ia, ib):
            dpos = position[ia] - position[ib]
            return np.hypot(dpos[..., 0], dpos[..., 1]) < intra_dish_sep

        if self.exclude_intradish:
            baseline_flag = ~_intra_dish(prodstack["input_a"], prodstack["input_b"])
        else:
            baseline_flag = np.ones(prodstack.size, dtype=bool)

        pol_uniq = [bp for bp in np.unique(baseline_pol) if "N" not in bp]
        pol_index = [
            np.flatnonzero((baseline_pol == up) & baseline_flag) for up in pol_uniq
        ]
        npol = len(pol_uniq)

        auto_flag = (prodstack["input_a"] == prodstack["input_b"]).astype(np.float32)

        if self.exclude_intradish and (np.sum(auto_flag) == npol):
            raise ValueError(
                "You have requested the exclusion of "
                "intradish baselines,  however it appears "
                "that the visibilities have already been stacked "
                "over dish, preventing calculation of the "
                "radiometric estimate."
            )

        # Dereference the weight dataset
        bweight = data.weight[:].local_array
        bflag = bweight > 0.0

        # Initialize arrays
        var = np.zeros((nfreq, npol, ntime), dtype=np.float32)
        counter = np.zeros((nfreq, npol, ntime), dtype=np.float32)

        # Average over selected baseline per polarization
        for pp, ipol in enumerate(pol_index):
            pcnt = cnt[ipol, :]
            pscale = 2.0 - auto_flag[ipol, np.newaxis]

            # Loop over frequencies to reduce memory usage
            for ff in range(nfreq):
                fslc = slice((ff % niff) * ntime, ((ff % niff) + 1) * ntime)
                pfcnt = pcnt[:, index_cnt[fslc]]

                pvar = tools.invert_no_zero(bweight[ff, ipol, :])
                pflag = bflag[ff, ipol, :].astype(np.float32)

                var[ff, pp, :] = np.sum(pfcnt**2 * pscale * pflag * pvar, axis=0)

                counter[ff, pp, :] = np.sum(pfcnt * pscale * pflag, axis=0)

        # Normalize
        var *= tools.invert_no_zero(counter**2)

        # Determine which of the stack indices correspond to autocorrelations
        auto_stack_id = np.flatnonzero(auto_flag)
        auto_input = prodstack["input_a"][auto_stack_id]
        auto_pol = input_pol[auto_input]

        auto_cnt = cnt[auto_stack_id, :][:, index_cnt]
        auto_cnt = np.swapaxes(auto_cnt.reshape(-1, niff, ntime), 0, 1)
        num_feed = auto_cnt * bflag[:, auto_stack_id, :].astype(np.float32)

        auto = data.vis[:, auto_stack_id, :].local_array.real

        # Construct the radiometric estimate of the noise by taking the sum
        # of the product of pairs of (possibly stacked) autocorrelations.
        radiometer = np.zeros((nfreq, npol, ntime), dtype=np.float32)
        radiometer_counter = np.zeros((nfreq, npol, ntime), dtype=np.float32)

        # CHORD: the same dish test for the pairs of autocorrelations
        auto_intra = _intra_dish(auto_input[:, np.newaxis], auto_input[np.newaxis, :])

        for ii, (ai, pi) in enumerate(zip(auto_input, auto_pol)):
            for jj, (aj, pj) in enumerate(zip(auto_input, auto_pol)):
                if self.exclude_intradish and auto_intra[ii, jj]:
                    # Exclude intradish baselines
                    continue

                # Combine XY and YX into single polarisation product
                pp = pol_uniq.index(pi + pj) if pi <= pj else pol_uniq.index(pj + pi)

                # Weight by the number of feeds that were averaged
                # together to obtain each stacked autocorrelation
                nsq = num_feed[:, ii, :] * num_feed[:, jj, :]

                radiometer[:, pp, :] += nsq * auto[:, ii, :] * auto[:, jj, :]

                radiometer_counter[:, pp, :] += nsq

        # Calculate number of independent samples from the
        # integration time, frequency resolution, and fraction of packets lost
        tint = np.median(np.abs(np.diff(data.time)))
        dnu = np.median(data.index_map["freq"]["width"]) * 1e6

        if ("flags" in data) and ("frac_lost" in data["flags"]):
            frac_lost = data["flags"]["frac_lost"][:].local_array
        else:
            frac_lost = np.zeros((1, 1), dtype=np.float32)

        # CHORD: the X-engine counts the valid samples of each integration, so use
        # that number when the data has it
        if ("flags" in data) and ("valid_fpga_count" in data["flags"]):
            nint = data["flags"]["valid_fpga_count"][:].local_array.astype(np.float64)
            nint = nint[:, np.newaxis, :]
        else:
            nint = dnu * tint * (1.0 - frac_lost[:, np.newaxis, :])

        # Normalize by the number of independent samples
        # and the total number of baselines squared
        radiometer *= tools.invert_no_zero(nint * radiometer_counter**2)

        # Create output container
        metrics = containers.SystemSensitivity(
            pol=np.array(pol_uniq, dtype="<U2"),
            axes_from=data,
            attrs_from=data,
            comm=data.comm,
            distributed=data.distributed,
        )

        metrics.redistribute("freq")

        # In order to write generic code for generating the radiometric
        # estimate of the sensitivity, we had to sum over the upper and lower triangle
        # of the visibility matrix.  Below we multiply by sqrt(2) in order to
        # obtain the sensitivity of the real component.
        metrics.radiometer[:] = np.sqrt(2.0 * radiometer)
        metrics.measured[:] = np.sqrt(2.0 * var)

        # Save the total number of baselines that were averaged in the weight dataset
        metrics.weight[:] = counter

        # Save the fraction of missing samples
        metrics.frac_lost[:] = frac_lost

        return metrics


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
