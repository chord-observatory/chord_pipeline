"""Tasks for RFI flagging of CHORD data.

- :class:`ComputeSystemSensitivity`: draco's
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
from draco.analysis import sensitivity
from draco.core import containers
from draco.util import tools


def _float_or_none(x):
    """Property type for an optional float (None is kept)."""
    return None if x is None else float(x)


class _DishesAsCylinders:
    """A view of a telescope in which every dish is a cylinder, for draco.

    draco's ``ComputeSystemSensitivity`` leaves out the baselines within a
    cylinder (``exclude_intracyl``) by comparing the east-west separation of the
    two feeds with half of ``cylinder_width``. In this view the inputs at the same
    position (the X and Y feeds of a dish) share an east-west coordinate, the
    dishes are numbered 0, 1, 2, ... in metres, and ``cylinder_width`` is 1 m, so
    the baselines left out are the autocorrelations and the X-Y products of each
    dish. Every other attribute is that of the telescope.
    """

    cylinder_width = 1.0

    def __init__(self, telescope):
        self._telescope = telescope
        # Positions to the millimetre: the two feeds of a dish are at one position
        pos = np.round(np.asarray(telescope.feedpositions, dtype=np.float64), 3)
        _, dish = np.unique(pos, axis=0, return_inverse=True)
        dish = np.asarray(dish).ravel().astype(np.float64)
        self.feedpositions = np.column_stack([dish, np.zeros_like(dish)])
        self.ndish = int(dish.max()) + 1 if dish.size else 0

    def __getattr__(self, name):
        if name == "_telescope":
            raise AttributeError(name)
        return getattr(self._telescope, name)


class ComputeSystemSensitivity(sensitivity.ComputeSystemSensitivity):
    """draco's ``ComputeSystemSensitivity`` for a dish array.

    The calculation is draco's, with two differences for CHORD:

    - The autocorrelations and the X-Y products of each dish are left out, as
      the CHIME daily pipeline leaves out the baselines within a cylinder
      (``exclude_intracyl: true``): their noise is not independent, which the
      radiometer equation assumes. This is done with draco's own
      ``exclude_intracyl`` code, on a view of the telescope in which every dish
      is a cylinder (the feeds at one position are one dish). It is always on and
      is not a config parameter.
    - The number of samples of each integration is the one the X-engine counted
      (``flags/valid_fpga_count``). draco uses channel width x median integration
      time x (1 - ``frac_lost``), but the X-engine integrations are not all of
      one length (they differ by a frame, 0.85 %), which biases the radiometric
      noise of the short ones. The radiometric noise is rescaled by
      ``sqrt(n_draco / valid_fpga_count)``; without ``valid_fpga_count`` in the
      data, draco's value is kept.

    As in draco, the inputs used are those with nonzero ``input_flags``: remove
    the bad inputs before this task (e.g. with ``SelectInputs``) or flag them in
    ``input_flags``. The bad inputs of the telescope are not used for this. The
    output is draco's: the polarisation products are XX, XY (XY and YX together)
    and YY, ``measured`` and ``radiometer`` are the noise of the real part, and
    ``weight`` counts each baseline twice (both triangles of the visibility
    matrix).
    """

    # Always on (with every dish a cylinder); not a config parameter
    exclude_intracyl = True

    def setup(self, telescope):
        """Save the telescope, as a view in which every dish is a cylinder.

        Parameters
        ----------
        telescope : TransitTelescope or ProductManager
            The telescope, e.g. from ``LoadPathfinderTelescope``.
        """
        super().setup(telescope)
        self.telescope = _DishesAsCylinders(self.telescope)
        self.log.debug(f"{self.telescope.ndish} dishes in the telescope")

    def process(self, data):
        """Compute the sensitivity.

        Parameters
        ----------
        data : CHORDTimeStream
            Timestream with the autocorrelations and the weights.

        Returns
        -------
        sens : draco.core.containers.SystemSensitivity
            As from draco, with the radiometric noise for the number of samples
            that the X-engine counted.
        """
        sens = super().process(data)

        if "flags/valid_fpga_count" not in data:
            self.log.warning(
                "No flags/valid_fpga_count in the data: keeping draco's number of "
                "samples (channel width x integration time x (1 - frac_lost))."
            )
            return sens

        # draco's number of samples, as computed in draco
        tint = np.median(np.abs(np.diff(data.time)))
        dnu = np.median(data.index_map["freq"]["width"]) * 1e6
        if "flags/frac_lost" in data:
            frac_lost = data["flags/frac_lost"][:].local_array
        else:
            frac_lost = np.zeros((1, 1), dtype=np.float32)
        n_draco = dnu * tint * (1.0 - frac_lost)

        # The number the X-engine counted
        n_xeng = data["flags/valid_fpga_count"][:].local_array.astype(np.float64)

        scale = np.sqrt(n_draco * tools.invert_no_zero(n_xeng))
        scale = np.broadcast_to(scale, n_xeng.shape)
        sens.radiometer[:].local_array[:] *= scale[:, np.newaxis, :]
        sens.attrs["nsamp"] = "valid_fpga_count"

        good = n_xeng > 0
        if good.any():
            dev = np.abs(scale[good] - 1.0)
            self.log.debug(
                "Radiometric noise rescaled for valid_fpga_count: median "
                f"|scale - 1| {np.median(dev):.2e}, max {dev.max():.2e}"
            )
        return sens


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
