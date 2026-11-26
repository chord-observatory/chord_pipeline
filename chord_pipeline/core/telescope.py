"""Simple models for the CHORD array.

Identical to telescope objects in
:py:mod:`drift.telescope.disharray.core.PolarisedDishArray` and
:py:mod:`drift.telescope.disharray.core.PolarisedDishArraySurvey`,
but with the following CHORD defaults:
    * Frequency channels set to the CHORD band (300-1500 MHz with 3276 channels)
    * Observatory location set to the approximate location of the CHORD array
      (currently using CHIME location).
    * Layout set to a grid with 6.3 m EW spacing and 8.5 m NS spacing.
    * Beam pattern set to Airy with a 6 m diameter dish, 50 percent aperture efficiency,
      and -60 dB of cross-polar leakage.
    * System temperature set to 30 K specification and integration time set to 1 day.

Note that these defaults represent a large telescope object to simulate. For smaller
scale runs reduce the number of frequency channels using the `channel_range`
configuration parameter.
"""

import abc
import logging
from typing import Optional

import numpy as np

from caput import config
from caput.pipeline import Task
from chord_util import layout as chord_layout

from drift.core.telescope import PolarisedTelescope
from drift.telescope.disharray.core import MultiElevationSurvey, DishArrayMixin
from drift.telescope.disharray.beams import rotate_thetaphi_beam

# CHIME coordinates
CHORD_LATITUDE = 49.3207092194
CHORD_LONGITUDE = -119.6236774310
CHORD_ALTITUDE = 555.372  # m


class _CHORDDefaults(DishArrayMixin, config.Reader, metaclass=abc.ABCMeta):
    """Mixin for a CHORD 528-element array.

    Identical to :py:class:`drift.telescope.disharray.core.DishArrayMixin`
    but with defaults mentioned above.
    """

    # Nominal frequency channels for the intensity mapping backend
    freq_start = config.Property(proptype=float, default=0.0)
    freq_end = config.Property(proptype=float, default=1600.0)
    num_freq = config.Property(proptype=int, default=8192)
    channel_range = config.Property(proptype=list, default=[1536, 7681])

    # Location of the array
    latitude = config.Property(proptype=float, default=CHORD_LATITUDE)
    longitude = config.Property(proptype=float, default=CHORD_LONGITUDE)
    altitude = config.Property(proptype=float, default=CHORD_ALTITUDE)

    # System temperature
    tsys_flat = config.Property(proptype=float, default=30.0)
    ndays = config.Property(proptype=int, default=90)

    # Include auto-correlations
    auto_correlations = config.Property(proptype=bool, default=True)

    # Diameter of the dish in the E-W and N-S direction
    min_u = config.Property(proptype=float, default=6.0)
    min_v = config.Property(proptype=float, default=6.0)

    # Pointing offset in degrees from zenith in the elevation direction
    elevation_pointing_offset = config.Property(proptype=float, default=0.0)

    # Array layout
    layout_spec = config.Property(
        proptype=dict,
        default={
            "type": "grid",
            "grid_ew": 22,
            "grid_ns": 24,
            "spacing_ew": 6.3,
            "spacing_ns": 8.5,
        },
    )

    # Primary beam
    beam_spec = config.Property(
        proptype=dict,
        default={
            "type": "airy",
            "diameter": 6,
            "aperture_efficiency": 0.50,
            "crosspol_type": "scaled",
            "crosspol_scale_dB": -60,
        },
    )

    def beam(
        self, feed_ind: int, freq_ind: int, angpos: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Primary beam pattern. If a beam_obj from a `:py:class:.CustomDishArray`
        that supports a pointing argument is detected, the offset pointing is
        passed through. Otherwise the evaluated HEALPix beam pattern of the
        `single_pointing_telescope` is directly rotated.

        Parameters
        ----------
        feed_ind
            Index of the feed to pass to the `beam_obj`.
        freq_ind
            Frequency index to pass to the `beam_obj`.

        Returns
        -------
            (npix, 2) Beam pattern in sky theta, phi directions. May be complex.
        """
        if angpos is None:
            angpos = self._angpos

        ddec = self.elevation_pointing_offset  # In degrees
        if ddec == 0.0:
            # There is no pointing offset so call the baseclass.
            return super().beam(feed_ind, freq_ind, angpos)
        elif hasattr(self, "beam_obj") and getattr(
            self.beam_obj, "supports_pointing", False
        ):
            # We're working on a CustomDishArray with a beam object that supports a pointing
            # argument.
            altaz_pointing = np.radians(np.array([90 + ddec, 180]))
            return self.beam_obj(
                self, feed_ind, freq_ind, angpos, altaz_pointing=altaz_pointing
            )
        else:
            # We manually rotate the beam which is assumed to be at the zenith
            # pointing in sky Eth, Eph.
            beam = super().beam(feed_ind, freq_ind, angpos)
            return rotate_thetaphi_beam(beam, np.radians(-ddec), angpos)


class CHORD(_CHORDDefaults, PolarisedTelescope):
    """Single pointing CHORD 528-element array."""

    pass


class CHORDSurvey(MultiElevationSurvey, CHORD):
    """Multi-pointing CHORD 528-element array."""

    pass


class _CHORD64Defaults(_CHORDDefaults):
    # Array layout
    layout_spec = config.Property(
        proptype=dict,
        default={
            "type": "grid",
            "grid_ew": 11,
            "grid_ns": 6,
            "spacing_ew": 6.3,
            "spacing_ns": 8.5,
        },
    )


class CHORD64(_CHORD64Defaults, PolarisedTelescope):
    """Single pointing CHORD 64-element array."""

    pass


class CHORD64Survey(MultiElevationSurvey, CHORD64):
    """Multi-pointing CHORD 64-element array."""

    pass


class _InputLayout:
    """Minimal stand-in for a drift layout object: feed positions and polarisations."""

    def __init__(self, feedpositions, polarisation):
        self.feedpositions = feedpositions
        self.polarisation = polarisation


class CHORDPathfinder(_CHORDDefaults, PolarisedTelescope):
    """The CHORD pathfinder: one feed per correlator input.

    Unlike the grid telescopes above, the feeds are the inputs of a layout file
    (see :mod:`chord_util.layout`), in the order of the data, and the bad inputs are
    removed from the feed mask.

    Attributes
    ----------
    layout_file : str, optional
        Layout file. Default the pathfinder layout of chord_util.
    bad_inputs : list of str
        Labels of the inputs to remove.
    freq_list : list of float, optional
        Explicit channel centres in MHz, for data with a selection of channels.
        Default the channels of the CHORD band.
    """

    layout_file = config.Property(proptype=str, default=None)
    bad_inputs = config.Property(proptype=list, default=[])
    freq_list = config.Property(proptype=list, default=None)

    def _finalise_config(self):
        super()._finalise_config()
        layout = chord_layout.read_layout(self.layout_file)
        self.layout_obj = _InputLayout(
            np.column_stack([layout["ew_m"], layout["ns_m"]]).astype(np.float64),
            np.asarray(layout["pol"]).astype(str),
        )
        self.input_labels = np.asarray(layout["label"]).astype(str)
        self.input_chan_id = np.asarray(layout["chan_id"]).astype(int)

    def calculate_frequencies(self):
        """Use the explicit channel centres, if there are any."""
        if self.freq_list is None:
            return super().calculate_frequencies()
        self._frequencies = np.array(self.freq_list, dtype=np.float64)

    @property
    def input_index(self):
        """Index map of the inputs, to match them with those of the data."""
        out = np.zeros(
            len(self.input_labels),
            dtype=[("chan_id", "u2"), ("correlator_input", "U32")],
        )
        out["chan_id"] = self.input_chan_id
        out["correlator_input"] = self.input_labels
        return out

    def _unique_beams(self):
        beam_map, beam_mask = super()._unique_beams()
        good = ~np.isin(self.input_labels, self.bad_inputs)
        return beam_map, beam_mask & good[:, None] & good[None, :]


def _first_time(fh):
    """UNIX time of the first sample of a timestream or an X-engine file."""
    if "index_map/time" in fh:
        t0 = fh["index_map/time"][0]
        return float(t0["ctime"] if t0.dtype.names else t0)
    # X-engine file: the first filled time bin (unfilled bins have zero time)
    t_ns = fh["time_center_t_inst_ns"][:]
    return float(t_ns[t_ns != 0][0]) * 1e-9


class LoadPathfinderTelescope(Task):
    """Make the CHORD pathfinder telescope.

    The layout is the same every night, and the pointing and the bad inputs are not.
    Both of these are taken here from ``data_file``, unless they are given explicitly:
    the pointing from its attributes, and the bad inputs from the feed mask history
    (:func:`chord_util.layout.bad_inputs_at`) at its first time.

    Attributes
    ----------
    layout_file : str, optional
        Layout file. Default the pathfinder layout of chord_util.
    data_file : str, optional
        A timestream or X-engine file of the data to analyse.
    dec_pointing : float, optional
        Declination of the pointing in degrees. Default from the attributes of
        ``data_file``.
    bad_inputs : list of str, optional
        Labels of the bad inputs. Default from the feed mask history at the time of
        ``data_file``.
    feedmask_file : str, optional
        Feed mask history. Default that of chord_util.
    freq_file : str, optional
        A file with the channels of the data, when they are a selection of the band
        (for ``CollateProducts``).
    freq_list : list of float, optional
        The channel centres in MHz, instead of ``freq_file``.

    Notes
    -----
    The pointing sets only the beam model of the telescope. The eigen-calibration
    uses the source's own coordinates.
    """

    layout_file = config.Property(proptype=str, default=None)
    data_file = config.Property(proptype=str, default=None)
    dec_pointing = config.Property(proptype=float, default=None)
    bad_inputs = config.Property(proptype=list, default=None)
    feedmask_file = config.Property(proptype=str, default=None)
    freq_file = config.Property(proptype=str, default=None)
    freq_list = config.Property(proptype=list, default=None)

    def setup(self):
        """Return the telescope."""
        import h5py
        from chord_ephem.observers import chord

        dec, time = self.dec_pointing, None
        if self.data_file is not None:
            with h5py.File(self.data_file, "r") as fh:
                if dec is None:
                    dec = chord_layout.pointing_dec(fh.attrs)
                time = _first_time(fh)
        if dec is None:
            raise ValueError("Give data_file or dec_pointing.")

        bad = self.bad_inputs
        if bad is None:
            if time is None:
                raise ValueError("Give data_file or bad_inputs.")
            bad = sorted(chord_layout.bad_inputs_at(time, self.feedmask_file))

        freq_list = self.freq_list
        if self.freq_file is not None:
            with h5py.File(self.freq_file, "r") as fh:
                freq_list = [float(f) for f in fh["index_map/freq"]["centre"][:]]

        logging.getLogger(__name__).info(
            f"Pointing Dec {dec:.3f} deg, {len(bad)} bad inputs: {bad}"
        )
        conf = {
            "bad_inputs": list(bad),
            "latitude": chord.latitude,
            "longitude": chord.longitude,
            "elevation_pointing_offset": dec - chord.latitude,
        }
        if self.layout_file is not None:
            conf["layout_file"] = self.layout_file
        if freq_list is not None:
            conf["freq_list"] = freq_list
        return CHORDPathfinder.from_config(conf)
