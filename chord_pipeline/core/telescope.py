"""Simple models for the CHORD array.

Identical to telescope objects in
:py:mod:`drift.telescope.custom_disharray.core.PolarisedDishArray` and
:py:mod:`drift.telescope.custom_disharray.core.PolarisedDishArraySurvey`,
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
from typing import Optional

import numpy as np

from caput import config

from drift.core.telescope import PolarisedTelescope
from drift.telescope.custom_disharray.core import MultiElevationSurvey, CustomDishArray
from drift.telescope.custom_disharray.beams import rotate_thetaphi_beam

# CHIME coordinates
CHORD_LATITUDE = 49.3207092194
CHORD_LONGITUDE = -119.6236774310
CHORD_ALTITUDE = 555.372  # m


class _CHORD64Defaults(CustomDishArray, config.Reader, metaclass=abc.ABCMeta):
    """Mixin for a CHORD 64-element array.

    Identical to :py:class:`drift.telescope.custom_disharray.core.CustomDishArray`
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
            "grid_ew": 11,
            "grid_ns": 6,
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


class _CHORDDefaults(_CHORD64Defaults):

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


class CHORD64(_CHORD64Defaults, PolarisedTelescope):
    """Single pointing CHORD 64-element array."""

    pass


class CHORD64Survey(MultiElevationSurvey, CHORD64):
    """Multi-pointing CHORD 64-element array."""

    pass


class CHORD(_CHORDDefaults, PolarisedTelescope):
    """Single pointing CHORD 64-element array."""

    pass


class CHORDSurvey(MultiElevationSurvey, CHORD):
    """Multi-pointing CHORD 64-element array."""

    pass
