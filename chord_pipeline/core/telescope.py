"""Simple models for the CHORD array.

Identical to telescope objects in
:py:mod:`drift.telescope.custom_disharray.core.PolarisedDishArray` and 
:py:mod:`drift.telescope.custom_disharray.core.PolarisedDishArraySurvey`,
but with the following CHORD defaults:
    * Frequency channels set to the CHORD band (300-1500 MHz with 2048 channels)
    * Observatory location set to the approximate location of the CHORD array
      (currently using CHIME location).
    * A grid array layout with 6.3 m EW spacing and 9.0 m NS spacing for the 
      :py:class:`CHORD64` and :py:class:`CHORD64Survey`.
    * An airy beam pattern with a 6 m diameter dish and -60 dB of cross-polar leakage.

Note that these defaults represent a large telescope object to simulate. For smaller
scale runs reduce the number of frequency channels using the `channel_range`
configuration parameter.
"""
import abc
from caput import config

from drift.core.telescope import PolarisedTelescope
from drift.telescope.custom_disharray.core import MultiElevationSurvey, CustomDishArray


# CHIME coordinates
CHORD_LATITUDE = 49.3207092194
CHORD_LONGITUDE = -119.6236774310
CHORD_ALTITUDE = 555.372  # m


class _CHORD64Defaults(CustomDishArray, config.Reader, metaclass=abc.ABCMeta):
    """Mixin for a CHORD 64-element array.
    
    Identical to :py:class:`drift.telescope.custom_disharray.core.CustomDishArray`
    but with defaults mentioned above.
    """

    freq_start = config.Property(proptype=float, default=300)
    freq_end = config.Property(proptype=float, default=1500)
    num_freq = config.Property(proptype=int, default=2048)

    latitude = config.Property(proptype=float, default=CHORD_LATITUDE)
    longitude = config.Property(proptype=float, default=CHORD_LONGITUDE)
    altitude = config.Property(proptype=float, default=CHORD_ALTITUDE)

    layout_spec = config.Property(
        proptype=dict,
        default={
            "type": "grid",
            "grid_ew": 11,
            "grid_ns": 6,
            "spacing_ew": 6.3,
            "spacing_ns": 9.0,
        },
    )

    beam_spec = config.Property(
        proptype=dict,
        default={
            "type": "airy",
            "diameter": 6,
            "crosspol_type": "scaled",
            "crosspol_scale_dB": -60,
        },
    )


class CHORD64(_CHORD64Defaults, PolarisedTelescope):
    """Single pointing CHORD 64-element array.
    """

    pass


class CHORD64Survey(MultiElevationSurvey, CHORD64):
    """Multi-pointing CHORD 64-element array.
    """

    pass
