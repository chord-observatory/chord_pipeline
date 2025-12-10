"""Container definitions for CHORD"""

from __future__ import annotations

from typing import Iterable, Mapping

import h5py
import numpy as np
from draco.core.containers import SiderealStream


class CHORDSiderealStream(SiderealStream):
    """Sidereal stream container specialized for CHORD data.

    Based on the :class:`draco.core.containers.SiderealStream`, with additional
    helpers for CHORD visibility data and metadata.
    """

    @classmethod
    def from_hdf5(
        cls,
        filename: str,
        *,
        ra_source: str = "ERA",
        weights: str | None = "vis_weights",
    ) -> "CHORDSiderealStream":
        """Load an unstacked CHORD visibility file into a SiderealStream.

        Parameters
        ----------
        filename
            Path to the CHORD visibility HDF5 file.
        ra_source
            Which RA-like dataset to use for the RA axis. `"ERA"` uses
            `bin_start_ERA_deg`/`bin_end_ERA_deg`; `"LAST"` uses the LAST
            equivalents. You can also pass the name of a dataset holding RA
            centres in degrees.
        weights
            Name of the weight dataset to read. Default "vis_weights".
            If None, weights are set to ones.
        """
        with h5py.File(filename, "r") as fh:
            # Check if index_map group exists
            if "index_map" not in fh:
                raise ValueError(
                    f"File {filename} is missing 'index_map' group; "
                    "not a valid CHORD visibility file."
                )
                
            freq = np.array(fh["index_map/freq"][:])
            prod = np.array(fh["index_map/prod"][:])

            # Input count: prefer attribute, fall back to prod map
            ninput = int(fh.attrs.get("num_elements", prod["input_b"].max() + 1))

            # Compute RA centres
            if ra_source.lower() == "era":
                start = np.array(fh["bin_start_ERA_deg"][:], dtype=float)
                end = np.array(fh["bin_end_ERA_deg"][:], dtype=float)
                ra = 0.5 * (start + end)
            elif ra_source.lower() == "last":
                start = np.array(fh["bin_start_LAST"][:], dtype=float)
                end = np.array(fh["bin_end_LAST"][:], dtype=float)
                ra = 0.5 * (start + end)
            else:
                ra = np.array(fh[ra_source][:], dtype=float)

            vis = fh["vis"][:]

            if weights and weights in fh:
                vis_weights = np.array(fh[weights][:], dtype=np.float32)
            else:
                vis_weights = np.ones_like(vis, dtype=np.float32)

        cont = cls(freq=freq, ra=ra, prod=prod, input=ninput)
        cont.vis[:] = vis
        cont.weight[:] = vis_weights

        # Attach additional data
        cont.attrs["source_file"] = filename
        cont.attrs["ra_source"] = ra_source

        # TODO: Attach other fields?
        # ...

        return cont
