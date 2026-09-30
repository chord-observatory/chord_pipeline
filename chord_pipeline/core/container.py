"""Container definitions for CHORD"""

from __future__ import annotations

import posixpath
from typing import ClassVar, Iterable, Mapping

import h5py
import numpy as np
from caput.containers import COMPRESSION, COMPRESSION_OPTS
from caput.util import typeutils
from chord_util.andata import (  # noqa: F401
    INPUT_TYPE_DISH,
    INPUT_TYPE_FAKE,
    INPUT_TYPE_RFI,
)
from draco.core.containers import SiderealStream, TimeStream, TODContainer


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


def _freq_time_flag(dtype):
    """Dataset spec for a (freq, time) quality dataset from the X-engine."""
    return {
        "axes": ["freq", "time"],
        "dtype": dtype,
        "initialise": False,
        "distributed": True,
        "distributed_axis": "freq",
    }


def _small_dset(dtype, axes):
    """Dataset spec for a small, non-distributed dataset."""
    return {
        "axes": list(axes),
        "dtype": dtype,
        "initialise": False,
        "distributed": False,
    }


class RawContainer(TODContainer):
    """Base class for using raw CHORD data via the draco containers API.

    The CHORD analogue of ch_pipeline's ``RawContainer``. This modifies the set of
    allowed group/dataset names to allow access to the ``flags`` group (and any
    other groups listed in `_allowed_groups` by subclasses), which the draco
    containers otherwise forbid.
    """

    _allowed_groups: ClassVar[list[str]] = ["flags"]

    def group_name_allowed(self, name: str) -> bool:
        """Allow access to the groups in `_allowed_groups`."""
        # Strip leading and trailing "/"
        name = name.strip("/")

        return name in self._allowed_groups

    def dataset_name_allowed(self, name: str) -> bool:
        """Allow access to datasets in the root group and the allowed groups."""
        parent_name, name = posixpath.split(name)
        return parent_name == "/" or self.group_name_allowed(parent_name)

    @classmethod
    def from_acq_h5(
        cls,
        acq_files: str | list[str],
        start: int | None = None,
        stop: int | None = None,
        **kwargs,
    ) -> "RawContainer":
        """Load from an HDF5 file on disk.

        This is a thin wrapper around `from_file` to support the
        `andata.BaseData.from_acq_h5` API. The main difference is supporting
        the `start` and `stop` parameters. All other parameters are passed
        straight into `from_file`.

        This reads files that are already in container format (e.g. a saved
        :class:`CHORDTimeStream`). To read the raw X-engine files use
        :meth:`CHORDTimeStream.from_xengine_files`.

        Parameters
        ----------
        acq_files
            Path or glob to files (or list of).
        start, stop
            Indices into the full set of files to select.
        **kwargs
            Additional keyword arguments to pass to `from_file`

        Returns
        -------
        cont
            A single container instance.
        """
        if start is None and stop is None:
            pass
        elif start is not None and stop is not None:
            kwargs["time_sel"] = slice(start, stop)
        else:
            raise ValueError(
                f"Got mixed types for start ({type(start)}) and stop ({type(stop)})"
            )

        return cls.from_file(acq_files, **kwargs)


class CHORDTimeStream(TimeStream, RawContainer):
    """A container for CHORD X-engine visibility data.

    The CHORD analogue of ch_pipeline's ``CHIMETimeStream``: a
    :class:`draco.core.containers.TimeStream` holding the contents of a
    :class:`chord_util.andata.CorrData`, so it can be used by the draco and
    ch_pipeline tasks. Create it with :meth:`from_corrdata` (from a CorrData) or
    :meth:`from_xengine_files` (straight from the X-engine files); saved files
    are loaded with ``from_file`` like any draco container.

    Layout
    ------
    Standard ``TimeStream`` datasets:

    - ``vis``, ``vis_weight`` : (freq, stack, time). The data is unstacked, so
      the ``stack`` axis is identical to ``prod``.
    - ``input_flags`` : (input, time). 1 for a good input, 0 for a bad one,
      from the X-engine's per-element flags.

    Data quality (under ``flags/``; (freq, time) unless noted):

    - ``frac_lost``, ``frac_rfi``, ``frac_rfi_only``, ``frac_pl``: fraction of
      each integration lost overall, to RFI excision, to RFI only, and to
      packet loss.
    - ``frames_added``: 1 where a frame was received.
    - ``valid_fpga_count``, ``rfi_fpga_count``, ``rfi_only_fpga_count``,
      ``pl_fpga_count``: FPGA tick counts.
    - ``element_flags`` : (freq, input, time) the X-engine per-element flags
      (1.0 good, 0.0 bad).

    Timing (``time`` axis, copied from the files): ``bin_*``,
    ``time_center_*``, ``fpga_start_tick``, ``frame_length_fpga_ticks``,
    ``rfi_frame_excision_*`` and ``file_index`` (the entry of
    ``index_map/file`` each sample came from).

    Per input (under ``input_info/``): ``label``, ``type`` (-1 fake, 0 dish,
    1 RFI monitor), ``pol``, ``dish_idx``, ``grid_x_idx``, ``grid_y_idx``,
    ``coelev_disp_deg``, ``feed_pos_disp_m`` and ``feed_positions_m``
    (input, xyz), ``main_array_grid_indices`` (input, grid_xy).

    Other: ``digital_gain`` (freq, input) holds the F-engine gains that were
    divided out, ``dish_positions_in_grid_coords`` (dish, xyz),
    ``config_json`` (config) the kotekan/FPGA configuration snapshots, and
    optionally the X-engine eigen-decomposition ``eval``, ``evec``, ``erms``.

    The ``input`` index map has fields ``chan_id`` (the element index in the
    full-array ``CHORDBeamformer`` ordering, ``dish_idx + 64 * pol``, which
    does not change when inputs are removed) and ``correlator_input`` (the
    input label, e.g. ``B4p1``).
    """

    _dataset_spec: ClassVar = {
        # Data quality
        **{
            f"flags/{name}": _freq_time_flag(dtype)
            for name, dtype in [
                ("frac_lost", np.float32),
                ("frac_rfi", np.float32),
                ("frac_rfi_only", np.float32),
                ("frac_pl", np.float32),
                ("frames_added", np.uint8),
                ("valid_fpga_count", np.uint64),
                ("rfi_fpga_count", np.uint64),
                ("rfi_only_fpga_count", np.uint64),
                ("pl_fpga_count", np.uint64),
            ]
        },
        "flags/element_flags": {
            "axes": ["freq", "input", "time"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
            "compression": COMPRESSION,
            "compression_opts": COMPRESSION_OPTS,
            "chunks": (16, 64, 1024),
        },
        # X-engine eigen-decomposition (optional)
        "eval": {
            "axes": ["freq", "ev", "time"],
            "dtype": np.float32,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "evec": {
            "axes": ["freq", "ev", "input", "time"],
            "dtype": np.complex64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
        "erms": _freq_time_flag(np.float32),
        # F-engine gains that were divided out of the data
        "digital_gain": {
            "axes": ["freq", "input"],
            "dtype": np.complex64,
            "initialise": False,
            "distributed": True,
            "distributed_axis": "freq",
        },
        # Timing and binning information
        **{
            name: _small_dset(dtype, ["time"])
            for name, dtype in [
                ("bin_ERA_deg", np.float64),
                ("bin_abs_index", np.uint64),
                ("bin_delta_ut1_inst", np.float64),
                ("bin_start_ERA_deg", np.float64),
                ("bin_end_ERA_deg", np.float64),
                ("bin_start_ERAL_deg", np.float64),
                ("bin_end_ERAL_deg", np.float64),
                ("bin_t_inst_ns", np.int64),
                ("bin_ut1_ns", np.int64),
                ("bin_xp_as", np.float64),
                ("bin_yp_as", np.float64),
                ("time_center_t_inst_ns", np.int64),
                ("time_center_ut1_ns", np.int64),
                ("fpga_start_tick", np.uint64),
                ("frame_length_fpga_ticks", np.uint64),
                ("rfi_frame_excision_enabled", bool),
                ("rfi_frame_excision_num", np.int32),
                ("file_index", np.int32),
            ]
        },
        "rfi_frame_excision_threshold": _small_dset(np.float32, ["time", "threshold"]),
        "rfi_frame_excision_fraction": _small_dset(np.float32, ["time", "threshold"]),
        # Per-input description
        **{
            f"input_info/{name}": _small_dset(dtype, ["input"])
            for name, dtype in [
                ("label", "U16"),
                ("type", np.int32),
                ("pol", np.int32),
                ("dish_idx", np.int64),
                ("grid_x_idx", np.int64),
                ("grid_y_idx", np.int64),
                ("coelev_disp_deg", np.float64),
            ]
        },
        "input_info/feed_pos_disp_m": _small_dset(np.float64, ["input", "xyz"]),
        "input_info/feed_positions_m": _small_dset(np.float64, ["input", "xyz"]),
        "input_info/main_array_grid_indices": _small_dset(np.int64, ["input", "grid_xy"]),
        # Telescope description
        "dish_positions_in_grid_coords": _small_dset(np.float64, ["dish", "xyz"]),
        # kotekan/FPGA configuration snapshots (JSON strings of varying length)
        "config_json": _small_dset(h5py.string_dtype(), ["config"]),
    }

    # Axes used by the CHORD datasets, in addition to the TimeStream axes. They
    # are not required when creating a container, but are copied when present in
    # `axes_from`/`copy_from` (as draco tasks do via `empty_like`), so that all
    # the datasets can be recreated. `ev` and `pol_product` only exist if the
    # eigen-decomposition or radiometer test were loaded.
    _optional_axes: ClassVar[tuple[str, ...]] = (
        "xyz",
        "grid_xy",
        "dish",
        "threshold",
        "config",
        "file",
        "ev",
        "pol_product",
    )

    # Groups that may hold datasets, in addition to the root group
    _allowed_groups: ClassVar[list[str]] = ["flags", "input_info"]

    def __init__(self, *args, **kwargs):
        # Add the optional axes that are given, or defined in the container the
        # axes are copied from, as axes of this instance (see
        # `ContainerPrototype.axes`)
        source = kwargs.get("axes_from", kwargs.get("copy_from", None))
        extra = [
            ax
            for ax in self._optional_axes
            if ax in kwargs or (source is not None and ax in source.index_map)
        ]
        if extra:
            self._axes = tuple(extra)

        super().__init__(*args, **kwargs)

    @classmethod
    def from_corrdata(cls, data) -> "CHORDTimeStream":
        """Turn a :class:`chord_util.andata.CorrData` into a CHORDTimeStream.

        This makes a shallow copy of `data`, whose internals are destroyed in the
        process, so it should be discarded afterwards.

        Parameters
        ----------
        data : chord_util.andata.CorrData
            The data to convert.

        Returns
        -------
        newdata : CHORDTimeStream
            The correlator data as a CHORDTimeStream object.
        """
        newdata = cls(data_group=data, distributed=data.distributed, comm=data.comm)

        storage = newdata._storage_root._get_storage()

        # Move the datasets that have a different location in a TimeStream
        if "/flags/vis_weight" in newdata:
            storage["vis_weight"] = storage["flags"].pop("vis_weight")
            storage["vis_weight"]._name = "/vis_weight"

        if "/flags/inputs" in newdata:
            storage["input_flags"] = storage["flags"].pop("inputs")
            storage["input_flags"]._name = "/input_flags"

        # The data is unstacked, so label the product axis `stack` as the
        # TimeStream spec expects
        for name in ["vis", "vis_weight"]:
            if name in newdata:
                axes = typeutils.bytes_to_unicode(newdata[name].attrs["axis"])
                newdata[name].attrs["axis"] = np.array(
                    ["stack" if ax == "prod" else ax for ax in axes]
                )

        # Remove anything not in the spec
        contains = set(newdata.datasets)
        for group in cls._allowed_groups:
            if group in newdata:
                contains |= {f"{group}/{name}" for name in newdata[group]}

        for name in contains:
            if name not in newdata.dataset_spec:
                del newdata[name]

        return newdata

    @classmethod
    def from_xengine_files(cls, acq_files, start=None, stop=None, **kwargs):
        """Read X-engine files straight into a CHORDTimeStream.

        Equivalent to ``from_corrdata(andata.CorrData.from_acq_h5(...))``.

        Parameters
        ----------
        acq_files : filename, list of filenames or filename pattern
            The X-engine files.
        start, stop : int, optional
            Range of (filled) time samples to read.
        **kwargs
            Passed to :meth:`chord_util.andata.CorrData.from_acq_h5` (e.g.
            ``freq_sel``, ``input_sel``, ``datasets``, ``distributed``).

        Returns
        -------
        ts : CHORDTimeStream
            The data.
        """
        from chord_util import andata

        data = andata.CorrData.from_acq_h5(acq_files, start=start, stop=stop, **kwargs)
        return cls.from_corrdata(data)

    @property
    def frac_lost(self):
        """The fraction of each integration that was lost."""
        return self["flags/frac_lost"]

    @property
    def flags(self):
        """The group of data quality flags."""
        return self["flags"]

    @property
    def input_type(self) -> np.ndarray:
        """Input type (-1 fake, 0 dish, 1 RFI monitor) of each input."""
        return self["input_info/type"][:]

    @property
    def is_rfi_monitor(self) -> np.ndarray:
        """Boolean mask selecting the RFI monitor inputs."""
        return self.input_type == INPUT_TYPE_RFI

    @property
    def is_dish(self) -> np.ndarray:
        """Boolean mask selecting the dish inputs."""
        return self.input_type == INPUT_TYPE_DISH

    @property
    def labels(self) -> np.ndarray:
        """Human readable label of each input (e.g. `B4p1`, `RFIA1p2`)."""
        return np.array(
            [
                lbl.decode() if isinstance(lbl, bytes) else str(lbl)
                for lbl in self.input["correlator_input"]
            ]
        )
