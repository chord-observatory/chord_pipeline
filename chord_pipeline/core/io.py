"""Tasks for IO.

Tasks for finding and loading the CHORD X-engine visibility files. These mirror
ch_pipeline's ``QueryDatabase`` and ``LoadCorrDataFiles``, but use
:mod:`chord_util.andata` in place of :mod:`ch_util.andata`.

A typical configuration to concatenate a whole acquisition into one timestream:

.. code-block:: yaml

    - type: chord_pipeline.core.io.QueryAcquisitionFiles
      out: filelist
      params:
        acquisition: acq_20260911_232919_986789057

    - type: chord_pipeline.core.io.LoadCorrDataFiles
      requires: filelist
      out: tstream
      params:
        save: true
        output_name: "tstream_{tag}.h5"
"""

from __future__ import annotations

import gc
import glob
import os
import re

import h5py
import numpy as np
from caput import config
from caput.pipeline import exceptions, tasklib
from chord_util import andata

from .container import CHORDTimeStream

# Where the pathfinder X-engine data lives on the cluster
DEFAULT_DATA_ROOT = "/project/rrg-kmsmith/chord-data/kotekan_vis_files/subset"

_FNAME_RE = re.compile(r"vis_(\d+)_\d{8}T_\d{6}_\d{9}\.h5$")


def _abs_file_idx(path):
    """The absolute file index encoded in an X-engine file name."""
    m = _FNAME_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


class QueryAcquisitionFiles(tasklib.base.MPILoggedTask):
    """Find the X-engine files of an acquisition.

    The CHORD analogue of ch_pipeline's ``QueryDatabase``: returns (from
    `setup`) the list of files to pass to :class:`LoadCorrDataFiles`.

    Attributes
    ----------
    acquisition : str
        Name of the acquisition directory (e.g.
        ``acq_20260911_232919_986789057``), or a full path to it. Glob patterns
        are allowed and all matching acquisitions are included.
    data_root : str
        Directory holding the acquisitions, used if `acquisition` is not a full
        path. Default is the pathfinder subset stream.
    file_range : list, optional
        ``[start, stop]`` to only use a range of the (sorted) files.
    """

    acquisition = config.Property(proptype=str)
    data_root = config.Property(proptype=str, default=DEFAULT_DATA_ROOT)
    file_range = config.Property(proptype=list, default=None)

    def setup(self):
        """Find the files.

        Returns
        -------
        files : list
            The files, sorted by their absolute file index.
        """
        files = None
        if self.comm.rank == 0:
            acq = self.acquisition
            if not os.path.isabs(acq):
                acq = os.path.join(self.data_root, acq)
            files = [f for d in sorted(glob.glob(acq)) for f in glob.glob(f"{d}/vis_*.h5")]
            files = sorted(files, key=_abs_file_idx)
            if self.file_range is not None:
                files = files[slice(*self.file_range)]
        files = self.comm.bcast(files, root=0)

        if not files:
            raise RuntimeError(f"No files found for acquisition {self.acquisition}.")

        self.log.info(f"Found {len(files)} files for {self.acquisition}.")
        return files


class LoadCorrDataFiles(tasklib.base.ContainerTask):
    """Load CHORD X-engine data from a file list passed into the setup routine.

    The CHORD analogue of ch_pipeline's ``LoadCorrDataFiles``. The files are read
    with :meth:`chord_util.andata.CorrData.from_acq_h5`, distributed over
    frequency. Unlike CHIME, where each file is output as a separate container, by
    default all the files are concatenated into a single timestream (set
    `files_per_container` to split them into groups of consecutive files).

    Attributes
    ----------
    files_per_container : int, optional
        Number of consecutive files to concatenate into each output. Default is
        all of them, i.e. a single timestream.
    freq_physical : list
        List of physical frequencies in MHz. Given highest priority.
    channel_range : list
        Range of frequency channel indices, either `[start, stop, step]`, `[start,
        stop]`, or `[stop]` is acceptable. Given second priority.
    channel_index : list
        List of frequency channel indices. Given third priority.
    input_type : str
        Which inputs to load: "all", "dish" (drop the RFI monitors) or "rfi".
        Default is "all".
    only_autos : bool
        Only load the autocorrelations (of the inputs selected by `input_type`).
    datasets : list, optional
        Datasets to load (e.g. "vis", "flags/vis_weight"). Default is those of
        :data:`chord_util.andata.CORR_DATASETS` and
        :data:`chord_util.andata.DERIVED_DATASETS` in the files, except
        :data:`chord_util.andata.PLACEHOLDER_DATASETS`.
    exclude_datasets : list
        Datasets not to load, e.g. ``[eval, evec, erms]``. Default is none.
    apply_gain : bool
        Divide out the F-engine digital gains. Default is True.
    normalise_gain : bool
        Normalise the digital gains by the median dish gain before dividing them
        out. Default is True.
    use_draco_container : bool
        Output a :class:`~chord_pipeline.core.container.CHORDTimeStream` rather
        than a :class:`~chord_util.andata.CorrData`. Default is True.
    """

    files_per_container = config.Property(proptype=int, default=None)

    freq_physical = config.Property(proptype=list, default=[])
    channel_range = config.Property(proptype=list, default=[])
    channel_index = config.Property(proptype=list, default=[])

    input_type = config.enum(["all", "dish", "rfi"], default="all")
    only_autos = config.Property(proptype=bool, default=False)

    datasets = config.Property(proptype=list, default=None)
    exclude_datasets = config.Property(proptype=list, default=[])

    apply_gain = config.Property(proptype=bool, default=True)
    normalise_gain = config.Property(proptype=bool, default=True)

    use_draco_container = config.Property(proptype=bool, default=True)

    files = None

    _group_ptr = 0

    def setup(self, files):
        """Set the list of files to load.

        Parameters
        ----------
        files : list
            List of X-engine file paths.
        """
        if not isinstance(files, list | tuple):
            raise RuntimeError("Argument must be list of files.")

        self.files = list(files)
        n = self.files_per_container or len(self.files)
        self._groups = [self.files[i : i + n] for i in range(0, len(self.files), n)]

        # Read the frequency, product and input layout from the first file, to set
        # up the selections
        layout = None
        if self.comm.rank == 0:
            with h5py.File(self.files[0], "r") as fh:
                layout = {
                    "freq": fh["index_map/freq"]["centre"][:],
                    "prod": fh["index_map/prod"][:],
                    "type": fh["index_map/type"][:],
                }
        layout = self.comm.bcast(layout, root=0)

        self._sel = {
            "freq_sel": self._freq_sel(layout["freq"]),
            "input_sel": self._input_sel(layout["type"]),
            "prod_sel": self._prod_sel(layout["prod"], layout["type"]),
        }
        if self._sel["prod_sel"] is not None:
            self._sel["input_sel"] = None

    def process(self):
        """Load in the next group of files.

        Returns
        -------
        ts : CHORDTimeStream or andata.CorrData
            The timestream. Return type depends on the value of
            `use_draco_container`.
        """
        if len(self._groups) == self._group_ptr:
            raise exceptions.PipelineStopIteration

        # Collect garbage to remove any prior CorrData objects
        gc.collect()

        # Fetch the next group of files
        group = self._groups[self._group_ptr]
        self._group_ptr += 1

        self.log.info(
            f"Reading group {self._group_ptr} of {len(self._groups)} "
            f"({len(group)} files, {os.path.basename(group[0])} ... "
            f"{os.path.basename(group[-1])})."
        )

        ts = andata.CorrData.from_acq_h5(
            group,
            datasets=self._datasets(),
            apply_gain=self.apply_gain,
            normalise_gain=self.normalise_gain,
            distributed=True,
            comm=self.comm,
            **self._sel,
        )

        # Tag with the acquisition name, plus the file range if the acquisition
        # was split up
        tag = ts.attrs["acquisition"]
        if len(self._groups) > 1:
            tag += f"_{_abs_file_idx(group[0])}-{_abs_file_idx(group[-1])}"
        ts.attrs["tag"] = tag

        # Store the name of the first file
        ts.attrs["filename"] = group[0]

        self.log.info(
            f"Loaded {ts.vis.global_shape} (freq, prod, time) visibilities, "
            f"crs_board_remap from {ts.attrs['crs_board_remap_source']}."
        )

        # Return timestream
        if self.use_draco_container:
            ts = CHORDTimeStream.from_corrdata(ts)

        return ts

    def _freq_sel(self, freq):
        """The frequency selection, following ch_pipeline's priorities."""
        if self.freq_physical:
            return sorted({int(np.argmin(np.abs(freq - f))) for f in self.freq_physical})
        if self.channel_range and (len(self.channel_range) <= 3):
            return slice(*self.channel_range)
        if self.channel_index:
            return list(self.channel_index)
        return None

    def _input_sel(self, input_type):
        """Indices of the inputs of the requested type, or None for all."""
        if self.input_type == "all":
            return None
        wanted = (
            andata.INPUT_TYPE_DISH if self.input_type == "dish" else andata.INPUT_TYPE_RFI
        )
        return np.flatnonzero(input_type == wanted)

    def _prod_sel(self, prod, input_type):
        """Indices of the autocorrelations (of the selected inputs), if requested."""
        if not self.only_autos:
            return None
        sel = prod["input_a"] == prod["input_b"]
        inputs = self._input_sel(input_type)
        if inputs is not None:
            sel &= np.isin(prod["input_a"], inputs)
        return np.flatnonzero(sel)

    def _datasets(self):
        """The datasets to load."""
        if self.datasets is not None:
            dsets = list(self.datasets)
        else:
            dsets = [
                d
                for d in list(andata.CORR_DATASETS) + list(andata.DERIVED_DATASETS)
                if d not in andata.PLACEHOLDER_DATASETS
            ]
        exclude = set(self.exclude_datasets)
        return [d for d in dsets if d not in exclude and d.split("/")[-1] not in exclude]
