"""Shared fixtures: synthetic X-engine acquisitions (see xengine_testdata.py)."""

import pathlib

import numpy as np
import pytest
from caput.util import mpitools
from drift.core.telescope import TransitTelescope

import xengine_testdata


@pytest.fixture(scope="session")
def acq(tmp_path_factory):
    """A synthetic acquisition, written once on rank 0 and shared with all ranks.

    Returns
    -------
    root : pathlib.Path
        Directory holding the acquisition directory.
    files : list of str
        The X-engine files.
    truth : dict
        The data they encode, see `xengine_testdata.make_xengine_files`.
    """
    out = None
    if mpitools.rank0:
        root = tmp_path_factory.mktemp("xengine")
        files, truth = xengine_testdata.make_xengine_files(root)
        out = (pathlib.Path(root), files, truth)
    out = mpitools.bcast(out, root=0)
    mpitools.barrier()
    return out


@pytest.fixture
def mpi_tmp_path(tmp_path_factory):
    """A temporary directory shared by all ranks."""
    dirname = None
    if mpitools.rank0:
        dirname = str(tmp_path_factory.mktemp("mpi"))
    return pathlib.Path(mpitools.bcast(dirname, root=0))


class StandInTelescope(TransitTelescope):
    """Just enough of a telescope for draco's ComputeSystemSensitivity.

    There is no telescope model for the pathfinder yet. Note that draco reads
    `cylinder_width`, which dish-array telescopes don't define.
    """

    input_index = None
    polarisation = None
    feedpositions = None
    feedmask = None
    cylinder_width = 0.0
    beamclass = None

    def __init__(self, ts):
        ninput = len(ts.input)
        self.input_index = np.array(ts.input["chan_id"], dtype=[("chan_id", "<u2")])
        self.polarisation = np.where(ts["input_info/pol"][:] == 0, "X", "Y")
        self.feedpositions = ts["input_info/feed_positions_m"][:, :2]
        self.feedmask = np.ones((ninput, ninput), dtype=bool)

    def beam(self, feed, freq):
        raise NotImplementedError

    def _transfer_single(self, *args):
        raise NotImplementedError

    @property
    def u_width(self):
        return 6.0

    @property
    def v_width(self):
        return 6.0
